"""Learned prior-guided refinement with ROI-based feature fusion."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .config import RefinementConfig
from .io_utils import IMAGE_EXTENSIONS, ensure_dir, resolve_path, write_json


def canonical_stem(path_or_name: str | Path) -> str:
    stem = Path(str(path_or_name)).stem
    for suffix in ["_sr", "_lr", "_gt", "_prior", "_final", "_x2", "_x3", "_x4"]:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    return stem


def read_gray_float(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return image.astype(np.float32) / 255.0


def save_gray_float(path: str | Path, image: np.ndarray) -> str:
    array = np.clip(image, 0.0, 1.0)
    output = (array * 255.0).round().astype(np.uint8)
    cv2.imwrite(str(path), output)
    return str(Path(path).resolve())


def load_json(path: Path) -> dict | list:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def resize_like(image: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    target_h, target_w = shape
    if image.shape[:2] == (target_h, target_w):
        return image
    return cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_CUBIC)


def normalize_boxes(
    boxes: list[list[int]],
    source_shape: tuple[int, int],
    target_shape: tuple[int, int],
) -> list[list[int]]:
    src_h, src_w = source_shape
    dst_h, dst_w = target_shape
    scale_x = float(dst_w) / max(1.0, float(src_w))
    scale_y = float(dst_h) / max(1.0, float(src_h))
    normalized: list[list[int]] = []
    for box in boxes:
        x, y, w, h = [int(v) for v in box]
        x = int(round(x * scale_x))
        y = int(round(y * scale_y))
        w = max(1, int(round(w * scale_x)))
        h = max(1, int(round(h * scale_y)))
        if x >= dst_w or y >= dst_h:
            continue
        w = min(w, dst_w - x)
        h = min(h, dst_h - y)
        if w > 0 and h > 0:
            normalized.append([x, y, w, h])
    return normalized


def make_labeled_strip(panels: list[tuple[str, np.ndarray]]) -> np.ndarray:
    if not panels:
        return np.zeros((32, 32), dtype=np.uint8)
    panel_h, panel_w = panels[0][1].shape[:2]
    pad = 8
    label_h = 40
    total_w = pad + len(panels) * (panel_w + pad)
    total_h = pad + panel_h + label_h + pad
    canvas = np.zeros((total_h, total_w), dtype=np.uint8)
    for index, (label, image) in enumerate(panels):
        x0 = pad + index * (panel_w + pad)
        y0 = pad
        canvas[y0:y0 + panel_h, x0:x0 + panel_w] = (np.clip(image, 0.0, 1.0) * 255.0).round().astype(np.uint8)
        cv2.rectangle(canvas, (x0, y0), (x0 + panel_w, y0 + panel_h), 180, 1)
        text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.95, 2)
        text_x = x0 + max(2, (panel_w - text_size[0]) // 2)
        text_y = y0 + panel_h + 28
        cv2.putText(
            canvas,
            label,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.95,
            255,
            2,
            cv2.LINE_AA,
        )
    return canvas


def find_image_by_stem(root: str | Path, stem: str) -> Path | None:
    root_path = Path(root).expanduser().resolve()
    if not root_path.exists():
        return None
    candidates = sorted(
        path for path in root_path.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    for candidate in candidates:
        if canonical_stem(candidate) == stem:
            return candidate.resolve()
    return None


def resolve_lr_image_path(
    image_result: dict,
    sr_image_path: Path,
    source_map: dict[str, str],
    dataset_roots: list[str] | None = None,
) -> Path | None:
    explicit = image_result.get("lr_image_path") or image_result.get("input_path")
    if explicit:
        candidate = Path(str(explicit)).expanduser()
        if candidate.exists():
            return candidate.resolve()

    mapped = source_map.get(str(sr_image_path.resolve()))
    if mapped:
        candidate = Path(mapped).expanduser()
        if candidate.exists():
            return candidate.resolve()

    stem = canonical_stem(sr_image_path)
    for root in dataset_roots or []:
        candidate = find_image_by_stem(root, stem)
        if candidate is not None:
            return candidate
    return None


def load_super_resolution_source_map(restoration_root: Path) -> dict[str, str]:
    pipeline_root = restoration_root.parent
    candidates: list[Path] = []
    direct_candidate = pipeline_root / "01_super_resolution" / "manifest.json"
    if direct_candidate.exists():
        candidates.append(direct_candidate)
    child_dirs = sorted(pipeline_root.iterdir()) if pipeline_root.exists() else []
    for child in child_dirs:
        candidate = child / "manifest.json"
        if candidate.exists() and candidate not in candidates:
            candidates.append(candidate)

    source_map: dict[str, str] = {}
    for candidate in candidates:
        try:
            payload = load_json(candidate)
        except Exception:
            continue
        if not isinstance(payload, dict) or payload.get("stage") != "super_resolution":
            continue
        for item in payload.get("items", []):
            input_path = item.get("input_path")
            output_path = item.get("output_path")
            if input_path and output_path:
                source_map[str(Path(output_path).expanduser().resolve())] = str(
                    Path(input_path).expanduser().resolve()
                )
    return source_map


def load_restoration_results(results_path: Path) -> list[dict]:
    data = load_json(results_path)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {results_path}")
    return data


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.InstanceNorm2d(channels, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.InstanceNorm2d(channels, affine=True),
        )
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class FeatureEncoder(nn.Module):
    def __init__(self, in_channels: int, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, channels, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            ResidualBlock(channels),
            ResidualBlock(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PriorTransformBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.condition = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels * 2, 3, padding=1),
        )
        self.norm = nn.InstanceNorm2d(channels, affine=False)
        self.delta = nn.Sequential(
            ResidualBlock(channels),
            nn.Conv2d(channels, channels, 3, padding=1),
        )
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 3, channels, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, 1, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        roi_features: torch.Tensor,
        prior_features: torch.Tensor,
        lr_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        condition_input = torch.cat([prior_features, lr_features], dim=1)
        gamma, beta = torch.chunk(self.condition(condition_input), 2, dim=1)
        modulated = self.norm(roi_features) * (1.0 + torch.tanh(gamma)) + beta
        delta = self.delta(modulated)
        gate = self.gate(torch.cat([roi_features, prior_features, lr_features], dim=1))
        return roi_features + gate * delta, gate


def extract_roi_batch(
    feature: torch.Tensor,
    batch_boxes: list[list[list[int]]],
    output_size: int,
) -> tuple[torch.Tensor, list[tuple[int, int, int, int, int]]]:
    rois: list[torch.Tensor] = []
    meta: list[tuple[int, int, int, int, int]] = []
    _, channels, height, width = feature.shape
    for batch_index, boxes in enumerate(batch_boxes):
        for box in boxes:
            x, y, w, h = [int(v) for v in box]
            x1 = max(0, min(width - 1, x))
            y1 = max(0, min(height - 1, y))
            x2 = max(x1 + 1, min(width, x1 + max(1, w)))
            y2 = max(y1 + 1, min(height, y1 + max(1, h)))
            patch = feature[batch_index : batch_index + 1, :, y1:y2, x1:x2]
            if patch.numel() == 0:
                patch = torch.zeros(
                    (1, channels, output_size, output_size),
                    dtype=feature.dtype,
                    device=feature.device,
                )
            else:
                patch = F.interpolate(
                    patch,
                    size=(output_size, output_size),
                    mode="bilinear",
                    align_corners=False,
                )
            rois.append(patch)
            meta.append((batch_index, x1, y1, x2, y2))
    if not rois:
        empty = torch.zeros(
            (0, channels, output_size, output_size),
            dtype=feature.dtype,
            device=feature.device,
        )
        return empty, meta
    return torch.cat(rois, dim=0), meta


def scatter_roi_batch(
    roi_features: torch.Tensor,
    roi_meta: list[tuple[int, int, int, int, int]],
    spatial_size: tuple[int, int],
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    _, channels, _, _ = roi_features.shape
    height, width = spatial_size
    accum = torch.zeros((batch_size, channels, height, width), dtype=roi_features.dtype, device=roi_features.device)
    weight = torch.zeros((batch_size, 1, height, width), dtype=roi_features.dtype, device=roi_features.device)
    for index, (batch_index, x1, y1, x2, y2) in enumerate(roi_meta):
        patch = F.interpolate(
            roi_features[index : index + 1],
            size=(y2 - y1, x2 - x1),
            mode="bilinear",
            align_corners=False,
        )
        accum[batch_index : batch_index + 1, :, y1:y2, x1:x2] += patch
        weight[batch_index : batch_index + 1, :, y1:y2, x1:x2] += 1.0
    return accum, weight


class LearnedPriorFusionNet(nn.Module):
    def __init__(self, base_channels: int = 24, roi_size: int = 48):
        super().__init__()
        self.roi_size = roi_size
        self.sr_encoder = FeatureEncoder(1, base_channels)
        self.prior_encoder = FeatureEncoder(1, base_channels)
        self.lr_encoder = FeatureEncoder(1, base_channels)
        self.base_fuse = nn.Sequential(
            nn.Conv2d(base_channels * 3, base_channels, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            ResidualBlock(base_channels),
        )
        self.prior_transform = PriorTransformBlock(base_channels)
        self.merge = nn.Sequential(
            nn.Conv2d(base_channels * 3 + 1, base_channels * 2, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            ResidualBlock(base_channels * 2),
            nn.Conv2d(base_channels * 2, base_channels, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.final_head = nn.Sequential(
            ResidualBlock(base_channels),
            nn.Conv2d(base_channels, 1, 1),
            nn.Sigmoid(),
        )
        self.confidence_head = nn.Sequential(
            nn.Conv2d(base_channels, base_channels // 2, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels // 2, 1, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        sr_image: torch.Tensor,
        prior_image: torch.Tensor,
        lr_image: torch.Tensor,
        boxes: list[list[list[int]]],
    ) -> dict[str, torch.Tensor]:
        sr_feat = self.sr_encoder(sr_image)
        prior_feat = self.prior_encoder(prior_image)
        lr_feat = self.lr_encoder(lr_image)
        base_feat = self.base_fuse(torch.cat([sr_feat, prior_feat, lr_feat], dim=1))

        roi_feat, roi_meta = extract_roi_batch(base_feat, boxes, self.roi_size)
        prior_roi, _ = extract_roi_batch(prior_feat, boxes, self.roi_size)
        lr_roi, _ = extract_roi_batch(lr_feat, boxes, self.roi_size)

        batch_size, _, height, width = base_feat.shape
        if roi_feat.shape[0] > 0:
            refined_roi, roi_gate = self.prior_transform(roi_feat, prior_roi, lr_roi)
            roi_accum, roi_weight = scatter_roi_batch(refined_roi, roi_meta, (height, width), batch_size)
            gate_accum, gate_weight = scatter_roi_batch(roi_gate, roi_meta, (height, width), batch_size)
            roi_map = roi_accum / roi_weight.clamp(min=1.0)
            gate_map = gate_accum / gate_weight.clamp(min=1.0)
        else:
            roi_map = torch.zeros_like(base_feat)
            gate_map = torch.zeros((batch_size, 1, height, width), dtype=base_feat.dtype, device=base_feat.device)

        merged = self.merge(torch.cat([base_feat, roi_map, prior_feat, gate_map], dim=1))
        final_image = self.final_head(merged)
        confidence = self.confidence_head(merged)
        return {
            "final": final_image,
            "confidence": confidence,
            "roi_gate": gate_map,
        }


@dataclass
class LearnedRefinementBundle:
    model: LearnedPriorFusionNet
    device: torch.device
    checkpoint_path: Path
    checkpoint: dict[str, Any]


def load_learned_refinement_bundle(
    checkpoint_path: str | Path,
    *,
    device_name: str | None = None,
    search_roots: list[str] | None = None,
    base_dir: str | Path | None = None,
) -> LearnedRefinementBundle:
    resolved_checkpoint = resolve_path(
        checkpoint_path,
        search_roots=search_roots,
        base_dir=base_dir,
    )
    checkpoint = torch.load(str(resolved_checkpoint), map_location="cpu")
    if checkpoint.get("checkpoint_type") != "learned_refinement_v1":
        raise RuntimeError(f"Unsupported learned refinement checkpoint: {resolved_checkpoint}")

    model_cfg = checkpoint.get("cfg", {}).get("model", {})
    model = LearnedPriorFusionNet(
        base_channels=int(model_cfg.get("base_channels", 24)),
        roi_size=int(model_cfg.get("roi_size", 48)),
    )
    state_dict = checkpoint.get("model_ema") or checkpoint.get("model")
    if not isinstance(state_dict, dict):
        raise RuntimeError("Learned refinement checkpoint is missing model weights.")
    model.load_state_dict(state_dict)
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.eval().to(device)
    return LearnedRefinementBundle(
        model=model,
        device=device,
        checkpoint_path=resolved_checkpoint,
        checkpoint=checkpoint,
    )


def run_learned_refinement(
    restoration_input: str | Path,
    output_dir: str | Path,
    config: RefinementConfig,
    *,
    search_roots: list[str] | None = None,
    base_dir: str | Path | None = None,
) -> dict[str, Any]:
    if not config.model_path:
        raise ValueError("refinement.model_path is required when refinement.mode='learned'.")

    resolved_input = resolve_path(
        restoration_input,
        search_roots=search_roots,
        base_dir=base_dir,
    )
    if resolved_input.is_dir():
        results_path = resolved_input / "retrieval_results_all.json"
        restoration_root = resolved_input
    else:
        results_path = resolved_input
        restoration_root = resolved_input.parent

    if not results_path.exists():
        raise FileNotFoundError(f"Missing retrieval results: {results_path}")

    bundle = load_learned_refinement_bundle(
        config.model_path,
        device_name=config.device,
        search_roots=search_roots,
        base_dir=base_dir,
    )
    results = load_restoration_results(results_path)
    resolved_output = ensure_dir(output_dir)
    source_map = load_super_resolution_source_map(restoration_root)
    dataset_roots = [config.lr_dir] if getattr(config, "lr_dir", None) else []
    print(f"[learned-refinement] restoration_input={resolved_input}")
    print(f"[learned-refinement] checkpoint={bundle.checkpoint_path}")
    print(f"[learned-refinement] total_images={len(results)} output_dir={resolved_output}")

    items: list[dict[str, Any]] = []
    for index, image_result in enumerate(results, start=1):
        image_name = image_result.get("image_name") or Path(image_result["image_path"]).stem
        print(f"[learned-refinement] ({index}/{len(results)}) image={image_name}")
        image_output_dir = ensure_dir(resolved_output / image_name)

        sr_image_path = resolve_path(
            image_result["image_path"],
            search_roots=search_roots,
            base_dir=base_dir,
        )
        prior_path_text = (
            image_result.get("paths", {})
            .get("reconstructed_ranks", {})
            .get(f"rank_{config.rank}")
        )
        if not prior_path_text:
            prior_path_text = str((restoration_root / image_name / f"reconstructed_rank{config.rank}.png").resolve())
        prior_path = resolve_path(prior_path_text, search_roots=search_roots, base_dir=base_dir)
        lr_image_path = resolve_lr_image_path(
            image_result,
            sr_image_path,
            source_map,
            dataset_roots=dataset_roots,
        )

        sr_image = read_gray_float(sr_image_path)
        prior_image = read_gray_float(prior_path)
        prior_image = resize_like(prior_image, sr_image.shape[:2])
        lr_image = read_gray_float(lr_image_path) if lr_image_path is not None else np.zeros_like(sr_image)
        lr_image = resize_like(lr_image, sr_image.shape[:2])
        boxes = [[int(v) for v in component["bbox"]] for component in image_result.get("components", [])]

        sr_tensor = torch.from_numpy(sr_image).unsqueeze(0).unsqueeze(0).to(bundle.device)
        prior_tensor = torch.from_numpy(prior_image).unsqueeze(0).unsqueeze(0).to(bundle.device)
        lr_tensor = torch.from_numpy(lr_image).unsqueeze(0).unsqueeze(0).to(bundle.device)

        with torch.no_grad():
            prediction = bundle.model(sr_tensor.float(), prior_tensor.float(), lr_tensor.float(), [boxes])
            final_image = prediction["final"].squeeze().detach().float().cpu().numpy()
            confidence = prediction["confidence"].squeeze().detach().float().cpu().numpy()

        final_path = save_gray_float(image_output_dir / "final_refined.png", final_image)
        confidence_path = save_gray_float(image_output_dir / "confidence_map.png", confidence)
        sr_copy_path = save_gray_float(image_output_dir / "sr_input.png", sr_image)
        prior_copy_path = save_gray_float(image_output_dir / "gan_prior.png", prior_image)
        lr_copy_path = save_gray_float(image_output_dir / "lr_input.png", lr_image) if lr_image_path is not None else None
        overview = make_labeled_strip(
            [
                ("LR", lr_image),
                ("SR", sr_image),
                ("PRIOR", prior_image),
                ("FINAL", final_image),
            ]
        )
        overview_path = save_gray_float(image_output_dir / "comparison_strip.png", overview.astype(np.float32) / 255.0)

        item_manifest = {
            "image_name": image_name,
            "lr_image_path": str(lr_image_path.resolve()) if lr_image_path is not None else None,
            "sr_image_path": str(sr_image_path.resolve()),
            "prior_image_path": str(prior_path.resolve()),
            "final_image_path": final_path,
            "confidence_map_path": confidence_path,
            "comparison_strip_path": overview_path,
            "lr_copy_path": lr_copy_path,
            "sr_copy_path": sr_copy_path,
            "gan_prior_copy_path": prior_copy_path,
            "num_components": len(boxes),
        }
        write_json(item_manifest, image_output_dir / "refinement_results.json")
        items.append(item_manifest)

    manifest = {
        "stage": "learned_refinement",
        "checkpoint_path": str(bundle.checkpoint_path.resolve()),
        "restoration_input": str(resolved_input),
        "restoration_results_path": str(results_path.resolve()),
        "output_dir": str(resolved_output),
        "count": len(items),
        "items": items,
    }
    write_json(manifest, resolved_output / "manifest.json")
    return manifest
