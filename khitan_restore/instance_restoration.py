"""Instance retrieval plus conditional restoration for component recovery."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch import nn

from .config import RestorationConfig
from .io_utils import ensure_dir, resolve_path, write_json
from .legacy_loader import load_retrieval_module


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
        self.conv = ConvBlock(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = nn.functional.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class ConditionalComponentRestorer(nn.Module):
    def __init__(self, in_channels: int = 4, base_channels: int = 32):
        super().__init__()
        self.enc1 = ConvBlock(in_channels, base_channels)
        self.enc2 = ConvBlock(base_channels, base_channels * 2)
        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4)
        self.enc4 = ConvBlock(base_channels * 4, base_channels * 8)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(base_channels * 8, base_channels * 16)
        self.up4 = UpBlock(base_channels * 16, base_channels * 8, base_channels * 8)
        self.up3 = UpBlock(base_channels * 8, base_channels * 4, base_channels * 4)
        self.up2 = UpBlock(base_channels * 4, base_channels * 2, base_channels * 2)
        self.up1 = UpBlock(base_channels * 2, base_channels, base_channels)
        self.out = nn.Sequential(
            nn.Conv2d(base_channels, base_channels // 2, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels // 2, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s1 = self.enc1(x)
        s2 = self.enc2(self.pool(s1))
        s3 = self.enc3(self.pool(s2))
        s4 = self.enc4(self.pool(s3))
        x = self.bottleneck(self.pool(s4))
        x = self.up4(x, s4)
        x = self.up3(x, s3)
        x = self.up2(x, s2)
        x = self.up1(x, s1)
        return self.out(x)


@dataclass
class InstanceRestorationBundle:
    module: Any
    device: torch.device
    checkpoint_path: Path
    checkpoint: dict[str, Any]
    restorer: ConditionalComponentRestorer
    bank_meta: list[dict[str, Any]]
    annotations_path: Path
    output_size: int
    topk: int


def _read_annotations(path: str | Path) -> dict[str, Any]:
    with open(Path(path).expanduser().resolve(), "r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_gray(path: str | Path, size: int | None = None, *, nearest: bool = False) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    if size is not None and image.shape[:2] != (size, size):
        interpolation = cv2.INTER_NEAREST if nearest else cv2.INTER_CUBIC
        image = cv2.resize(image, (size, size), interpolation=interpolation)
    return image


def _build_instance_bank(
    annotations_path: str | Path,
    *,
    splits: set[str] | None = None,
    output_size: int = 128,
    feature_mask_size: int = 64,
) -> tuple[Any, list[dict[str, Any]]]:
    module = load_retrieval_module()
    payload = _read_annotations(annotations_path)
    raw_items = payload.get("items", [])
    if splits:
        raw_items = [item for item in raw_items if item.get("split") in splits]
    if not raw_items:
        raise RuntimeError(f"No component instances found in {annotations_path}")

    bank_meta: list[dict[str, Any]] = []
    for index, item in enumerate(raw_items):
        mask_image = _read_gray(item["mask_path"], output_size, nearest=True)
        soft_image = _read_gray(item["soft_path"], output_size, nearest=False)
        norm_mask = module.normalize_mask(mask_image, size=feature_mask_size)
        ys, xs = np.where(norm_mask > 0)
        if len(xs) > 0 and len(ys) > 0:
            width = xs.max() - xs.min() + 1
            height = ys.max() - ys.min() + 1
            aspect = float(width) / max(1.0, float(height))
        else:
            aspect = 1.0

        bank_meta.append(
            {
                "comp_id": index,
                "img": soft_image,
                "mask": norm_mask,
                "fill": module.fill_ratio(norm_mask),
                "aspect": aspect,
                "blank": module.fill_ratio(norm_mask) < 0.003,
                "instance_name": item["name"],
                "component_key": item.get("component_key"),
                "source_comp_id": item.get("comp_id"),
                "mask_path": item["mask_path"],
                "soft_path": item["soft_path"],
                "soft_image": soft_image,
                "mask_image": mask_image,
                "bbox": item.get("bbox"),
                "split": item.get("split"),
            }
        )
    return module, bank_meta


def _resolve_instance_annotations(
    checkpoint: dict[str, Any],
    *,
    search_roots: list[str] | None = None,
    base_dir: str | Path | None = None,
) -> Path:
    cfg = checkpoint.get("cfg", {})
    data_cfg = cfg.get("data", {})
    annotations_path = data_cfg.get("annotations_path")
    if not annotations_path:
        raise RuntimeError("Instance restoration checkpoint is missing cfg.data.annotations_path.")
    return resolve_path(
        annotations_path,
        search_roots=search_roots,
        base_dir=base_dir,
    )


def _candidate_with_metadata(candidate: dict[str, Any], bank_meta: list[dict[str, Any]]) -> dict[str, Any]:
    bank_item = bank_meta[int(candidate["comp_id"])]
    enriched = dict(candidate)
    enriched["instance_name"] = bank_item["instance_name"]
    enriched["component_key"] = bank_item.get("component_key")
    enriched["source_comp_id"] = bank_item.get("source_comp_id")
    enriched["soft_path"] = bank_item["soft_path"]
    enriched["mask_path"] = bank_item["mask_path"]
    return enriched


def _component_tensor(image: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(image.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)


def _crop_foreground(image: np.ndarray) -> np.ndarray:
    if image is None or image.size == 0:
        return image
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ys, xs = np.where(binary > 0)
    if len(xs) == 0 or len(ys) == 0:
        return gray
    x1, x2 = xs.min(), xs.max() + 1
    y1, y2 = ys.min(), ys.max() + 1
    return gray[y1:y2, x1:x2].copy()


def _paste_component_soft(canvas: np.ndarray, comp_img: np.ndarray, bbox: list[int]) -> np.ndarray:
    x, y, w, h = [int(value) for value in bbox]
    if w <= 0 or h <= 0:
        return canvas

    fg_crop = _crop_foreground(comp_img)
    if fg_crop is None or fg_crop.size == 0:
        return canvas

    resized = cv2.resize(fg_crop, (w, h), interpolation=cv2.INTER_CUBIC).astype(np.float32) / 255.0
    _, binary = cv2.threshold(fg_crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    matte = cv2.resize(binary.astype(np.float32) / 255.0, (w, h), interpolation=cv2.INTER_LINEAR)
    matte = cv2.GaussianBlur(matte, (0, 0), 0.7)
    blended = np.clip(resized * np.clip(matte, 0.0, 1.0), 0.0, 1.0)

    y2 = min(canvas.shape[0], y + h)
    x2 = min(canvas.shape[1], x + w)
    if y2 <= y or x2 <= x:
        return canvas

    blended = blended[: y2 - y, : x2 - x]
    region = canvas[y:y2, x:x2].astype(np.float32) / 255.0
    region = np.maximum(region, blended)
    canvas[y:y2, x:x2] = (np.clip(region, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    return canvas


def _build_model_input(
    query_patch: np.ndarray,
    query_mask: np.ndarray,
    prior_soft: np.ndarray,
    prior_mask: np.ndarray,
    *,
    size: int,
) -> torch.Tensor:
    if query_patch.shape[:2] != (size, size):
        query_patch = cv2.resize(query_patch, (size, size), interpolation=cv2.INTER_CUBIC)
    if query_mask.shape[:2] != (size, size):
        query_mask = cv2.resize(query_mask, (size, size), interpolation=cv2.INTER_NEAREST)
    if prior_soft.shape[:2] != (size, size):
        prior_soft = cv2.resize(prior_soft, (size, size), interpolation=cv2.INTER_CUBIC)
    if prior_mask.shape[:2] != (size, size):
        prior_mask = cv2.resize(prior_mask, (size, size), interpolation=cv2.INTER_NEAREST)
    stacked = np.stack(
        [
            query_patch.astype(np.float32) / 255.0,
            query_mask.astype(np.float32) / 255.0,
            prior_soft.astype(np.float32) / 255.0,
            prior_mask.astype(np.float32) / 255.0,
        ],
        axis=0,
    )
    return torch.from_numpy(stacked).unsqueeze(0)


def load_instance_restoration_bundle(
    checkpoint_path: str | Path,
    config: RestorationConfig,
    *,
    search_roots: list[str] | None = None,
    base_dir: str | Path | None = None,
) -> InstanceRestorationBundle:
    ckpt_path = resolve_path(checkpoint_path, search_roots=search_roots, base_dir=base_dir)
    checkpoint = torch.load(str(ckpt_path), map_location="cpu")
    checkpoint_type = checkpoint.get("checkpoint_type")
    if checkpoint_type != "instance_retrieval_restorer_v1":
        raise RuntimeError(f"Unsupported instance restoration checkpoint type: {checkpoint_type}")

    device_name = config.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    annotations_path = _resolve_instance_annotations(
        checkpoint,
        search_roots=search_roots,
        base_dir=base_dir,
    )
    model_cfg = checkpoint.get("cfg", {}).get("model", {})
    output_size = int(model_cfg.get("output_size", 128))
    topk = int(checkpoint.get("cfg", {}).get("inference", {}).get("topk", config.topk))
    bank_splits_raw = checkpoint.get("cfg", {}).get("inference", {}).get("bank_splits")
    bank_splits = set(bank_splits_raw) if bank_splits_raw else None
    module, bank_meta = _build_instance_bank(
        annotations_path,
        splits=bank_splits,
        output_size=output_size,
    )

    restorer = ConditionalComponentRestorer(
        in_channels=int(model_cfg.get("input_channels", 4)),
        base_channels=int(model_cfg.get("base_channels", 32)),
    )
    state_dict = checkpoint.get("restorer_ema") or checkpoint.get("restorer")
    if not isinstance(state_dict, dict):
        raise RuntimeError("Instance restoration checkpoint is missing restorer weights.")
    restorer.load_state_dict(state_dict)
    restorer.eval().to(device)

    print(f"[restoration] device={device} ckpt={ckpt_path}")
    print(f"[restoration] instance_annotations={annotations_path}")
    print(f"[restoration] instance_bank_count={len(bank_meta)}")
    return InstanceRestorationBundle(
        module=module,
        device=device,
        checkpoint_path=ckpt_path,
        checkpoint=checkpoint,
        restorer=restorer,
        bank_meta=bank_meta,
        annotations_path=annotations_path,
        output_size=output_size,
        topk=topk,
    )


def _save_reconstructed_views_from_patches(
    module: Any,
    item: dict[str, Any],
    components_out: list[dict[str, Any]],
    out_dir: str | Path,
) -> None:
    gray = module.read_gray_image(item["image_path"])
    if gray is None:
        raise ValueError(f"Could not read original image: {item['image_path']}")

    height, width = gray.shape[:2]
    recon = np.zeros((height, width), dtype=np.uint8)
    for component in sorted(
        components_out,
        key=lambda value: (value["bbox"][3] * value["bbox"][2], value["bbox"][1], value["bbox"][0]),
    ):
        restored_path = component.get("restored_patch_path")
        if not restored_path:
            continue
        restored_patch = module.read_gray_image(restored_path)
        if restored_patch is None:
            continue
        recon = _paste_component_soft(recon, restored_patch, component["bbox"])

    module.save_gray(str(Path(out_dir) / "reconstructed_rank1.png"), recon)
    overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    recon_col = cv2.cvtColor(recon, cv2.COLOR_GRAY2BGR)
    overlay = cv2.addWeighted(overlay, 0.65, recon_col, 0.85, 0)
    box_vis = module.draw_bbox_overlay(gray, components_out)
    cv2.imwrite(str(Path(out_dir) / "reconstructed_overlay.png"), overlay)
    cv2.imwrite(str(Path(out_dir) / "original_with_boxes.png"), box_vis)


def run_instance_component_restoration(
    segment_json_path: str | Path,
    output_dir: str | Path,
    config: RestorationConfig,
    *,
    search_roots: list[str] | None = None,
    base_dir: str | Path | None = None,
) -> dict:
    resolved_segment_json = resolve_path(segment_json_path, search_roots=search_roots, base_dir=base_dir)
    resolved_output = ensure_dir(output_dir)
    print(f"[restoration] segment_json={resolved_segment_json}")
    bundle = load_instance_restoration_bundle(
        config.ckpt_path,
        config,
        search_roots=search_roots,
        base_dir=base_dir,
    )
    module = bundle.module
    segmentation_items = module.load_segmentation_results(str(resolved_segment_json))
    segmentation_items = module.filter_segmentation_items(
        segmentation_items,
        image_name=config.image_name,
        image_path=config.image_path,
    )
    if not segmentation_items:
        raise RuntimeError("No segmentation items matched the restoration filters.")
    print(f"[restoration] total_images={len(segmentation_items)} output_dir={resolved_output}")

    preview_count = min(64, len(bundle.bank_meta))
    if config.save_bank_preview and preview_count > 0:
        preview_tensor = torch.stack(
            [
                _component_tensor(bundle.bank_meta[index]["soft_image"]).squeeze(0)
                for index in range(preview_count)
            ],
            dim=0,
        )
        module.save_grid(
            preview_tensor,
            str(resolved_output / "prototype_bank_preview.png"),
            nrow=8,
            title="Instance bank preview",
        )

    all_results = []
    manifest_items = []
    for image_index, item in enumerate(segmentation_items, start=1):
        image_name = item.get("image_name") or Path(item.get("image_path", "image")).stem
        print(f"[restoration] ({image_index}/{len(segmentation_items)}) image={image_name}")
        image_dir = ensure_dir(resolved_output / image_name)
        component_root = ensure_dir(image_dir / "components")
        components_out = []
        image_shape = item.get("image_shape")
        components = item.get("components", [])
        print(f"[restoration] image={image_name} components={len(components)}")

        for component_index, component in enumerate(components, start=1):
            idx = int(component.get("index", len(components_out)))
            bbox = [int(value) for value in component["bbox"]]
            print(
                f"[restoration] image={image_name} component=({component_index}/{len(components)}) "
                f"bbox={bbox}"
            )
            query_patch_vis, query_mask, query_source = module.load_query_patch_and_mask(
                component,
                item,
                query_size=bundle.output_size,
            )
            topk_items = module.score_query_to_bank(
                query_mask,
                bbox,
                bundle.bank_meta,
                image_shape=image_shape,
                topk=max(1, min(config.topk, bundle.topk)),
            )
            topk_items = [_candidate_with_metadata(candidate, bundle.bank_meta) for candidate in topk_items]

            component_dir = ensure_dir(component_root / f"comp_{idx:03d}")
            module.save_gray(str(component_dir / "query_patch.png"), query_patch_vis)
            module.save_gray(str(component_dir / "query_mask_norm64.png"), query_mask)
            for rank, candidate in enumerate(topk_items, start=1):
                instance_index = int(candidate["comp_id"])
                module.save_gray(
                    str(component_dir / f"top{rank}_instance_{instance_index:04d}.png"),
                    bundle.bank_meta[instance_index]["img"],
                )

            module.render_component_topk_board(
                query_patch_vis,
                topk_items,
                bundle.bank_meta,
                str(component_dir / "topk_board.png"),
            )

            best_candidate = topk_items[0]
            best_index = int(best_candidate["comp_id"])
            prior_soft = bundle.bank_meta[best_index]["soft_image"]
            prior_mask = bundle.bank_meta[best_index]["mask_image"]
            model_input = _build_model_input(
                query_patch_vis,
                cv2.resize(query_mask, (bundle.output_size, bundle.output_size), interpolation=cv2.INTER_NEAREST),
                prior_soft,
                prior_mask,
                size=bundle.output_size,
            ).to(bundle.device)
            with torch.no_grad():
                restored = bundle.restorer(model_input).squeeze().detach().float().cpu().clamp_(0.0, 1.0).numpy()
            restored_uint8 = (restored * 255.0).round().astype(np.uint8)
            restored_path = component_dir / "restored_patch.png"
            module.save_gray(str(restored_path), restored_uint8)

            query_meta = {
                "bbox": bbox,
                "source": query_source,
            }
            if image_shape is not None and len(image_shape) >= 2:
                image_h = int(image_shape[0])
                image_w = int(image_shape[1])
                x, y, w, h = bbox
                query_meta["center_norm"] = [
                    float((x + 0.5 * w) / max(1.0, image_w)),
                    float((y + 0.5 * h) / max(1.0, image_h)),
                ]
                query_meta["area_ratio"] = float((w * h) / max(1.0, image_w * image_h))
                query_meta["aspect"] = float(w / max(1.0, h))

            components_out.append(
                {
                    "index": idx,
                    "bbox": bbox,
                    "saved_patch": component.get("saved_patch"),
                    "query": query_meta,
                    "topk": topk_items,
                    "restored_patch_path": str(restored_path.resolve()),
                    "retrieval_mode": "instance",
                }
            )

        _save_reconstructed_views_from_patches(module, item, components_out, image_dir)
        image_result = {
            "image_name": image_name,
            "image_path": item.get("image_path"),
            "image_shape": item.get("image_shape"),
            "num_components": len(components_out),
            "bank_mode": "instance_retrieval_restorer",
            "uses_mapping_net": False,
            "paths": {
                "image_dir": str(image_dir.resolve()),
                "retrieval_results": str((image_dir / "retrieval_results.json").resolve()),
                "reconstructed_ranks": {
                    "rank_1": str((image_dir / "reconstructed_rank1.png").resolve()),
                },
                "reconstructed_overlay": str((image_dir / "reconstructed_overlay.png").resolve()),
                "original_with_boxes": str((image_dir / "original_with_boxes.png").resolve()),
            },
            "components": components_out,
        }
        all_results.append(image_result)
        write_json(image_result, image_dir / "retrieval_results.json")
        manifest_items.append(
            {
                "image_name": image_name,
                "image_path": item.get("image_path"),
                "image_dir": str(image_dir.resolve()),
                "retrieval_results": str((image_dir / "retrieval_results.json").resolve()),
                "reconstructed_rank1": str((image_dir / "reconstructed_rank1.png").resolve()),
            }
        )

    print(f"[restoration] completed count={len(all_results)}")
    all_results_path = write_json(all_results, resolved_output / "retrieval_results_all.json")
    manifest = {
        "stage": "restoration",
        "device": str(bundle.device),
        "segment_json": str(resolved_segment_json),
        "output_dir": str(resolved_output),
        "checkpoint_path": str(bundle.checkpoint_path.resolve()),
        "config_path": str(bundle.annotations_path.resolve()),
        "generator_name": "instance_retrieval_restorer",
        "num_components": len(bundle.bank_meta),
        "count": len(all_results),
        "results_path": str(all_results_path),
        "items": manifest_items,
    }
    write_json(manifest, resolved_output / "manifest.json")
    return manifest
