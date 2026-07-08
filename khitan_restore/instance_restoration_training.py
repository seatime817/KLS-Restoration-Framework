"""Training for instance retrieval plus conditional component restoration."""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .component_annotations import build_component_annotations, write_component_annotations
from .instance_restoration import (
    ConditionalComponentRestorer,
    _build_instance_bank,
    _build_model_input,
)
from .io_utils import ensure_dir, load_structured_file, resolve_path, write_json
from .legacy_loader import load_retrieval_module


@dataclass
class InstanceRestorationDataConfig:
    components_root: str = "dataset/components"
    annotations_path: str = "dataset/components/component_annotations.json"
    vocab_path: str = "dataset/components/component_vocab.json"
    auto_build_annotations: bool = True
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    train_split: str = "train"
    val_split: str = "val"
    output_size: int = 128
    feature_mask_size: int = 64
    neighbor_topk: int = 8
    neighbor_random_topn: int = 3


@dataclass
class InstanceRestorationModelConfig:
    input_channels: int = 4
    base_channels: int = 32
    output_size: int = 128


@dataclass
class InstanceRestorationLossConfig:
    lambda_l1: float = 5.0
    lambda_bce: float = 2.0
    lambda_ssim: float = 1.2
    lambda_edge: float = 0.8
    lambda_mask_consistency: float = 1.0


@dataclass
class InstanceRestorationTrainConfig:
    device: str | None = None
    seed: int = 1234
    batch_size: int = 12
    val_batch_size: int = 12
    num_workers: int = 0
    max_epochs: int = 200
    max_steps: int | None = None
    learning_rate: float = 2.0e-4
    weight_decay: float = 1.0e-5
    grad_clip: float = 2.5
    ema_decay: float = 0.999
    log_every: int = 25
    validate_every: int = 100
    save_every: int = 250
    preview_count: int = 12
    resume_path: str | None = None


@dataclass
class InstanceRestorationOutputConfig:
    root_dir: str = "runs/instance_restoration_training"
    checkpoints_dir: str = "checkpoints"
    previews_dir: str = "previews"
    metrics_name: str = "metrics_history.json"
    best_checkpoint_name: str = "instance_restoration_best.pth"
    last_checkpoint_name: str = "instance_restoration_last.pth"
    resolved_config_name: str = "resolved_instance_restoration_config.json"


@dataclass
class InstanceRestorationTrainingConfig:
    data: InstanceRestorationDataConfig = field(default_factory=InstanceRestorationDataConfig)
    model: InstanceRestorationModelConfig = field(default_factory=InstanceRestorationModelConfig)
    loss: InstanceRestorationLossConfig = field(default_factory=InstanceRestorationLossConfig)
    train: InstanceRestorationTrainConfig = field(default_factory=InstanceRestorationTrainConfig)
    output: InstanceRestorationOutputConfig = field(default_factory=InstanceRestorationOutputConfig)
    config_base_dir: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_instance_restoration_training_config(
    config_path: str | Path | None = None,
) -> InstanceRestorationTrainingConfig:
    defaults = InstanceRestorationTrainingConfig().to_dict()
    if config_path is None:
        return InstanceRestorationTrainingConfig()

    resolved = Path(config_path).expanduser().resolve()
    raw = load_structured_file(resolved)
    merged = _deep_merge(defaults, raw)
    return InstanceRestorationTrainingConfig(
        data=InstanceRestorationDataConfig(**merged.get("data", {})),
        model=InstanceRestorationModelConfig(**merged.get("model", {})),
        loss=InstanceRestorationLossConfig(**merged.get("loss", {})),
        train=InstanceRestorationTrainConfig(**merged.get("train", {})),
        output=InstanceRestorationOutputConfig(**merged.get("output", {})),
        config_base_dir=str(resolved.parent),
    )


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_annotation_files(
    config: InstanceRestorationTrainingConfig,
) -> tuple[Path, Path, Path]:
    search_roots = [config.config_base_dir] if config.config_base_dir else None
    components_root = resolve_path(
        config.data.components_root,
        search_roots=search_roots,
        base_dir=config.config_base_dir,
    )
    annotations_path = resolve_path(
        config.data.annotations_path,
        search_roots=search_roots,
        base_dir=config.config_base_dir,
        allow_missing=True,
    )
    vocab_path = resolve_path(
        config.data.vocab_path,
        search_roots=search_roots,
        base_dir=config.config_base_dir,
        allow_missing=True,
    )
    if config.data.auto_build_annotations and (not annotations_path.exists() or not vocab_path.exists()):
        annotations, vocab = build_component_annotations(
            components_root,
            train_ratio=config.data.train_ratio,
            val_ratio=config.data.val_ratio,
            test_ratio=config.data.test_ratio,
        )
        annotations_path, vocab_path = write_component_annotations(annotations, vocab, components_root)
    return components_root, annotations_path, vocab_path


def _read_annotations(path: str | Path) -> dict[str, Any]:
    with open(Path(path).expanduser().resolve(), "r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_gray(path: str | Path, size: int, *, nearest: bool) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    if image.shape[:2] != (size, size):
        interpolation = cv2.INTER_NEAREST if nearest else cv2.INTER_CUBIC
        image = cv2.resize(image, (size, size), interpolation=interpolation)
    return image


def _degrade_component(image: np.ndarray, rng: random.Random) -> np.ndarray:
    degraded = image.astype(np.float32).copy()
    size = degraded.shape[0]

    if rng.random() < 0.85:
        scale = rng.uniform(0.35, 0.80)
        small = max(16, int(round(size * scale)))
        degraded = cv2.resize(degraded, (small, small), interpolation=cv2.INTER_AREA)
        degraded = cv2.resize(degraded, (size, size), interpolation=cv2.INTER_CUBIC)

    if rng.random() < 0.70:
        sigma = rng.uniform(0.6, 2.2)
        degraded = cv2.GaussianBlur(degraded, (0, 0), sigma)

    if rng.random() < 0.55:
        kernel_size = rng.choice([2, 3, 4])
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        if rng.random() < 0.5:
            degraded = cv2.erode(degraded.astype(np.uint8), kernel, iterations=1).astype(np.float32)
        else:
            degraded = cv2.dilate(degraded.astype(np.uint8), kernel, iterations=1).astype(np.float32)

    if rng.random() < 0.55:
        for _ in range(rng.randint(1, 3)):
            x1 = rng.randint(0, size - 1)
            y1 = rng.randint(0, size - 1)
            x2 = min(size, x1 + rng.randint(size // 10, size // 4))
            y2 = min(size, y1 + rng.randint(2, size // 12))
            degraded[y1:y2, x1:x2] *= rng.uniform(0.0, 0.35)

    if rng.random() < 0.45:
        shift_x = rng.randint(-size // 16, size // 16)
        shift_y = rng.randint(-size // 16, size // 16)
        matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        degraded = cv2.warpAffine(
            degraded,
            matrix,
            (size, size),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

    if rng.random() < 0.75:
        noise = np.random.normal(0.0, rng.uniform(4.0, 18.0), degraded.shape).astype(np.float32)
        degraded = degraded + noise

    return np.clip(degraded, 0.0, 255.0).astype(np.uint8)


def _binary_focal_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    pred = pred.clamp(1.0e-6, 1.0 - 1.0e-6)
    pt = pred * target + (1.0 - pred) * (1.0 - target)
    alpha_t = alpha * target + (1.0 - alpha) * (1.0 - target)
    focal_weight = alpha_t * torch.pow(1.0 - pt, gamma)
    bce = -(target * pred.log() + (1.0 - target) * (1.0 - pred).log())
    return (focal_weight * bce).mean()


def _ssim_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    kernel = 7
    padding = kernel // 2
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    mu_x = F.avg_pool2d(pred, kernel, stride=1, padding=padding)
    mu_y = F.avg_pool2d(target, kernel, stride=1, padding=padding)
    sigma_x = F.avg_pool2d(pred * pred, kernel, stride=1, padding=padding) - mu_x * mu_x
    sigma_y = F.avg_pool2d(target * target, kernel, stride=1, padding=padding) - mu_y * mu_y
    sigma_xy = F.avg_pool2d(pred * target, kernel, stride=1, padding=padding) - mu_x * mu_y
    numerator = (2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2)
    denominator = (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2)
    return 1.0 - (numerator / (denominator + 1.0e-6)).mean()


def _sobel_edges(image: torch.Tensor) -> torch.Tensor:
    kernel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=image.device,
        dtype=image.dtype,
    ).view(1, 1, 3, 3)
    kernel_y = torch.tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
        device=image.device,
        dtype=image.dtype,
    ).view(1, 1, 3, 3)
    grad_x = F.conv2d(image, kernel_x, padding=1)
    grad_y = F.conv2d(image, kernel_y, padding=1)
    return torch.sqrt(grad_x.pow(2) + grad_y.pow(2) + 1.0e-6)


class InstanceConditionDataset(Dataset):
    def __init__(
        self,
        *,
        annotations_path: str | Path,
        split: str,
        bank_meta: list[dict[str, Any]],
        output_size: int,
        feature_mask_size: int,
        neighbor_topk: int,
        neighbor_random_topn: int,
        random_neighbor: bool,
        seed: int,
    ):
        self.module = load_retrieval_module()
        payload = _read_annotations(annotations_path)
        self.items = [item for item in payload["items"] if item.get("split") == split]
        if not self.items:
            raise ValueError(f"No items found for split={split}")
        self.bank_meta = bank_meta
        self.output_size = output_size
        self.feature_mask_size = feature_mask_size
        self.neighbor_random_topn = neighbor_random_topn
        self.random_neighbor = random_neighbor
        self.rng = random.Random(seed)

        self._mask_cache = {
            item["name"]: _read_gray(item["mask_path"], output_size, nearest=True)
            for item in self.items
        }
        self._soft_cache = {
            item["name"]: _read_gray(item["soft_path"], output_size, nearest=False)
            for item in self.items
        }
        self._neighbor_cache = self._build_neighbor_cache(neighbor_topk)

    def _build_neighbor_cache(self, neighbor_topk: int) -> dict[str, list[int]]:
        neighbor_cache: dict[str, list[int]] = {}
        for item in self.items:
            mask_image = self._mask_cache[item["name"]]
            query_mask = self.module.normalize_mask(mask_image, size=self.feature_mask_size)
            topk_items = self.module.score_query_to_bank(
                query_mask,
                item.get("bbox", [0, 0, self.output_size, self.output_size]),
                self.bank_meta,
                image_shape=None,
                topk=max(neighbor_topk + 1, 2),
            )
            neighbors: list[int] = []
            for candidate in topk_items:
                bank_item = self.bank_meta[int(candidate["comp_id"])]
                if bank_item["instance_name"] == item["name"]:
                    continue
                neighbors.append(int(candidate["comp_id"]))
            if not neighbors:
                for index, bank_item in enumerate(self.bank_meta):
                    if bank_item["instance_name"] != item["name"]:
                        neighbors.append(index)
                        break
            neighbor_cache[item["name"]] = neighbors
        return neighbor_cache

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.items[index]
        target_soft = self._soft_cache[item["name"]]
        target_mask = self._mask_cache[item["name"]]
        degraded = _degrade_component(target_soft, self.rng)
        degraded_mask = self.module.normalize_mask(degraded, size=self.output_size)

        neighbors = self._neighbor_cache[item["name"]]
        if self.random_neighbor:
            usable = neighbors[: max(1, min(self.neighbor_random_topn, len(neighbors)))]
            neighbor_index = usable[self.rng.randrange(len(usable))]
        else:
            neighbor_index = neighbors[0]
        prior_item = self.bank_meta[neighbor_index]
        model_input = _build_model_input(
            degraded,
            degraded_mask,
            prior_item["soft_image"],
            prior_item["mask_image"],
            size=self.output_size,
        ).squeeze(0)
        return {
            "input": model_input,
            "target_soft": torch.from_numpy(target_soft.astype(np.float32) / 255.0).unsqueeze(0),
            "target_mask": torch.from_numpy(target_mask.astype(np.float32) / 255.0).unsqueeze(0),
            "query": torch.from_numpy(degraded.astype(np.float32) / 255.0).unsqueeze(0),
            "prior": torch.from_numpy(prior_item["soft_image"].astype(np.float32) / 255.0).unsqueeze(0),
            "name": item["name"],
            "neighbor_name": prior_item["instance_name"],
            "component_key": item.get("component_key"),
        }


def _save_preview(batch: dict[str, Any], prediction: torch.Tensor, output_path: Path, count: int) -> None:
    rows = []
    limit = min(count, int(prediction.shape[0]))
    for index in range(limit):
        rows.append(
            (
                str(batch["component_key"][index]),
                batch["query"][index],
                batch["prior"][index],
                prediction[index],
                batch["target_soft"][index],
            )
        )
    size = rows[0][1].shape[-1] if rows else 128
    pad = 8
    text_h = 28
    canvas = Image.new("L", (pad + 4 * (size + pad), pad + len(rows) * (size + text_h + pad)), color=255)
    draw = ImageDraw.Draw(canvas)
    for row_index, (component_key, query, prior, pred, target) in enumerate(rows):
        y0 = pad + row_index * (size + text_h + pad)
        draw.text((pad, y0), component_key, fill=0)
        image_y = y0 + text_h
        for col_index, (title, tensor) in enumerate(
            [("query", query), ("prior", prior), ("pred", pred), ("target", target)]
        ):
            image = Image.fromarray((tensor.squeeze().detach().cpu().clamp(0.0, 1.0).numpy() * 255.0).round().astype(np.uint8), mode="L")
            x0 = pad + col_index * (size + pad)
            canvas.paste(image, (x0, image_y))
            draw.rectangle([x0, image_y, x0 + size, image_y + size], outline=0, width=1)
            draw.text((x0 + 4, image_y + 4), title, fill=0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def _build_checkpoint_payload(
    config: InstanceRestorationTrainingConfig,
    *,
    step: int,
    epoch: int,
    best_score: float,
    restorer: nn.Module,
    restorer_ema: nn.Module,
    metrics_history: list[dict[str, Any]],
) -> dict[str, Any]:
    runtime_cfg = {
        "data": {
            "components_root": config.data.components_root,
            "annotations_path": config.data.annotations_path,
            "vocab_path": config.data.vocab_path,
            "output_size": config.data.output_size,
        },
        "model": {
            "input_channels": config.model.input_channels,
            "base_channels": config.model.base_channels,
            "output_size": config.model.output_size,
        },
        "inference": {
            "topk": config.data.neighbor_topk,
            "bank_splits": None,
        },
        "loss": asdict(config.loss),
        "train": {
            "seed": config.train.seed,
            "batch_size": config.train.batch_size,
            "learning_rate": config.train.learning_rate,
            "ema_decay": config.train.ema_decay,
        },
    }
    return {
        "checkpoint_type": "instance_retrieval_restorer_v1",
        "step": step,
        "epoch": epoch,
        "best_score": best_score,
        "cfg": runtime_cfg,
        "train_cfg": config.to_dict(),
        "restorer": restorer.state_dict(),
        "restorer_ema": restorer_ema.state_dict(),
        "metrics_history": metrics_history,
    }


def _ema_update(source: nn.Module, target: nn.Module, decay: float) -> None:
    with torch.no_grad():
        for source_param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.mul_(decay).add_(source_param.data, alpha=1.0 - decay)
        for source_buffer, target_buffer in zip(source.buffers(), target.buffers()):
            target_buffer.copy_(source_buffer)


def _evaluate(
    *,
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    preview_path: Path | None,
    preview_count: int,
) -> dict[str, float]:
    model.eval()
    totals = Counter()
    preview_saved = False
    with torch.no_grad():
        for batch in dataloader:
            model_input = batch["input"].to(device=device, dtype=torch.float32)
            target_soft = batch["target_soft"].to(device=device, dtype=torch.float32)
            target_mask = batch["target_mask"].to(device=device, dtype=torch.float32)
            pred = model(model_input)
            totals["l1"] += float(F.l1_loss(pred, target_soft).item()) * pred.shape[0]
            totals["bce"] += float(F.binary_cross_entropy(pred, target_mask).item()) * pred.shape[0]
            totals["ssim"] += float((1.0 - _ssim_loss(pred, target_soft)).item()) * pred.shape[0]
            pred_mask = (pred > 0.5).float()
            intersection = (pred_mask * target_mask).flatten(1).sum(dim=1)
            union = ((pred_mask + target_mask) > 0).float().flatten(1).sum(dim=1)
            totals["iou"] += float(((intersection + 1.0e-6) / (union + 1.0e-6)).mean().item()) * pred.shape[0]
            totals["count"] += pred.shape[0]

            if preview_path is not None and not preview_saved:
                preview_batch = {
                    "query": batch["query"][:preview_count],
                    "prior": batch["prior"][:preview_count],
                    "target_soft": batch["target_soft"][:preview_count],
                    "component_key": batch["component_key"][:preview_count],
                }
                _save_preview(preview_batch, pred[:preview_count].cpu(), preview_path, preview_count)
                preview_saved = True

    count = max(1, int(totals["count"]))
    metrics = {
        "val_l1": totals["l1"] / count,
        "val_bce": totals["bce"] / count,
        "val_ssim": totals["ssim"] / count,
        "val_iou": totals["iou"] / count,
    }
    metrics["score"] = 1.5 * metrics["val_iou"] + 1.2 * metrics["val_ssim"] - 0.8 * metrics["val_l1"] - 0.4 * metrics["val_bce"]
    return metrics


def run_instance_restoration_training(
    config_path: str | Path | None = None,
    *,
    output_root: str | Path | None = None,
    resume_path: str | Path | None = None,
) -> dict[str, Any]:
    config = load_instance_restoration_training_config(config_path)
    if output_root is not None:
        config.output.root_dir = str(output_root)
    if resume_path is not None:
        config.train.resume_path = str(resume_path)

    _set_seed(config.train.seed)
    search_roots = [config.config_base_dir] if config.config_base_dir else None
    output_dir = ensure_dir(
        resolve_path(
            config.output.root_dir,
            search_roots=search_roots,
            base_dir=config.config_base_dir,
            allow_missing=True,
        )
    )
    checkpoints_dir = ensure_dir(output_dir / config.output.checkpoints_dir)
    previews_dir = ensure_dir(output_dir / config.output.previews_dir)
    write_json(config.to_dict(), output_dir / config.output.resolved_config_name)

    components_root, annotations_path, _ = _resolve_annotation_files(config)
    train_module, train_bank_meta = _build_instance_bank(
        annotations_path,
        splits={config.data.train_split},
        output_size=config.data.output_size,
        feature_mask_size=config.data.feature_mask_size,
    )
    train_dataset = InstanceConditionDataset(
        annotations_path=annotations_path,
        split=config.data.train_split,
        bank_meta=train_bank_meta,
        output_size=config.data.output_size,
        feature_mask_size=config.data.feature_mask_size,
        neighbor_topk=config.data.neighbor_topk,
        neighbor_random_topn=config.data.neighbor_random_topn,
        random_neighbor=True,
        seed=config.train.seed,
    )
    val_dataset = InstanceConditionDataset(
        annotations_path=annotations_path,
        split=config.data.val_split,
        bank_meta=train_bank_meta,
        output_size=config.data.output_size,
        feature_mask_size=config.data.feature_mask_size,
        neighbor_topk=config.data.neighbor_topk,
        neighbor_random_topn=1,
        random_neighbor=False,
        seed=config.train.seed,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=config.train.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.train.val_batch_size,
        shuffle=False,
        num_workers=config.train.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    device_name = config.train.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    model = ConditionalComponentRestorer(
        in_channels=config.model.input_channels,
        base_channels=config.model.base_channels,
    ).to(device)
    model_ema = ConditionalComponentRestorer(
        in_channels=config.model.input_channels,
        base_channels=config.model.base_channels,
    ).to(device)
    model_ema.load_state_dict(model.state_dict())
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.train.learning_rate,
        weight_decay=config.train.weight_decay,
    )

    global_step = 0
    best_score = -float("inf")
    metrics_history: list[dict[str, Any]] = []
    if config.train.resume_path:
        resume_resolved = resolve_path(config.train.resume_path, search_roots=search_roots, base_dir=config.config_base_dir)
        checkpoint = torch.load(str(resume_resolved), map_location="cpu")
        model.load_state_dict(checkpoint["restorer"])
        model_ema.load_state_dict(checkpoint.get("restorer_ema", checkpoint["restorer"]))
        global_step = int(checkpoint.get("step", 0))
        best_score = float(checkpoint.get("best_score", best_score))
        metrics_history = list(checkpoint.get("metrics_history", []))

    print(f"[instance-train] device={device}")
    print(f"[instance-train] annotations={annotations_path}")
    print(f"[instance-train] components_root={components_root}")
    print(f"[instance-train] train={len(train_dataset)} val={len(val_dataset)} bank={len(train_bank_meta)}")

    train_start = time.time()
    stop_training = False
    for epoch in range(config.train.max_epochs):
        model.train()
        for batch in train_loader:
            global_step += 1
            model_input = batch["input"].to(device=device, dtype=torch.float32)
            target_soft = batch["target_soft"].to(device=device, dtype=torch.float32)
            target_mask = batch["target_mask"].to(device=device, dtype=torch.float32)
            pred = model(model_input)
            losses = {
                "l1": F.l1_loss(pred, target_soft),
                "bce": F.binary_cross_entropy(pred, target_mask),
                "focal": _binary_focal_loss(pred, target_mask),
                "ssim": _ssim_loss(pred, target_soft),
                "edge": F.l1_loss(_sobel_edges(pred), _sobel_edges(target_soft)),
                "mask_consistency": F.l1_loss(pred * target_mask, target_soft * target_mask),
            }
            total = (
                config.loss.lambda_l1 * losses["l1"]
                + config.loss.lambda_bce * (losses["bce"] + losses["focal"])
                + config.loss.lambda_ssim * losses["ssim"]
                + config.loss.lambda_edge * losses["edge"]
                + config.loss.lambda_mask_consistency * losses["mask_consistency"]
            )
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            if config.train.grad_clip and config.train.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.train.grad_clip)
            optimizer.step()
            _ema_update(model, model_ema, config.train.ema_decay)

            if global_step % config.train.log_every == 0 or global_step == 1:
                elapsed = time.time() - train_start
                print(
                    f"[instance-train] step={global_step} epoch={epoch + 1} elapsed={elapsed:.1f}s "
                    f"total={float(total.item()):.4f} l1={float(losses['l1'].item()):.4f} "
                    f"bce={float(losses['bce'].item()):.4f} focal={float(losses['focal'].item()):.4f} "
                    f"ssim={float(losses['ssim'].item()):.4f} edge={float(losses['edge'].item()):.4f}"
                )

            if global_step % config.train.validate_every == 0 or global_step == 1:
                preview_path = previews_dir / f"preview_step_{global_step:07d}.png"
                val_metrics = _evaluate(
                    model=model_ema,
                    dataloader=val_loader,
                    device=device,
                    preview_path=preview_path,
                    preview_count=config.train.preview_count,
                )
                val_metrics["step"] = global_step
                val_metrics["epoch"] = epoch + 1
                metrics_history.append(val_metrics)
                write_json(metrics_history, output_dir / config.output.metrics_name)
                print(
                    f"[instance-train] validation l1={val_metrics['val_l1']:.4f} "
                    f"bce={val_metrics['val_bce']:.4f} ssim={val_metrics['val_ssim']:.4f} "
                    f"iou={val_metrics['val_iou']:.4f} score={val_metrics['score']:.4f}"
                )
                payload = _build_checkpoint_payload(
                    config,
                    step=global_step,
                    epoch=epoch + 1,
                    best_score=best_score,
                    restorer=model,
                    restorer_ema=model_ema,
                    metrics_history=metrics_history,
                )
                torch.save(payload, checkpoints_dir / config.output.last_checkpoint_name)
                if val_metrics["score"] > best_score:
                    best_score = float(val_metrics["score"])
                    payload["best_score"] = best_score
                    torch.save(payload, checkpoints_dir / config.output.best_checkpoint_name)
                    print(f"[instance-train] new_best score={best_score:.4f}")

            if global_step % config.train.save_every == 0:
                payload = _build_checkpoint_payload(
                    config,
                    step=global_step,
                    epoch=epoch + 1,
                    best_score=best_score,
                    restorer=model,
                    restorer_ema=model_ema,
                    metrics_history=metrics_history,
                )
                torch.save(payload, checkpoints_dir / f"instance_restoration_step_{global_step:07d}.pth")

            if config.train.max_steps is not None and global_step >= config.train.max_steps:
                stop_training = True
                break
        if stop_training:
            break

    final_payload = _build_checkpoint_payload(
        config,
        step=global_step,
        epoch=epoch + 1 if global_step > 0 else 0,
        best_score=best_score,
        restorer=model,
        restorer_ema=model_ema,
        metrics_history=metrics_history,
    )
    torch.save(final_payload, checkpoints_dir / config.output.last_checkpoint_name)
    summary = {
        "stage": "instance_restoration_training",
        "device": str(device),
        "output_dir": str(output_dir),
        "annotations_path": str(annotations_path),
        "train_count": len(train_dataset),
        "val_count": len(val_dataset),
        "bank_count": len(train_bank_meta),
        "global_step": global_step,
        "best_score": best_score,
        "best_checkpoint": str((checkpoints_dir / config.output.best_checkpoint_name).resolve()),
        "last_checkpoint": str((checkpoints_dir / config.output.last_checkpoint_name).resolve()),
    }
    write_json(summary, output_dir / "training_summary.json")
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train instance retrieval + conditional restoration.")
    parser.add_argument("--config", type=str, default=None, help="YAML or JSON config path.")
    parser.add_argument("--output", type=str, default=None, help="Override output root.")
    parser.add_argument("--resume", type=str, default=None, help="Resume checkpoint path.")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    summary = run_instance_restoration_training(
        args.config,
        output_root=args.output,
        resume_path=args.resume,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
