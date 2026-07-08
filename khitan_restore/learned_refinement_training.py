"""Training for ROI-based learned refinement."""

from __future__ import annotations

import argparse
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
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .io_utils import ensure_dir, load_structured_file, resolve_path, write_json
from .learned_refinement import (
    LearnedPriorFusionNet,
    canonical_stem,
    load_restoration_results,
    make_labeled_strip,
    normalize_boxes,
    read_gray_float,
    resize_like,
)


@dataclass
class LearnedRefinementDataConfig:
    restoration_results_path: str = "runs/server_sy/03_restoration/retrieval_results_all.json"
    lr_dir: str = "dataset/data/train/trainL"
    gt_dir: str = "dataset/data/train/trainH"
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    resize_height: int = 512
    resize_width: int = 512
    prior_rank: int = 1
    max_images: int | None = None


@dataclass
class LearnedRefinementModelConfig:
    base_channels: int = 24
    roi_size: int = 48


@dataclass
class LearnedRefinementLossConfig:
    lambda_l1: float = 4.0
    lambda_bce: float = 2.0
    lambda_dice: float = 1.5
    lambda_edge: float = 0.8
    lambda_confidence: float = 0.6
    lambda_prior_suppress: float = 0.4


@dataclass
class LearnedRefinementTrainConfig:
    device: str | None = None
    seed: int = 1234
    batch_size: int = 2
    val_batch_size: int = 2
    num_workers: int = 0
    max_epochs: int = 80
    max_steps: int | None = 12000
    learning_rate: float = 2.0e-4
    weight_decay: float = 1.0e-5
    grad_clip: float = 2.0
    ema_decay: float = 0.999
    log_every: int = 20
    validate_every: int = 100
    save_every: int = 250
    preview_count: int = 4
    resume_path: str | None = None


@dataclass
class LearnedRefinementOutputConfig:
    root_dir: str = "runs/learned_refinement_training"
    checkpoints_dir: str = "checkpoints"
    previews_dir: str = "previews"
    metrics_name: str = "metrics_history.json"
    best_checkpoint_name: str = "learned_refinement_best.pth"
    last_checkpoint_name: str = "learned_refinement_last.pth"
    resolved_config_name: str = "resolved_learned_refinement_config.json"


@dataclass
class LearnedRefinementTrainingConfig:
    data: LearnedRefinementDataConfig = field(default_factory=LearnedRefinementDataConfig)
    model: LearnedRefinementModelConfig = field(default_factory=LearnedRefinementModelConfig)
    loss: LearnedRefinementLossConfig = field(default_factory=LearnedRefinementLossConfig)
    train: LearnedRefinementTrainConfig = field(default_factory=LearnedRefinementTrainConfig)
    output: LearnedRefinementOutputConfig = field(default_factory=LearnedRefinementOutputConfig)
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


def _from_dict(data: dict[str, Any]) -> LearnedRefinementTrainingConfig:
    return LearnedRefinementTrainingConfig(
        data=LearnedRefinementDataConfig(**data.get("data", {})),
        model=LearnedRefinementModelConfig(**data.get("model", {})),
        loss=LearnedRefinementLossConfig(**data.get("loss", {})),
        train=LearnedRefinementTrainConfig(**data.get("train", {})),
        output=LearnedRefinementOutputConfig(**data.get("output", {})),
        config_base_dir=data.get("config_base_dir"),
    )


def load_learned_refinement_training_config(
    config_path: str | Path | None = None,
) -> LearnedRefinementTrainingConfig:
    defaults = LearnedRefinementTrainingConfig().to_dict()
    if config_path is None:
        return _from_dict(defaults)

    resolved = Path(config_path).expanduser().resolve()
    raw = load_structured_file(resolved)
    merged = _deep_merge(defaults, raw)
    merged["config_base_dir"] = str(resolved.parent)
    return _from_dict(merged)


@dataclass
class RefinementRecord:
    image_name: str
    canonical_name: str
    lr_path: Path
    sr_path: Path
    prior_path: Path
    gt_path: Path
    boxes: list[list[int]]


def _index_images(root: str | Path) -> dict[str, Path]:
    root_path = Path(root).expanduser().resolve()
    if not root_path.exists():
        raise FileNotFoundError(f"Image directory not found: {root_path}")
    mapping: dict[str, Path] = {}
    for path in sorted(root_path.rglob("*")):
        if path.is_file():
            mapping.setdefault(canonical_stem(path), path.resolve())
    return mapping


def _resolve_results_path(
    configured_path: str | Path,
    *,
    search_roots: list[str] | None,
    base_dir: str | Path | None,
) -> Path:
    resolved = resolve_path(configured_path, search_roots=search_roots, base_dir=base_dir)
    if resolved.is_dir():
        candidate = resolved / "retrieval_results_all.json"
        if candidate.exists():
            return candidate
    return resolved


def build_refinement_records(
    config: LearnedRefinementTrainingConfig,
) -> list[RefinementRecord]:
    search_roots: list[str] = []
    if config.config_base_dir:
        search_roots.append(config.config_base_dir)
    base_dir = config.config_base_dir

    results_path = _resolve_results_path(
        config.data.restoration_results_path,
        search_roots=search_roots,
        base_dir=base_dir,
    )
    results = load_restoration_results(results_path)
    lr_index = _index_images(resolve_path(config.data.lr_dir, search_roots=search_roots, base_dir=base_dir))
    gt_index = _index_images(resolve_path(config.data.gt_dir, search_roots=search_roots, base_dir=base_dir))

    records: list[RefinementRecord] = []
    for result in results:
        sr_path = resolve_path(result["image_path"], search_roots=search_roots, base_dir=base_dir)
        canonical_name = canonical_stem(result.get("image_name") or sr_path)
        lr_path = lr_index.get(canonical_name)
        gt_path = gt_index.get(canonical_name)
        if lr_path is None or gt_path is None:
            continue

        prior_path_text = (
            result.get("paths", {})
            .get("reconstructed_ranks", {})
            .get(f"rank_{config.data.prior_rank}")
        )
        if not prior_path_text:
            continue
        prior_path = resolve_path(prior_path_text, search_roots=search_roots, base_dir=base_dir)
        boxes = [[int(v) for v in component["bbox"]] for component in result.get("components", [])]
        if not boxes:
            continue

        records.append(
            RefinementRecord(
                image_name=str(result.get("image_name") or sr_path.stem),
                canonical_name=canonical_name,
                lr_path=lr_path,
                sr_path=sr_path,
                prior_path=prior_path,
                gt_path=gt_path,
                boxes=boxes,
            )
        )

    records = sorted(records, key=lambda item: item.canonical_name)
    if config.data.max_images is not None:
        records = records[: int(config.data.max_images)]
    if not records:
        raise RuntimeError("No learned refinement training records could be built.")
    return records


def split_records(
    records: list[RefinementRecord],
    *,
    train_ratio: float,
    val_ratio: float,
) -> tuple[list[RefinementRecord], list[RefinementRecord], list[RefinementRecord]]:
    total = len(records)
    if total == 1:
        return records[:], records[:], records[:]
    if total == 2:
        return records[:1], records[1:], records[1:]
    train_count = max(1, int(round(total * train_ratio)))
    val_count = max(1, int(round(total * val_ratio)))
    if train_count + val_count >= total:
        val_count = max(1, total - train_count - 1)
    train_records = records[:train_count]
    val_records = records[train_count : train_count + val_count]
    test_records = records[train_count + val_count :]
    if not val_records:
        val_records = train_records[-1:]
        train_records = train_records[:-1]
    if not test_records:
        test_records = val_records[-1:]
    return train_records, val_records, test_records


class LearnedRefinementDataset(Dataset):
    def __init__(
        self,
        records: list[RefinementRecord],
        *,
        resize_height: int,
        resize_width: int,
    ) -> None:
        self.records = records
        self.resize_height = resize_height
        self.resize_width = resize_width

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        sr = read_gray_float(record.sr_path)
        prior = read_gray_float(record.prior_path)
        prior = resize_like(prior, sr.shape[:2])
        lr = read_gray_float(record.lr_path)
        gt = read_gray_float(record.gt_path)
        lr = resize_like(lr, sr.shape[:2])
        gt = resize_like(gt, sr.shape[:2])

        original_shape = sr.shape[:2]
        if self.resize_height > 0 and self.resize_width > 0:
            target_shape = (self.resize_height, self.resize_width)
            sr = resize_like(sr, target_shape)
            prior = resize_like(prior, target_shape)
            lr = resize_like(lr, target_shape)
            gt = resize_like(gt, target_shape)
            boxes = normalize_boxes(record.boxes, original_shape, target_shape)
        else:
            target_shape = original_shape
            boxes = [list(box) for box in record.boxes]

        gt_mask = (gt > 0.5).astype(np.float32)
        return {
            "image_name": record.image_name,
            "sr": torch.from_numpy(sr).unsqueeze(0),
            "prior": torch.from_numpy(prior).unsqueeze(0),
            "lr": torch.from_numpy(lr).unsqueeze(0),
            "gt": torch.from_numpy(gt).unsqueeze(0),
            "gt_mask": torch.from_numpy(gt_mask).unsqueeze(0),
            "boxes": boxes,
        }


def _collate_refinement(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "image_name": [item["image_name"] for item in batch],
        "sr": torch.stack([item["sr"] for item in batch], dim=0),
        "prior": torch.stack([item["prior"] for item in batch], dim=0),
        "lr": torch.stack([item["lr"] for item in batch], dim=0),
        "gt": torch.stack([item["gt"] for item in batch], dim=0),
        "gt_mask": torch.stack([item["gt_mask"] for item in batch], dim=0),
        "boxes": [item["boxes"] for item in batch],
    }


def _dice_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = pred.flatten(1)
    target = target.flatten(1)
    inter = (pred * target).sum(dim=1)
    denom = pred.sum(dim=1) + target.sum(dim=1)
    return 1.0 - ((2.0 * inter + 1.0e-6) / (denom + 1.0e-6)).mean()


def _sobel_edges(image: torch.Tensor) -> torch.Tensor:
    kernel_x = torch.tensor(
        [[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]],
        dtype=image.dtype,
        device=image.device,
    ).unsqueeze(0)
    kernel_y = torch.tensor(
        [[[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]],
        dtype=image.dtype,
        device=image.device,
    ).unsqueeze(0)
    grad_x = F.conv2d(image, kernel_x, padding=1)
    grad_y = F.conv2d(image, kernel_y, padding=1)
    return torch.sqrt(grad_x.pow(2) + grad_y.pow(2) + 1.0e-6)


def _ssim_like(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
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
    return (numerator / (denominator + 1.0e-6)).mean()


def _ema_update(source: nn.Module, target: nn.Module, decay: float) -> None:
    with torch.no_grad():
        for source_param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.mul_(decay).add_(source_param.data, alpha=1.0 - decay)
        for source_buffer, target_buffer in zip(source.buffers(), target.buffers()):
            target_buffer.copy_(source_buffer)


def _save_preview(
    batch: dict[str, Any],
    prediction: torch.Tensor,
    output_path: Path,
    count: int,
) -> None:
    rows: list[np.ndarray] = []
    limit = min(count, int(prediction.shape[0]))
    for index in range(limit):
        panels = [
            ("LR", batch["lr"][index].squeeze().cpu().numpy()),
            ("SR", batch["sr"][index].squeeze().cpu().numpy()),
            ("PRIOR", batch["prior"][index].squeeze().cpu().numpy()),
            ("PRED", prediction[index].squeeze().detach().cpu().clamp(0.0, 1.0).numpy()),
            ("GT", batch["gt"][index].squeeze().cpu().numpy()),
        ]
        rows.append(make_labeled_strip(panels))
    if not rows:
        return
    canvas = np.vstack(rows)
    save_gray_float(output_path, canvas.astype(np.float32) / 255.0)


def save_gray_float(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    output = (np.clip(image, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    cv2.imwrite(str(path), output)


def _build_checkpoint_payload(
    config: LearnedRefinementTrainingConfig,
    *,
    step: int,
    epoch: int,
    best_score: float,
    model: nn.Module,
    model_ema: nn.Module,
    metrics_history: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "checkpoint_type": "learned_refinement_v1",
        "step": step,
        "epoch": epoch,
        "best_score": best_score,
        "cfg": {
            "model": {
                "base_channels": config.model.base_channels,
                "roi_size": config.model.roi_size,
            },
            "data": {
                "resize_height": config.data.resize_height,
                "resize_width": config.data.resize_width,
            },
        },
        "train_cfg": config.to_dict(),
        "model": model.state_dict(),
        "model_ema": model_ema.state_dict(),
        "metrics_history": metrics_history,
    }


def _evaluate(
    *,
    model: LearnedPriorFusionNet,
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
            sr = batch["sr"].to(device=device, dtype=torch.float32)
            prior = batch["prior"].to(device=device, dtype=torch.float32)
            lr = batch["lr"].to(device=device, dtype=torch.float32)
            gt = batch["gt"].to(device=device, dtype=torch.float32)
            gt_mask = batch["gt_mask"].to(device=device, dtype=torch.float32)
            prediction = model(sr, prior, lr, batch["boxes"])
            pred = prediction["final"]
            confidence = prediction["confidence"]
            totals["l1"] += float(F.l1_loss(pred, gt).item()) * pred.shape[0]
            totals["bce"] += float(F.binary_cross_entropy(pred, gt_mask).item()) * pred.shape[0]
            totals["dice"] += float(_dice_loss(pred, gt_mask).item()) * pred.shape[0]
            totals["ssim"] += float(_ssim_like(pred, gt).item()) * pred.shape[0]
            pred_mask = (pred > 0.5).float()
            intersection = (pred_mask * gt_mask).flatten(1).sum(dim=1)
            union = ((pred_mask + gt_mask) > 0).float().flatten(1).sum(dim=1)
            totals["iou"] += float(((intersection + 1.0e-6) / (union + 1.0e-6)).mean().item()) * pred.shape[0]
            totals["conf"] += float(F.binary_cross_entropy(confidence, gt_mask).item()) * pred.shape[0]
            totals["count"] += pred.shape[0]

            if preview_path is not None and not preview_saved:
                _save_preview(batch, pred.cpu(), preview_path, preview_count)
                preview_saved = True

    count = max(1, int(totals["count"]))
    metrics = {
        "val_l1": totals["l1"] / count,
        "val_bce": totals["bce"] / count,
        "val_dice": totals["dice"] / count,
        "val_ssim": totals["ssim"] / count,
        "val_iou": totals["iou"] / count,
        "val_conf": totals["conf"] / count,
    }
    metrics["score"] = (
        1.7 * metrics["val_iou"]
        + 1.1 * metrics["val_ssim"]
        - 0.9 * metrics["val_l1"]
        - 0.5 * metrics["val_bce"]
        - 0.3 * metrics["val_conf"]
    )
    return metrics


def run_learned_refinement_training(
    config_path: str | Path | None = None,
    *,
    output_root: str | Path | None = None,
    resume_path: str | Path | None = None,
) -> dict[str, Any]:
    config = load_learned_refinement_training_config(config_path)
    if output_root is not None:
        config.output.root_dir = str(output_root)
    if resume_path is not None:
        config.train.resume_path = str(resume_path)

    records = build_refinement_records(config)
    train_records, val_records, test_records = split_records(
        records,
        train_ratio=config.data.train_ratio,
        val_ratio=config.data.val_ratio,
    )

    output_dir = ensure_dir(resolve_path(config.output.root_dir, base_dir=config.config_base_dir, allow_missing=True))
    checkpoints_dir = ensure_dir(output_dir / config.output.checkpoints_dir)
    previews_dir = ensure_dir(output_dir / config.output.previews_dir)
    write_json(config.to_dict(), output_dir / config.output.resolved_config_name)

    train_dataset = LearnedRefinementDataset(
        train_records,
        resize_height=config.data.resize_height,
        resize_width=config.data.resize_width,
    )
    val_dataset = LearnedRefinementDataset(
        val_records,
        resize_height=config.data.resize_height,
        resize_width=config.data.resize_width,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=config.train.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=_collate_refinement,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.train.val_batch_size,
        shuffle=False,
        num_workers=config.train.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=_collate_refinement,
    )

    device_name = config.train.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    model = LearnedPriorFusionNet(
        base_channels=config.model.base_channels,
        roi_size=config.model.roi_size,
    ).to(device)
    model_ema = LearnedPriorFusionNet(
        base_channels=config.model.base_channels,
        roi_size=config.model.roi_size,
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
        checkpoint = torch.load(
            str(resolve_path(config.train.resume_path, base_dir=config.config_base_dir)),
            map_location="cpu",
        )
        model.load_state_dict(checkpoint["model"])
        model_ema.load_state_dict(checkpoint.get("model_ema", checkpoint["model"]))
        global_step = int(checkpoint.get("step", 0))
        best_score = float(checkpoint.get("best_score", best_score))
        metrics_history = list(checkpoint.get("metrics_history", []))

    print(f"[learned-refine-train] device={device}")
    print(
        f"[learned-refine-train] train={len(train_dataset)} val={len(val_dataset)} "
        f"test={len(test_records)} output_dir={output_dir}"
    )

    train_start = time.time()
    stop_training = False
    for epoch in range(config.train.max_epochs):
        model.train()
        for batch in train_loader:
            global_step += 1
            sr = batch["sr"].to(device=device, dtype=torch.float32)
            prior = batch["prior"].to(device=device, dtype=torch.float32)
            lr = batch["lr"].to(device=device, dtype=torch.float32)
            gt = batch["gt"].to(device=device, dtype=torch.float32)
            gt_mask = batch["gt_mask"].to(device=device, dtype=torch.float32)

            prediction = model(sr, prior, lr, batch["boxes"])
            pred = prediction["final"]
            confidence = prediction["confidence"]

            losses = {
                "l1": F.l1_loss(pred, gt),
                "bce": F.binary_cross_entropy(pred, gt_mask),
                "dice": _dice_loss(pred, gt_mask),
                "edge": F.l1_loss(_sobel_edges(pred), _sobel_edges(gt)),
                "confidence": F.binary_cross_entropy(confidence, gt_mask),
                "prior_suppress": ((pred * (1.0 - gt_mask)) * prior).mean(),
            }
            total = (
                config.loss.lambda_l1 * losses["l1"]
                + config.loss.lambda_bce * losses["bce"]
                + config.loss.lambda_dice * losses["dice"]
                + config.loss.lambda_edge * losses["edge"]
                + config.loss.lambda_confidence * losses["confidence"]
                + config.loss.lambda_prior_suppress * losses["prior_suppress"]
            )

            optimizer.zero_grad(set_to_none=True)
            total.backward()
            if config.train.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_clip)
            optimizer.step()
            _ema_update(model, model_ema, config.train.ema_decay)

            if global_step % config.train.log_every == 0 or global_step == 1:
                elapsed = time.time() - train_start
                print(
                    f"[learned-refine-train] step={global_step} epoch={epoch + 1} elapsed={elapsed:.1f}s "
                    f"total={float(total.item()):.4f} l1={float(losses['l1'].item()):.4f} "
                    f"bce={float(losses['bce'].item()):.4f} dice={float(losses['dice'].item()):.4f} "
                    f"edge={float(losses['edge'].item()):.4f} conf={float(losses['confidence'].item()):.4f} "
                    f"prior_sup={float(losses['prior_suppress'].item()):.4f}"
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
                    f"[learned-refine-train] validation l1={val_metrics['val_l1']:.4f} "
                    f"bce={val_metrics['val_bce']:.4f} dice={val_metrics['val_dice']:.4f} "
                    f"ssim={val_metrics['val_ssim']:.4f} iou={val_metrics['val_iou']:.4f} "
                    f"score={val_metrics['score']:.4f}"
                )
                payload = _build_checkpoint_payload(
                    config,
                    step=global_step,
                    epoch=epoch + 1,
                    best_score=best_score,
                    model=model,
                    model_ema=model_ema,
                    metrics_history=metrics_history,
                )
                torch.save(payload, checkpoints_dir / config.output.last_checkpoint_name)
                if val_metrics["score"] > best_score:
                    best_score = float(val_metrics["score"])
                    payload["best_score"] = best_score
                    torch.save(payload, checkpoints_dir / config.output.best_checkpoint_name)
                    print(f"[learned-refine-train] new_best score={best_score:.4f}")

            if global_step % config.train.save_every == 0:
                payload = _build_checkpoint_payload(
                    config,
                    step=global_step,
                    epoch=epoch + 1,
                    best_score=best_score,
                    model=model,
                    model_ema=model_ema,
                    metrics_history=metrics_history,
                )
                torch.save(payload, checkpoints_dir / f"learned_refinement_step_{global_step:07d}.pth")

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
        model=model,
        model_ema=model_ema,
        metrics_history=metrics_history,
    )
    torch.save(final_payload, checkpoints_dir / config.output.last_checkpoint_name)
    summary = {
        "stage": "learned_refinement_training",
        "device": str(device),
        "output_dir": str(output_dir),
        "records_total": len(records),
        "train_count": len(train_dataset),
        "val_count": len(val_dataset),
        "test_count": len(test_records),
        "global_step": global_step,
        "best_score": best_score,
        "best_checkpoint": str((checkpoints_dir / config.output.best_checkpoint_name).resolve()),
        "last_checkpoint": str((checkpoints_dir / config.output.last_checkpoint_name).resolve()),
    }
    write_json(summary, output_dir / "training_summary.json")
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train learned ROI-based refinement.")
    parser.add_argument("--config", type=str, default=None, help="YAML or JSON config path.")
    parser.add_argument("--output", type=str, default=None, help="Override output root.")
    parser.add_argument("--resume", type=str, default=None, help="Resume checkpoint path.")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    result = run_learned_refinement_training(args.config, output_root=args.output, resume_path=args.resume)
    print(result)


if __name__ == "__main__":
    main()
