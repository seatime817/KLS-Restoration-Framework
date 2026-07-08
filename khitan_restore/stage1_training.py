"""Stage1 training for component codebook + StyleGAN restoration."""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .component_annotations import build_component_annotations, write_component_annotations
from .io_utils import ensure_dir, load_structured_file, resolve_path, write_json
from .legacy_loader import load_component_stylegan_module


@dataclass
class Stage1DataConfig:
    components_root: str = "dataset/components"
    annotations_path: str = "dataset/components/component_annotations.json"
    vocab_path: str = "dataset/components/component_vocab.json"
    auto_build_annotations: bool = True
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    train_split: str = "train"
    val_split: str = "val"
    image_size: int = 128


@dataclass
class Stage1ModelConfig:
    num_components: int | None = None
    codebook_dim: int = 512
    w_dim: int = 512
    base_channels: int = 512
    discriminator_base_channels: int = 64
    use_noise_in_blocks: bool = False
    legacy_output_head: bool = False


@dataclass
class Stage1LossConfig:
    lambda_soft_l1: float = 6.0
    lambda_mask_bce: float = 2.0
    lambda_mask_focal: float = 1.5
    lambda_dice: float = 1.5
    lambda_ssim: float = 1.0
    lambda_edge: float = 0.5
    lambda_adv: float = 0.2
    lambda_diversity: float = 0.02
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    diversity_margin: float = 0.20


@dataclass
class Stage1TrainConfig:
    device: str | None = None
    seed: int = 1234
    batch_size: int = 16
    val_batch_size: int = 16
    num_workers: int = 0
    max_epochs: int = 200
    max_steps: int | None = None
    generator_lr: float = 1.0e-4
    discriminator_lr: float = 2.0e-4
    weight_decay: float = 0.0
    beta1: float = 0.0
    beta2: float = 0.99
    grad_clip: float = 5.0
    balanced_sampling: bool = True
    gan_start_step: int = 500
    gan_ramp_steps: int = 1000
    d_reg_every: int = 16
    r1_gamma: float = 10.0
    ema_decay: float = 0.999
    log_every: int = 25
    validate_every: int = 200
    save_every: int = 500
    preview_count: int = 12
    resume_path: str | None = None


@dataclass
class Stage1OutputConfig:
    root_dir: str = "runs/stage1_training"
    checkpoints_dir: str = "checkpoints"
    previews_dir: str = "previews"
    metrics_name: str = "metrics_history.json"
    best_checkpoint_name: str = "stage1_best.pth"
    last_checkpoint_name: str = "stage1_last.pth"
    resolved_config_name: str = "resolved_stage1_config.json"


@dataclass
class Stage1TrainingConfig:
    data: Stage1DataConfig = field(default_factory=Stage1DataConfig)
    model: Stage1ModelConfig = field(default_factory=Stage1ModelConfig)
    loss: Stage1LossConfig = field(default_factory=Stage1LossConfig)
    train: Stage1TrainConfig = field(default_factory=Stage1TrainConfig)
    output: Stage1OutputConfig = field(default_factory=Stage1OutputConfig)
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


def load_stage1_training_config(config_path: str | Path | None = None) -> Stage1TrainingConfig:
    defaults = Stage1TrainingConfig().to_dict()
    if config_path is None:
        return Stage1TrainingConfig()

    resolved = Path(config_path).expanduser().resolve()
    raw = load_structured_file(resolved)
    merged = _deep_merge(defaults, raw)
    return Stage1TrainingConfig(
        data=Stage1DataConfig(**merged.get("data", {})),
        model=Stage1ModelConfig(**merged.get("model", {})),
        loss=Stage1LossConfig(**merged.get("loss", {})),
        train=Stage1TrainConfig(**merged.get("train", {})),
        output=Stage1OutputConfig(**merged.get("output", {})),
        config_base_dir=str(resolved.parent),
    )


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_grayscale_tensor(path: str | Path, size: int, *, nearest: bool) -> torch.Tensor:
    image = Image.open(path).convert("L")
    if image.size != (size, size):
        resample = Image.NEAREST if nearest else Image.BILINEAR
        image = image.resize((size, size), resample=resample)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).unsqueeze(0)


class ComponentPairDataset(Dataset):
    def __init__(
        self,
        annotations_path: str | Path,
        *,
        split: str,
        image_size: int,
    ):
        resolved = Path(annotations_path).expanduser().resolve()
        with open(resolved, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        self.items = [item for item in payload["items"] if item.get("split") == split]
        if not self.items:
            raise ValueError(f"No component items found for split={split} in {resolved}")
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.items[index]
        mask = _load_grayscale_tensor(item["mask_path"], self.image_size, nearest=True)
        soft = _load_grayscale_tensor(item["soft_path"], self.image_size, nearest=False)
        return {
            "comp_id": int(item["comp_id"]),
            "mask": mask,
            "soft": soft,
            "name": item["name"],
            "component_key": item["component_key"],
        }


def _build_sampler(dataset: ComponentPairDataset) -> WeightedRandomSampler | None:
    counts = Counter(int(item["comp_id"]) for item in dataset.items)
    if not counts:
        return None
    weights = [1.0 / counts[int(item["comp_id"])] for item in dataset.items]
    return WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)


def _build_dataloaders(
    config: Stage1TrainingConfig,
    annotations_path: Path,
) -> tuple[ComponentPairDataset, ComponentPairDataset, DataLoader, DataLoader]:
    train_dataset = ComponentPairDataset(
        annotations_path,
        split=config.data.train_split,
        image_size=config.data.image_size,
    )
    val_dataset = ComponentPairDataset(
        annotations_path,
        split=config.data.val_split,
        image_size=config.data.image_size,
    )
    sampler = _build_sampler(train_dataset) if config.train.balanced_sampling else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
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
    return train_dataset, val_dataset, train_loader, val_loader


def _resolve_data_files(
    config: Stage1TrainingConfig,
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

    if not annotations_path.exists():
        raise FileNotFoundError(f"Missing component annotations: {annotations_path}")
    if not vocab_path.exists():
        raise FileNotFoundError(f"Missing component vocab: {vocab_path}")
    return components_root, annotations_path, vocab_path


def _load_vocab(vocab_path: Path) -> dict[str, Any]:
    with open(vocab_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _generator_forward(
    comp_ids: torch.Tensor,
    comp_codebook: nn.Module,
    mapping_net: nn.Module,
    stylegan: nn.Module,
) -> torch.Tensor:
    codes = comp_codebook(comp_ids)
    w = mapping_net(codes.squeeze(-1).squeeze(-1))
    return stylegan(codes, w)


def _binary_focal_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    alpha: float,
    gamma: float,
) -> torch.Tensor:
    pred = pred.clamp(1.0e-6, 1.0 - 1.0e-6)
    pt = pred * target + (1.0 - pred) * (1.0 - target)
    alpha_t = alpha * target + (1.0 - alpha) * (1.0 - target)
    focal_weight = alpha_t * torch.pow(1.0 - pt, gamma)
    bce = -(target * pred.log() + (1.0 - target) * (1.0 - pred).log())
    return (focal_weight * bce).mean()


def _dice_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    pred_flat = pred.flatten(1)
    target_flat = target.flatten(1)
    intersection = (pred_flat * target_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
    score = (2.0 * intersection + eps) / (union + eps)
    return 1.0 - score.mean()


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
    ssim_map = numerator / (denominator + 1.0e-6)
    return 1.0 - ssim_map.mean()


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


def _codebook_diversity_loss(
    comp_codebook: nn.Module,
    margin: float,
) -> torch.Tensor:
    embeddings = F.normalize(comp_codebook.codes.weight, dim=1)
    similarity = embeddings @ embeddings.t()
    eye = torch.eye(similarity.shape[0], device=similarity.device, dtype=torch.bool)
    off_diagonal = similarity.masked_select(~eye)
    if off_diagonal.numel() == 0:
        return similarity.new_tensor(0.0)
    return F.relu(off_diagonal - margin).mean()


def _gan_ramp_weight(step: int, start_step: int, ramp_steps: int) -> float:
    if step < start_step:
        return 0.0
    if ramp_steps <= 0:
        return 1.0
    return min(1.0, (step - start_step) / float(ramp_steps))


def _batch_iou(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred_mask = pred > 0.5
    target_mask = target > 0.5
    intersection = (pred_mask & target_mask).float().flatten(1).sum(dim=1)
    union = (pred_mask | target_mask).float().flatten(1).sum(dim=1)
    score = (intersection + 1.0e-6) / (union + 1.0e-6)
    return float(score.mean().item())


def _ema_update(source: nn.Module, target: nn.Module, decay: float) -> None:
    with torch.no_grad():
        for source_param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.mul_(decay).add_(source_param.data, alpha=1.0 - decay)
        for source_buffer, target_buffer in zip(source.buffers(), target.buffers()):
            target_buffer.copy_(source_buffer)


def _format_metrics(metrics: dict[str, float]) -> str:
    return " ".join(f"{key}={value:.4f}" for key, value in metrics.items())


def _to_uint8_image(tensor: torch.Tensor) -> Image.Image:
    array = tensor.detach().float().cpu().squeeze().clamp(0.0, 1.0).numpy()
    return Image.fromarray((array * 255.0).round().astype(np.uint8), mode="L")


def _save_preview(
    batch: dict[str, Any],
    prediction: torch.Tensor,
    output_path: Path,
) -> None:
    count = min(int(prediction.shape[0]), len(batch["name"]))
    rows = []
    for index in range(count):
        pred_img = _to_uint8_image(prediction[index])
        soft_img = _to_uint8_image(batch["soft"][index])
        mask_img = _to_uint8_image(batch["mask"][index])
        rows.append(
            (
                str(batch["component_key"][index]),
                int(batch["comp_id"][index]),
                pred_img,
                soft_img,
                mask_img,
            )
        )

    tile_size = rows[0][2].size[0] if rows else 128
    pad = 8
    text_h = 28
    canvas = Image.new(
        "L",
        (pad + 3 * (tile_size + pad), pad + len(rows) * (tile_size + text_h + pad)),
        color=255,
    )
    draw = ImageDraw.Draw(canvas)
    for row_index, (component_key, comp_id, pred_img, soft_img, mask_img) in enumerate(rows):
        y0 = pad + row_index * (tile_size + text_h + pad)
        draw.text((pad, y0), f"id={comp_id} {component_key}", fill=0)
        image_y = y0 + text_h
        for col_index, (title, image) in enumerate(
            [("pred", pred_img), ("soft", soft_img), ("mask", mask_img)]
        ):
            x0 = pad + col_index * (tile_size + pad)
            canvas.paste(image, (x0, image_y))
            draw.rectangle([x0, image_y, x0 + tile_size, image_y + tile_size], outline=0, width=1)
            draw.text((x0 + 4, image_y + 4), title, fill=0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def _build_checkpoint_payload(
    config: Stage1TrainingConfig,
    *,
    step: int,
    epoch: int,
    best_score: float,
    num_components: int,
    comp_codebook: nn.Module,
    mapping_net: nn.Module,
    stylegan: nn.Module,
    stylegan_ema: nn.Module,
    discriminator: nn.Module,
    optimizer_g: torch.optim.Optimizer,
    optimizer_d: torch.optim.Optimizer,
    metrics_history: list[dict[str, Any]],
) -> dict[str, Any]:
    runtime_cfg = {
        "data": {
            "components_root": config.data.components_root,
            "annotations_path": config.data.annotations_path,
            "vocab_path": config.data.vocab_path,
        },
        "model": {
            "num_components": num_components,
            "codebook_dim": config.model.codebook_dim,
            "w_dim": config.model.w_dim,
            "stylegan_channels": config.model.base_channels,
            "use_noise_in_blocks": config.model.use_noise_in_blocks,
            "legacy_output_head": config.model.legacy_output_head,
        },
        "train": {
            "seed": config.train.seed,
            "batch_size": config.train.batch_size,
            "generator_lr": config.train.generator_lr,
            "discriminator_lr": config.train.discriminator_lr,
            "ema_decay": config.train.ema_decay,
        },
        "loss": asdict(config.loss),
    }
    return {
        "step": step,
        "epoch": epoch,
        "best_score": best_score,
        "cfg": runtime_cfg,
        "train_cfg": config.to_dict(),
        "comp_codebook": comp_codebook.state_dict(),
        "mapping_net": mapping_net.state_dict(),
        "stylegan": stylegan.state_dict(),
        "stylegan_ema": stylegan_ema.state_dict(),
        "discriminator": discriminator.state_dict(),
        "optimizer_g": optimizer_g.state_dict(),
        "optimizer_d": optimizer_d.state_dict(),
        "metrics_history": metrics_history,
    }


def _save_checkpoint(payload: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return path


def _evaluate(
    *,
    comp_codebook: nn.Module,
    mapping_net: nn.Module,
    stylegan_ema: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    loss_config: Stage1LossConfig,
    preview_path: Path | None,
    preview_count: int,
) -> dict[str, float]:
    comp_codebook.eval()
    mapping_net.eval()
    stylegan_ema.eval()

    totals = Counter()
    preview_saved = False
    with torch.no_grad():
        for batch in dataloader:
            comp_ids = batch["comp_id"].to(device=device, dtype=torch.long)
            mask = batch["mask"].to(device=device, dtype=torch.float32)
            soft = batch["soft"].to(device=device, dtype=torch.float32)
            fake = _generator_forward(comp_ids, comp_codebook, mapping_net, stylegan_ema)

            totals["soft_l1"] += float(F.l1_loss(fake, soft).item()) * fake.shape[0]
            totals["mask_bce"] += float(F.binary_cross_entropy(fake, mask).item()) * fake.shape[0]
            totals["mask_focal"] += float(
                _binary_focal_loss(
                    fake,
                    mask,
                    alpha=loss_config.focal_alpha,
                    gamma=loss_config.focal_gamma,
                ).item()
            ) * fake.shape[0]
            totals["dice"] += float(_dice_loss(fake, mask).item()) * fake.shape[0]
            totals["ssim"] += float((1.0 - _ssim_loss(fake, soft)).item()) * fake.shape[0]
            totals["edge"] += float(F.l1_loss(_sobel_edges(fake), _sobel_edges(soft)).item()) * fake.shape[0]
            totals["iou"] += _batch_iou(fake, mask) * fake.shape[0]
            totals["count"] += fake.shape[0]

            if preview_path is not None and not preview_saved:
                preview_batch = {
                    "comp_id": batch["comp_id"][:preview_count],
                    "component_key": batch["component_key"][:preview_count],
                    "soft": batch["soft"][:preview_count],
                    "mask": batch["mask"][:preview_count],
                    "name": batch["name"][:preview_count],
                }
                _save_preview(preview_batch, fake[:preview_count].cpu(), preview_path)
                preview_saved = True

    count = max(1, int(totals["count"]))
    metrics = {
        "val_soft_l1": totals["soft_l1"] / count,
        "val_mask_bce": totals["mask_bce"] / count,
        "val_mask_focal": totals["mask_focal"] / count,
        "val_dice": totals["dice"] / count,
        "val_ssim": totals["ssim"] / count,
        "val_edge": totals["edge"] / count,
        "val_iou": totals["iou"] / count,
    }
    metrics["score"] = (
        1.75 * metrics["val_iou"]
        + 1.25 * metrics["val_ssim"]
        - 0.75 * metrics["val_soft_l1"]
        - 0.50 * metrics["val_mask_bce"]
        - 0.50 * metrics["val_dice"]
    )
    return metrics


def run_stage1_training(
    config_path: str | Path | None = None,
    *,
    output_root: str | Path | None = None,
    resume_path: str | Path | None = None,
) -> dict[str, Any]:
    config = load_stage1_training_config(config_path)
    if output_root is not None:
        config.output.root_dir = str(output_root)
    if resume_path is not None:
        config.train.resume_path = str(resume_path)

    _set_seed(config.train.seed)
    search_roots = [config.config_base_dir] if config.config_base_dir else None
    output_dir = resolve_path(
        config.output.root_dir,
        search_roots=search_roots,
        base_dir=config.config_base_dir,
        allow_missing=True,
    )
    output_dir = ensure_dir(output_dir)
    checkpoints_dir = ensure_dir(output_dir / config.output.checkpoints_dir)
    previews_dir = ensure_dir(output_dir / config.output.previews_dir)
    write_json(config.to_dict(), output_dir / config.output.resolved_config_name)

    components_root, annotations_path, vocab_path = _resolve_data_files(config)
    vocab = _load_vocab(vocab_path)
    num_components = config.model.num_components or int(vocab["num_components"])

    train_dataset, val_dataset, train_loader, val_loader = _build_dataloaders(config, annotations_path)
    stylegan_module = load_component_stylegan_module()

    device_name = config.train.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    print(f"[stage1-train] device={device}")
    print(f"[stage1-train] annotations={annotations_path}")
    print(f"[stage1-train] vocab={vocab_path}")
    print(
        "[stage1-train] dataset "
        f"train={len(train_dataset)} val={len(val_dataset)} num_components={num_components}"
    )

    comp_codebook = stylegan_module.ComponentCodebook(
        num_entries=num_components,
        dim=config.model.codebook_dim,
    ).to(device)
    mapping_net = stylegan_module.MappingNetwork(
        z_dim=config.model.codebook_dim,
        w_dim=config.model.w_dim,
    ).to(device)
    stylegan = stylegan_module.ComponentStyleGAN(
        codebook_dim=config.model.codebook_dim,
        w_dim=config.model.w_dim,
        base_ch=config.model.base_channels,
        use_noise_in_blocks=config.model.use_noise_in_blocks,
        legacy_output_head=config.model.legacy_output_head,
    ).to(device)
    stylegan_ema = stylegan_module.ComponentStyleGAN(
        codebook_dim=config.model.codebook_dim,
        w_dim=config.model.w_dim,
        base_ch=config.model.base_channels,
        use_noise_in_blocks=config.model.use_noise_in_blocks,
        legacy_output_head=config.model.legacy_output_head,
    ).to(device)
    stylegan_ema.load_state_dict(stylegan.state_dict())
    discriminator = stylegan_module.StyleGANDiscriminator(
        in_ch=1,
        base_ch=config.model.discriminator_base_channels,
    ).to(device)

    optimizer_g = torch.optim.Adam(
        list(comp_codebook.parameters()) + list(mapping_net.parameters()) + list(stylegan.parameters()),
        lr=config.train.generator_lr,
        betas=(config.train.beta1, config.train.beta2),
        weight_decay=config.train.weight_decay,
    )
    optimizer_d = torch.optim.Adam(
        discriminator.parameters(),
        lr=config.train.discriminator_lr,
        betas=(config.train.beta1, config.train.beta2),
        weight_decay=config.train.weight_decay,
    )

    start_step = 0
    start_epoch = 0
    best_score = -float("inf")
    metrics_history: list[dict[str, Any]] = []
    if config.train.resume_path:
        resume_resolved = resolve_path(
            config.train.resume_path,
            search_roots=search_roots,
            base_dir=config.config_base_dir,
        )
        print(f"[stage1-train] resume={resume_resolved}")
        checkpoint = torch.load(str(resume_resolved), map_location="cpu")
        comp_codebook.load_state_dict(checkpoint["comp_codebook"])
        mapping_net.load_state_dict(checkpoint["mapping_net"])
        stylegan.load_state_dict(checkpoint["stylegan"])
        stylegan_ema.load_state_dict(checkpoint.get("stylegan_ema", checkpoint["stylegan"]))
        if "discriminator" in checkpoint:
            discriminator.load_state_dict(checkpoint["discriminator"])
        if "optimizer_g" in checkpoint:
            optimizer_g.load_state_dict(checkpoint["optimizer_g"])
        if "optimizer_d" in checkpoint:
            optimizer_d.load_state_dict(checkpoint["optimizer_d"])
        start_step = int(checkpoint.get("step", 0))
        start_epoch = int(checkpoint.get("epoch", 0))
        best_score = float(checkpoint.get("best_score", best_score))
        metrics_history = list(checkpoint.get("metrics_history", []))

    global_step = start_step
    train_start = time.time()
    stop_training = False

    for epoch in range(start_epoch, config.train.max_epochs):
        comp_codebook.train()
        mapping_net.train()
        stylegan.train()
        discriminator.train()

        for batch in train_loader:
            global_step += 1
            comp_ids = batch["comp_id"].to(device=device, dtype=torch.long)
            mask = batch["mask"].to(device=device, dtype=torch.float32)
            soft = batch["soft"].to(device=device, dtype=torch.float32)

            gan_weight = config.loss.lambda_adv * _gan_ramp_weight(
                global_step,
                config.train.gan_start_step,
                config.train.gan_ramp_steps,
            )

            d_loss_value = 0.0
            if gan_weight > 0.0:
                optimizer_d.zero_grad(set_to_none=True)
                with torch.no_grad():
                    fake_detached = _generator_forward(comp_ids, comp_codebook, mapping_net, stylegan)
                real_logits = discriminator(soft)
                fake_logits = discriminator(fake_detached)
                d_loss = F.softplus(fake_logits).mean() + F.softplus(-real_logits).mean()

                if config.train.d_reg_every > 0 and global_step % config.train.d_reg_every == 0:
                    soft_r1 = soft.detach().requires_grad_(True)
                    real_logits_r1 = discriminator(soft_r1)
                    gradients = torch.autograd.grad(
                        outputs=real_logits_r1.sum(),
                        inputs=soft_r1,
                        create_graph=True,
                    )[0]
                    r1_penalty = gradients.pow(2).flatten(1).sum(dim=1).mean()
                    d_loss = d_loss + 0.5 * config.train.r1_gamma * config.train.d_reg_every * r1_penalty

                d_loss.backward()
                optimizer_d.step()
                d_loss_value = float(d_loss.item())

            optimizer_g.zero_grad(set_to_none=True)
            fake = _generator_forward(comp_ids, comp_codebook, mapping_net, stylegan)

            losses = {
                "soft_l1": F.l1_loss(fake, soft),
                "mask_bce": F.binary_cross_entropy(fake, mask),
                "mask_focal": _binary_focal_loss(
                    fake,
                    mask,
                    alpha=config.loss.focal_alpha,
                    gamma=config.loss.focal_gamma,
                ),
                "dice": _dice_loss(fake, mask),
                "ssim": _ssim_loss(fake, soft),
                "edge": F.l1_loss(_sobel_edges(fake), _sobel_edges(soft)),
                "diversity": _codebook_diversity_loss(
                    comp_codebook,
                    margin=config.loss.diversity_margin,
                ),
            }
            if gan_weight > 0.0:
                losses["adv"] = F.softplus(-discriminator(fake)).mean()
            else:
                losses["adv"] = fake.new_tensor(0.0)

            total_g = (
                config.loss.lambda_soft_l1 * losses["soft_l1"]
                + config.loss.lambda_mask_bce * losses["mask_bce"]
                + config.loss.lambda_mask_focal * losses["mask_focal"]
                + config.loss.lambda_dice * losses["dice"]
                + config.loss.lambda_ssim * losses["ssim"]
                + config.loss.lambda_edge * losses["edge"]
                + gan_weight * losses["adv"]
                + config.loss.lambda_diversity * losses["diversity"]
            )
            total_g.backward()
            if config.train.grad_clip and config.train.grad_clip > 0:
                nn.utils.clip_grad_norm_(
                    list(comp_codebook.parameters()) + list(mapping_net.parameters()) + list(stylegan.parameters()),
                    max_norm=config.train.grad_clip,
                )
            optimizer_g.step()
            _ema_update(stylegan, stylegan_ema, decay=config.train.ema_decay)

            if global_step % config.train.log_every == 0 or global_step == 1:
                metrics = {
                    "g_total": float(total_g.item()),
                    "d_total": d_loss_value,
                    "soft_l1": float(losses["soft_l1"].item()),
                    "mask_bce": float(losses["mask_bce"].item()),
                    "mask_focal": float(losses["mask_focal"].item()),
                    "dice": float(losses["dice"].item()),
                    "ssim_loss": float(losses["ssim"].item()),
                    "edge": float(losses["edge"].item()),
                    "adv": float(losses["adv"].item()),
                    "gan_weight": float(gan_weight),
                }
                elapsed = time.time() - train_start
                print(
                    f"[stage1-train] step={global_step} epoch={epoch + 1} "
                    f"elapsed={elapsed:.1f}s {_format_metrics(metrics)}"
                )

            if global_step % config.train.validate_every == 0 or global_step == 1:
                preview_path = previews_dir / f"preview_step_{global_step:07d}.png"
                val_metrics = _evaluate(
                    comp_codebook=comp_codebook,
                    mapping_net=mapping_net,
                    stylegan_ema=stylegan_ema,
                    dataloader=val_loader,
                    device=device,
                    loss_config=config.loss,
                    preview_path=preview_path,
                    preview_count=config.train.preview_count,
                )
                val_metrics["step"] = global_step
                val_metrics["epoch"] = epoch + 1
                metrics_history.append(val_metrics)
                write_json(metrics_history, output_dir / config.output.metrics_name)
                print(f"[stage1-train] validation {_format_metrics(val_metrics)}")

                checkpoint_payload = _build_checkpoint_payload(
                    config,
                    step=global_step,
                    epoch=epoch + 1,
                    best_score=best_score,
                    num_components=num_components,
                    comp_codebook=comp_codebook,
                    mapping_net=mapping_net,
                    stylegan=stylegan,
                    stylegan_ema=stylegan_ema,
                    discriminator=discriminator,
                    optimizer_g=optimizer_g,
                    optimizer_d=optimizer_d,
                    metrics_history=metrics_history,
                )
                _save_checkpoint(
                    checkpoint_payload,
                    checkpoints_dir / config.output.last_checkpoint_name,
                )
                if val_metrics["score"] > best_score:
                    best_score = float(val_metrics["score"])
                    checkpoint_payload["best_score"] = best_score
                    best_path = _save_checkpoint(
                        checkpoint_payload,
                        checkpoints_dir / config.output.best_checkpoint_name,
                    )
                    print(f"[stage1-train] new_best score={best_score:.4f} path={best_path}")

            if global_step % config.train.save_every == 0:
                checkpoint_payload = _build_checkpoint_payload(
                    config,
                    step=global_step,
                    epoch=epoch + 1,
                    best_score=best_score,
                    num_components=num_components,
                    comp_codebook=comp_codebook,
                    mapping_net=mapping_net,
                    stylegan=stylegan,
                    stylegan_ema=stylegan_ema,
                    discriminator=discriminator,
                    optimizer_g=optimizer_g,
                    optimizer_d=optimizer_d,
                    metrics_history=metrics_history,
                )
                step_path = checkpoints_dir / f"stage1_step_{global_step:07d}.pth"
                _save_checkpoint(checkpoint_payload, step_path)
                print(f"[stage1-train] checkpoint={step_path}")

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
        num_components=num_components,
        comp_codebook=comp_codebook,
        mapping_net=mapping_net,
        stylegan=stylegan,
        stylegan_ema=stylegan_ema,
        discriminator=discriminator,
        optimizer_g=optimizer_g,
        optimizer_d=optimizer_d,
        metrics_history=metrics_history,
    )
    last_path = _save_checkpoint(final_payload, checkpoints_dir / config.output.last_checkpoint_name)

    summary = {
        "stage": "stage1_training",
        "device": str(device),
        "output_dir": str(output_dir),
        "components_root": str(components_root),
        "annotations_path": str(annotations_path),
        "vocab_path": str(vocab_path),
        "num_components": num_components,
        "train_count": len(train_dataset),
        "val_count": len(val_dataset),
        "global_step": global_step,
        "best_score": best_score,
        "last_checkpoint": str(last_path),
        "best_checkpoint": str((checkpoints_dir / config.output.best_checkpoint_name).resolve()),
        "metrics_path": str((output_dir / config.output.metrics_name).resolve()),
    }
    write_json(summary, output_dir / "training_summary.json")
    return summary


def build_stage1_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train Stage1 component codebook + StyleGAN checkpoint.",
    )
    parser.add_argument("--config", type=str, default=None, help="YAML or JSON config path.")
    parser.add_argument("--output", type=str, default=None, help="Optional output root override.")
    parser.add_argument("--resume", type=str, default=None, help="Resume checkpoint path.")
    return parser


def main() -> None:
    parser = build_stage1_arg_parser()
    args = parser.parse_args()
    summary = run_stage1_training(
        args.config,
        output_root=args.output,
        resume_path=args.resume,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
