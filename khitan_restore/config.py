"""Configuration dataclasses for the unified pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .io_utils import load_structured_file


@dataclass
class SuperResolutionConfig:
    enabled: bool = True
    device: str | None = None
    task: str = "real_sr"
    scale: int = 4
    noise: int = 15
    jpeg: int = 40
    training_patch_size: int = 64
    large_model: bool = False
    model_source: str = "pretrained"
    pretrained_model_path: str = (
        "model_zoo/swinir/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth"
    )
    custom_model_path: str | None = None
    model_path: str = "model_zoo/swinir/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth"
    auto_download: bool = True
    tile: int | None = None
    tile_overlap: int = 32


@dataclass
class DatasetConfig:
    root_dir: str | None = None
    data_dir: str | None = None
    components_dir: str | None = None
    checkpoints_dir: str | None = None


@dataclass
class SegmentationConfig:
    enabled: bool = True
    save_patches: bool = True
    save_debug: bool = False
    config_overrides: dict[str, Any] = field(default_factory=dict)


@dataclass
class RestorationConfig:
    enabled: bool = True
    ckpt_path: str = "checkpoints/stage1_best.pth"
    config_path: str = "test/cospnet.yaml"
    device: str | None = None
    use_ema: bool = True
    topk: int = 5
    bank_batch_size: int = 64
    seed: int = 1234
    save_bank_preview: bool = True
    image_name: str | None = None
    image_path: str | None = None
    num_components: int | None = None


@dataclass
class RefinementConfig:
    enabled: bool = True
    mode: str = "rule"
    model_path: str | None = None
    device: str | None = None
    lr_dir: str | None = None
    rank: int = 1
    prior_threshold: float = 0.15
    base_prior_weight: float = 0.40
    edge_prior_weight: float = 0.35
    disagreement_penalty: float = 0.30
    detail_boost: float = 0.15
    mask_blur_sigma: float = 2.0
    final_sharpen: float = 0.20
    fusion_scales: list[float] = field(default_factory=lambda: [1.0, 0.5])
    min_component_area: int = 16


@dataclass
class OutputConfig:
    root_dir: str = "runs/default"
    super_resolution_dir: str = "01_super_resolution"
    segmentation_dir: str = "02_segmentation"
    restoration_dir: str = "03_restoration"
    refinement_dir: str = "04_refinement"
    manifest_name: str = "pipeline_manifest.json"


@dataclass
class PipelineConfig:
    input_path: str = "sample_data/input.png"
    search_roots: list[str] = field(default_factory=list)
    config_base_dir: str | None = None
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    super_resolution: SuperResolutionConfig = field(default_factory=SuperResolutionConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    restoration: RestorationConfig = field(default_factory=RestorationConfig)
    refinement: RefinementConfig = field(default_factory=RefinementConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

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


def _from_dict(data: dict[str, Any]) -> PipelineConfig:
    dataset = DatasetConfig(**data.get("dataset", {}))
    sr_data = dict(data.get("super_resolution", {}))
    legacy_model_path = sr_data.get("model_path")
    model_source = str(sr_data.get("model_source", "pretrained")).strip().lower()
    if legacy_model_path and not sr_data.get("pretrained_model_path"):
        sr_data["pretrained_model_path"] = legacy_model_path
    if legacy_model_path and model_source == "custom" and not sr_data.get("custom_model_path"):
        sr_data["custom_model_path"] = legacy_model_path
    super_resolution = SuperResolutionConfig(**sr_data)
    segmentation = SegmentationConfig(**data.get("segmentation", {}))
    restoration = RestorationConfig(**data.get("restoration", {}))
    refinement = RefinementConfig(**data.get("refinement", {}))
    output = OutputConfig(**data.get("output", {}))
    return PipelineConfig(
        input_path=data.get("input_path", PipelineConfig.input_path),
        search_roots=list(data.get("search_roots", [])),
        config_base_dir=data.get("config_base_dir"),
        dataset=dataset,
        super_resolution=super_resolution,
        segmentation=segmentation,
        restoration=restoration,
        refinement=refinement,
        output=output,
    )


def load_pipeline_config(config_path: str | Path | None = None) -> PipelineConfig:
    defaults = PipelineConfig().to_dict()
    if config_path is None:
        return _from_dict(defaults)

    resolved = Path(config_path).expanduser().resolve()
    raw = load_structured_file(resolved)
    merged = _deep_merge(defaults, raw)
    merged["config_base_dir"] = str(resolved.parent)
    return _from_dict(merged)
