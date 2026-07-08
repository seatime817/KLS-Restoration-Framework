"""SwinIR wrapper used by the unified pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import requests
import torch

from .config import SuperResolutionConfig
from .io_utils import ensure_dir, list_images, resolve_path, write_json
from .legacy_loader import load_swinir_module


def _download_model_weights(model_path: Path) -> None:
    url = (
        "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/"
        f"{model_path.name}"
    )
    model_path.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, allow_redirects=True, timeout=120)
    response.raise_for_status()
    model_path.write_bytes(response.content)


def _build_args(config: SuperResolutionConfig, model_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        task=config.task,
        scale=config.scale,
        noise=config.noise,
        jpeg=config.jpeg,
        training_patch_size=config.training_patch_size,
        large_model=config.large_model,
        model_path=str(model_path),
        folder_lq=None,
        folder_gt=None,
        tile=config.tile,
        tile_overlap=config.tile_overlap,
    )


def _resolve_model_path(
    config: SuperResolutionConfig,
    *,
    search_roots: list[str] | None = None,
    base_dir: str | Path | None = None,
) -> tuple[str, Path]:
    model_source = str(config.model_source or "pretrained").strip().lower()
    if model_source not in {"pretrained", "custom"}:
        raise ValueError(
            "super_resolution.model_source must be either 'pretrained' or 'custom'"
        )

    if model_source == "pretrained":
        configured_path = config.pretrained_model_path or config.model_path
    else:
        configured_path = config.custom_model_path or config.model_path
        if not configured_path:
            raise ValueError(
                "super_resolution.custom_model_path is required when model_source='custom'"
            )

    model_path = resolve_path(
        configured_path,
        search_roots=search_roots,
        base_dir=base_dir,
        allow_missing=True,
    )
    return model_source, model_path


def _read_input_image(image_path: Path, task: str) -> np.ndarray:
    if task in {"gray_dn", "jpeg_car"}:
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        return image.astype(np.float32)[..., None] / 255.0

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    return image.astype(np.float32) / 255.0


def _save_output_image(output_tensor: torch.Tensor, output_path: Path) -> None:
    output = output_tensor.data.squeeze().float().cpu().clamp_(0, 1).numpy()
    if output.ndim == 3:
        if output.shape[0] == 1:
            output = output[0]
        else:
            output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
    output = (output * 255.0).round().astype(np.uint8)
    cv2.imwrite(str(output_path), output)


def run_super_resolution(
    input_path: str | Path,
    output_dir: str | Path,
    config: SuperResolutionConfig,
    *,
    search_roots: list[str] | None = None,
    base_dir: str | Path | None = None,
) -> dict:
    legacy = load_swinir_module()
    input_resolved = resolve_path(input_path, search_roots=search_roots, base_dir=base_dir)
    output_resolved = ensure_dir(output_dir)
    image_output_dir = ensure_dir(output_resolved / "images")
    image_paths = list_images(input_resolved)

    model_source, model_path = _resolve_model_path(
        config,
        search_roots=search_roots,
        base_dir=base_dir,
    )
    if not model_path.exists():
        if model_source == "pretrained" and config.auto_download:
            _download_model_weights(model_path)
        else:
            raise FileNotFoundError(f"SwinIR model not found: {model_path}")

    args = _build_args(config, model_path)
    device_name = config.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    print(f"[super-resolution] input={input_resolved}")
    print(
        f"[super-resolution] device={device} model_source={model_source} model={model_path}"
    )
    print(f"[super-resolution] total_images={len(image_paths)}")
    _, _, _, window_size = legacy.setup(args)
    model = legacy.define_model(args)
    model.eval()
    model = model.to(device)

    items: list[dict] = []
    for index, image_path in enumerate(image_paths, start=1):
        print(f"[super-resolution] ({index}/{len(image_paths)}) processing {image_path}")
        image = _read_input_image(image_path, config.task)
        image_tensor = np.transpose(
            image if image.shape[2] == 1 else image[:, :, [2, 1, 0]],
            (2, 0, 1),
        )
        image_tensor = torch.from_numpy(image_tensor).float().unsqueeze(0).to(device)

        with torch.no_grad():
            _, _, h_old, w_old = image_tensor.size()
            h_pad = (h_old // window_size + 1) * window_size - h_old
            w_pad = (w_old // window_size + 1) * window_size - w_old
            image_tensor = torch.cat([image_tensor, torch.flip(image_tensor, [2])], 2)[
                :, :, : h_old + h_pad, :
            ]
            image_tensor = torch.cat([image_tensor, torch.flip(image_tensor, [3])], 3)[
                :, :, :, : w_old + w_pad
            ]
            output_tensor = legacy.test(image_tensor, model, args, window_size)
            output_tensor = output_tensor[..., : h_old * config.scale, : w_old * config.scale]

        output_path = image_output_dir / f"{image_path.stem}_sr.png"
        _save_output_image(output_tensor, output_path)
        items.append(
            {
                "image_name": image_path.stem,
                "input_path": str(image_path.resolve()),
                "output_path": str(output_path.resolve()),
                "output_name": output_path.name,
                "scale": config.scale,
                "task": config.task,
            }
        )
    print(f"[super-resolution] completed count={len(items)} output_dir={image_output_dir}")

    manifest = {
        "stage": "super_resolution",
        "device": str(device),
        "input_path": str(input_resolved),
        "output_dir": str(output_resolved),
        "output_image_dir": str(image_output_dir),
        "model_source": model_source,
        "model_path": str(model_path.resolve()),
        "count": len(items),
        "items": items,
    }
    write_json(manifest, output_resolved / "manifest.json")
    return manifest
