"""Segmentation wrapper used by the unified pipeline."""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path

from .config import SegmentationConfig
from .io_utils import ensure_dir, list_images, resolve_path, write_json
from .legacy_loader import load_segment_module


def _build_segment_config(config: SegmentationConfig):
    legacy = load_segment_module()
    valid_fields = {item.name for item in fields(legacy.SegmentConfig)}
    unknown = sorted(set(config.config_overrides) - valid_fields)
    if unknown:
        unknown_text = ", ".join(unknown)
        raise ValueError(f"Unknown segmentation config keys: {unknown_text}")
    return legacy.SegmentConfig(**config.config_overrides)


def run_segmentation(
    input_path: str | Path,
    output_dir: str | Path,
    config: SegmentationConfig,
    *,
    search_roots: list[str] | None = None,
    base_dir: str | Path | None = None,
) -> dict:
    legacy = load_segment_module()
    resolved_input = resolve_path(input_path, search_roots=search_roots, base_dir=base_dir)
    resolved_output = ensure_dir(output_dir)
    segment_config = _build_segment_config(config)
    image_paths = list_images(resolved_input)
    print(f"[segmentation] input={resolved_input}")
    print(f"[segmentation] total_images={len(image_paths)} output_dir={resolved_output}")

    items = []
    for index, image_path in enumerate(image_paths, start=1):
        print(f"[segmentation] ({index}/{len(image_paths)}) processing {image_path}")
        result = legacy.segment_single_image(
            image_path=str(image_path.resolve()),
            cfg=segment_config,
            output_dir=str(resolved_output),
            save_patches=config.save_patches,
            save_debug=config.save_debug,
        )
        items.append(result)
    print(f"[segmentation] completed count={len(items)}")

    payload = {
        "stage": "segmentation",
        "input_path": str(resolved_input),
        "output_dir": str(resolved_output),
        "count": len(items),
        "results": items,
        "items": items,
    }
    segment_json = write_json(payload, resolved_output / "segment_results.json")
    write_json(
        [
            {
                "image_name": item["image_name"],
                "image_path": item["image_path"],
                "num_components": item["num_components"],
            }
            for item in items
        ],
        resolved_output / "segment_summary.json",
    )

    manifest = {
        "stage": "segmentation",
        "input_path": str(resolved_input),
        "output_dir": str(resolved_output),
        "segment_json": str(segment_json),
        "count": len(items),
        "items": items,
    }
    write_json(manifest, resolved_output / "manifest.json")
    return manifest
