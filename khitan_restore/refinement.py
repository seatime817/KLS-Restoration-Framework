"""Prior-guided refinement that fuses SwinIR output with GAN-generated text priors."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .config import RefinementConfig
from .io_utils import ensure_dir, resolve_path, write_json


def _read_gray_float(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return image.astype(np.float32) / 255.0


def _save_gray_float(path: str | Path, image: np.ndarray) -> str:
    array = np.clip(image, 0.0, 1.0)
    output = (array * 255.0).round().astype(np.uint8)
    cv2.imwrite(str(path), output)
    return str(Path(path).resolve())


def _load_json(path: Path) -> dict | list:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _normalize_map(image: np.ndarray) -> np.ndarray:
    minimum = float(image.min())
    maximum = float(image.max())
    if maximum - minimum < 1e-6:
        return np.zeros_like(image, dtype=np.float32)
    return ((image - minimum) / (maximum - minimum)).astype(np.float32)


def _edge_map(image: np.ndarray) -> np.ndarray:
    grad_x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(grad_x, grad_y)
    return _normalize_map(magnitude)


def _adain_match(prior_patch: np.ndarray, sr_patch: np.ndarray, mask: np.ndarray) -> np.ndarray:
    valid = mask > 1e-3
    if valid.sum() < 4:
        return prior_patch.copy()

    prior_values = prior_patch[valid]
    sr_values = sr_patch[valid]
    prior_mean = float(prior_values.mean())
    sr_mean = float(sr_values.mean())
    prior_std = float(prior_values.std())
    sr_std = float(sr_values.std())

    normalized = (prior_patch - prior_mean) / max(prior_std, 1e-6)
    matched = normalized * max(sr_std, 1e-6) + sr_mean
    return np.clip(matched, 0.0, 1.0).astype(np.float32)


def _resize_patch(image: np.ndarray, scale: float) -> np.ndarray:
    if abs(scale - 1.0) < 1e-6:
        return image
    height, width = image.shape[:2]
    scaled_width = max(4, int(round(width * scale)))
    scaled_height = max(4, int(round(height * scale)))
    return cv2.resize(image, (scaled_width, scaled_height), interpolation=cv2.INTER_CUBIC)


def _resize_back(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    width, height = size
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)


def _build_alpha_map(sr_patch: np.ndarray, prior_patch: np.ndarray, config: RefinementConfig) -> np.ndarray:
    smooth_prior = cv2.GaussianBlur(prior_patch, (0, 0), 0.9)
    smooth_sr = cv2.GaussianBlur(sr_patch, (0, 0), 0.6)
    prior_mask = (smooth_prior > config.prior_threshold).astype(np.float32)
    if float(prior_mask.sum()) < 1.0:
        return np.zeros_like(sr_patch, dtype=np.float32)

    sr_edges = _edge_map(smooth_sr)
    prior_edges = _edge_map(smooth_prior)
    disagreement = np.abs(sr_edges - prior_edges)

    alpha = config.base_prior_weight * prior_mask
    alpha += config.edge_prior_weight * prior_edges * prior_mask
    alpha *= 1.0 - config.disagreement_penalty * disagreement
    alpha = np.clip(alpha, 0.0, 0.95)

    if config.mask_blur_sigma > 0:
        alpha = cv2.GaussianBlur(alpha, (0, 0), config.mask_blur_sigma)

    alpha = np.maximum(alpha, 0.12 * prior_mask)
    return np.clip(alpha, 0.0, 0.95).astype(np.float32)


def _fuse_component_patch(
    sr_patch: np.ndarray,
    prior_patch: np.ndarray,
    config: RefinementConfig,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = sr_patch.shape[:2]
    fused_patches: list[np.ndarray] = []
    alpha_patches: list[np.ndarray] = []

    for scale in config.fusion_scales:
        scaled_sr = _resize_patch(sr_patch, scale)
        scaled_prior = _resize_patch(prior_patch, scale)
        alpha = _build_alpha_map(scaled_sr, scaled_prior, config)
        matched_prior = _adain_match(scaled_prior, scaled_sr, alpha)
        prior_smooth = cv2.GaussianBlur(matched_prior, (0, 0), 1.2)
        prior_detail = matched_prior - prior_smooth
        fused = scaled_sr * (1.0 - alpha) + matched_prior * alpha
        fused = fused + config.detail_boost * alpha * prior_detail
        fused = np.clip(fused, 0.0, 1.0).astype(np.float32)

        fused_patches.append(_resize_back(fused, (width, height)))
        alpha_patches.append(_resize_back(alpha, (width, height)))

    fused_patch = np.mean(np.stack(fused_patches, axis=0), axis=0).astype(np.float32)
    alpha_patch = np.max(np.stack(alpha_patches, axis=0), axis=0).astype(np.float32)
    return np.clip(fused_patch, 0.0, 1.0), np.clip(alpha_patch, 0.0, 0.95)


def _unsharp_mask(image: np.ndarray, amount: float) -> np.ndarray:
    if amount <= 0:
        return image
    blurred = cv2.GaussianBlur(image, (0, 0), 1.2)
    sharpened = image + amount * (image - blurred)
    return np.clip(sharpened, 0.0, 1.0).astype(np.float32)


def _otsu_threshold(image: np.ndarray) -> float:
    image_uint8 = (np.clip(image, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    threshold, _ = cv2.threshold(image_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return float(threshold) / 255.0


def _remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)
    if int(binary.sum()) == 0:
        return np.zeros_like(binary, dtype=np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    cleaned = np.zeros_like(binary, dtype=np.uint8)
    for label_index in range(1, num_labels):
        area = int(stats[label_index, cv2.CC_STAT_AREA])
        if area >= min_area:
            cleaned[labels == label_index] = 1
    return cleaned


def _keep_components_touching_seed(mask: np.ndarray, seed: np.ndarray) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)
    seed_binary = (seed > 0).astype(np.uint8)
    if int(binary.sum()) == 0 or int(seed_binary.sum()) == 0:
        return np.zeros_like(binary, dtype=np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    kept = np.zeros_like(binary, dtype=np.uint8)
    for label_index in range(1, num_labels):
        if int(stats[label_index, cv2.CC_STAT_AREA]) <= 0:
            continue
        component_mask = labels == label_index
        if np.any(seed_binary[component_mask] > 0):
            kept[component_mask] = 1
    return kept


def _component_text_mask(
    sr_patch: np.ndarray,
    prior_patch: np.ndarray,
    fused_patch: np.ndarray,
    alpha_patch: np.ndarray,
    config: RefinementConfig,
) -> np.ndarray:
    smooth_prior = cv2.GaussianBlur(prior_patch, (0, 0), 0.9)
    support = np.maximum(smooth_prior, 0.80 * fused_patch)
    support_threshold = max(config.prior_threshold * 0.80, min(0.55, _otsu_threshold(support) * 0.72))
    matte = np.clip((support - support_threshold) / max(1.0e-6, 1.0 - support_threshold), 0.0, 1.0)
    matte = np.maximum(
        matte,
        0.55 * (smooth_prior >= max(config.prior_threshold, 0.10)).astype(np.float32),
    )
    matte = np.maximum(matte, 0.25 * np.clip(alpha_patch, 0.0, 1.0))
    matte = cv2.GaussianBlur(matte.astype(np.float32), (0, 0), 0.9)
    return np.clip(matte, 0.0, 1.0).astype(np.float32)


def _clean_final_text(final_soft: np.ndarray, prior_image: np.ndarray, config: RefinementConfig) -> np.ndarray:
    smooth_soft = cv2.GaussianBlur(final_soft, (0, 0), 0.6)
    smooth_prior = cv2.GaussianBlur(prior_image, (0, 0), 0.8)
    support = np.maximum(smooth_soft, 0.65 * smooth_prior)
    support_threshold = max(config.prior_threshold * 0.85, min(0.50, _otsu_threshold(support) * 0.78))
    support_mask = (support >= support_threshold).astype(np.uint8)
    support_mask = _remove_small_components(support_mask, max(8, config.min_component_area // 2))
    if int(support_mask.sum()) == 0:
        return np.zeros_like(final_soft, dtype=np.float32)

    support_values = smooth_soft[support_mask > 0]
    lo = float(np.percentile(support_values, 10))
    hi = float(np.percentile(support_values, 99))
    if hi - lo < 1.0e-6:
        normalized = np.clip(smooth_soft, 0.0, 1.0)
    else:
        normalized = np.clip((smooth_soft - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)

    high_threshold = max(0.52, min(0.82, _otsu_threshold(normalized) * 1.05))
    low_threshold = max(0.30, min(high_threshold - 0.08, high_threshold * 0.72))

    strong_mask = ((normalized >= high_threshold) & (support_mask > 0)).astype(np.uint8)
    weak_mask = ((normalized >= low_threshold) & (support_mask > 0)).astype(np.uint8)
    strong_mask = _remove_small_components(strong_mask, max(6, config.min_component_area // 3))
    connected_mask = _keep_components_touching_seed(weak_mask, strong_mask)
    connected_mask = _remove_small_components(connected_mask, max(8, config.min_component_area // 2))

    if int(connected_mask.sum()) == 0:
        connected_mask = strong_mask if int(strong_mask.sum()) > 0 else support_mask

    edge_band = np.clip((normalized - low_threshold) / max(1.0e-6, high_threshold - low_threshold), 0.0, 1.0)
    clean = edge_band * connected_mask.astype(np.float32)
    clean = np.clip(np.maximum(clean, 0.92 * strong_mask.astype(np.float32)), 0.0, 1.0)
    clean = cv2.GaussianBlur(clean.astype(np.float32), (0, 0), 0.45)
    return np.clip(clean, 0.0, 1.0).astype(np.float32)


def _resize_like(image: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    target_h, target_w = shape
    if image.shape[:2] == (target_h, target_w):
        return image
    return cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_CUBIC)


def _make_labeled_strip(panels: list[tuple[str, np.ndarray]]) -> np.ndarray:
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


def _comparison_strip(
    lr_image: np.ndarray | None,
    sr_image: np.ndarray,
    prior_image: np.ndarray,
    final_image: np.ndarray,
) -> np.ndarray:
    panels: list[tuple[str, np.ndarray]] = []
    base_shape = sr_image.shape[:2]
    if lr_image is not None:
        panels.append(("LR", _resize_like(lr_image, base_shape)))
    panels.extend(
        [
            ("SR", sr_image),
            ("PRIOR", prior_image),
            ("FINAL", final_image),
        ]
    )
    return _make_labeled_strip(panels)


def _alpha_overview_panel(
    lr_image: np.ndarray | None,
    sr_image: np.ndarray,
    prior_image: np.ndarray,
    final_image: np.ndarray,
) -> np.ndarray:
    panels: list[tuple[str, np.ndarray]] = []
    base_shape = sr_image.shape[:2]
    if lr_image is not None:
        panels.append(("LR", _resize_like(lr_image, base_shape)))
    panels.extend(
        [
            ("SR", sr_image),
            ("PRIOR", prior_image),
            ("FINAL", final_image),
        ]
    )
    return _make_labeled_strip(panels)


def _load_super_resolution_source_map(restoration_root: Path) -> dict[str, str]:
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
            payload = _load_json(candidate)
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


def _resolve_lr_image_path(
    image_result: dict,
    sr_image_path: Path,
    source_map: dict[str, str],
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

    sr_stem = sr_image_path.stem
    base_stem = sr_stem[:-3] if sr_stem.endswith("_sr") else sr_stem
    for output_path, input_path in source_map.items():
        output_stem = Path(output_path).stem
        if output_stem == sr_stem or output_stem == f"{base_stem}_sr":
            candidate = Path(input_path).expanduser()
            if candidate.exists():
                return candidate.resolve()
    return None


def _load_results(results_path: Path) -> list[dict]:
    data = _load_json(results_path)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {results_path}")
    return data


def run_prior_guided_refinement(
    restoration_input: str | Path,
    output_dir: str | Path,
    config: RefinementConfig,
    *,
    search_roots: list[str] | None = None,
    base_dir: str | Path | None = None,
) -> dict:
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

    results = _load_results(results_path)
    resolved_output = ensure_dir(output_dir)
    sr_source_map = _load_super_resolution_source_map(restoration_root)
    print(f"[refinement] restoration_input={resolved_input}")
    print(f"[refinement] total_images={len(results)} output_dir={resolved_output}")
    items: list[dict] = []

    for index, image_result in enumerate(results, start=1):
        image_name = image_result.get("image_name") or Path(image_result["image_path"]).stem
        print(f"[refinement] ({index}/{len(results)}) image={image_name}")
        image_output_dir = ensure_dir(resolved_output / image_name)
        image_restore_dir = restoration_root / image_name
        sr_image_path = resolve_path(
            image_result["image_path"],
            search_roots=search_roots,
            base_dir=base_dir,
        )
        lr_image_path = _resolve_lr_image_path(image_result, sr_image_path, sr_source_map)
        prior_path = image_restore_dir / f"reconstructed_rank{config.rank}.png"
        if not prior_path.exists():
            raise FileNotFoundError(f"Missing reconstructed prior image: {prior_path}")

        sr_image = _read_gray_float(sr_image_path)
        lr_image = _read_gray_float(lr_image_path) if lr_image_path is not None else None
        prior_image = _read_gray_float(prior_path)
        if sr_image.shape != prior_image.shape:
            prior_image = cv2.resize(
                prior_image,
                (sr_image.shape[1], sr_image.shape[0]),
                interpolation=cv2.INTER_CUBIC,
            )

        final_image = np.zeros_like(prior_image, dtype=np.float32)
        alpha_canvas = np.zeros_like(sr_image, dtype=np.float32)
        component_metrics: list[dict] = []

        components = image_result.get("components", [])
        components = sorted(
            components,
            key=lambda component: int(component["bbox"][2]) * int(component["bbox"][3]),
            reverse=True,
        )

        for component in components:
            x, y, width, height = [int(value) for value in component["bbox"]]
            if width * height < config.min_component_area:
                continue

            sr_patch = sr_image[y:y + height, x:x + width]
            prior_patch = prior_image[y:y + height, x:x + width]
            if sr_patch.size == 0 or prior_patch.size == 0:
                continue

            fused_patch, alpha_patch = _fuse_component_patch(sr_patch, prior_patch, config)
            text_mask = _component_text_mask(sr_patch, prior_patch, fused_patch, alpha_patch, config)
            guided_patch = 0.92 * fused_patch + 0.08 * cv2.GaussianBlur(prior_patch, (0, 0), 0.8)
            clean_patch = guided_patch * text_mask
            region = final_image[y:y + height, x:x + width]
            final_image[y:y + height, x:x + width] = np.maximum(region, clean_patch)
            alpha_canvas[y:y + height, x:x + width] = np.maximum(
                alpha_canvas[y:y + height, x:x + width],
                alpha_patch,
            )

            component_metrics.append(
                {
                    "index": int(component.get("index", len(component_metrics))),
                    "bbox": [x, y, width, height],
                    "mean_alpha": float(alpha_patch.mean()),
                    "max_alpha": float(alpha_patch.max()),
                    "prior_fill": float((prior_patch > config.prior_threshold).mean()),
                    "text_fill": float((text_mask > 0.15).mean()),
                }
            )

        final_soft = _unsharp_mask(final_image, config.final_sharpen)
        final_soft = cv2.GaussianBlur(final_soft, (0, 0), 0.35)
        final_soft = np.clip(final_soft, 0.0, 1.0).astype(np.float32)
        final_image = _clean_final_text(final_soft, prior_image, config)

        final_path = _save_gray_float(image_output_dir / "final_refined.png", final_image)
        final_soft_path = _save_gray_float(image_output_dir / "final_soft.png", final_soft)
        alpha_only_path = _save_gray_float(image_output_dir / "alpha_only.png", alpha_canvas)
        sr_copy_path = _save_gray_float(image_output_dir / "sr_input.png", sr_image)
        lr_copy_path = (
            _save_gray_float(image_output_dir / "lr_input.png", lr_image)
            if lr_image is not None
            else None
        )
        prior_copy_path = _save_gray_float(image_output_dir / "gan_prior.png", prior_image)
        alpha_path = _save_gray_float(
            image_output_dir / "alpha_map.png",
            _alpha_overview_panel(lr_image, sr_image, prior_image, final_image).astype(np.float32) / 255.0,
        )
        comparison_path = _save_gray_float(
            image_output_dir / "comparison_strip.png",
            _comparison_strip(lr_image, sr_image, prior_image, final_image).astype(np.float32) / 255.0,
        )

        item_manifest = {
            "image_name": image_name,
            "lr_image_path": str(lr_image_path.resolve()) if lr_image_path is not None else None,
            "sr_image_path": str(sr_image_path.resolve()),
            "prior_image_path": str(prior_path.resolve()),
            "final_image_path": final_path,
            "final_soft_path": final_soft_path,
            "alpha_map_path": alpha_path,
            "alpha_only_path": alpha_only_path,
            "comparison_strip_path": comparison_path,
            "lr_copy_path": lr_copy_path,
            "sr_copy_path": sr_copy_path,
            "gan_prior_copy_path": prior_copy_path,
            "num_components": len(component_metrics),
            "mean_alpha": float(alpha_canvas.mean()),
            "components": component_metrics,
        }
        write_json(item_manifest, image_output_dir / "refinement_results.json")
        items.append(item_manifest)
    print(f"[refinement] completed count={len(items)}")

    manifest = {
        "stage": "refinement",
        "restoration_input": str(resolved_input),
        "restoration_results_path": str(results_path.resolve()),
        "output_dir": str(resolved_output),
        "count": len(items),
        "items": items,
    }
    write_json(manifest, resolved_output / "manifest.json")
    return manifest


def run_refinement(
    restoration_input: str | Path,
    output_dir: str | Path,
    config: RefinementConfig,
    *,
    search_roots: list[str] | None = None,
    base_dir: str | Path | None = None,
) -> dict[str, Any]:
    mode = str(getattr(config, "mode", "rule") or "rule").strip().lower()
    if mode == "learned":
        from .learned_refinement import run_learned_refinement

        return run_learned_refinement(
            restoration_input,
            output_dir,
            config,
            search_roots=search_roots,
            base_dir=base_dir,
        )
    return run_prior_guided_refinement(
        restoration_input,
        output_dir,
        config,
        search_roots=search_roots,
        base_dir=base_dir,
    )
