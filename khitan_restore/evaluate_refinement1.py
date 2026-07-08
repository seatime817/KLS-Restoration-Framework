"""Evaluate refinement outputs against GT images."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from utils.util_calculate_psnr_ssim import calculate_psnr, calculate_ssim

from .io_utils import IMAGE_EXTENSIONS, ensure_dir, resolve_path, write_json
from .learned_refinement import canonical_stem


def _load_json(path: Path) -> dict | list:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _resolve_manifest(input_path: str | Path) -> Path:
    path = Path(input_path).expanduser().resolve()
    if path.is_dir():
        candidate = path / "manifest.json"
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"Could not find manifest.json under {path}")
    return path


def _index_gt(gt_dir: str | Path) -> dict[str, Path]:
    root = Path(gt_dir).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"GT directory not found: {root}")
    mapping: dict[str, Path] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            mapping.setdefault(canonical_stem(path), path.resolve())
    return mapping


def _read_gray_uint8(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return image


def _binary_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    _, pred_bin = cv2.threshold(pred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, gt_bin = cv2.threshold(gt, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    pred_mask = pred_bin > 0
    gt_mask = gt_bin > 0
    tp = float(np.logical_and(pred_mask, gt_mask).sum())
    fp = float(np.logical_and(pred_mask, np.logical_not(gt_mask)).sum())
    fn = float(np.logical_and(np.logical_not(pred_mask), gt_mask).sum())
    union = float(np.logical_or(pred_mask, gt_mask).sum())
    iou = tp / max(1.0, union)
    precision = tp / max(1.0, tp + fp)
    recall = tp / max(1.0, tp + fn)
    f1 = 2.0 * precision * recall / max(1.0e-6, precision + recall)
    return {
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _metric_row(pred_path: Path, gt_path: Path, crop_border: int) -> dict[str, float]:
    pred = _read_gray_uint8(pred_path)
    gt = _read_gray_uint8(gt_path)
    if pred.shape != gt.shape:
        pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_CUBIC)
    pred_hwc = pred[..., None]
    gt_hwc = gt[..., None]
    row = {
        "psnr": float(calculate_psnr(pred_hwc, gt_hwc, crop_border=crop_border, input_order="HWC")),
        "ssim": float(calculate_ssim(pred_hwc, gt_hwc, crop_border=crop_border, input_order="HWC")),
    }
    row.update(_binary_metrics(pred, gt))
    return row


def _mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = sorted(rows[0].keys())
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}


def run_evaluate_refinement(
    input_path: str | Path,
    gt_dir: str | Path,
    output_path: str | Path,
    *,
    crop_border: int = 0,
) -> dict[str, Any]:
    manifest_path = _resolve_manifest(input_path)
    manifest = _load_json(manifest_path)
    if not isinstance(manifest, dict):
        raise ValueError(f"Expected dict manifest in {manifest_path}")
    items = manifest.get("items", [])
    gt_index = _index_gt(gt_dir)

    final_rows: list[dict[str, float]] = []
    sr_rows: list[dict[str, float]] = []
    prior_rows: list[dict[str, float]] = []
    details: list[dict[str, Any]] = []

    for item in items:
        image_name = str(item.get("image_name") or Path(item.get("final_image_path", "")).stem)
        key = canonical_stem(image_name)
        gt_path = gt_index.get(key)
        if gt_path is None:
            continue

        final_path = Path(item["final_image_path"]).expanduser().resolve()
        sr_path = Path(item["sr_copy_path"]).expanduser().resolve() if item.get("sr_copy_path") else None
        prior_path = Path(item["gan_prior_copy_path"]).expanduser().resolve() if item.get("gan_prior_copy_path") else None

        final_metrics = _metric_row(final_path, gt_path, crop_border)
        detail = {
            "image_name": image_name,
            "gt_path": str(gt_path),
            "final_image_path": str(final_path),
            "final": final_metrics,
        }
        final_rows.append(final_metrics)

        if sr_path is not None and sr_path.exists():
            sr_metrics = _metric_row(sr_path, gt_path, crop_border)
            sr_rows.append(sr_metrics)
            detail["sr"] = sr_metrics
        if prior_path is not None and prior_path.exists():
            prior_metrics = _metric_row(prior_path, gt_path, crop_border)
            prior_rows.append(prior_metrics)
            detail["prior"] = prior_metrics
        details.append(detail)

    summary = {
        "stage": "evaluate_refinement",
        "manifest_path": str(manifest_path),
        "gt_dir": str(Path(gt_dir).expanduser().resolve()),
        "count": len(details),
        "final_mean": _mean_metrics(final_rows),
        "sr_mean": _mean_metrics(sr_rows) if sr_rows else None,
        "prior_mean": _mean_metrics(prior_rows) if prior_rows else None,
        "items": details,
    }
    write_json(summary, output_path)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate refinement outputs against GT.")
    parser.add_argument("--input", type=str, required=True, help="Refinement output dir or manifest.json path.")
    parser.add_argument("--gt-dir", type=str, required=True, help="GT image directory.")
    parser.add_argument("--output", type=str, required=True, help="Output JSON path.")
    parser.add_argument("--crop-border", type=int, default=0, help="Crop border for PSNR/SSIM.")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    output_path = resolve_path(args.output, allow_missing=True)
    ensure_dir(Path(output_path).parent)
    result = run_evaluate_refinement(
        args.input,
        args.gt_dir,
        output_path,
        crop_border=args.crop_border,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
