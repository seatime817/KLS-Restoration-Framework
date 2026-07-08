"""Evaluate refinement outputs against GT images."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps
from skimage import feature
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchvision.transforms.functional import pil_to_tensor

from utils.util_calculate_psnr_ssim import calculate_psnr, calculate_ssim

from .io_utils import IMAGE_EXTENSIONS, ensure_dir, resolve_path, write_json
from .learned_refinement import canonical_stem


try:
    BICUBIC = Image.Resampling.BICUBIC
except AttributeError:
    BICUBIC = Image.BICUBIC


def _load_json(path: Path) -> Union[Dict[str, Any], List[Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _resolve_manifest(input_path: Union[str, Path]) -> Path:
    path = Path(input_path).expanduser().resolve()
    if path.is_dir():
        candidate = path / "manifest.json"
        if candidate.exists():
            return candidate
        raise FileNotFoundError("Could not find manifest.json under {}".format(path))
    return path


def _index_gt(gt_dir: Union[str, Path]) -> Dict[str, Path]:
    root = Path(gt_dir).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError("GT directory not found: {}".format(root))

    mapping = {}  # type: Dict[str, Path]
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            mapping.setdefault(canonical_stem(path), path.resolve())
    return mapping


def _read_gray_uint8(path: Union[str, Path]) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Could not read image: {}".format(path))
    return image


def _read_rgb_pil(path: Union[str, Path]) -> Image.Image:
    image = Image.open(path)
    image = ImageOps.exif_transpose(image)
    return image.convert("RGB")


def _pil_to_float_tensor(img: Image.Image) -> torch.Tensor:
    return pil_to_tensor(img).float() / 255.0


def _get_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("You set --device cuda, but CUDA is not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _binary_metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    _, pred_bin = cv2.threshold(pred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, gt_bin = cv2.threshold(gt, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    pred_mask = pred_bin > 0
    gt_mask = gt_bin > 0

    tp = float(np.logical_and(pred_mask, gt_mask).sum())
    fp = float(np.logical_and(pred_mask, np.logical_not(gt_mask)).sum())
    fn = float(np.logical_and(np.logical_not(pred_mask), gt_mask).sum())
    union = float(np.logical_or(pred_mask, gt_mask).sum())

    if tp == 0 and fp == 0 and fn == 0:
        return {
            "iou": 1.0,
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
            "dice": 1.0,
        }

    iou = tp / max(1.0, union)
    precision = tp / max(1.0, tp + fp)
    recall = tp / max(1.0, tp + fn)
    f1 = 2.0 * precision * recall / max(1.0e-6, precision + recall)
    dice = 2.0 * tp / max(1.0, 2.0 * tp + fp + fn)

    return {
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "dice": float(dice),
    }


def _compute_stsc(pred: np.ndarray, gt: np.ndarray, edge_sigma: float = 1.0) -> float:
    pred_gray = pred.astype(np.float32) / 255.0
    gt_gray = gt.astype(np.float32) / 255.0

    pred_edge = feature.canny(pred_gray, sigma=edge_sigma)
    gt_edge = feature.canny(gt_gray, sigma=edge_sigma)

    tp = float(np.logical_and(pred_edge, gt_edge).sum())
    fp = float(np.logical_and(pred_edge, np.logical_not(gt_edge)).sum())
    fn = float(np.logical_and(np.logical_not(pred_edge), gt_edge).sum())

    if tp == 0 and fp == 0 and fn == 0:
        return 1.0

    return float(2.0 * tp / (2.0 * tp + fp + fn + 1.0e-8))


@torch.no_grad()
def _compute_lpips(
    pred_path: Path,
    gt_path: Path,
    lpips_metric: LearnedPerceptualImagePatchSimilarity,
    device: torch.device,
) -> float:
    pred_img = _read_rgb_pil(pred_path)
    gt_img = _read_rgb_pil(gt_path)

    if pred_img.size != gt_img.size:
        pred_img = pred_img.resize(gt_img.size, resample=BICUBIC)

    pred_t = _pil_to_float_tensor(pred_img).unsqueeze(0).to(device)
    gt_t = _pil_to_float_tensor(gt_img).unsqueeze(0).to(device)

    value = lpips_metric(pred_t, gt_t)
    return float(value.detach().cpu().item())


def _metric_row(
    pred_path: Path,
    gt_path: Path,
    crop_border: int,
    lpips_metric: LearnedPerceptualImagePatchSimilarity,
    device: torch.device,
    edge_sigma: float,
) -> Dict[str, float]:
    pred = _read_gray_uint8(pred_path)
    gt = _read_gray_uint8(gt_path)

    if pred.shape != gt.shape:
        pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_CUBIC)

    pred_hwc = pred[..., None]
    gt_hwc = gt[..., None]

    row = {
        "psnr": float(calculate_psnr(pred_hwc, gt_hwc, crop_border=crop_border, input_order="HWC")),
        "ssim": float(calculate_ssim(pred_hwc, gt_hwc, crop_border=crop_border, input_order="HWC")),
        "lpips": float(_compute_lpips(pred_path, gt_path, lpips_metric, device)),
        "stsc": float(_compute_stsc(pred, gt, edge_sigma=edge_sigma)),
    }

    row.update(_binary_metrics(pred, gt))
    return row


def _mean_metrics(rows: List[Dict[str, float]]) -> Dict[str, float]:
    if not rows:
        return {}

    keys = sorted(rows[0].keys())
    result = {}
    for key in keys:
        result[key] = float(np.mean([row[key] for row in rows]))
    return result


def run_evaluate_refinement(
    input_path: Union[str, Path],
    gt_dir: Union[str, Path],
    output_path: Union[str, Path],
    crop_border: int = 0,
    lpips_net: str = "alex",
    edge_sigma: float = 1.0,
    device: str = "auto",
) -> Dict[str, Any]:
    manifest_path = _resolve_manifest(input_path)
    manifest = _load_json(manifest_path)

    if not isinstance(manifest, dict):
        raise ValueError("Expected dict manifest in {}".format(manifest_path))

    items = manifest.get("items", [])
    gt_index = _index_gt(gt_dir)

    device_obj = _get_device(device)
    lpips_metric = LearnedPerceptualImagePatchSimilarity(
        net_type=lpips_net,
        normalize=True,
    ).to(device_obj)
    lpips_metric.eval()

    final_rows = []  # type: List[Dict[str, float]]
    sr_rows = []  # type: List[Dict[str, float]]
    prior_rows = []  # type: List[Dict[str, float]]
    details = []  # type: List[Dict[str, Any]]

    for item in items:
        image_name = str(item.get("image_name") or Path(item.get("final_image_path", "")).stem)
        key = canonical_stem(image_name)
        gt_path = gt_index.get(key)

        if gt_path is None:
            continue

        final_path = Path(item["final_image_path"]).expanduser().resolve()
        sr_path = Path(item["sr_copy_path"]).expanduser().resolve() if item.get("sr_copy_path") else None
        prior_path = Path(item["gan_prior_copy_path"]).expanduser().resolve() if item.get("gan_prior_copy_path") else None

        final_metrics = _metric_row(
            final_path,
            gt_path,
            crop_border,
            lpips_metric,
            device_obj,
            edge_sigma,
        )

        detail = {
            "image_name": image_name,
            "gt_path": str(gt_path),
            "final_image_path": str(final_path),
            "final": final_metrics,
        }
        final_rows.append(final_metrics)

        if sr_path is not None and sr_path.exists():
            sr_metrics = _metric_row(
                sr_path,
                gt_path,
                crop_border,
                lpips_metric,
                device_obj,
                edge_sigma,
            )
            sr_rows.append(sr_metrics)
            detail["sr"] = sr_metrics

        if prior_path is not None and prior_path.exists():
            prior_metrics = _metric_row(
                prior_path,
                gt_path,
                crop_border,
                lpips_metric,
                device_obj,
                edge_sigma,
            )
            prior_rows.append(prior_metrics)
            detail["prior"] = prior_metrics

        details.append(detail)

    summary = {
        "stage": "evaluate_refinement",
        "manifest_path": str(manifest_path),
        "gt_dir": str(Path(gt_dir).expanduser().resolve()),
        "count": len(details),
        "crop_border": crop_border,
        "lpips_net": lpips_net,
        "edge_sigma": edge_sigma,
        "device": str(device_obj),
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
    parser.add_argument("--lpips-net", type=str, default="alex", choices=["alex", "vgg", "squeeze"], help="LPIPS backbone.")
    parser.add_argument("--edge-sigma", type=float, default=1.0, help="Sigma for StSc Canny edge detector.")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Device for LPIPS.")
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    output_path = resolve_path(args.output, allow_missing=True)
    ensure_dir(Path(output_path).parent)

    result = run_evaluate_refinement(
        args.input,
        args.gt_dir,
        output_path,
        crop_border=args.crop_border,
        lpips_net=args.lpips_net,
        edge_sigma=args.edge_sigma,
        device=args.device,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()