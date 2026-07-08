"""Build weak component annotations from auto-segmented component filenames."""

from __future__ import annotations

import argparse
import hashlib
from collections import Counter
from pathlib import Path
from typing import Any

from .io_utils import ensure_dir, write_json


def parse_component_filename(path: str | Path) -> dict[str, Any]:
    file_path = Path(path).expanduser().resolve()
    parts = file_path.stem.split("_")
    if len(parts) < 10 or parts[0] != "autoseg":
        raise ValueError(f"Unsupported component filename: {file_path.name}")

    bbox = [int(value) for value in parts[2:6]]
    component_tokens = parts[6:-1]
    if not component_tokens:
        raise ValueError(f"Missing component tokens in filename: {file_path.name}")

    return {
        "name": file_path.name,
        "stem": file_path.stem,
        "source_image_id": parts[1],
        "bbox": bbox,
        "component_tokens": component_tokens,
        "component_key": "|".join(component_tokens),
        "filename_hash": parts[-1],
    }


def _ratio_to_bucket(source_image_id: str) -> float:
    digest = hashlib.md5(source_image_id.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def choose_split(
    source_image_id: str,
    *,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> str:
    total = train_ratio + val_ratio + test_ratio
    if total <= 0:
        raise ValueError("Split ratios must sum to a positive value.")

    train_cutoff = train_ratio / total
    val_cutoff = (train_ratio + val_ratio) / total
    bucket = _ratio_to_bucket(source_image_id)
    if bucket < train_cutoff:
        return "train"
    if bucket < val_cutoff:
        return "val"
    return "test"


def _build_vocab(
    component_keys: list[str],
) -> tuple[dict[str, int], list[dict[str, Any]]]:
    counter = Counter(component_keys)
    ordered_keys = sorted(counter)
    key_to_id = {component_key: index for index, component_key in enumerate(ordered_keys)}
    vocab = [
        {
            "comp_id": key_to_id[component_key],
            "component_key": component_key,
            "component_tokens": component_key.split("|"),
            "count": counter[component_key],
        }
        for component_key in ordered_keys
    ]
    return key_to_id, vocab


def build_component_annotations(
    components_root: str | Path,
    *,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> tuple[dict[str, Any], dict[str, Any]]:
    root = Path(components_root).expanduser().resolve()
    mask_dir = root / "mask"
    soft_dir = root / "soft"
    if not mask_dir.exists():
        raise FileNotFoundError(f"Missing mask directory: {mask_dir}")
    if not soft_dir.exists():
        raise FileNotFoundError(f"Missing soft directory: {soft_dir}")

    mask_files = sorted(mask_dir.glob("*.png"))
    if not mask_files:
        raise FileNotFoundError(f"No component masks found under: {mask_dir}")

    parsed_items: list[dict[str, Any]] = []
    component_keys: list[str] = []
    split_counter: Counter[str] = Counter()
    for mask_path in mask_files:
        soft_path = soft_dir / mask_path.name
        if not soft_path.exists():
            raise FileNotFoundError(f"Missing paired soft component: {soft_path}")

        item = parse_component_filename(mask_path)
        split = choose_split(
            item["source_image_id"],
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
        )
        item["split"] = split
        item["label_source"] = "filename_pattern"
        item["mask_path"] = str(mask_path.resolve())
        item["soft_path"] = str(soft_path.resolve())
        parsed_items.append(item)
        component_keys.append(item["component_key"])
        split_counter[split] += 1

    key_to_id, vocab_items = _build_vocab(component_keys)
    for item in parsed_items:
        item["comp_id"] = key_to_id[item["component_key"]]

    annotations = {
        "version": 1,
        "label_source": "filename_pattern",
        "components_root": str(root),
        "mask_dir": str(mask_dir),
        "soft_dir": str(soft_dir),
        "num_items": len(parsed_items),
        "num_components": len(vocab_items),
        "split_counts": dict(split_counter),
        "items": parsed_items,
    }
    vocab = {
        "version": 1,
        "label_source": "filename_pattern",
        "num_components": len(vocab_items),
        "items": vocab_items,
    }
    return annotations, vocab


def write_component_annotations(
    annotations: dict[str, Any],
    vocab: dict[str, Any],
    output_dir: str | Path,
) -> tuple[Path, Path]:
    resolved_output = ensure_dir(output_dir)
    annotations_path = write_json(
        annotations,
        resolved_output / "component_annotations.json",
    )
    vocab_path = write_json(
        vocab,
        resolved_output / "component_vocab.json",
    )
    return annotations_path, vocab_path


def build_component_annotations_cli() -> None:
    parser = argparse.ArgumentParser(
        description="Build weak comp_id annotations from component mask/soft filenames.",
    )
    parser.add_argument(
        "--components-root",
        type=str,
        default="dataset/components",
        help="Root directory that contains mask/ and soft/.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Where to write component_annotations.json and component_vocab.json.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    args = parser.parse_args()

    annotations, vocab = build_component_annotations(
        args.components_root,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )
    output_dir = args.output_dir or args.components_root
    annotations_path, vocab_path = write_component_annotations(
        annotations,
        vocab,
        output_dir,
    )
    print(f"[component-annotations] annotations={annotations_path}")
    print(f"[component-annotations] vocab={vocab_path}")
    print(
        "[component-annotations] "
        f"num_items={annotations['num_items']} num_components={annotations['num_components']} "
        f"split_counts={annotations['split_counts']}"
    )

