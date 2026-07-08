"""Shared filesystem helpers for the unified pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}


def ensure_dir(path: str | Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def unique_paths(paths: Iterable[str | Path]) -> list[Path]:
    seen: set[str] = set()
    result: list[Path] = []
    for raw_path in paths:
        candidate = Path(raw_path).expanduser()
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        result.append(candidate)
    return result


def resolve_path(
    raw_path: str | Path,
    *,
    search_roots: Iterable[str | Path] | None = None,
    base_dir: str | Path | None = None,
    allow_missing: bool = False,
) -> Path:
    if raw_path is None or str(raw_path).strip() == "":
        raise ValueError("Path is empty.")

    raw_text = str(raw_path).strip()
    candidate = Path(raw_path).expanduser()
    candidates: list[Path] = []

    if candidate.is_absolute():
        candidates.append(candidate)
    else:
        if raw_text.startswith("."):
            if base_dir is not None:
                candidates.append(Path(base_dir).expanduser() / candidate)
            candidates.append(PROJECT_ROOT / candidate)
            candidates.append(Path.cwd() / candidate)
        else:
            candidates.append(PROJECT_ROOT / candidate)
            candidates.append(Path.cwd() / candidate)
            if base_dir is not None:
                candidates.append(Path(base_dir).expanduser() / candidate)
        for root in search_roots or []:
            candidates.append(Path(root).expanduser() / candidate)

    ordered = unique_paths(candidates)
    for item in ordered:
        if item.exists():
            return item.resolve()

    if allow_missing and ordered:
        if candidate.is_absolute():
            return candidate.resolve()
        if raw_text.startswith(".") and base_dir is not None:
            return (Path(base_dir).expanduser() / candidate).resolve()
        return (PROJECT_ROOT / candidate).resolve()

    searched = "\n".join(f"- {item}" for item in ordered)
    raise FileNotFoundError(f"Could not resolve path: {raw_path}\nSearched:\n{searched}")


def list_images(input_path: str | Path) -> list[Path]:
    resolved = Path(input_path).expanduser().resolve()
    if resolved.is_file():
        if resolved.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image file: {resolved}")
        return [resolved]

    if not resolved.exists():
        raise FileNotFoundError(f"Input path does not exist: {resolved}")

    images = [
        path
        for path in sorted(resolved.rglob("*"))
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if not images:
        raise FileNotFoundError(f"No images found under: {resolved}")
    return images


def load_structured_file(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    with open(resolved, "r", encoding="utf-8") as handle:
        if resolved.suffix.lower() in {".yaml", ".yml"}:
            return yaml.safe_load(handle) or {}
        if resolved.suffix.lower() == ".json":
            return json.load(handle)
    raise ValueError(f"Unsupported config format: {resolved.suffix}")


def write_json(data: Any, path: str | Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with open(resolved, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
    return resolved
