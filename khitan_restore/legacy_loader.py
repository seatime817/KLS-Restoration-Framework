"""Load legacy experimental modules behind a clean interface."""

from __future__ import annotations

import importlib.util
import sys
from functools import lru_cache
from pathlib import Path

from .io_utils import PROJECT_ROOT


def _load_module(module_name: str, file_path: Path):
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    if str(file_path.parent) not in sys.path:
        sys.path.insert(0, str(file_path.parent))

    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {file_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=1)
def load_swinir_module():
    return _load_module("legacy_main_test_swinir", PROJECT_ROOT / "main_test_swinir.py")


@lru_cache(maxsize=1)
def load_segment_module():
    return _load_module(
        "legacy_segment_core",
        PROJECT_ROOT / "khitan_auto_component_segmenter_paper_experiment" / "segment_core.py",
    )


@lru_cache(maxsize=1)
def load_component_stylegan_module():
    return _load_module(
        "legacy_component_stylegan",
        PROJECT_ROOT / "test" / "component_stylegan.py",
    )


@lru_cache(maxsize=1)
def load_retrieval_module():
    return _load_module(
        "legacy_stage1_retrieval",
        PROJECT_ROOT / "test" / "test_stage1_gan_best_retrieval_modified.py",
    )
