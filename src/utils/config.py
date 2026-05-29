"""
config.py — Loads hyperparameters from configs/config.yaml and exposes a
typed namespace (CFG) used throughout the project.

Usage:
    from src.utils.config import CFG
    print(CFG.data.random_seed)
    print(CFG.models.lstm.hidden_size)
"""
from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml

# Project root = two levels up from this file (src/utils/config.py)
ROOT: Path = Path(__file__).resolve().parents[2]
_CONFIG_PATH: Path = ROOT / "configs" / "config.yaml"


def _to_namespace(obj: Any) -> Any:
    """Recursively convert dicts to SimpleNamespace for dot-access."""
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_namespace(v) for v in obj]
    return obj


def _load_config(path: Path) -> SimpleNamespace:
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    ns = _to_namespace(raw)
    # Resolve relative paths against project root
    ns.data.raw_path     = str(ROOT / ns.data.raw_path)
    ns.data.processed_dir = str(ROOT / ns.data.processed_dir)
    ns.outputs.models_dir  = str(ROOT / ns.outputs.models_dir)
    ns.outputs.logs_dir    = str(ROOT / ns.outputs.logs_dir)
    ns.outputs.figures_dir = str(ROOT / ns.outputs.figures_dir)
    ns.outputs.reports_dir = str(ROOT / ns.outputs.reports_dir)
    return ns


CFG: SimpleNamespace = _load_config(_CONFIG_PATH)

# Convenience: number of classes derived from label list
CFG.labels.num_classes = len(CFG.labels.codes)

# Device — evaluated lazily so importing this module is side-effect-free
import torch as _torch


def get_device() -> _torch.device:
    """Return the best available device (CUDA → CPU)."""
    if _torch.cuda.is_available():
        name = _torch.cuda.get_device_name(0)
        return _torch.device("cuda")
    return _torch.device("cpu")
