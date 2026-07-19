"""Configuration loading: configs/default.yaml -> nested attribute access."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import yaml

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"


def _to_ns(obj):
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _to_ns(v) for k, v in obj.items()})
    return obj


def load_config(path: str | Path | None = None) -> SimpleNamespace:
    with open(path or DEFAULT_PATH) as f:
        return _to_ns(yaml.safe_load(f))
