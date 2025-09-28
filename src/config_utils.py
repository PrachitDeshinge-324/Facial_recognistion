"""Configuration helpers shared across the face-recognition utilities."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "config.json"


def load_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> Mapping[str, Any]:
    """Read `config.json` if present, otherwise return an empty mapping."""

    path = Path(config_path)
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)
