"""
Loads config/config.yaml into a plain Python object.

Why a loader instead of reading YAML everywhere?
  * One place validates the file, so a typo gives a clear error message
    instead of a mysterious KeyError deep inside the video loop.
  * `cfg.drowsiness.ear_threshold` is nicer to read than
    `cfg["drowsiness"]["ear_threshold"]`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

# Project root = the folder that contains config/, src/, models/ ...
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


class ConfigSection:
    """A dict wrapper that also supports attribute access (cfg.camera.width)."""

    def __init__(self, data: Dict[str, Any]):
        self._data = data
        for key, value in data.items():
            # Nested dicts become nested ConfigSections, recursively.
            setattr(self, key, ConfigSection(value) if isinstance(value, dict) else value)

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def to_dict(self) -> Dict[str, Any]:
        return dict(self._data)

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def __repr__(self) -> str:  # helps when debugging in a REPL
        return f"ConfigSection({list(self._data)})"


# The sections we absolutely need. Missing ones are a hard error.
REQUIRED_SECTIONS = ("camera", "model", "landmarks", "drowsiness", "heatmap", "alarm", "ui")


def load_config(path: str | Path | None = None) -> ConfigSection:
    """Read and validate the YAML config.

    Raises FileNotFoundError or ValueError with an actionable message.
    """
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH

    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}\n"
            "Make sure you run the app from the project root, or pass an explicit path."
        )

    try:
        with open(config_path, "r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise ValueError(f"config.yaml is not valid YAML:\n{exc}") from exc

    if not isinstance(raw, dict):
        raise ValueError(f"config.yaml must be a mapping at the top level, got {type(raw).__name__}")

    missing = [name for name in REQUIRED_SECTIONS if name not in raw]
    if missing:
        raise ValueError(f"config.yaml is missing required section(s): {', '.join(missing)}")

    _validate(raw)
    return ConfigSection(raw)


def _validate(raw: Dict[str, Any]) -> None:
    """Catch the mistakes that would otherwise show up as weird behaviour."""
    drowsiness = raw["drowsiness"]

    weights = drowsiness.get("weights", {})
    total = sum(float(v) for v in weights.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            f"drowsiness.weights must sum to 1.0 but they sum to {total:.3f}. "
            f"Current values: {weights}"
        )

    if not 0.0 < float(drowsiness["ear_threshold"]) < 1.0:
        raise ValueError("drowsiness.ear_threshold must be between 0 and 1 (typical value: 0.21)")

    if drowsiness["warning_score"] >= drowsiness["drowsy_score"]:
        raise ValueError("drowsiness.warning_score must be lower than drowsiness.drowsy_score")

    imgsz = int(raw["model"]["imgsz"])
    if imgsz % 32 != 0:
        raise ValueError(f"model.imgsz must be a multiple of 32, got {imgsz}")

    alpha = float(raw["heatmap"]["alpha"])
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("heatmap.alpha must be between 0.0 and 1.0")


def resolve_path(relative: str | Path) -> Path:
    """Turn a config path like 'models/x.pt' into an absolute path.

    Absolute paths are returned unchanged, so users can point anywhere on disk.
    """
    p = Path(relative)
    return p if p.is_absolute() else (PROJECT_ROOT / p)
