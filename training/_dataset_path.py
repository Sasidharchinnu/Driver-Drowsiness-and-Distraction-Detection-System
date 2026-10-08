"""
Make Ultralytics find our dataset, regardless of its global settings.

The problem
-----------
In data.yaml, `path: dataset` looks like it is relative to data.yaml. It is not.
Ultralytics resolves it against a GLOBAL setting, `datasets_dir`, stored in
~/Library/Application Support/Ultralytics/settings.json (or the equivalent on
Linux/Windows). That setting points at whatever project you trained last, so on
a machine that has trained anything before, the path silently resolves to the
wrong place and training dies with "images not found".

The fix
-------
Write a resolved copy of data.yaml in which `path` is an ABSOLUTE path to this
project's dataset folder, and hand that copy to Ultralytics. An absolute path
ignores `datasets_dir` entirely, so training works on any machine with no
global configuration.

The generated file is training/.data.resolved.yaml. It is derived, so you never
edit it - edit training/data.yaml instead.
"""

from __future__ import annotations

from pathlib import Path

import yaml

TRAINING_DIR = Path(__file__).resolve().parent
SOURCE_YAML = TRAINING_DIR / "data.yaml"
RESOLVED_YAML = TRAINING_DIR / ".data.resolved.yaml"


def resolved_data_yaml() -> Path:
    """Return the path to a data.yaml whose `path` key is absolute."""
    if not SOURCE_YAML.exists():
        raise FileNotFoundError(f"{SOURCE_YAML} is missing.")

    with open(SOURCE_YAML, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    # Turn a relative dataset root into an absolute one, anchored at training/.
    dataset_root = Path(data.get("path", "dataset"))
    if not dataset_root.is_absolute():
        dataset_root = (TRAINING_DIR / dataset_root).resolve()
    data["path"] = str(dataset_root)

    with open(RESOLVED_YAML, "w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)

    return RESOLVED_YAML


def dataset_root() -> Path:
    """The absolute folder that holds images/ and labels/."""
    with open(SOURCE_YAML, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    root = Path(data.get("path", "dataset"))
    return root if root.is_absolute() else (TRAINING_DIR / root).resolve()
