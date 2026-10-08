"""
Download the YOLO face-detection weights used by the app.

Why we need this
----------------
The stock YOLO models shipped by Ultralytics (yolov8n.pt / yolo11n.pt) are trained
on the COCO dataset, whose 80 classes contain "person" but NOT "face", "eye" or
"mouth".  So a stock model cannot give us the boxes this project needs.

We therefore download a *real* YOLO model that was actually fine-tuned for face
detection, and we keep the option of training our own 5-class model
(see ../training/).

Run it with:
    python tools/download_models.py
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

# Project root = one level above this file's folder (tools/ -> project root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

# Candidate sources, tried in order.  Each one is a genuine YOLO checkpoint that
# was fine-tuned on a face-detection dataset.
#   repo_id  : the Hugging Face model repository
#   filename : the checkpoint inside that repository
#   save_as  : the local filename we store it under
FACE_MODEL_SOURCES = [
    {
        "repo_id": "AdamCodd/YOLOv11n-face-detection",
        "filename": "model.pt",
        "save_as": "yolov11n-face.pt",
        "license": "Apache-2.0",
        "note": "YOLO11n fine-tuned on WIDER FACE.",
    },
    {
        "repo_id": "arnabdhar/YOLOv8-Face-Detection",
        "filename": "model.pt",
        "save_as": "yolov8n-face.pt",
        "license": "AGPL-3.0",
        "note": "YOLOv8n fine-tuned on a Roboflow face dataset.",
    },
]


def download_face_model(force: bool = False) -> Path:
    """Fetch the first face model that downloads successfully.

    Returns the local path to the checkpoint.
    Raises RuntimeError if every source fails (e.g. no internet).
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # If we already have one of the checkpoints, reuse it instead of re-downloading.
    if not force:
        for source in FACE_MODEL_SOURCES:
            local = MODELS_DIR / source["save_as"]
            if local.exists() and local.stat().st_size > 100_000:
                print(f"[ok] Already present: {local}")
                return local

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover - dependency is in requirements.txt
        raise RuntimeError(
            "huggingface_hub is not installed. Run: pip install -r requirements.txt"
        ) from exc

    errors = []
    for source in FACE_MODEL_SOURCES:
        target = MODELS_DIR / source["save_as"]
        print(f"[..] Downloading {source['repo_id']}/{source['filename']} ({source['license']})")
        try:
            cached = hf_hub_download(repo_id=source["repo_id"], filename=source["filename"])
            shutil.copyfile(cached, target)
            print(f"[ok] Saved -> {target}  ({target.stat().st_size / 1e6:.1f} MB)")
            print(f"     {source['note']}")
            return target
        except Exception as exc:  # network error, gated repo, etc.
            print(f"[!!] Failed: {exc}")
            errors.append(f"{source['repo_id']}: {exc}")

    raise RuntimeError(
        "Could not download any face-detection model.\n"
        "Reasons:\n  - " + "\n  - ".join(errors) + "\n"
        "Fix: check your internet connection, or train your own model with "
        "`python training/train.py` and point config/config.yaml at the result."
    )


def download_base_model() -> Path:
    """Download the plain COCO YOLO11n checkpoint.

    We do not use it for face/eye/mouth detection (it has no such classes), but it
    is the *starting point* for fine-tuning our own model in training/train.py.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    target = MODELS_DIR / "yolo11n.pt"
    if target.exists():
        print(f"[ok] Already present: {target}")
        return target

    from ultralytics import YOLO  # triggers Ultralytics' own download

    print("[..] Downloading yolo11n.pt (COCO base weights, used for fine-tuning)")
    model = YOLO("yolo11n.pt")
    weights = Path(model.ckpt_path) if getattr(model, "ckpt_path", None) else Path("yolo11n.pt")
    if weights.exists() and weights.resolve() != target.resolve():
        shutil.copyfile(weights, target)
    print(f"[ok] Saved -> {target}")
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description="Download YOLO weights for the project.")
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    parser.add_argument("--base", action="store_true", help="also fetch COCO yolo11n.pt for training")
    args = parser.parse_args()

    try:
        download_face_model(force=args.force)
        if args.base:
            download_base_model()
    except RuntimeError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1

    print("\nAll set. Now run:  streamlit run app.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
