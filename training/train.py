"""
Fine-tune YOLO on the custom 5-class drowsiness dataset.

What "fine-tuning" means here
-----------------------------
We do NOT train from random weights - that would need hundreds of thousands of
images. We start from `yolo11n.pt`, which already learned generic visual
features (edges, textures, shapes, "object-ness") from COCO's 118k images, and
we only re-teach the final layers to recognise our 5 classes. This is transfer
learning, and it is why a few thousand of your own frames is enough.

The 5 classes (see training/data.yaml):
    0 face          3 mouth_open (yawn)
    1 eye_open      4 mouth_closed
    2 eye_closed

Once trained, point the app at the result:
    model.weights: "training/runs/drowsiness/weights/best.pt"   # in config.yaml

Typical run:
    python training/train.py --epochs 100 --batch 16

On Apple Silicon this uses the `mps` GPU automatically. 100 epochs on ~2000
frames takes roughly 25-40 minutes on an M2.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.detector import select_device                      # noqa: E402
from training._dataset_path import dataset_root, resolved_data_yaml   # noqa: E402

TRAINING_DIR = PROJECT_ROOT / "training"
DATA_YAML = TRAINING_DIR / "data.yaml"
DATASET_ROOT = dataset_root()


def verify_dataset() -> bool:
    """Fail early with a clear message instead of 40 minutes into training."""
    if not DATA_YAML.exists():
        print(f"ERROR: {DATA_YAML} is missing.")
        return False

    problems = []
    counts = {}
    for split_name in ("train", "val"):
        images = DATASET_ROOT / "images" / split_name
        labels = DATASET_ROOT / "labels" / split_name
        if not images.exists():
            problems.append(f"missing folder: {images}")
            continue

        image_files = list(images.glob("*.jpg")) + list(images.glob("*.png"))
        label_files = list(labels.glob("*.txt")) if labels.exists() else []
        counts[split_name] = len(image_files)

        if not image_files:
            problems.append(f"no images in {images}")
        # Every image needs a matching label file with the same stem.
        image_stems = {path.stem for path in image_files}
        label_stems = {path.stem for path in label_files}
        orphans = image_stems - label_stems
        if orphans:
            problems.append(f"{len(orphans)} image(s) in {split_name} have no .txt label")

    if problems:
        print("ERROR: the dataset is not ready:\n  - " + "\n  - ".join(problems))
        print("\nBuild it first:")
        print("  python training/auto_label.py --session normal   --frames 400")
        print("  python training/auto_label.py --session yawning  --frames 300")
        print("  python training/auto_label.py --session eyesshut --frames 300")
        print("  python training/auto_label.py --split")
        return False

    print(f"Dataset OK: {counts.get('train', 0)} train / {counts.get('val', 0)} val images")
    if counts.get("train", 0) < 300:
        print("  WARNING: fewer than 300 training images. Expect a weak model.")
    return True


def train(args) -> int:
    from ultralytics import YOLO

    device = select_device(args.device)
    print(f"\nStarting from : {args.model}")
    print(f"Device        : {device}")
    print(f"Epochs        : {args.epochs}   batch: {args.batch}   imgsz: {args.imgsz}\n")

    model = YOLO(args.model)

    results = model.train(
        # An absolute path, so Ultralytics' global datasets_dir cannot mislead it.
        data=str(resolved_data_yaml()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        project=str(TRAINING_DIR / "runs"),
        name=args.name,
        exist_ok=True,
        patience=args.patience,      # early stop if val mAP stops improving
        optimizer="auto",
        lr0=args.lr,
        seed=args.seed,
        val=True,
        plots=True,                  # writes PR curves / confusion matrix as PNGs

        # ---- Data augmentation ------------------------------------------
        # Augmentation randomly alters each training image so the model sees
        # more variety than we actually recorded. The settings below are
        # deliberately tuned for faces in a car:
        hsv_h=0.015,     # tiny hue shift - skin tone must stay realistic
        hsv_s=0.7,       # saturation: washed-out vs vivid camera sensors
        hsv_v=0.4,       # brightness: day, dusk, tunnel, night
        degrees=10.0,    # small rotation - a driver's head tilts, it does not invert
        translate=0.1,   # the face is not always centred
        scale=0.4,       # closer to / further from the camera
        fliplr=0.5,      # mirror: left and right eyes must both be learned
        flipud=0.0,      # never flip vertically - an upside-down driver is not a case
        mosaic=1.0,      # stitch 4 images together: strong regulariser
        close_mosaic=10, # switch mosaic off for the last 10 epochs to settle
        erasing=0.2,     # randomly blank patches - mimics sunglasses / occlusion
    )

    best_weights = TRAINING_DIR / "runs" / args.name / "weights" / "best.pt"
    print("\n" + "=" * 62)
    print("Training finished.")
    print(f"Best weights: {best_weights}")

    if best_weights.exists() and args.install:
        target = PROJECT_ROOT / "models" / "drowsiness-yolo.pt"
        shutil.copyfile(best_weights, target)
        print(f"Copied to   : {target}")
        print("\nNow set this line in config/config.yaml:")
        print('  weights: "models/drowsiness-yolo.pt"')
    else:
        print("\nTo use it, set this line in config/config.yaml:")
        print(f'  weights: "training/runs/{args.name}/weights/best.pt"')

    print("\nInspect the training curves and confusion matrix here:")
    print(f"  {TRAINING_DIR / 'runs' / args.name}")
    print("=" * 62)

    # `results` carries the final validation metrics.
    try:
        metrics = results.results_dict
        print("\nFinal validation metrics:")
        for key in ("metrics/precision(B)", "metrics/recall(B)",
                    "metrics/mAP50(B)", "metrics/mAP50-95(B)"):
            if key in metrics:
                print(f"  {key:24s} {metrics[key]:.4f}")
    except Exception:
        pass   # metrics are a nice-to-have, never a reason to fail

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="yolo11n.pt",
                        help="starting checkpoint (n=nano fastest, s/m = more accurate)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="auto", help="auto | mps | cuda | cpu")
    parser.add_argument("--name", default="drowsiness", help="run name under training/runs/")
    parser.add_argument("--patience", type=int, default=25, help="early-stopping patience")
    parser.add_argument("--lr", type=float, default=0.01, help="initial learning rate")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-verify", action="store_true")
    parser.add_argument("--install", action="store_true",
                        help="copy the best weights into models/ when finished")
    args = parser.parse_args()

    if not args.skip_verify and not verify_dataset():
        return 1
    return train(args)


if __name__ == "__main__":
    raise SystemExit(main())
