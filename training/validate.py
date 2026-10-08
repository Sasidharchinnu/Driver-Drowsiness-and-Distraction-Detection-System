"""
Evaluate a trained model on the held-out test split.

What the numbers mean
---------------------
  Precision  of everything the model called an "eye_closed", what fraction
             really was one? Low precision = false alarms.
  Recall     of all the real "eye_closed" boxes, what fraction did it find?
             Low recall = missed micro-sleeps, which is the dangerous failure.
  mAP50      mean Average Precision at IoU 0.5 - the standard single-number
             detection score. A box counts as correct if it overlaps the truth
             by at least 50%.
  mAP50-95   the average of mAP at IoU 0.50, 0.55 ... 0.95. Much stricter,
             because it also rewards tight, well-placed boxes.

For this application, RECALL on eye_closed matters more than precision: a
false alarm is annoying, a missed micro-sleep is a crash.

Run:
    python training/validate.py --weights training/runs/drowsiness/weights/best.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.detector import select_device                # noqa: E402
from training._dataset_path import resolved_data_yaml  # noqa: E402
CLASS_NAMES = ["face", "eye_open", "eye_closed", "mouth_open", "mouth_closed"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", default="training/runs/drowsiness/weights/best.pt")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    weights = Path(args.weights)
    if not weights.is_absolute():
        weights = PROJECT_ROOT / weights
    if not weights.exists():
        print(f"ERROR: weights not found: {weights}")
        print("Train a model first:  python training/train.py")
        return 1

    from ultralytics import YOLO

    model = YOLO(str(weights))
    print(f"Validating {weights.name} on the '{args.split}' split...\n")

    metrics = model.val(
        data=str(resolved_data_yaml()),
        split=args.split,
        imgsz=args.imgsz,
        device=select_device(args.device),
        plots=True,
        project=str(PROJECT_ROOT / "training" / "runs"),
        name="validate",
        exist_ok=True,
    )

    print("\n" + "=" * 58)
    print("OVERALL")
    print(f"  Precision   {metrics.box.mp:.4f}")
    print(f"  Recall      {metrics.box.mr:.4f}")
    print(f"  mAP50       {metrics.box.map50:.4f}")
    print(f"  mAP50-95    {metrics.box.map:.4f}")

    print("\nPER CLASS")
    print(f"  {'class':14s} {'precision':>10s} {'recall':>8s} {'mAP50':>8s}")
    for index, class_id in enumerate(metrics.box.ap_class_index):
        name = CLASS_NAMES[int(class_id)] if int(class_id) < len(CLASS_NAMES) else str(class_id)
        print(f"  {name:14s} {metrics.box.p[index]:10.4f} "
              f"{metrics.box.r[index]:8.4f} {metrics.box.ap50[index]:8.4f}")

    # The safety-critical check.
    try:
        closed_index = list(metrics.box.ap_class_index).index(2)   # 2 = eye_closed
        recall = metrics.box.r[closed_index]
        print("\nSAFETY CHECK — recall on 'eye_closed'")
        if recall >= 0.90:
            print(f"  {recall:.4f}  good: micro-sleeps are reliably caught.")
        elif recall >= 0.75:
            print(f"  {recall:.4f}  acceptable, but record more eyes-closed data.")
        else:
            print(f"  {recall:.4f}  TOO LOW — the model will miss micro-sleeps.")
            print("  Record more 'eyesshut' sessions and retrain.")
    except (ValueError, IndexError):
        pass

    print("=" * 58)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
