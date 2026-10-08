"""
Build a labelled YOLO dataset from your own webcam, automatically.

The problem
-----------
Training a 5-class model needs thousands of labelled boxes. Drawing them by
hand in LabelImg takes days.

The trick
---------
We already have two components that produce accurate boxes:
  * the pretrained YOLO FACE model -> the `face` box
  * MediaPipe FaceMesh             -> exact eye and mouth contours

So we let those two label the data for us. This is called **auto-labelling** or
weak supervision: a slow/complex teacher (YOLO-face + FaceMesh) generates the
training labels for a fast/simple student (our single 5-class YOLO model).

The open/closed class is decided by the same EAR and MAR maths the live app
uses, so the labels are consistent with how the system will be judged.

How to record a good dataset
----------------------------
Run this several times and vary the conditions, otherwise the model only
learns "you, in this chair, in this light":

    python training/auto_label.py --session normal    --frames 400
    python training/auto_label.py --session blinking  --frames 400
    python training/auto_label.py --session yawning   --frames 300
    python training/auto_label.py --session eyesshut  --frames 300
    python training/auto_label.py --session glasses   --frames 300
    python training/auto_label.py --session dim_light --frames 300
    python training/auto_label.py --session side_angle --frames 300

Then split and train:

    python training/auto_label.py --split
    python training/train.py

Always sanity-check a few labels before training:

    python training/auto_label.py --preview
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.camera import CameraError, WebcamStream          # noqa: E402
from src.config import load_config                        # noqa: E402
from src.detector import YOLODetector                     # noqa: E402
from src.landmarks import FaceMeshAnalyzer                # noqa: E402

DATASET_ROOT = PROJECT_ROOT / "training" / "dataset"
RAW_DIR = DATASET_ROOT / "raw"                # everything lands here first
CLASS_NAMES = ["face", "eye_open", "eye_closed", "mouth_open", "mouth_closed"]

FACE = 0
EYE_OPEN, EYE_CLOSED = 1, 2
MOUTH_OPEN, MOUTH_CLOSED = 3, 4


def to_yolo_line(
    class_id: int, box: Tuple[int, int, int, int], image_width: int, image_height: int
) -> str:
    """Convert a pixel box (x1,y1,x2,y2) into one YOLO label line.

    YOLO wants the CENTRE of the box plus its size, all divided by the image
    dimensions so the numbers sit between 0 and 1 and survive any resizing.
    """
    x1, y1, x2, y2 = box
    x_center = ((x1 + x2) / 2.0) / image_width
    y_center = ((y1 + y2) / 2.0) / image_height
    width = (x2 - x1) / image_width
    height = (y2 - y1) / image_height

    # Clamp: a box must stay inside the image or Ultralytics rejects the label.
    x_center = min(max(x_center, 0.0), 1.0)
    y_center = min(max(y_center, 0.0), 1.0)
    width = min(max(width, 0.0), 1.0)
    height = min(max(height, 0.0), 1.0)
    return f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"


def label_one_frame(
    frame: np.ndarray,
    detector: YOLODetector,
    mesh: FaceMeshAnalyzer,
    ear_threshold: float,
    mar_threshold: float,
    allow_multi_face: bool = False,
) -> Optional[List[str]]:
    """Produce the YOLO label lines for one frame, or None if it is unusable.

    Why we skip frames containing more than one face
    ------------------------------------------------
    We can only measure eyes and mouth for ONE face per frame (MediaPipe runs on
    the driver's crop). If a second person is in shot, their eyes and mouth would
    be left unlabelled - and in YOLO training, "unlabelled" means "background".
    The model would literally be taught that eyes are not eyes, which quietly
    poisons the dataset.

    So by default a multi-face frame is rejected. Record your training sessions
    alone in the frame. Pass --allow-multi-face only if you accept that cost.
    """
    height, width = frame.shape[:2]

    detections = detector.detect(frame)
    face = detections.primary_face()
    if face is None:
        return None            # no face -> nothing to learn from, skip the frame

    if len(detections.faces) > 1 and not allow_multi_face:
        return None            # see the docstring: avoid mislabelled background

    # Run FaceMesh on the face crop, exactly as the live pipeline does.
    margin_x, margin_y = int(face.width * 0.15), int(face.height * 0.15)
    x1 = max(0, face.x1 - margin_x)
    y1 = max(0, face.y1 - margin_y)
    x2 = min(width, face.x2 + margin_x)
    y2 = min(height, face.y2 + margin_y)
    if x2 - x1 < 40 or y2 - y1 < 40:
        return None

    crop_rgb = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
    landmarks = mesh.analyze(crop_rgb, offset=(x1, y1), full_frame_shape=(height, width))
    if not landmarks.found:
        return None            # landmarks failed -> we cannot label eyes/mouth

    lines = [to_yolo_line(FACE, face.box, width, height)]

    # --- eyes: class decided per eye by that eye's own EAR ---
    for box, ear in (
        (landmarks.left_eye_box, landmarks.left_ear),
        (landmarks.right_eye_box, landmarks.right_ear),
    ):
        if box is None:
            continue
        class_id = EYE_CLOSED if ear < ear_threshold else EYE_OPEN
        lines.append(to_yolo_line(class_id, box, width, height))

    # --- mouth: open (yawn) vs closed, decided by MAR ---
    if landmarks.mouth_box is not None:
        class_id = MOUTH_OPEN if landmarks.mar > mar_threshold else MOUTH_CLOSED
        lines.append(to_yolo_line(class_id, landmarks.mouth_box, width, height))

    return lines


def record(
    session: str,
    frames: int,
    every: int,
    camera_index: Optional[int],
    allow_multi_face: bool = False,
) -> None:
    """Capture frames from the webcam and write image + label pairs."""
    cfg = load_config()
    images_dir = RAW_DIR / "images"
    labels_dir = RAW_DIR / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    detector = YOLODetector(
        weights=cfg.model.weights,
        device=cfg.model.device,
        confidence=cfg.model.confidence,
        iou=cfg.model.iou,
        imgsz=cfg.model.imgsz,
    )
    mesh = FaceMeshAnalyzer(refine=cfg.landmarks.refine)

    ear_threshold = float(cfg.drowsiness.ear_threshold)
    mar_threshold = float(cfg.drowsiness.mar_threshold)

    print(f"\nRecording session '{session}': target {frames} labelled frames.")
    print("Look at the camera and act out the behaviour for this session.")
    print("Make sure you are ALONE in the frame - multi-face frames are skipped.")
    print("Press Ctrl+C to stop early.\n")

    camera = WebcamStream(
        index=camera_index if camera_index is not None else cfg.camera.index,
        width=cfg.camera.width,
        height=cfg.camera.height,
        fps=cfg.camera.fps,
        flip_horizontal=cfg.camera.flip_horizontal,
    )

    saved = skipped = seen = 0
    try:
        camera.start()
        # Give the user a moment to get into position.
        for countdown in (3, 2, 1):
            print(f"  starting in {countdown}...", flush=True)
            time.sleep(1.0)

        while saved < frames:
            ok, frame = camera.read()
            if not ok or frame is None:
                time.sleep(0.01)
                continue

            seen += 1
            # Sub-sampling: consecutive webcam frames are nearly identical, and
            # near-duplicates teach the model nothing while inflating the set.
            if seen % every != 0:
                continue

            lines = label_one_frame(
                frame, detector, mesh, ear_threshold, mar_threshold, allow_multi_face
            )
            if lines is None:
                skipped += 1
                continue

            stem = f"{session}_{saved:05d}"
            cv2.imwrite(str(images_dir / f"{stem}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
            (labels_dir / f"{stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
            saved += 1

            if saved % 25 == 0:
                print(f"  saved {saved}/{frames} "
                      f"(skipped {skipped} frames: no face, or more than one face)")

    except KeyboardInterrupt:
        print("\n  stopped by user")
    except CameraError as exc:
        print(f"\nCamera error: {exc}")
    finally:
        camera.stop()
        mesh.close()

    print(f"\nDone. {saved} labelled frames in {images_dir}")
    total = len(list(images_dir.glob('*.jpg')))
    print(f"Dataset now holds {total} frames in total.")
    if total < 500:
        print("Tip: aim for 1500+ frames across varied lighting, angles and glasses.")


def split(train_ratio: float, val_ratio: float, seed: int) -> None:
    """Shuffle the raw pool into train/val/test folders."""
    images_dir = RAW_DIR / "images"
    labels_dir = RAW_DIR / "labels"
    stems = sorted(path.stem for path in images_dir.glob("*.jpg"))

    if not stems:
        print(f"No images found in {images_dir}. Record some first:")
        print("  python training/auto_label.py --session normal --frames 400")
        return

    random.Random(seed).shuffle(stems)
    total = len(stems)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    partitions = {
        "train": stems[:train_end],
        "val": stems[train_end:val_end],
        "test": stems[val_end:],
    }

    for name, members in partitions.items():
        image_out = DATASET_ROOT / "images" / name
        label_out = DATASET_ROOT / "labels" / name
        # Start clean so re-running --split never mixes two different splits.
        for folder in (image_out, label_out):
            if folder.exists():
                shutil.rmtree(folder)
            folder.mkdir(parents=True, exist_ok=True)

        for stem in members:
            shutil.copyfile(images_dir / f"{stem}.jpg", image_out / f"{stem}.jpg")
            shutil.copyfile(labels_dir / f"{stem}.txt", label_out / f"{stem}.txt")
        print(f"  {name:5s}: {len(members)} frames")

    print(f"\nSplit {total} frames. Now train:\n  python training/train.py")
    _print_class_balance(partitions, labels_dir)


def _print_class_balance(partitions: dict, labels_dir: Path) -> None:
    """Report how many boxes of each class exist.

    A badly skewed dataset (say 2000 eye_open and 30 eye_closed) trains a model
    that never predicts the rare class, so it is worth seeing before training.
    """
    counts = {name: 0 for name in CLASS_NAMES}
    for members in partitions.values():
        for stem in members:
            for line in (labels_dir / f"{stem}.txt").read_text().splitlines():
                if line.strip():
                    counts[CLASS_NAMES[int(line.split()[0])]] += 1

    print("\nClass balance (number of boxes):")
    largest = max(counts.values()) or 1
    for name, count in counts.items():
        bar = "#" * int(30 * count / largest)
        print(f"  {name:14s} {count:6d}  {bar}")

    rare = [name for name, count in counts.items() if count < largest * 0.05 and count >= 0]
    if rare:
        print(f"\n  WARNING: under-represented classes: {', '.join(rare)}")
        print("  Record more sessions acting out those states before training.")


def preview(count: int) -> None:
    """Draw the saved labels back onto the saved images, to verify them."""
    images_dir = RAW_DIR / "images"
    labels_dir = RAW_DIR / "labels"
    stems = sorted(path.stem for path in images_dir.glob("*.jpg"))
    if not stems:
        print(f"No images in {images_dir} yet.")
        return

    out_dir = DATASET_ROOT / "preview"
    out_dir.mkdir(parents=True, exist_ok=True)
    colors = [(0, 255, 0), (255, 200, 0), (0, 0, 255), (0, 0, 255), (0, 220, 255)]

    for stem in random.sample(stems, min(count, len(stems))):
        image = cv2.imread(str(images_dir / f"{stem}.jpg"))
        height, width = image.shape[:2]
        for line in (labels_dir / f"{stem}.txt").read_text().splitlines():
            if not line.strip():
                continue
            class_id, xc, yc, bw, bh = line.split()
            class_id = int(class_id)
            # Undo the normalisation to get pixels back.
            xc, yc, bw, bh = float(xc) * width, float(yc) * height, float(bw) * width, float(bh) * height
            x1, y1 = int(xc - bw / 2), int(yc - bh / 2)
            x2, y2 = int(xc + bw / 2), int(yc + bh / 2)
            cv2.rectangle(image, (x1, y1), (x2, y2), colors[class_id], 2)
            cv2.putText(image, CLASS_NAMES[class_id], (x1, max(12, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, colors[class_id], 1, cv2.LINE_AA)
        cv2.imwrite(str(out_dir / f"{stem}_labelled.jpg"), image)

    print(f"Wrote {min(count, len(stems))} annotated previews to {out_dir}")
    print("Open them and confirm the boxes sit on the right features.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--session", type=str, help="name for this recording session")
    parser.add_argument("--frames", type=int, default=300, help="labelled frames to save")
    parser.add_argument("--every", type=int, default=3, help="keep 1 frame out of every N")
    parser.add_argument("--camera", type=int, default=None, help="camera index override")
    parser.add_argument("--allow-multi-face", action="store_true",
                        help="keep frames with several faces (lowers label quality)")
    parser.add_argument("--split", action="store_true", help="split raw pool into train/val/test")
    parser.add_argument("--train-ratio", type=float, default=0.75)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--preview", action="store_true", help="render labels onto images to check them")
    parser.add_argument("--preview-count", type=int, default=12)
    args = parser.parse_args()

    if args.split:
        split(args.train_ratio, args.val_ratio, args.seed)
    elif args.preview:
        preview(args.preview_count)
    elif args.session:
        record(args.session, args.frames, args.every, args.camera, args.allow_multi_face)
    else:
        parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
