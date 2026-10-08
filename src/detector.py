"""
YOLO object detection wrapper (Ultralytics).

What YOLO gives us
------------------
YOLO ("You Only Look Once") is a single-stage detector: one forward pass over
the image produces every bounding box, class label and confidence at once.
That is why it is fast enough for a live webcam.

Two supported model types
-------------------------
1. FACE-ONLY model (the default, `models/yolov11n-face.pt`)
   Classes: {0: 'face'}. YOLO finds the face box; MediaPipe FaceMesh then
   locates eyes and mouth *inside* that box (see src/landmarks.py).

2. CUSTOM 5-CLASS model (what you get from `python training/train.py`)
   Classes: face, eye_open, eye_closed, mouth_open, mouth_closed.
   Here YOLO detects eyes and mouth natively and we can score drowsiness
   from YOLO alone, with FaceMesh only refining the EAR value.

The class is written so the rest of the app does not care which one is loaded:
`detector.has_eye_classes` tells the pipeline what it is working with.
"""

from __future__ import annotations

import platform
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import resolve_path

# Class names our custom-trained model uses. Keep in sync with training/data.yaml.
CUSTOM_CLASS_NAMES = ["face", "eye_open", "eye_closed", "mouth_open", "mouth_closed"]

EYE_CLASSES = {"eye_open", "eye_closed", "eye"}
MOUTH_CLASSES = {"mouth_open", "mouth_closed", "mouth", "yawn"}
FACE_CLASSES = {"face", "head", "person"}


class ModelError(RuntimeError):
    """Raised when the YOLO checkpoint cannot be found or loaded."""


@dataclass
class Detection:
    """One bounding box produced by YOLO."""

    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    class_id: int
    class_name: str

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def area(self) -> int:
        return max(0, self.width) * max(0, self.height)

    @property
    def center(self) -> Tuple[int, int]:
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)

    @property
    def box(self) -> Tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)

    def label(self, show_confidence: bool = True) -> str:
        return f"{self.class_name} {self.confidence:.2f}" if show_confidence else self.class_name


@dataclass
class DetectionResult:
    """Everything YOLO found in one frame, already sorted into categories."""

    faces: List[Detection] = field(default_factory=list)
    eyes: List[Detection] = field(default_factory=list)
    mouths: List[Detection] = field(default_factory=list)
    others: List[Detection] = field(default_factory=list)
    inference_ms: float = 0.0

    @property
    def all(self) -> List[Detection]:
        return self.faces + self.eyes + self.mouths + self.others

    @property
    def face_count(self) -> int:
        return len(self.faces)

    def primary_face(self) -> Optional[Detection]:
        """The driver = the largest face box (the person closest to the camera).

        Passengers sitting further back produce smaller boxes, so this simple
        rule is both fast and reliable for an in-car camera.
        """
        return max(self.faces, key=lambda d: d.area) if self.faces else None


def select_device(preference: str = "auto") -> str:
    """Choose the compute device.

    'auto' resolves to:
        mps  - Apple Silicon GPU (M1/M2/M3/M4). This is the fast path on a Mac.
        cuda - NVIDIA GPU
        cpu  - everything else
    """
    if preference and preference != "auto":
        return preference

    try:
        import torch
    except ImportError:  # pragma: no cover
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"
    if platform.machine() == "arm64" and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class YOLODetector:
    """Loads a YOLO checkpoint once and runs it on every frame."""

    def __init__(
        self,
        weights: str,
        device: str = "auto",
        confidence: float = 0.35,
        iou: float = 0.45,
        imgsz: int = 480,
        max_detections: int = 5,
        half_precision: bool = False,
    ):
        self.weights_path = resolve_path(weights)
        self.device = select_device(device)
        self.confidence = confidence
        self.iou = iou
        self.imgsz = imgsz
        self.max_detections = max_detections
        self.half_precision = half_precision and self.device == "cuda"  # fp16 is CUDA-only here

        self.model = self._load_model()
        self.class_names: Dict[int, str] = dict(self.model.names)

        lowered = {name.lower() for name in self.class_names.values()}
        self.has_eye_classes = bool(lowered & EYE_CLASSES)
        self.has_mouth_classes = bool(lowered & MOUTH_CLASSES)

        self._warmup()

    # ------------------------------------------------------------------ load
    def _load_model(self):
        if not self.weights_path.exists():
            raise ModelError(
                f"YOLO weights not found at: {self.weights_path}\n\n"
                "Fix it with one of these:\n"
                "  1. python tools/download_models.py       (downloads a face model)\n"
                "  2. python training/train.py              (train your own 5-class model)\n"
                "  3. Edit model.weights in config/config.yaml to point at your .pt file"
            )

        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ModelError(
                "Ultralytics is not installed. Run: pip install -r requirements.txt"
            ) from exc

        try:
            return YOLO(str(self.weights_path))
        except Exception as exc:
            raise ModelError(
                f"Failed to load the YOLO checkpoint {self.weights_path.name}: {exc}\n"
                "The file may be corrupted - delete it and re-run tools/download_models.py"
            ) from exc

    def _warmup(self) -> None:
        """Run one dummy inference.

        The very first YOLO call compiles kernels and is 10-50x slower than the
        rest. Doing it here means the first *real* frame is already fast, so the
        FPS counter does not start with a terrible number.
        """
        dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
        try:
            self.model.predict(dummy, imgsz=self.imgsz, device=self.device, verbose=False)
        except Exception:
            # MPS occasionally refuses on the very first call; fall back to CPU
            # rather than killing the app.
            if self.device != "cpu":
                self.device = "cpu"
                self.model.predict(dummy, imgsz=self.imgsz, device="cpu", verbose=False)

    # ------------------------------------------------------------- inference
    def detect(self, frame: np.ndarray) -> DetectionResult:
        """Run YOLO on one BGR frame and return categorised detections."""
        if frame is None or frame.size == 0:
            return DetectionResult()

        try:
            results = self.model.predict(
                frame,
                imgsz=self.imgsz,
                conf=self.confidence,
                iou=self.iou,
                max_det=self.max_detections,
                device=self.device,
                half=self.half_precision,
                verbose=False,
            )
        except Exception as exc:
            # A single bad frame must never crash the live loop.
            if self.device != "cpu":
                self.device = "cpu"  # permanent downgrade, it will keep working
                return self.detect(frame)
            raise ModelError(f"YOLO inference failed: {exc}") from exc

        if not results:
            return DetectionResult()

        result = results[0]
        # Ultralytics reports per-stage timings in milliseconds.
        inference_ms = float(result.speed.get("inference", 0.0)) if hasattr(result, "speed") else 0.0
        output = DetectionResult(inference_ms=inference_ms)

        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return output

        frame_h, frame_w = frame.shape[:2]
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        class_ids = boxes.cls.cpu().numpy().astype(int)

        for (x1, y1, x2, y2), conf, class_id in zip(xyxy, confs, class_ids):
            class_name = self.class_names.get(int(class_id), f"class_{class_id}")
            detection = Detection(
                # Clamp to the frame so drawing and cropping can never go out of bounds.
                x1=int(max(0, min(x1, frame_w - 1))),
                y1=int(max(0, min(y1, frame_h - 1))),
                x2=int(max(0, min(x2, frame_w - 1))),
                y2=int(max(0, min(y2, frame_h - 1))),
                confidence=float(conf),
                class_id=int(class_id),
                class_name=class_name,
            )
            if detection.area <= 0:
                continue  # degenerate box, ignore

            lowered = class_name.lower()
            if lowered in FACE_CLASSES:
                output.faces.append(detection)
            elif lowered in EYE_CLASSES:
                output.eyes.append(detection)
            elif lowered in MOUTH_CLASSES:
                output.mouths.append(detection)
            else:
                output.others.append(detection)

        return output

    # ------------------------------------------------------------------ info
    def describe(self) -> Dict[str, object]:
        """Human-readable summary, shown in the dashboard sidebar."""
        return {
            "weights": self.weights_path.name,
            "device": self.device,
            "classes": list(self.class_names.values()),
            "num_classes": len(self.class_names),
            "imgsz": self.imgsz,
            "confidence": self.confidence,
            "native_eye_detection": self.has_eye_classes,
            "native_mouth_detection": self.has_mouth_classes,
        }


# -----------------------------------------------------------------------------
# Helpers for the custom 5-class model
# -----------------------------------------------------------------------------
def eyes_closed_from_detections(result: DetectionResult) -> Optional[bool]:
    """Decide open vs closed from YOLO's own eye classes.

    Only usable with the custom-trained model, which has separate `eye_open`
    and `eye_closed` classes. Returns None when the model has no eye classes or
    saw no eyes, so the caller can fall back to the landmark EAR.

    We compare the total confidence of each side rather than just counting
    boxes: one confident `eye_closed` should outweigh a hesitant `eye_open`.
    """
    if not result.eyes:
        return None

    closed_score = sum(d.confidence for d in result.eyes if d.class_name.lower() == "eye_closed")
    open_score = sum(d.confidence for d in result.eyes if d.class_name.lower() == "eye_open")
    if closed_score == 0 and open_score == 0:
        return None
    return closed_score > open_score


def mouth_open_from_detections(result: DetectionResult) -> Optional[bool]:
    """Same idea for the mouth: is YOLO calling it open (a yawn) or closed?"""
    if not result.mouths:
        return None

    open_score = sum(
        d.confidence for d in result.mouths if d.class_name.lower() in ("mouth_open", "yawn")
    )
    closed_score = sum(
        d.confidence for d in result.mouths if d.class_name.lower() == "mouth_closed"
    )
    if open_score == 0 and closed_score == 0:
        return None
    return open_score > closed_score


def detections_inside(
    candidates: List[Detection], container: Detection, min_overlap: float = 0.5
) -> List[Detection]:
    """Keep only the detections that lie mostly inside `container`.

    Used to attach eye/mouth boxes to the DRIVER's face rather than a
    passenger's, by checking how much of each small box falls inside the face.
    """
    kept = []
    for candidate in candidates:
        # Intersection rectangle between the candidate and the container.
        ix1 = max(candidate.x1, container.x1)
        iy1 = max(candidate.y1, container.y1)
        ix2 = min(candidate.x2, container.x2)
        iy2 = min(candidate.y2, container.y2)
        if ix2 <= ix1 or iy2 <= iy1:
            continue
        intersection = (ix2 - ix1) * (iy2 - iy1)
        if candidate.area > 0 and intersection / candidate.area >= min_overlap:
            kept.append(candidate)
    return kept
