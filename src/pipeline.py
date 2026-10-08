"""
The pipeline: one object that owns every stage and processes one frame at a time.

Per-frame flow
--------------
    webcam frame (BGR)
        |
        v
    [1] YOLO  -> face boxes + confidences
        |
        v
    [2] pick the largest face = the driver
        |
        v
    [3] crop that face, run MediaPipe FaceMesh on the crop
        -> 478 landmarks -> EAR, MAR, eye boxes, mouth box, head tilt
        |
        v
    [4] DrowsinessEngine.update(EAR, MAR) -> score + state + counters
        |
        v
    [5] heatmap intensities from EAR / MAR / score -> Gaussian attention map
        |
        v
    [6] draw boxes, labels, HUD, blend the heatmap
        |
        v
    [7] AlarmSystem.update(is_drowsy) -> sound if needed
        |
        v
    annotated frame + metrics -> the dashboard

Why crop before FaceMesh (step 3)?
  * Accuracy: the eyes occupy far more pixels in a 200x200 face crop than in the
    full 640x480 frame, so the landmarks are more precise and EAR is less noisy.
  * Speed: MediaPipe processes a smaller image.
  * Consistency: YOLO decides who the driver is, so FaceMesh cannot wander off
    to a passenger's face.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Optional, Tuple

import cv2
import numpy as np

from .alarm import AlarmSystem
from .camera import CameraError, WebcamStream
from .detector import (
    Detection,
    DetectionResult,
    ModelError,
    YOLODetector,
    detections_inside,
    eyes_closed_from_detections,
    mouth_open_from_detections,
)
from .drowsiness import DrowsinessEngine, DrowsinessMetrics, DrowsinessState
from .heatmap import AttentionHeatmap, intensity_from_ear, intensity_from_mar
from .landmarks import FaceMeshAnalyzer, LandmarkResult
from .overlay import draw_detections, draw_hud, draw_landmarks

# How much context to include around the YOLO face box before running FaceMesh.
# YOLO face boxes are tight; FaceMesh wants a little breathing room to find the
# full head outline, so we grow the box by 15% on each side.
FACE_CROP_MARGIN = 0.15


@dataclass
class FrameResult:
    """Everything produced from one frame, handed to the UI."""

    frame: np.ndarray                                   # annotated, ready to show
    raw_frame: np.ndarray                               # clean copy, no drawing
    heatmap_view: Optional[np.ndarray] = None           # standalone heatmap panel
    metrics: DrowsinessMetrics = field(default_factory=DrowsinessMetrics)
    detections: DetectionResult = field(default_factory=DetectionResult)
    landmarks: Optional[LandmarkResult] = None
    fps: float = 0.0
    inference_ms: float = 0.0
    total_ms: float = 0.0
    alarm_fired: bool = False


class FPSCounter:
    """Smoothed frames-per-second, measured from real wall-clock timestamps."""

    def __init__(self, smoothing: float = 0.9, window: int = 30):
        self.smoothing = smoothing
        self._timestamps: Deque[float] = deque(maxlen=window)
        self._value = 0.0

    def tick(self, now: Optional[float] = None) -> float:
        now = now if now is not None else time.perf_counter()
        self._timestamps.append(now)
        if len(self._timestamps) < 2:
            return 0.0

        elapsed = self._timestamps[-1] - self._timestamps[0]
        if elapsed <= 0:
            return self._value

        instant = (len(self._timestamps) - 1) / elapsed
        # Exponential smoothing keeps the readout from jittering every frame.
        self._value = (
            instant if self._value == 0.0
            else self.smoothing * self._value + (1.0 - self.smoothing) * instant
        )
        return self._value

    @property
    def value(self) -> float:
        return self._value

    def reset(self) -> None:
        self._timestamps.clear()
        self._value = 0.0


class DrowsinessPipeline:
    """Owns the camera, model, engine, heatmap and alarm for one session."""

    def __init__(self, cfg, camera_index: Optional[int] = None):
        self.cfg = cfg

        # --- model (may raise ModelError - the UI shows that message) ---
        self.detector = YOLODetector(
            weights=cfg.model.weights,
            device=cfg.model.device,
            confidence=cfg.model.confidence,
            iou=cfg.model.iou,
            imgsz=cfg.model.imgsz,
            max_detections=cfg.model.max_detections,
            half_precision=cfg.model.half_precision,
        )

        # --- landmarks (optional, but needed for EAR/MAR with a face-only model) ---
        self.mesh: Optional[FaceMeshAnalyzer] = None
        if cfg.landmarks.enabled:
            self.mesh = FaceMeshAnalyzer(
                max_faces=cfg.landmarks.max_faces,
                min_detection_confidence=cfg.landmarks.min_detection_confidence,
                min_tracking_confidence=cfg.landmarks.min_tracking_confidence,
                refine=cfg.landmarks.refine,
            )

        self.engine = DrowsinessEngine(cfg.drowsiness)
        self.heatmap = AttentionHeatmap(cfg.heatmap)
        self.alarm = AlarmSystem(cfg.alarm)
        self.fps_counter = FPSCounter(smoothing=cfg.ui.fps_smoothing)

        self.camera_index = camera_index if camera_index is not None else cfg.camera.index
        self.camera: Optional[WebcamStream] = None
        self.frames_processed = 0
        self._last_good_landmarks: Optional[LandmarkResult] = None

    # --------------------------------------------------------------- camera
    def start_camera(self) -> None:
        """Open the webcam. Raises CameraError with a user-facing message."""
        if self.camera is not None and self.camera.is_running:
            return
        self.camera = WebcamStream(
            index=self.camera_index,
            width=self.cfg.camera.width,
            height=self.cfg.camera.height,
            fps=self.cfg.camera.fps,
            flip_horizontal=self.cfg.camera.flip_horizontal,
            warmup_frames=self.cfg.camera.warmup_frames,
        )
        self.camera.start()
        self.fps_counter.reset()

    def stop_camera(self) -> None:
        """Always release the device - a held camera blocks every other app."""
        if self.camera is not None:
            self.camera.stop()
            self.camera = None

    # -------------------------------------------------------------- one frame
    def process_frame(self, frame: np.ndarray) -> FrameResult:
        """Run the full detection + scoring + drawing chain on one BGR frame."""
        started = time.perf_counter()
        raw_frame = frame.copy()

        # ---- [1] YOLO ----------------------------------------------------
        detections = self.detector.detect(frame)
        primary_face = detections.primary_face()

        # ---- [2][3] landmarks inside the driver's face -------------------
        landmarks: Optional[LandmarkResult] = None
        if primary_face is not None and self.mesh is not None:
            landmarks = self._analyze_face(frame, primary_face)

        landmarks_ok = landmarks is not None and landmarks.found

        # ---- [2b] native YOLO eye/mouth boxes, if this model has them ------
        # With the custom 5-class model, YOLO detects eyes and mouth directly.
        # We keep only the boxes that fall inside the driver's face, so a
        # passenger's eyes can never be attributed to the driver.
        yolo_eyes: list = []
        yolo_mouths: list = []
        if primary_face is not None and self.detector.has_eye_classes:
            yolo_eyes = detections_inside(detections.eyes, primary_face)
            yolo_mouths = detections_inside(detections.mouths, primary_face)

        # A face counts as usable if EITHER source can tell us about the eyes.
        face_found = primary_face is not None and (landmarks_ok or bool(yolo_eyes))

        # ---- [3b] pick the EAR / MAR values to score --------------------
        ear_value, mar_value = self._resolve_measurements(landmarks, yolo_eyes, yolo_mouths)

        # ---- [4] drowsiness scoring --------------------------------------
        metrics = self.engine.update(
            ear=ear_value if face_found else None,
            mar=mar_value if face_found else None,
            face_detected=face_found,
            face_count=detections.face_count,
            head_tilt=landmarks.head_tilt if landmarks_ok else 0.0,
        )

        # ---- [4b] choose which boxes to draw and heat up ------------------
        # Native YOLO boxes win when the model produces them; otherwise we use
        # the landmark-derived boxes. Either way these are measured, never
        # assumed, positions.
        left_eye_box, right_eye_box, mouth_box = self._resolve_boxes(
            landmarks if landmarks_ok else None, yolo_eyes, yolo_mouths
        )

        # ---- [5] attention heatmap ---------------------------------------
        heatmap_view = None
        if self.cfg.heatmap.enabled:
            attention = self._build_attention(
                frame.shape, primary_face, metrics,
                left_eye_box, right_eye_box, mouth_box,
                ear_value, mar_value,
            )
            heatmap_view = self.heatmap.standalone_view(attention, raw_frame)
            frame = self.heatmap.overlay(frame, attention)

        # ---- [6] drawing ---------------------------------------------------
        if self.cfg.ui.show_boxes:
            frame = draw_detections(
                frame,
                faces=detections.faces,
                primary_face=primary_face,
                metrics=metrics,
                left_eye_box=left_eye_box,
                right_eye_box=right_eye_box,
                mouth_box=mouth_box,
                show_labels=self.cfg.ui.show_labels,
                show_confidence=self.cfg.ui.show_confidence,
                left_ear=landmarks.left_ear if landmarks_ok else None,
                right_ear=landmarks.right_ear if landmarks_ok else None,
            )
        if self.cfg.ui.show_landmarks and landmarks_ok and landmarks is not None:
            draw_landmarks(frame, landmarks.points)

        fps = self.fps_counter.tick()
        if self.cfg.ui.show_hud:
            frame = draw_hud(frame, metrics, fps)

        # ---- [7] alarm -----------------------------------------------------
        alarm_fired = self.alarm.update(metrics.state == DrowsinessState.DROWSY)

        self.frames_processed += 1
        return FrameResult(
            frame=frame,
            raw_frame=raw_frame,
            heatmap_view=heatmap_view,
            metrics=metrics,
            detections=detections,
            landmarks=landmarks,
            fps=fps,
            inference_ms=detections.inference_ms,
            total_ms=(time.perf_counter() - started) * 1000.0,
            alarm_fired=alarm_fired,
        )

    def read_and_process(self) -> Optional[FrameResult]:
        """Grab the newest webcam frame and process it. None if none is ready."""
        if self.camera is None:
            raise CameraError("The camera has not been started. Call start_camera() first.")
        ok, frame = self.camera.read()
        if not ok or frame is None:
            return None
        return self.process_frame(frame)

    # ------------------------------------------------------------- internals
    def _analyze_face(self, frame: np.ndarray, face: Detection) -> Optional[LandmarkResult]:
        """Crop the face box (with margin) and run FaceMesh on that crop."""
        height, width = frame.shape[:2]
        margin_x = int(face.width * FACE_CROP_MARGIN)
        margin_y = int(face.height * FACE_CROP_MARGIN)

        x1 = max(0, face.x1 - margin_x)
        y1 = max(0, face.y1 - margin_y)
        x2 = min(width, face.x2 + margin_x)
        y2 = min(height, face.y2 + margin_y)

        # A box too small to contain usable eye pixels: skip rather than guess.
        if x2 - x1 < 40 or y2 - y1 < 40:
            return None

        crop = frame[y1:y2, x1:x2]
        # MediaPipe expects RGB; OpenCV gives us BGR.
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

        result = self.mesh.analyze(
            crop_rgb,
            offset=(x1, y1),              # shift coordinates back to full-frame space
            full_frame_shape=(height, width),
        )

        if result.found:
            self._last_good_landmarks = result
        return result

    def _resolve_measurements(
        self,
        landmarks: Optional[LandmarkResult],
        yolo_eyes: list,
        yolo_mouths: list,
    ) -> Tuple[Optional[float], Optional[float]]:
        """Decide which EAR and MAR numbers to score this frame.

        Priority:
          1. MediaPipe landmarks - a real, continuous measurement. Always best.
          2. YOLO's eye_open / eye_closed classes - only a binary answer, so we
             synthesise an EAR just either side of the threshold. This is what
             keeps the system working when landmarks are disabled and the custom
             5-class model is loaded.
        """
        if landmarks is not None and landmarks.found:
            return landmarks.ear, landmarks.mar

        ear_value = mar_value = None

        closed = eyes_closed_from_detections(DetectionResult(eyes=list(yolo_eyes)))
        if closed is not None:
            # A binary class cannot give a real ratio, so we place the value
            # clearly on the correct side of the threshold.
            ear_value = self.engine.ear_threshold * (0.5 if closed else 1.6)

        mouth_open = mouth_open_from_detections(DetectionResult(mouths=list(yolo_mouths)))
        if mouth_open is not None:
            mar_value = self.engine.mar_threshold * (1.4 if mouth_open else 0.3)

        return ear_value, mar_value

    def _resolve_boxes(
        self,
        landmarks: Optional[LandmarkResult],
        yolo_eyes: list,
        yolo_mouths: list,
    ) -> Tuple[Optional[tuple], Optional[tuple], Optional[tuple]]:
        """Pick the eye and mouth boxes to draw: YOLO's if it has them."""
        if yolo_eyes:
            # Sort left to right so the labels stay on consistent sides.
            ordered = sorted(yolo_eyes, key=lambda d: d.center[0])
            left_box = ordered[0].box
            right_box = ordered[-1].box if len(ordered) > 1 else None
            mouth_box = (
                max(yolo_mouths, key=lambda d: d.confidence).box if yolo_mouths else None
            )
            return left_box, right_box, mouth_box

        if landmarks is not None and landmarks.found:
            return landmarks.left_eye_box, landmarks.right_eye_box, landmarks.mouth_box

        return None, None, None

    def _build_attention(
        self,
        frame_shape: Tuple[int, ...],
        face: Optional[Detection],
        metrics: DrowsinessMetrics,
        left_eye_box: Optional[tuple],
        right_eye_box: Optional[tuple],
        mouth_box: Optional[tuple],
        ear_value: Optional[float],
        mar_value: Optional[float],
    ) -> np.ndarray:
        """Convert the current measurements into heatmap intensities."""
        eye_boxes = [left_eye_box, right_eye_box]
        eye_intensity = (
            intensity_from_ear(ear_value, self.engine.ear_threshold)
            if ear_value is not None else 0.0
        )
        mouth_intensity = (
            intensity_from_mar(mar_value, self.engine.mar_threshold)
            if mar_value is not None else 0.0
        )

        return self.heatmap.build(
            frame_shape=frame_shape,
            eye_boxes=eye_boxes,
            eye_intensity=eye_intensity,
            mouth_box=mouth_box,
            mouth_intensity=mouth_intensity,
            face_box=face.box if face is not None else None,
            # The face-wide glow tracks the overall score, so the driver's whole
            # head warms up as fatigue builds.
            face_intensity=metrics.score / 100.0,
        )

    # ------------------------------------------------------------- lifecycle
    def reset_session(self) -> None:
        """Zero all counters and history without reopening the camera."""
        self.engine.reset()
        self.heatmap.reset()
        self.alarm.reset()
        self.fps_counter.reset()
        self.frames_processed = 0

    def close(self) -> None:
        """Release every resource. Safe to call twice."""
        self.stop_camera()
        if self.mesh is not None:
            self.mesh.close()
            self.mesh = None

    def describe(self) -> Dict[str, object]:
        """Diagnostics for the sidebar."""
        info = {"model": self.detector.describe(), "alarm": self.alarm.describe()}
        if self.camera is not None and self.camera.info is not None:
            cam = self.camera.info
            info["camera"] = {
                "index": cam.index,
                "resolution": f"{cam.width}x{cam.height}",
                "driver_fps": round(cam.fps, 1),
                "backend": cam.backend,
            }
        info["landmarks"] = "MediaPipe FaceMesh" if self.mesh is not None else "disabled"
        return info

    def __enter__(self) -> "DrowsinessPipeline":
        self.start_camera()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
