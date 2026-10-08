"""
Facial landmarks with MediaPipe FaceMesh -> EAR and MAR.

Why we need landmarks at all
----------------------------
YOLO gives us a *box*. A box cannot tell you whether an eye is open, because an
open eye and a closed eye occupy almost the same rectangle. To measure eyelid
closure you need the actual eyelid points. MediaPipe FaceMesh gives 478 3D
landmarks per face in real time on CPU, which is exactly what we need.

EAR - Eye Aspect Ratio  (Soukupova & Cech, 2016)
------------------------------------------------
Take 6 points around the eye:

        p2   p3
    p1            p4        EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
        p6   p5

  * p1, p4 are the eye corners  -> the horizontal width (denominator)
  * p2,p3,p5,p6 are the eyelids -> the vertical height (numerator)

Open eye  -> tall and wide  -> EAR around 0.25 - 0.35
Closed eye-> flat           -> EAR drops sharply toward 0.0 - 0.15

Dividing by the width is the clever bit: it makes EAR scale-invariant, so
leaning closer to the camera does not change the value.

MAR - Mouth Aspect Ratio
------------------------
The same idea for the mouth: vertical lip opening divided by mouth width.
A yawn is a big, sustained MAR. Talking gives short MAR spikes, which is why
the drowsiness engine also requires a minimum duration (see config.yaml).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

# -----------------------------------------------------------------------------
# FaceMesh landmark indices.
# These come from MediaPipe's canonical 468-point face model - they are fixed
# positions in the mesh topology, not values we invented.
# Order matters: [p1, p2, p3, p4, p5, p6] as drawn in the diagram above.
# -----------------------------------------------------------------------------
LEFT_EYE_EAR = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_EAR = [33, 160, 158, 133, 153, 144]

# Full eye contours, used to draw an accurate eye bounding box.
LEFT_EYE_CONTOUR = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE_CONTOUR = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

# Mouth: two corners for the width, three upper/lower lip pairs for the height.
MOUTH_CORNERS = [61, 291]
MOUTH_VERTICAL_PAIRS = [(81, 178), (13, 14), (311, 402)]
MOUTH_CONTOUR = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
                 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]

# Points used to estimate head pose (a nodding head is another fatigue sign).
NOSE_TIP = 1
CHIN = 152
FOREHEAD = 10


@dataclass
class LandmarkResult:
    """Everything we measured from one face."""

    found: bool = False
    left_ear: float = 0.0
    right_ear: float = 0.0
    ear: float = 0.0                 # average of both eyes
    mar: float = 0.0
    left_eye_box: Optional[Tuple[int, int, int, int]] = None
    right_eye_box: Optional[Tuple[int, int, int, int]] = None
    mouth_box: Optional[Tuple[int, int, int, int]] = None
    left_eye_center: Optional[Tuple[int, int]] = None
    right_eye_center: Optional[Tuple[int, int]] = None
    mouth_center: Optional[Tuple[int, int]] = None
    head_tilt: float = 0.0           # degrees; 0 = upright
    points: List[Tuple[int, int]] = field(default_factory=list)  # all landmarks, for drawing


def _euclidean(a: np.ndarray, b: np.ndarray) -> float:
    """Straight-line distance between two 2D points."""
    return float(np.linalg.norm(a - b))


def compute_ear(points: np.ndarray, indices: List[int]) -> float:
    """Eye Aspect Ratio for one eye. See the module docstring for the formula."""
    try:
        p1, p2, p3, p4, p5, p6 = (points[i] for i in indices)
    except IndexError:
        return 0.0

    vertical = _euclidean(p2, p6) + _euclidean(p3, p5)
    horizontal = _euclidean(p1, p4)
    if horizontal < 1e-6:
        return 0.0  # degenerate face, avoid dividing by zero
    return vertical / (2.0 * horizontal)


def compute_mar(points: np.ndarray) -> float:
    """Mouth Aspect Ratio: average lip gap divided by mouth width."""
    try:
        left_corner, right_corner = points[MOUTH_CORNERS[0]], points[MOUTH_CORNERS[1]]
        width = _euclidean(left_corner, right_corner)
        if width < 1e-6:
            return 0.0
        gaps = [_euclidean(points[top], points[bottom]) for top, bottom in MOUTH_VERTICAL_PAIRS]
        return float(np.mean(gaps)) / width
    except IndexError:
        return 0.0


def _bounding_box(
    points: np.ndarray, indices: List[int], padding: int, frame_shape: Tuple[int, int]
) -> Optional[Tuple[int, int, int, int]]:
    """Tight box around a set of landmarks, padded and clamped to the frame."""
    try:
        selected = points[indices]
    except IndexError:
        return None
    if selected.size == 0:
        return None

    height, width = frame_shape
    x1 = int(max(0, selected[:, 0].min() - padding))
    y1 = int(max(0, selected[:, 1].min() - padding))
    x2 = int(min(width - 1, selected[:, 0].max() + padding))
    y2 = int(min(height - 1, selected[:, 1].max() + padding))
    return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None


class FaceMeshAnalyzer:
    """Wraps MediaPipe FaceMesh and turns its landmarks into EAR / MAR / boxes."""

    def __init__(
        self,
        max_faces: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        refine: bool = True,
    ):
        try:
            import mediapipe as mp
        except ImportError as exc:
            raise RuntimeError(
                "MediaPipe is not installed. Run: pip install -r requirements.txt\n"
                "Note: MediaPipe supports Python 3.9-3.12. On 3.13+ it is unavailable,\n"
                "so create the venv with python3.11."
            ) from exc

        self._mp = mp
        self._mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,        # video mode: tracks between frames, much faster
            max_num_faces=max_faces,
            refine_landmarks=refine,        # adds iris points, sharpens the eyelids
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def analyze(
        self,
        frame_rgb: np.ndarray,
        offset: Tuple[int, int] = (0, 0),
        full_frame_shape: Optional[Tuple[int, int]] = None,
    ) -> LandmarkResult:
        """Find landmarks in an RGB image and compute all metrics.

        `offset` and `full_frame_shape` let us run FaceMesh on a *cropped* face
        ROI (faster and more accurate) while reporting coordinates in full-frame
        space, so the boxes line up with the YOLO boxes when we draw them.
        """
        result = LandmarkResult()
        if frame_rgb is None or frame_rgb.size == 0:
            return result

        crop_h, crop_w = frame_rgb.shape[:2]
        out_shape = full_frame_shape if full_frame_shape else (crop_h, crop_w)

        # MediaPipe expects a contiguous, writeable=False array for best speed.
        frame_rgb.flags.writeable = False
        mesh_result = self._mesh.process(frame_rgb)
        frame_rgb.flags.writeable = True

        if not mesh_result.multi_face_landmarks:
            return result

        landmarks = mesh_result.multi_face_landmarks[0].landmark

        # MediaPipe returns normalised coordinates (0-1) relative to the crop.
        # Scale by the crop size, then shift by the crop's position in the frame.
        offset_x, offset_y = offset
        points = np.array(
            [[lm.x * crop_w + offset_x, lm.y * crop_h + offset_y] for lm in landmarks],
            dtype=np.float32,
        )

        result.found = True
        result.left_ear = compute_ear(points, LEFT_EYE_EAR)
        result.right_ear = compute_ear(points, RIGHT_EYE_EAR)
        result.ear = (result.left_ear + result.right_ear) / 2.0
        result.mar = compute_mar(points)

        # Pad eye boxes a little so the drawn rectangle contains the whole eye.
        eye_padding = max(3, int(0.02 * crop_w))
        mouth_padding = max(4, int(0.02 * crop_w))
        result.left_eye_box = _bounding_box(points, LEFT_EYE_CONTOUR, eye_padding, out_shape)
        result.right_eye_box = _bounding_box(points, RIGHT_EYE_CONTOUR, eye_padding, out_shape)
        result.mouth_box = _bounding_box(points, MOUTH_CONTOUR, mouth_padding, out_shape)

        result.left_eye_center = _center_of(points, LEFT_EYE_CONTOUR)
        result.right_eye_center = _center_of(points, RIGHT_EYE_CONTOUR)
        result.mouth_center = _center_of(points, MOUTH_CONTOUR)
        result.head_tilt = _head_tilt(points)
        result.points = [(int(x), int(y)) for x, y in points]

        return result

    def close(self) -> None:
        """Release MediaPipe's native resources."""
        if getattr(self, "_mesh", None) is not None:
            self._mesh.close()
            self._mesh = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def _center_of(points: np.ndarray, indices: List[int]) -> Optional[Tuple[int, int]]:
    """Average position of a landmark group, used to place heatmap blobs."""
    try:
        selected = points[indices]
    except IndexError:
        return None
    return (int(selected[:, 0].mean()), int(selected[:, 1].mean()))


def _head_tilt(points: np.ndarray) -> float:
    """Angle of the forehead-chin axis away from vertical, in degrees.

    A drowsy driver's head drops forward or sideways, so a large sustained tilt
    is a useful secondary cue. Positive = leaning one way, negative = the other.
    """
    try:
        forehead, chin = points[FOREHEAD], points[CHIN]
    except IndexError:
        return 0.0
    dx = float(chin[0] - forehead[0])
    dy = float(chin[1] - forehead[1])
    if abs(dy) < 1e-6:
        return 0.0
    return float(np.degrees(np.arctan2(dx, dy)))
