"""
All the drawing: YOLO boxes, labels, confidence scores and the on-frame HUD.

Everything here is pure presentation - it reads the metrics and paints pixels.
No detection logic lives in this file, which keeps the algorithm easy to follow.

Colour convention (OpenCV uses B, G, R order, not R, G, B):
    green  = alert / healthy
    amber  = warning
    red    = drowsy / danger
    cyan   = eye regions
    yellow = mouth region
    grey   = passengers (faces that are not the driver)
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

from .detector import Detection
from .drowsiness import DrowsinessMetrics, DrowsinessState

# --- palette (B, G, R) -------------------------------------------------------
GREEN = (80, 220, 100)
AMBER = (0, 190, 255)
RED = (60, 60, 255)
CYAN = (255, 220, 80)
YELLOW = (60, 230, 250)
GREY = (150, 150, 150)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
PANEL = (35, 32, 30)

FONT = cv2.FONT_HERSHEY_SIMPLEX

STATE_COLORS = {
    DrowsinessState.ALERT: GREEN,
    DrowsinessState.WARNING: AMBER,
    DrowsinessState.DROWSY: RED,
    DrowsinessState.NO_FACE: GREY,
}


def state_color(state: DrowsinessState) -> Tuple[int, int, int]:
    return STATE_COLORS.get(state, GREY)


# -----------------------------------------------------------------------------
# Primitives
# -----------------------------------------------------------------------------
def draw_label(
    frame: np.ndarray,
    text: str,
    origin: Tuple[int, int],
    color: Tuple[int, int, int],
    scale: float = 0.5,
    thickness: int = 1,
    above: bool = True,
) -> None:
    """Draw text on a filled background chip so it stays readable on any image."""
    x, y = origin
    (tw, th), baseline = cv2.getTextSize(text, FONT, scale, thickness)
    pad = 4

    if above:
        # Put the chip above the point, unless that would go off the top edge.
        top = y - th - baseline - pad * 2
        if top < 0:
            top = y
        bottom = top + th + baseline + pad * 2
    else:
        top, bottom = y, y + th + baseline + pad * 2

    right = min(x + tw + pad * 2, frame.shape[1])
    cv2.rectangle(frame, (x, top), (right, bottom), color, cv2.FILLED)
    cv2.putText(
        frame, text, (x + pad, bottom - baseline - pad), FONT, scale, BLACK, thickness, cv2.LINE_AA
    )


def draw_corner_box(
    frame: np.ndarray,
    box: Tuple[int, int, int, int],
    color: Tuple[int, int, int],
    thickness: int = 2,
    corner_ratio: float = 0.22,
) -> None:
    """A bracket-style box (corners only).

    Used for the face because it keeps the driver's features visible instead of
    boxing them in with a solid rectangle.
    """
    x1, y1, x2, y2 = box
    corner_w = max(8, int((x2 - x1) * corner_ratio))
    corner_h = max(8, int((y2 - y1) * corner_ratio))

    for (cx, cy, dx, dy) in (
        (x1, y1, 1, 1),    # top-left
        (x2, y1, -1, 1),   # top-right
        (x1, y2, 1, -1),   # bottom-left
        (x2, y2, -1, -1),  # bottom-right
    ):
        cv2.line(frame, (cx, cy), (cx + dx * corner_w, cy), color, thickness, cv2.LINE_AA)
        cv2.line(frame, (cx, cy), (cx, cy + dy * corner_h), color, thickness, cv2.LINE_AA)


def draw_progress_bar(
    frame: np.ndarray,
    top_left: Tuple[int, int],
    size: Tuple[int, int],
    fraction: float,
    color: Tuple[int, int, int],
    background: Tuple[int, int, int] = (70, 70, 70),
) -> None:
    """A horizontal filled bar, used for the score and the channel breakdown."""
    x, y = top_left
    width, height = size
    fraction = float(np.clip(fraction, 0.0, 1.0))
    cv2.rectangle(frame, (x, y), (x + width, y + height), background, cv2.FILLED)
    if fraction > 0:
        cv2.rectangle(frame, (x, y), (x + int(width * fraction), y + height), color, cv2.FILLED)
    cv2.rectangle(frame, (x, y), (x + width, y + height), (110, 110, 110), 1)


# -----------------------------------------------------------------------------
# Detection drawing
# -----------------------------------------------------------------------------
def draw_detections(
    frame: np.ndarray,
    faces: Sequence[Detection],
    primary_face: Optional[Detection],
    metrics: DrowsinessMetrics,
    left_eye_box: Optional[Tuple[int, int, int, int]] = None,
    right_eye_box: Optional[Tuple[int, int, int, int]] = None,
    mouth_box: Optional[Tuple[int, int, int, int]] = None,
    show_labels: bool = True,
    show_confidence: bool = True,
    left_ear: Optional[float] = None,
    right_ear: Optional[float] = None,
) -> np.ndarray:
    """Draw every detected region with its label and confidence."""
    color = state_color(metrics.state)

    # --- faces (YOLO) ---
    for face in faces:
        is_driver = primary_face is not None and face is primary_face
        box_color = color if is_driver else GREY
        draw_corner_box(frame, face.box, box_color, thickness=2 if is_driver else 1)
        if show_labels:
            name = "DRIVER" if is_driver else "passenger"
            text = f"{name} {face.confidence:.2f}" if show_confidence else name
            draw_label(frame, text, (face.x1, face.y1), box_color, scale=0.5)

    # --- eyes ---
    # Labels are kept short: two eye boxes sit only a few centimetres apart, so
    # long text would overlap into an unreadable mess.
    eye_color = RED if metrics.eyes_closed else CYAN
    eye_state = "shut" if metrics.eyes_closed else "open"
    for box, side, side_ear in (
        (left_eye_box, "L", left_ear),
        (right_eye_box, "R", right_ear),
    ):
        if box is None:
            continue
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), eye_color, 1, cv2.LINE_AA)
        if show_labels:
            text = f"{side} {eye_state}"
            if show_confidence:
                # For a landmark-derived box, the number that matters is that
                # eye's OWN EAR - it is the actual evidence for open vs shut.
                value = side_ear if side_ear is not None else metrics.ear
                text += f" {value:.2f}"
            draw_label(frame, text, (box[0], box[1]), eye_color, scale=0.36)

    # --- mouth ---
    if mouth_box is not None:
        mouth_color = RED if metrics.is_yawning else YELLOW
        cv2.rectangle(
            frame, (mouth_box[0], mouth_box[1]), (mouth_box[2], mouth_box[3]),
            mouth_color, 1, cv2.LINE_AA,
        )
        if show_labels:
            text = "YAWN" if metrics.is_yawning else "mouth"
            if show_confidence:
                text += f" {metrics.mar:.2f}"
            draw_label(frame, text, (mouth_box[0], mouth_box[3]), mouth_color, scale=0.38, above=False)

    return frame


def draw_landmarks(frame: np.ndarray, points: Sequence[Tuple[int, int]], step: int = 3) -> None:
    """Scatter the FaceMesh points. `step` thins them so the face stays visible."""
    for point in points[::step]:
        cv2.circle(frame, point, 1, (0, 255, 0), -1)


# -----------------------------------------------------------------------------
# HUD
# -----------------------------------------------------------------------------
def draw_hud(frame: np.ndarray, metrics: DrowsinessMetrics, fps: float) -> np.ndarray:
    """The translucent stats panel in the top-left plus the status banner."""
    color = state_color(metrics.state)
    height, width = frame.shape[:2]

    # --- translucent panel, drawn by blending a filled rect over a copy ---
    panel_w, panel_h = 232, 150
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (10 + panel_w, 10 + panel_h), PANEL, cv2.FILLED)
    cv2.addWeighted(overlay, 0.62, frame, 0.38, 0, frame)
    cv2.rectangle(frame, (10, 10), (10 + panel_w, 10 + panel_h), color, 1)

    # --- status line ---
    cv2.putText(frame, metrics.state.value, (20, 36), FONT, 0.72, color, 2, cv2.LINE_AA)
    cv2.putText(frame, f"{metrics.score:5.1f}", (150, 36), FONT, 0.6, WHITE, 1, cv2.LINE_AA)

    # --- score bar ---
    draw_progress_bar(frame, (20, 44), (panel_w - 20, 8), metrics.score / 100.0, color)

    # --- numbers ---
    rows = [
        f"EAR {metrics.ear:.3f}   MAR {metrics.mar:.3f}",
        f"Closed {metrics.closure_duration:4.1f}s  PERCLOS {metrics.perclos * 100:4.1f}%",
        f"Blinks {metrics.blink_count:<4d} Yawns {metrics.yawn_count}",
        f"Micro-sleeps {metrics.microsleep_count:<3d} FPS {fps:4.1f}",
    ]
    for index, row in enumerate(rows):
        cv2.putText(frame, row, (20, 74 + index * 19), FONT, 0.42, WHITE, 1, cv2.LINE_AA)

    # --- full-width banner when the alarm is live ---
    if metrics.state == DrowsinessState.DROWSY:
        cv2.rectangle(frame, (0, 0), (width - 1, height - 1), RED, 6)
        banner = "!! WAKE UP - DROWSINESS DETECTED !!"
        (tw, th), _ = cv2.getTextSize(banner, FONT, 0.78, 2)
        bx = (width - tw) // 2
        cv2.rectangle(frame, (bx - 12, height - 58), (bx + tw + 12, height - 18), RED, cv2.FILLED)
        cv2.putText(frame, banner, (bx, height - 30), FONT, 0.78, WHITE, 2, cv2.LINE_AA)

    elif metrics.state == DrowsinessState.NO_FACE:
        message = "No driver detected - please face the camera"
        (tw, _), _ = cv2.getTextSize(message, FONT, 0.6, 2)
        cv2.putText(
            frame, message, ((width - tw) // 2, height - 30), FONT, 0.6, AMBER, 2, cv2.LINE_AA
        )

    elif metrics.face_count > 1:
        cv2.putText(
            frame, f"{metrics.face_count} faces - scoring the largest (driver)",
            (14, height - 16), FONT, 0.45, AMBER, 1, cv2.LINE_AA,
        )

    return frame


def draw_error_frame(width: int, height: int, message: str) -> np.ndarray:
    """A black placeholder frame carrying an error message.

    Used so the dashboard always has something to display, even when the camera
    has failed - an empty video panel looks like a crash to the user.
    """
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:] = (28, 26, 24)
    cv2.putText(frame, "CAMERA UNAVAILABLE", (24, 48), FONT, 0.8, RED, 2, cv2.LINE_AA)

    # Wrap the message so long text does not run off the edge.
    words, line, y = message.split(), "", 92
    for word in words:
        candidate = f"{line} {word}".strip()
        if cv2.getTextSize(candidate, FONT, 0.45, 1)[0][0] > width - 48:
            cv2.putText(frame, line, (24, y), FONT, 0.45, WHITE, 1, cv2.LINE_AA)
            y += 22
            line = word
        else:
            line = candidate
    if line:
        cv2.putText(frame, line, (24, y), FONT, 0.45, WHITE, 1, cv2.LINE_AA)
    return frame
