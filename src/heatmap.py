"""
Attention heatmap generation.

What this shows
---------------
A colour overlay whose brightness says "this is the region currently driving the
drowsiness score". It is computed from the live measurements, not painted on:

  * Each eye gets a Gaussian blob whose height grows as EAR falls.
    Wide-open eyes -> nearly invisible. Closed eyes -> hot red.
  * The mouth gets a blob whose height grows with MAR, so a yawn lights up.
  * The whole face gets a wide, low blob proportional to the overall score, so
    a generally drowsy driver glows even between events.

How the maths works
-------------------
A 2D Gaussian centred at (cx, cy):

    G(x, y) = A * exp( -( (x-cx)^2 / (2*sx^2) + (y-cy)^2 / (2*sy^2) ) )

  A  = amplitude  -> how intense (driven by the measurements)
  sx = sigma_x    -> how wide (driven by the size of the detected region)

We add the blobs together, normalise to 0-255, apply an OpenCV colormap, and
alpha-blend onto the frame. A temporal decay term keeps the map from flickering:

    map_now = max(new_map, previous_map * decay)

Performance note
----------------
Building a Gaussian over a full 640x480 frame every frame is wasteful. We only
evaluate it inside a small window around the centre (3 sigma covers >99% of a
Gaussian), which makes the whole thing cost well under a millisecond.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple

import cv2
import numpy as np

# Friendly names -> OpenCV constants, so config.yaml can say "JET".
COLORMAPS = {
    "JET": cv2.COLORMAP_JET,
    "TURBO": cv2.COLORMAP_TURBO,
    "HOT": cv2.COLORMAP_HOT,
    "INFERNO": cv2.COLORMAP_INFERNO,
    "MAGMA": cv2.COLORMAP_MAGMA,
    "PLASMA": cv2.COLORMAP_PLASMA,
    "VIRIDIS": cv2.COLORMAP_VIRIDIS,
}


class AttentionHeatmap:
    """Builds and blends the attention overlay. One instance per session."""

    def __init__(self, cfg):
        """`cfg` is the `heatmap` section of config.yaml."""
        self.enabled = bool(cfg.enabled)
        self.alpha = float(cfg.alpha)
        self.colormap = COLORMAPS.get(str(cfg.colormap).upper(), cv2.COLORMAP_JET)
        self.eye_sigma_scale = float(cfg.eye_sigma_scale)
        self.mouth_sigma_scale = float(cfg.mouth_sigma_scale)
        self.face_sigma_scale = float(cfg.face_sigma_scale)
        self.decay = float(cfg.decay)

        self._accumulator: Optional[np.ndarray] = None  # previous frame's map

    def reset(self) -> None:
        """Forget the temporal history (used by the dashboard Reset button)."""
        self._accumulator = None

    # ------------------------------------------------------------ the maths
    @staticmethod
    def _add_gaussian(
        canvas: np.ndarray,
        center: Tuple[int, int],
        sigma_x: float,
        sigma_y: float,
        amplitude: float,
    ) -> None:
        """Add one 2D Gaussian blob to `canvas`, in place.

        We evaluate it only within +/-3 sigma of the centre, because beyond that
        a Gaussian contributes less than 1% - a big speed win.
        """
        if amplitude <= 0.0:
            return

        height, width = canvas.shape
        cx, cy = int(center[0]), int(center[1])
        sigma_x = max(float(sigma_x), 1.0)
        sigma_y = max(float(sigma_y), 1.0)

        # The window we actually compute, clipped to the canvas.
        half_w = int(3 * sigma_x)
        half_h = int(3 * sigma_y)
        x1, x2 = max(0, cx - half_w), min(width, cx + half_w + 1)
        y1, y2 = max(0, cy - half_h), min(height, cy + half_h + 1)
        if x2 <= x1 or y2 <= y1:
            return  # the blob is entirely off-screen

        # Build coordinate grids relative to the centre, then apply the formula.
        xs = np.arange(x1, x2, dtype=np.float32) - cx
        ys = np.arange(y1, y2, dtype=np.float32) - cy
        gauss_x = np.exp(-(xs ** 2) / (2.0 * sigma_x ** 2))
        gauss_y = np.exp(-(ys ** 2) / (2.0 * sigma_y ** 2))
        # Outer product gives the full 2D blob in one vectorised operation.
        blob = amplitude * np.outer(gauss_y, gauss_x)

        canvas[y1:y2, x1:x2] += blob

    # ----------------------------------------------------------- build a map
    def build(
        self,
        frame_shape: Tuple[int, int],
        eye_boxes: Sequence[Optional[Tuple[int, int, int, int]]] = (),
        eye_intensity: float = 0.0,
        mouth_box: Optional[Tuple[int, int, int, int]] = None,
        mouth_intensity: float = 0.0,
        face_box: Optional[Tuple[int, int, int, int]] = None,
        face_intensity: float = 0.0,
    ) -> np.ndarray:
        """Return a float32 attention map in 0-1, the same size as the frame."""
        height, width = frame_shape[:2]
        canvas = np.zeros((height, width), dtype=np.float32)

        # --- broad face-level glow (lowest priority, drawn widest) ---
        if face_box is not None and face_intensity > 0:
            fx1, fy1, fx2, fy2 = face_box
            self._add_gaussian(
                canvas,
                center=((fx1 + fx2) // 2, (fy1 + fy2) // 2),
                sigma_x=(fx2 - fx1) * self.face_sigma_scale * 0.5,
                sigma_y=(fy2 - fy1) * self.face_sigma_scale * 0.5,
                amplitude=face_intensity * 0.5,  # kept dim so eyes stay dominant
            )

        # --- the eyes: the most important regions ---
        for box in eye_boxes:
            if box is None or eye_intensity <= 0:
                continue
            ex1, ey1, ex2, ey2 = box
            self._add_gaussian(
                canvas,
                center=((ex1 + ex2) // 2, (ey1 + ey2) // 2),
                sigma_x=max((ex2 - ex1) * self.eye_sigma_scale, 6.0),
                sigma_y=max((ey2 - ey1) * self.eye_sigma_scale * 1.6, 6.0),
                amplitude=eye_intensity,
            )

        # --- the mouth: lights up during a yawn ---
        if mouth_box is not None and mouth_intensity > 0:
            mx1, my1, mx2, my2 = mouth_box
            self._add_gaussian(
                canvas,
                center=((mx1 + mx2) // 2, (my1 + my2) // 2),
                sigma_x=max((mx2 - mx1) * self.mouth_sigma_scale, 8.0),
                sigma_y=max((my2 - my1) * self.mouth_sigma_scale, 8.0),
                amplitude=mouth_intensity,
            )

        # --- temporal smoothing: fade the old map, keep whichever is hotter ---
        if self._accumulator is not None and self._accumulator.shape == canvas.shape:
            canvas = np.maximum(canvas, self._accumulator * self.decay)
        self._accumulator = canvas

        return np.clip(canvas, 0.0, 1.0)

    # --------------------------------------------------------------- render
    def colorize(self, attention: np.ndarray) -> np.ndarray:
        """Turn a 0-1 attention map into a BGR colour image.

        Speed trick: we blur and colourise at HALF resolution, then scale back
        up. A heatmap is smooth by nature, so the upscale is invisible, but it
        makes this step roughly 4x cheaper - which matters at 30 FPS.
        """
        height, width = attention.shape[:2]
        small = cv2.resize(attention, (width // 2, height // 2), interpolation=cv2.INTER_AREA)
        as_bytes = np.clip(small * 255.0, 0, 255).astype(np.uint8)
        # Blur so the blobs merge into smooth regions instead of hard discs.
        as_bytes = cv2.GaussianBlur(as_bytes, (0, 0), sigmaX=2.5, sigmaY=2.5)
        colored = cv2.applyColorMap(as_bytes, self.colormap)
        return cv2.resize(colored, (width, height), interpolation=cv2.INTER_LINEAR)

    def overlay(self, frame: np.ndarray, attention: np.ndarray) -> np.ndarray:
        """Blend the heatmap onto the frame, weighted by local intensity.

        We do NOT blend uniformly: the blend weight at each pixel is
        `alpha * attention`, so cold areas stay perfectly sharp and only hot
        areas get tinted. A flat blend would wash out the whole image.
        """
        if not self.enabled or attention is None:
            return frame

        # Nothing is hot enough to be visible - skip all the work.
        if float(attention.max()) < 0.02:
            return frame

        colored = self.colorize(attention)
        # Per-pixel blend weight, broadcast from HxW to HxWx3.
        # Everything stays float32 so numpy never promotes to slower float64.
        weight = (attention * self.alpha).astype(np.float32)[:, :, np.newaxis]
        blended = frame.astype(np.float32) * (1.0 - weight) + colored.astype(np.float32) * weight
        return blended.astype(np.uint8)

    def standalone_view(
        self, attention: np.ndarray, frame: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """The heatmap on its own, for the separate dashboard panel.

        If a frame is supplied we keep a faint greyscale version underneath so
        you can still tell where the face is.
        """
        colored = self.colorize(attention)
        if frame is None:
            return colored
        grey = cv2.cvtColor(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        return cv2.addWeighted(grey, 0.35, colored, 0.65, 0)


def intensity_from_ear(ear: float, ear_threshold: float) -> float:
    """Map EAR to a 0-1 'how closed is this eye' intensity.

    ear >= 1.6*threshold (wide open)  -> 0.15  (a faint always-on marker)
    ear == threshold     (closing)    -> ~0.6
    ear <= 0.5*threshold (shut)       -> 1.0
    """
    if ear <= 0:
        return 0.0
    open_reference = ear_threshold * 1.6
    closed_reference = ear_threshold * 0.5
    if ear >= open_reference:
        return 0.15
    if ear <= closed_reference:
        return 1.0
    # Linear ramp between the two reference points, inverted (lower EAR = hotter).
    ratio = (open_reference - ear) / max(open_reference - closed_reference, 1e-6)
    return float(np.clip(0.15 + ratio * 0.85, 0.0, 1.0))


def intensity_from_mar(mar: float, mar_threshold: float) -> float:
    """Map MAR to a 0-1 'how wide is this mouth' intensity."""
    if mar <= 0:
        return 0.0
    if mar >= mar_threshold * 1.5:
        return 1.0
    return float(np.clip(mar / max(mar_threshold * 1.5, 1e-6), 0.0, 1.0))
