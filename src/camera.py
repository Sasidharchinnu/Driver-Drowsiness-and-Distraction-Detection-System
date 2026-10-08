"""
Webcam capture with real error handling.

The tricky parts this module hides from the rest of the app:
  * macOS needs the AVFoundation backend, otherwise OpenCV can be very slow.
  * macOS asks for camera permission the first time. If the user says "No",
    OpenCV does not raise - it just hands back blank/failed frames forever.
    We detect that and report it as a permission problem.
  * A USB camera can be unplugged mid-drive. We detect consecutive read
    failures and surface a clean error instead of crashing.
  * Grabbing frames in a background thread keeps the UI smooth: the UI always
    reads the newest frame instead of draining a stale buffer.
"""

from __future__ import annotations

import platform
import threading
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np


class CameraError(RuntimeError):
    """Raised for any camera problem, carrying a message we can show the user."""


@dataclass
class CameraInfo:
    """What we learned about the camera after opening it."""

    index: int
    width: int
    height: int
    fps: float
    backend: str


def _preferred_backend() -> int:
    """Pick the fastest OpenCV capture backend for this operating system."""
    system = platform.system()
    if system == "Darwin":
        return cv2.CAP_AVFOUNDATION  # native macOS capture - required on Apple Silicon
    if system == "Windows":
        return cv2.CAP_DSHOW
    return cv2.CAP_V4L2  # Linux


def list_available_cameras(max_index: int = 5) -> List[int]:
    """Probe camera indices 0..max_index-1 and return the ones that actually work.

    Used by the UI to populate the camera picker. Probing is slightly slow
    (each open takes ~0.1-0.3 s), so we keep max_index small.
    """
    backend = _preferred_backend()
    found: List[int] = []
    for index in range(max_index):
        cap = cv2.VideoCapture(index, backend)
        try:
            if cap.isOpened():
                ok, frame = cap.read()
                if ok and frame is not None and frame.size > 0:
                    found.append(index)
        except cv2.error:
            pass
        finally:
            cap.release()
    return found


class WebcamStream:
    """Threaded webcam reader.

    Usage:
        cam = WebcamStream(index=0, width=640, height=480)
        cam.start()                       # raises CameraError on failure
        ok, frame = cam.read()
        cam.stop()

    Or as a context manager, which always releases the device:
        with WebcamStream(0) as cam:
            ok, frame = cam.read()
    """

    # If this many reads in a row fail, we declare the camera disconnected.
    MAX_CONSECUTIVE_FAILURES = 30

    def __init__(
        self,
        index: int = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        flip_horizontal: bool = True,
        warmup_frames: int = 5,
    ):
        self.index = index
        self.width = width
        self.height = height
        self.fps = fps
        self.flip_horizontal = flip_horizontal
        self.warmup_frames = warmup_frames

        self._capture: Optional[cv2.VideoCapture] = None
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._frame_id = 0           # increments on every new frame
        self._last_read_id = -1      # so read() can tell "new" from "repeat"
        self._running = False
        self._failure_count = 0
        self._error: Optional[str] = None
        self.info: Optional[CameraInfo] = None

    # ------------------------------------------------------------------ open
    def start(self) -> CameraInfo:
        """Open the device, verify we can actually read pixels, start the thread."""
        if self._running:
            return self.info  # already started; harmless no-op

        backend = _preferred_backend()
        self._capture = cv2.VideoCapture(self.index, backend)

        if not self._capture.isOpened():
            self._capture.release()
            self._capture = None
            raise CameraError(self._open_failure_message())

        # Ask the driver for our preferred format. These are *requests*: the
        # driver may return the nearest supported mode instead.
        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._capture.set(cv2.CAP_PROP_FPS, self.fps)
        self._capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # always give us the freshest frame

        # Warm-up: the first frames are often black while auto-exposure settles.
        first_good: Optional[np.ndarray] = None
        deadline = time.time() + 5.0  # never block the app for more than 5 seconds
        while time.time() < deadline:
            ok, frame = self._capture.read()
            if ok and frame is not None and frame.size > 0:
                first_good = frame
                if self.warmup_frames <= 0:
                    break
                self.warmup_frames -= 1
            time.sleep(0.02)

        if first_good is None:
            self._capture.release()
            self._capture = None
            raise CameraError(
                f"Camera {self.index} opened but returned no usable frames.\n\n"
                "On macOS this almost always means camera permission was denied.\n"
                "Fix: System Settings -> Privacy & Security -> Camera, and enable\n"
                "the app you launched this from (Terminal, iTerm, or your IDE).\n"
                "You must fully quit and reopen that app after granting permission."
            )

        actual_h, actual_w = first_good.shape[:2]
        self.info = CameraInfo(
            index=self.index,
            width=actual_w,
            height=actual_h,
            fps=float(self._capture.get(cv2.CAP_PROP_FPS)) or float(self.fps),
            backend=self._capture.getBackendName(),
        )

        self._latest_frame = self._prepare(first_good)
        self._frame_id = 1
        self._failure_count = 0
        self._error = None
        self._running = True

        self._thread = threading.Thread(target=self._reader_loop, name="WebcamStream", daemon=True)
        self._thread.start()
        return self.info

    def _open_failure_message(self) -> str:
        """Build a message that tells the user what to actually do."""
        available = list_available_cameras(max_index=4)
        if not available:
            return (
                f"Could not open camera index {self.index}, and no cameras were found at all.\n\n"
                "Check that:\n"
                "  1. A webcam is connected and not in use by Zoom/Teams/FaceTime.\n"
                "  2. Camera permission is granted: System Settings -> Privacy & "
                "Security -> Camera.\n"
                "  3. You restarted the terminal/IDE after granting permission."
            )
        return (
            f"Could not open camera index {self.index}.\n\n"
            f"Cameras that DO work right now: {available}.\n"
            f"Set camera.index to one of those in config/config.yaml, "
            f"or pick it from the sidebar."
        )

    # --------------------------------------------------------------- reading
    def _prepare(self, frame: np.ndarray) -> np.ndarray:
        """Mirror the frame if configured, so the driver sees themselves naturally."""
        return cv2.flip(frame, 1) if self.flip_horizontal else frame

    def _reader_loop(self) -> None:
        """Background thread: keep only the most recent frame."""
        while self._running:
            capture = self._capture
            if capture is None:
                break
            try:
                ok, frame = capture.read()
            except cv2.error as exc:
                ok, frame = False, None
                self._error = f"OpenCV error while reading camera: {exc}"

            if ok and frame is not None and frame.size > 0:
                with self._lock:
                    self._latest_frame = self._prepare(frame)
                    self._frame_id += 1
                self._failure_count = 0
            else:
                self._failure_count += 1
                if self._failure_count >= self.MAX_CONSECUTIVE_FAILURES:
                    self._error = (
                        f"Lost the camera after {self._failure_count} failed reads. "
                        "It was probably unplugged or taken over by another app."
                    )
                    self._running = False
                    break
                time.sleep(0.01)

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Return (success, frame). The frame is a copy, safe to draw on.

        Raises CameraError if the camera died while streaming.
        """
        if self._error:
            raise CameraError(self._error)
        with self._lock:
            if self._latest_frame is None:
                return False, None
            self._last_read_id = self._frame_id
            return True, self._latest_frame.copy()

    def has_new_frame(self) -> bool:
        """True if a frame arrived since the last read(). Avoids re-processing."""
        with self._lock:
            return self._frame_id != self._last_read_id

    @property
    def is_running(self) -> bool:
        return self._running and self._error is None

    # --------------------------------------------------------------- closing
    def stop(self) -> None:
        """Release the device. Safe to call more than once."""
        self._running = False
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._thread = None
        if self._capture is not None:
            self._capture.release()
            self._capture = None
        with self._lock:
            self._latest_frame = None

    # Context-manager support guarantees the camera is released even on error.
    def __enter__(self) -> "WebcamStream":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    def __del__(self):  # last-resort cleanup if someone forgets stop()
        try:
            self.stop()
        except Exception:
            pass
