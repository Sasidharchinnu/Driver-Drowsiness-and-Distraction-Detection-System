"""
The drowsiness algorithm: turn EAR / MAR over time into a 0-100 score.

The big idea
------------
A single frame tells you almost nothing. Everybody blinks; everybody opens their
mouth. Drowsiness is a *temporal* pattern, so this module is a small state
machine that remembers what happened over the last minute.

Four independent evidence channels
----------------------------------
1. PERCLOS (40%)  - the fraction of time the eyes were closed over a rolling
                    60 s window. This is the measure NHTSA studies found
                    correlates best with real driver fatigue.
2. CLOSURE (35%)  - how long the eyes are closed RIGHT NOW. This is what
                    catches a microsleep in progress, and it is the channel
                    that can raise the alarm within ~1 second.
3. YAWN (15%)     - yawns per minute. A slower, earlier warning sign.
4. BLINK (10%)    - blink rate outside the normal 8-28 per minute range.
                    Both unusually slow AND unusually fast blinking correlate
                    with fatigue.

score = 100 * (0.40*perclos + 0.35*closure + 0.15*yawn + 0.10*blink)

Each channel is normalised to 0-1 before weighting, so no single channel can
dominate. The final score is exponentially smoothed to stop it flickering
between frames.

Blink vs microsleep
-------------------
Both are "eyes closed". We tell them apart by DURATION, measured when the eyes
reopen:
    closure <= 0.40 s  -> normal blink   (counted, harmless)
    closure >= 1.20 s  -> microsleep     (dangerous, forces DROWSY immediately)
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Deque, Dict, List, Optional, Tuple


class DrowsinessState(str, Enum):
    """The three states shown on the dashboard."""

    ALERT = "ALERT"        # score below the warning band
    WARNING = "WARNING"    # early fatigue signs
    DROWSY = "DROWSY"      # alarm territory
    NO_FACE = "NO FACE"    # nobody in front of the camera


@dataclass
class DrowsinessMetrics:
    """A full snapshot of the driver's state for one frame."""

    state: DrowsinessState = DrowsinessState.NO_FACE
    score: float = 0.0                 # 0-100, smoothed
    raw_score: float = 0.0             # 0-100, unsmoothed (useful for plots)
    ear: float = 0.0
    mar: float = 0.0
    eyes_closed: bool = False
    closure_duration: float = 0.0      # seconds the eyes have been shut right now
    longest_closure: float = 0.0       # worst closure seen this session
    blink_count: int = 0
    yawn_count: int = 0
    is_yawning: bool = False
    yawn_duration: float = 0.0
    perclos: float = 0.0               # 0-1 fraction of closed time in the window
    blink_rate: float = 0.0            # blinks per minute
    yawn_rate: float = 0.0             # yawns per minute
    microsleep_count: int = 0
    head_tilt: float = 0.0
    face_detected: bool = False
    face_count: int = 0
    alarm_active: bool = False
    # How much each channel contributed, 0-1. Displayed as bars in the dashboard
    # and reused to weight the attention heatmap.
    contributions: Dict[str, float] = field(default_factory=dict)


class DrowsinessEngine:
    """Stateful analyser. Call `update()` once per frame."""

    def __init__(self, cfg):
        """`cfg` is the `drowsiness` section of config.yaml."""
        self.ear_threshold = float(cfg.ear_threshold)
        self.ear_consec_frames = int(cfg.ear_consec_frames)
        self.blink_max_duration = float(cfg.blink_max_duration)
        self.microsleep_duration = float(cfg.microsleep_duration)
        self.mar_threshold = float(cfg.mar_threshold)
        self.yawn_min_duration = float(cfg.yawn_min_duration)
        self.perclos_window = float(cfg.perclos_window)
        self.perclos_threshold = float(cfg.perclos_threshold)
        self.warning_score = float(cfg.warning_score)
        self.drowsy_score = float(cfg.drowsy_score)
        self.smoothing = float(cfg.score_smoothing)
        self.blink_rate_min = float(cfg.normal_blink_rate_min)
        self.blink_rate_max = float(cfg.normal_blink_rate_max)

        weights = cfg.weights
        self.weights = {
            "perclos": float(weights.perclos),
            "closure": float(weights.closure),
            "yawn": float(weights.yawn),
            "blink": float(weights.blink),
        }

        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self) -> None:
        """Clear all history. Called at session start and by the Reset button."""
        self.session_start = time.time()

        # (timestamp, is_closed) samples inside the rolling PERCLOS window
        self._closure_samples: Deque[Tuple[float, bool]] = deque()
        self._blink_times: Deque[float] = deque()
        self._yawn_times: Deque[float] = deque()

        self._closed_frames = 0          # consecutive frames under the EAR threshold
        self._closure_start: Optional[float] = None
        self._yawn_start: Optional[float] = None
        self._counted_current_yawn = False

        self.blink_count = 0
        self.yawn_count = 0
        self.microsleep_count = 0
        self.longest_closure = 0.0

        self._smoothed_score = 0.0
        self._last_state = DrowsinessState.NO_FACE
        self.history: List[Tuple[float, float]] = []  # (elapsed_seconds, score) for the chart

    # ----------------------------------------------------------------- update
    def update(
        self,
        ear: Optional[float],
        mar: Optional[float],
        face_detected: bool,
        face_count: int = 0,
        head_tilt: float = 0.0,
        now: Optional[float] = None,
    ) -> DrowsinessMetrics:
        """Feed one frame of measurements in, get the current assessment out."""
        now = now if now is not None else time.time()

        # ---- No face: do not accumulate evidence, but do not lose history ----
        if not face_detected or ear is None:
            # An in-progress closure is abandoned - the face left the frame, we
            # cannot claim the eyes are shut.
            self._closure_start = None
            self._closed_frames = 0
            self._yawn_start = None
            self._counted_current_yawn = False
            # Decay the score toward zero so a stale DROWSY does not stick.
            self._smoothed_score *= 0.95
            return DrowsinessMetrics(
                state=DrowsinessState.NO_FACE,
                score=self._smoothed_score,
                raw_score=0.0,
                blink_count=self.blink_count,
                yawn_count=self.yawn_count,
                microsleep_count=self.microsleep_count,
                longest_closure=self.longest_closure,
                face_detected=False,
                face_count=face_count,
                perclos=self._perclos(now),
                contributions={},
            )

        mar = mar or 0.0

        # ------------------------- EYES: closure, blinks, microsleeps --------
        eye_is_closed_now = ear < self.ear_threshold
        self._closure_samples.append((now, eye_is_closed_now))
        self._trim(self._closure_samples, now)

        closure_duration = 0.0
        if eye_is_closed_now:
            self._closed_frames += 1
            # Require N consecutive frames so one noisy landmark frame cannot
            # invent a blink.
            if self._closed_frames >= self.ear_consec_frames and self._closure_start is None:
                self._closure_start = now
            if self._closure_start is not None:
                closure_duration = now - self._closure_start
                self.longest_closure = max(self.longest_closure, closure_duration)
        else:
            # The eyes just reopened: classify the closure that ended.
            if self._closure_start is not None:
                duration = now - self._closure_start
                if duration <= self.blink_max_duration:
                    self.blink_count += 1
                    self._blink_times.append(now)
                elif duration >= self.microsleep_duration:
                    self.microsleep_count += 1
                else:
                    # Between a blink and a microsleep: a slow, heavy blink.
                    # Still counted as a blink, because it is one.
                    self.blink_count += 1
                    self._blink_times.append(now)
            self._closure_start = None
            self._closed_frames = 0

        self._trim(self._blink_times, now)

        # ------------------------------------- MOUTH: yawn detection ---------
        is_yawning = False
        yawn_duration = 0.0
        if mar > self.mar_threshold:
            if self._yawn_start is None:
                self._yawn_start = now
                self._counted_current_yawn = False
            yawn_duration = now - self._yawn_start
            # Only a mouth held open long enough counts - this rejects talking.
            if yawn_duration >= self.yawn_min_duration:
                is_yawning = True
                if not self._counted_current_yawn:
                    self.yawn_count += 1
                    self._yawn_times.append(now)
                    self._counted_current_yawn = True
        else:
            self._yawn_start = None
            self._counted_current_yawn = False

        self._trim(self._yawn_times, now)

        # ------------------------------------------ SCORE: four channels -----
        perclos = self._perclos(now)
        window_minutes = max(self._elapsed_window(now) / 60.0, 1e-6)
        blink_rate = len(self._blink_times) / window_minutes
        yawn_rate = len(self._yawn_times) / window_minutes

        contributions = {
            "perclos": self._perclos_contribution(perclos),
            "closure": self._closure_contribution(closure_duration),
            "yawn": self._yawn_contribution(yawn_rate, is_yawning),
            "blink": self._blink_contribution(blink_rate, now),
        }

        raw_score = 100.0 * sum(self.weights[k] * v for k, v in contributions.items())
        raw_score = max(0.0, min(100.0, raw_score))

        # Exponential smoothing: new = a*raw + (1-a)*old.
        # Small `a` = slow, stable needle. Large `a` = twitchy but responsive.
        self._smoothed_score = (
            self.smoothing * raw_score + (1.0 - self.smoothing) * self._smoothed_score
        )

        # ------------------------------------------------ decide the state ---
        state = self._classify(self._smoothed_score, closure_duration)

        self.history.append((now - self.session_start, self._smoothed_score))
        if len(self.history) > 3000:          # keep memory bounded (~2-3 minutes)
            self.history = self.history[-3000:]
        self._last_state = state

        return DrowsinessMetrics(
            state=state,
            score=self._smoothed_score,
            raw_score=raw_score,
            ear=ear,
            mar=mar,
            eyes_closed=eye_is_closed_now,
            closure_duration=closure_duration,
            longest_closure=self.longest_closure,
            blink_count=self.blink_count,
            yawn_count=self.yawn_count,
            is_yawning=is_yawning,
            yawn_duration=yawn_duration,
            perclos=perclos,
            blink_rate=blink_rate,
            yawn_rate=yawn_rate,
            microsleep_count=self.microsleep_count,
            head_tilt=head_tilt,
            face_detected=True,
            face_count=face_count,
            alarm_active=state == DrowsinessState.DROWSY,
            contributions=contributions,
        )

    # ------------------------------------------------------- score channels
    def _perclos_contribution(self, perclos: float) -> float:
        """PERCLOS scaled so that hitting the threshold (20%) gives a full 1.0.

        Below the threshold it rises linearly, so mild closure still registers.
        """
        return min(1.0, perclos / max(self.perclos_threshold, 1e-6))

    def _closure_contribution(self, closure_duration: float) -> float:
        """Current closure length, 0 at a normal blink, 1 at a full microsleep.

        We start counting only *after* blink_max_duration, so ordinary blinking
        never pushes this channel above zero.
        """
        if closure_duration <= self.blink_max_duration:
            return 0.0
        span = max(self.microsleep_duration - self.blink_max_duration, 1e-6)
        return min(1.0, (closure_duration - self.blink_max_duration) / span)

    def _yawn_contribution(self, yawn_rate: float, is_yawning: bool) -> float:
        """3 yawns per minute saturates the channel; yawning right now adds 0.3."""
        rate_part = min(1.0, yawn_rate / 3.0)
        return min(1.0, rate_part + (0.3 if is_yawning else 0.0))

    def _blink_contribution(self, blink_rate: float, now: float) -> float:
        """How far the blink rate sits outside the healthy 8-28/min band.

        We stay silent for the first 15 seconds: with only a few seconds of data
        the rate estimate is meaningless and would produce a false alarm.
        """
        if now - self.session_start < 15.0:
            return 0.0
        if self.blink_rate_min <= blink_rate <= self.blink_rate_max:
            return 0.0
        if blink_rate < self.blink_rate_min:
            # Too few blinks = the "staring" pattern of a fatigued driver.
            return min(1.0, (self.blink_rate_min - blink_rate) / max(self.blink_rate_min, 1e-6))
        # Too many blinks = fighting to stay awake.
        return min(1.0, (blink_rate - self.blink_rate_max) / max(self.blink_rate_max, 1e-6))

    # ------------------------------------------------------------- helpers
    def _classify(self, score: float, closure_duration: float) -> DrowsinessState:
        """Map the score to a state, with one safety override.

        Override: if the eyes have been shut longer than microsleep_duration we
        declare DROWSY immediately, no matter what the smoothed score says.
        A microsleep at 100 km/h covers ~33 m per second - we cannot wait for a
        rolling average to catch up.
        """
        if closure_duration >= self.microsleep_duration:
            return DrowsinessState.DROWSY
        if score >= self.drowsy_score:
            return DrowsinessState.DROWSY
        if score >= self.warning_score:
            return DrowsinessState.WARNING
        return DrowsinessState.ALERT

    def _perclos(self, now: float) -> float:
        """PERCLOS = closed samples / total samples inside the rolling window."""
        self._trim(self._closure_samples, now)
        if not self._closure_samples:
            return 0.0
        closed = sum(1 for _, is_closed in self._closure_samples if is_closed)
        return closed / len(self._closure_samples)

    def _elapsed_window(self, now: float) -> float:
        """Length of data we actually have, capped at the window size.

        Early in a session we have only a few seconds. Dividing counts by the
        full 60 s would understate the rate, so we divide by what we really have.
        """
        return min(now - self.session_start, self.perclos_window)

    def _trim(self, buffer: Deque, now: float) -> None:
        """Drop samples that fell out of the rolling window."""
        cutoff = now - self.perclos_window
        while buffer:
            item = buffer[0]
            timestamp = item[0] if isinstance(item, tuple) else item
            if timestamp < cutoff:
                buffer.popleft()
            else:
                break

    # --------------------------------------------------------------- summary
    def session_summary(self) -> Dict[str, float]:
        """End-of-session statistics, shown when the user stops the camera."""
        elapsed = max(time.time() - self.session_start, 1e-6)
        return {
            "duration_seconds": elapsed,
            "total_blinks": self.blink_count,
            "total_yawns": self.yawn_count,
            "microsleeps": self.microsleep_count,
            "longest_closure": self.longest_closure,
            "avg_blink_rate": self.blink_count / (elapsed / 60.0),
            "peak_score": max((s for _, s in self.history), default=0.0),
        }
