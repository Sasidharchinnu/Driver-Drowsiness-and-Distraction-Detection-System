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
                    60 s window. Only closures LONGER than a normal blink are
                    counted, so ordinary blinking never raises it.
2. CLOSURE (35%)  - how long the eyes are closed RIGHT NOW. This is what
                    catches a microsleep in progress, and it is the channel
                    that can raise the alarm within ~1 second.
3. YAWN (15%)     - yawns in the last minute. A slower, earlier warning sign.
4. BLINK (10%)    - blink rate outside the normal 8-28 per minute range.
                    Both unusually slow AND unusually fast blinking correlate
                    with fatigue.

score = 100 * (0.40*perclos + 0.35*closure + 0.15*yawn + 0.10*blink)

Each channel is normalised to 0-1 before weighting, so no single channel can
dominate. The final score is exponentially smoothed to stop it flickering
between frames, and the state bands use hysteresis so the label does not
bounce between ALERT and WARNING when the score sits on a boundary.

Making blinks harmless
----------------------
* Personal calibration: everyone's open-eye EAR is different (0.22 for some
  people, 0.35 for others), so a single fixed threshold misfires. The engine
  tracks this driver's open-eye EAR over the last 30 s and sets "closed" at a
  fraction of it. The configured ear_threshold is used until enough data
  has been collected, and as the fallback when calibration is switched off.
* Hysteresis: an eye counts as closed below the threshold, but must rise a
  little ABOVE it to count as open again, so landmark jitter cannot invent
  extra blinks.
* Rates and PERCLOS are measured over at least 30 s, so two blinks in the
  first few seconds of a session cannot look like a fatigue pattern.

Blink vs microsleep
-------------------
Both are "eyes closed". We tell them apart by DURATION:
    closure <= 0.40 s  -> normal blink   (counted, harmless)
    closure >= 1.20 s  -> microsleep     (dangerous, forces DROWSY immediately)

Yawns
-----
A yawn is a strong, visible fatigue sign, so it escalates the state directly
instead of only adding its 15% share to the score:
    yawn in progress (mouth wide > yawn_min_duration)  -> at least WARNING
    mouth held wide open for yawn_alarm_duration (2.5 s) -> DROWSY + alarm
    3 or more yawns within the last minute             -> at least WARNING
The score is raised to match, so the gauge and the state always agree.

A yawn starts when MAR rises above mar_threshold and ends only when it falls
clearly below it (hysteresis) for longer than a short grace period. Brief
dips, or FaceMesh losing the face for a moment while the head tilts back, do
not cut the yawn short. Eyes naturally squeeze shut during a yawn, so eye
closure is not scored while a yawn is in progress.

Head falling (nodding off)
--------------------------
When sleep wins, the neck muscles relax and the head drops - usually forward
onto the chest, sometimes sideways or back. FaceMesh gives us the head's pitch
(forward / back) and roll (sideways) every frame.

* Personal posture: every camera sits at a different angle, so "upright" is
  learned as the median pose over the last 30 s while the head is steady. The
  head counts as falling when it leaves that posture by more than
  head_drop_angle (forward), head_back_angle (back) or head_side_angle (sideways).
* head down, any time                       -> at least WARNING
* head down for head_drop_duration (1.5 s)  -> DROWSY + alarm
* A falling head often takes the face out of view. If the face vanishes
  while the head was already dropping, that is treated as the head still being
  down (for up to head_lost_hold seconds) instead of "no face", so the alarm
  still sounds.
* Tilting the head back is ignored while yawning - that is part of a yawn.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np


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
    head_tilt: float = 0.0             # roll, degrees, relative to the learned upright pose
    head_pitch: float = 0.0            # degrees, + = forward, relative to the upright pose
    head_down: bool = False            # the head is dropped right now
    head_down_duration: float = 0.0    # seconds it has been down
    head_direction: str = ""           # "forward", "backward" or "sideways"
    head_nod_count: int = 0            # head drops this session
    head_calibrated: bool = False      # True once the upright pose has been learned
    face_detected: bool = False
    face_count: int = 0
    alarm_active: bool = False
    # True when DROWSY comes from an event that is already time-qualified (a
    # microsleep or a long yawn), so the alarm may sound without extra delay.
    urgent: bool = False
    ear_threshold: float = 0.0         # the threshold actually used this frame
    calibrated: bool = False           # True once the personal threshold is active
    # How much each channel contributed, 0-1. Displayed as bars in the dashboard
    # and reused to weight the attention heatmap.
    contributions: Dict[str, float] = field(default_factory=dict)


class DrowsinessEngine:
    """Stateful analyser. Call `update()` once per frame."""

    # Rates and PERCLOS are never computed over less data than this, so the
    # first seconds of a session cannot produce inflated numbers.
    MIN_RATE_WINDOW = 30.0
    # Blink rate is only judged once this much data exists.
    BLINK_RATE_WARMUP = 30.0
    # Personal EAR calibration.
    CALIBRATION_WINDOW = 30.0          # s of open-eye samples kept
    CALIBRATION_MIN_SAMPLES = 45       # ~1.5 s at 30 FPS before it switches on
    CALIBRATION_PERCENTILE = 80        # "typical open eye" = 80th percentile
    EYE_REOPEN_MARGIN = 0.02           # hysteresis: reopen above threshold + this
    # Yawn hysteresis: once started, the yawn lasts until MAR drops below this
    # fraction of the threshold for longer than the grace period.
    YAWN_RELEASE_RATIO = 0.80
    YAWN_GRACE = 0.40                  # s
    # MAR is smoothed before yawn detection: talking makes MAR swing up and
    # down several times a second and averages out low, a yawn stays high.
    MAR_SMOOTHING = 0.15
    # State hysteresis: leave a band only when the score is this far below it.
    STATE_HYSTERESIS = 6.0
    # Head pose calibration: median of steady-head samples over this window.
    POSE_WINDOW = 30.0                 # s
    POSE_MIN_SAMPLES = 30              # ~1 s at 30 FPS before pitch is judged
    # Once down, the head counts as up again only below this fraction of the
    # threshold, and only after this grace period (noisy frames).
    HEAD_RELEASE_RATIO = 0.80
    HEAD_GRACE = 0.30                  # s
    # A head drop this long counts as a nod in the session statistics.
    HEAD_NOD_MIN = 0.50                # s
    # If the face vanishes within this long after the head started falling,
    # the head is assumed to still be down.
    HEAD_FALL_MEMORY = 0.80            # s

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
        self.yawn_alarm_duration = float(cfg.get("yawn_alarm_duration", 2.5))
        self.frequent_yawns = int(cfg.get("frequent_yawns", 3))
        self.auto_calibrate = bool(cfg.get("auto_calibrate", True))
        self.ear_closed_ratio = float(cfg.get("ear_closed_ratio", 0.72))
        self.head_drop_angle = float(cfg.get("head_drop_angle", 20.0))
        self.head_back_angle = float(cfg.get("head_back_angle", 30.0))
        self.head_side_angle = float(cfg.get("head_side_angle", 30.0))
        self.head_drop_duration = float(cfg.get("head_drop_duration", 1.5))
        self.head_lost_hold = float(cfg.get("head_lost_hold", 8.0))

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

        # Closures longer than a blink, as (start, end) intervals, for PERCLOS.
        self._long_closures: Deque[Tuple[float, float]] = deque()
        self._blink_times: Deque[float] = deque()
        self._yawn_times: Deque[float] = deque()
        self._open_ear_samples: Deque[Tuple[float, float]] = deque()

        self._eye_closed = False         # hysteresis state of the eyes
        self._closed_frames = 0          # consecutive frames under the EAR threshold
        self._closure_start: Optional[float] = None

        self._yawn_start: Optional[float] = None
        self._mouth_last_open = 0.0
        self._smoothed_mar = 0.0
        self._raw_open_since: Optional[float] = None   # when the mouth really opened
        self._raw_last_high = 0.0
        self._counted_current_yawn = False

        self._pose_samples: Deque[Tuple[float, float, float]] = deque()  # (t, pitch, roll)
        self._head_down_start: Optional[float] = None
        self._head_last_down = 0.0
        self._head_direction = ""
        self._head_falling_at = float("-inf")   # last time the head was half-way down
        self._last_face_time = float("-inf")
        self._counted_current_nod = False

        self.blink_count = 0
        self.yawn_count = 0
        self.microsleep_count = 0
        self.head_nod_count = 0
        self.longest_closure = 0.0

        self._smoothed_score = 0.0
        self._last_state = DrowsinessState.ALERT
        self.history: List[Tuple[float, float]] = []  # (elapsed_seconds, score) for the chart

    @property
    def recent_yawns(self) -> int:
        """Yawns inside the rolling window (the last minute)."""
        return len(self._yawn_times)

    # ------------------------------------------------------- calibration
    @property
    def calibrated(self) -> bool:
        return self.auto_calibrate and len(self._open_ear_samples) >= self.CALIBRATION_MIN_SAMPLES

    @property
    def active_ear_threshold(self) -> float:
        """The EAR below which an eye counts as closed for THIS driver."""
        if not self.calibrated:
            return self.ear_threshold
        baseline = float(np.percentile([e for _, e in self._open_ear_samples],
                                       self.CALIBRATION_PERCENTILE))
        return float(np.clip(baseline * self.ear_closed_ratio, 0.12, 0.30))

    def _record_open_ear(self, ear: float, now: float) -> None:
        self._open_ear_samples.append((now, ear))
        cutoff = now - self.CALIBRATION_WINDOW
        while self._open_ear_samples and self._open_ear_samples[0][0] < cutoff:
            self._open_ear_samples.popleft()

    # ----------------------------------------------------------------- update
    def update(
        self,
        ear: Optional[float],
        mar: Optional[float],
        face_detected: bool,
        face_count: int = 0,
        head_tilt: float = 0.0,
        now: Optional[float] = None,
        head_pitch: Optional[float] = None,
    ) -> DrowsinessMetrics:
        """Feed one frame of measurements in, get the current assessment out."""
        now = now if now is not None else time.time()
        threshold = self.active_ear_threshold

        # ---- No face: do not accumulate evidence, but do not lose history ----
        if not face_detected or ear is None:
            # An in-progress closure is abandoned - the face left the frame, we
            # cannot claim the eyes are shut.
            self._closure_start = None
            self._closed_frames = 0
            self._eye_closed = False
            # A yawn survives a brief loss (head tilted back); a long one ends it.
            if self._yawn_start is not None and now - self._mouth_last_open > self.YAWN_GRACE * 2:
                self._end_yawn()

            # The face may have left the frame BECAUSE the head fell.
            lost_head = self._head_lost_while_falling(now)
            if lost_head is not None:
                return lost_head
            # Decay the score toward zero so a stale DROWSY does not stick.
            self._smoothed_score *= 0.95
            self._last_state = DrowsinessState.ALERT
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
                ear_threshold=threshold,
                calibrated=self.calibrated,
                contributions={},
            )

        mar = mar or 0.0
        self._last_face_time = now

        # ------------------------------------- MOUTH: yawn detection ---------
        # Checked first, because a yawn changes how we read the eyes.
        is_yawning, yawn_duration = self._update_yawn(mar, now)
        mouth_wide = self._yawn_start is not None

        # ------------------------- EYES: closure, blinks, microsleeps --------
        if self._eye_closed:
            raw_closed = ear < threshold + self.EYE_REOPEN_MARGIN
        else:
            raw_closed = ear < threshold
        self._eye_closed = raw_closed

        # Eyes squeeze shut during a yawn - that is the yawn, not sleep.
        eye_is_closed_now = raw_closed and not mouth_wide

        if not raw_closed and not mouth_wide:
            self._record_open_ear(ear, now)

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
        elif raw_closed:
            # Eyes shut because of a yawn: drop the closure without scoring it.
            self._closure_start = None
            self._closed_frames = 0
        else:
            self._finish_closure(now)

        # ------------------------------------------- HEAD: falling / nodding --
        head_down, head_duration, rel_pitch, rel_roll = self._update_head(
            head_pitch, head_tilt, now, yawning=mouth_wide)

        self._trim_times(self._blink_times, now)
        self._trim_times(self._yawn_times, now)
        self._trim_intervals(now)

        # ------------------------------------------ SCORE: four channels -----
        perclos = self._perclos(now, ongoing=closure_duration)
        rate_minutes = self._rate_window(now) / 60.0
        blink_rate = len(self._blink_times) / rate_minutes
        yawn_rate = len(self._yawn_times) / rate_minutes

        contributions = {
            "perclos": self._perclos_contribution(perclos),
            "closure": self._closure_contribution(closure_duration),
            "yawn": self._yawn_contribution(len(self._yawn_times), is_yawning),
            "blink": self._blink_contribution(blink_rate, now),
        }

        raw_score = 100.0 * sum(self.weights[k] * v for k, v in contributions.items())
        raw_score = max(raw_score, self._yawn_score_floor(is_yawning, yawn_duration))
        raw_score = max(raw_score, self._head_score_floor(head_down, head_duration))
        raw_score = max(0.0, min(100.0, raw_score))

        # Exponential smoothing: new = a*raw + (1-a)*old.
        # Small `a` = slow, stable needle. Large `a` = twitchy but responsive.
        self._smoothed_score = (
            self.smoothing * raw_score + (1.0 - self.smoothing) * self._smoothed_score
        )

        # ------------------------------------------------ decide the state ---
        long_yawn = is_yawning and yawn_duration >= self.yawn_alarm_duration
        head_fell = head_down and head_duration >= self.head_drop_duration
        urgent = closure_duration >= self.microsleep_duration or long_yawn or head_fell
        state = DrowsinessState.DROWSY if urgent else self._classify(
            self._smoothed_score, closure_duration)

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
            eyes_closed=raw_closed,
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
            head_tilt=rel_roll,
            head_pitch=rel_pitch,
            head_down=head_down,
            head_down_duration=head_duration,
            head_direction=self._head_direction if head_down else "",
            head_nod_count=self.head_nod_count,
            head_calibrated=self.head_calibrated,
            face_detected=True,
            face_count=face_count,
            alarm_active=state == DrowsinessState.DROWSY,
            urgent=urgent,
            ear_threshold=threshold,
            calibrated=self.calibrated,
            contributions=contributions,
        )

    # ------------------------------------------------------- eyes & mouth
    def _finish_closure(self, now: float) -> None:
        """The eyes just reopened: classify the closure that ended."""
        if self._closure_start is not None:
            duration = now - self._closure_start
            if duration >= self.microsleep_duration:
                self.microsleep_count += 1
            else:
                # Normal blinks and slow, heavy blinks both count as blinks.
                self.blink_count += 1
                self._blink_times.append(now)
            if duration > self.blink_max_duration:
                self._long_closures.append((self._closure_start, now))
        self._closure_start = None
        self._closed_frames = 0

    def _update_yawn(self, mar: float, now: float) -> Tuple[bool, float]:
        """Hysteresis yawn tracker. Returns (is_yawning, yawn_duration).

        Detection uses the smoothed MAR (rejects talking), but the duration is
        timed from the moment the raw MAR first went high, so the smoothing
        lag does not delay the alarm.
        """
        if mar > self.mar_threshold:
            if self._raw_open_since is None:
                self._raw_open_since = now
            self._raw_last_high = now
        elif (self._raw_open_since is not None and self._yawn_start is None
              and now - self._raw_last_high > self.YAWN_GRACE):
            self._raw_open_since = None

        self._smoothed_mar += self.MAR_SMOOTHING * (mar - self._smoothed_mar)
        mar = self._smoothed_mar
        if self._yawn_start is None:
            if mar > self.mar_threshold:
                self._yawn_start = self._raw_open_since or now
                self._mouth_last_open = now
                self._counted_current_yawn = False
        else:
            if mar > self.mar_threshold * self.YAWN_RELEASE_RATIO:
                self._mouth_last_open = now
            elif now - self._mouth_last_open > self.YAWN_GRACE:
                self._end_yawn()

        if self._yawn_start is None:
            return False, 0.0

        # Time the yawn up to the last frame the mouth was really open, so the
        # grace period after it closes cannot stretch a 2.4 s yawn past 2.5 s.
        duration = max(0.0, self._raw_last_high - self._yawn_start)
        # Only a mouth held open long enough counts - this rejects talking.
        is_yawning = duration >= self.yawn_min_duration
        if is_yawning and not self._counted_current_yawn:
            self.yawn_count += 1
            self._yawn_times.append(now)
            self._counted_current_yawn = True
        return is_yawning, duration

    def _end_yawn(self) -> None:
        self._yawn_start = None
        self._raw_open_since = None
        self._counted_current_yawn = False

    # ------------------------------------------------------------- head
    @property
    def head_calibrated(self) -> bool:
        return len(self._pose_samples) >= self.POSE_MIN_SAMPLES

    def _pose_baseline(self) -> Tuple[Optional[float], float]:
        """This driver's upright (pitch, roll).

        Pitch depends heavily on where the camera sits, so it is not judged at
        all until calibrated. Roll is 0 for an upright head on any camera, so
        0 is a safe default until then.
        """
        if not self.head_calibrated:
            return None, 0.0
        pitches = [p for _, p, _ in self._pose_samples]
        rolls = [r for _, _, r in self._pose_samples]
        return float(np.median(pitches)), float(np.median(rolls))

    def _update_head(
        self, pitch: Optional[float], roll: float, now: float, yawning: bool
    ) -> Tuple[bool, float, float, float]:
        """Track head drops. Returns (head_down, duration, rel_pitch, rel_roll)."""
        base_pitch, base_roll = self._pose_baseline()
        rel_pitch = pitch - base_pitch if pitch is not None and base_pitch is not None else 0.0
        rel_roll = roll - base_roll

        # Hysteresis: a dropped head must come clearly back up to count as up.
        scale = self.HEAD_RELEASE_RATIO if self._head_down_start is not None else 1.0
        direction = ""
        if rel_pitch >= self.head_drop_angle * scale:
            direction = "forward"
        elif rel_pitch <= -self.head_back_angle * scale and not yawning:
            direction = "backward"
        elif abs(rel_roll) >= self.head_side_angle * scale:
            direction = "sideways"

        # Remember when the head was at least half-way down, so a face that
        # vanishes mid-fall can be recognised as a fallen head.
        if (rel_pitch >= self.head_drop_angle * 0.5
                or abs(rel_roll) >= self.head_side_angle * 0.5):
            self._head_falling_at = now

        if direction:
            if self._head_down_start is None:
                self._head_down_start = now
                self._counted_current_nod = False
            self._head_last_down = now
            self._head_direction = direction
        elif self._head_down_start is not None and now - self._head_last_down > self.HEAD_GRACE:
            self._end_head_drop()

        # Learn the upright posture only from a steady, clearly-upright head.
        if (pitch is not None and self._head_down_start is None
                and abs(rel_pitch) < self.head_drop_angle * 0.5
                and abs(rel_roll) < self.head_side_angle * 0.5):
            self._pose_samples.append((now, pitch, roll))
            cutoff = now - self.POSE_WINDOW
            while self._pose_samples and self._pose_samples[0][0] < cutoff:
                self._pose_samples.popleft()

        duration = self._head_duration()
        return self._head_down_start is not None, duration, rel_pitch, rel_roll

    def _head_duration(self) -> float:
        if self._head_down_start is None:
            return 0.0
        duration = max(0.0, self._head_last_down - self._head_down_start)
        if duration >= self.HEAD_NOD_MIN and not self._counted_current_nod:
            self.head_nod_count += 1
            self._counted_current_nod = True
        return duration

    def _end_head_drop(self) -> None:
        self._head_down_start = None
        self._head_direction = ""
        self._counted_current_nod = False

    def _head_lost_while_falling(self, now: float) -> Optional[DrowsinessMetrics]:
        """Metrics for "the face vanished because the head fell", else None."""
        if now - self._last_face_time > self.head_lost_hold:
            self._end_head_drop()
            return None
        falling = (self._head_down_start is not None
                   or self._last_face_time - self._head_falling_at <= self.HEAD_FALL_MEMORY)
        if not falling:
            return None

        if self._head_down_start is None:
            self._head_down_start = self._head_falling_at
            self._head_direction = "forward"
            self._counted_current_nod = False
        self._head_last_down = now
        duration = self._head_duration()

        raw_score = self._head_score_floor(True, duration)
        self._smoothed_score = (
            self.smoothing * raw_score + (1.0 - self.smoothing) * self._smoothed_score
        )
        urgent = duration >= self.head_drop_duration
        state = DrowsinessState.DROWSY if urgent else DrowsinessState.WARNING
        self._last_state = state
        self.history.append((now - self.session_start, self._smoothed_score))
        return DrowsinessMetrics(
            state=state,
            score=self._smoothed_score,
            raw_score=raw_score,
            longest_closure=self.longest_closure,
            blink_count=self.blink_count,
            yawn_count=self.yawn_count,
            microsleep_count=self.microsleep_count,
            head_down=True,
            head_down_duration=duration,
            head_direction=self._head_direction,
            head_nod_count=self.head_nod_count,
            head_calibrated=self.head_calibrated,
            face_detected=False,
            alarm_active=urgent,
            urgent=urgent,
            perclos=self._perclos(now),
            ear_threshold=self.active_ear_threshold,
            calibrated=self.calibrated,
            contributions={},
        )

    def _head_score_floor(self, head_down: bool, duration: float) -> float:
        """Head down -> WARNING at once, climbing toward DROWSY; the exact
        head_drop_duration rule in update() makes the DROWSY call."""
        if not head_down:
            return 0.0
        progress = min(1.0, duration / max(self.head_drop_duration, 1e-6))
        low, top = self.warning_score + 2.0, self.drowsy_score - 2.0
        return low + (top - low) * progress

    # ------------------------------------------------------- score channels
    def _perclos_contribution(self, perclos: float) -> float:
        """PERCLOS scaled so that hitting the threshold (20%) gives a full 1.0."""
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

    def _yawn_contribution(self, recent_yawns: int, is_yawning: bool) -> float:
        """One yawn is normal; repeated yawning in the last minute is the signal.

        1 yawn -> 0.25, 2 -> 0.6, 3+ -> 1.0. Yawning right now adds 0.2.
        """
        levels = {0: 0.0, 1: 0.25, 2: 0.6}
        rate_part = levels.get(recent_yawns, 1.0)
        return min(1.0, rate_part + (0.2 if is_yawning else 0.0))

    def _yawn_score_floor(self, is_yawning: bool, yawn_duration: float) -> float:
        """Minimum score implied by yawning (see "Yawns" in the module docstring).

        While the mouth is held open the floor climbs from the WARNING line at
        yawn_min_duration to just below the DROWSY line; crossing into DROWSY is
        left to the exact yawn_alarm_duration rule in update().
        """
        floor = 0.0
        if len(self._yawn_times) >= self.frequent_yawns:
            floor = self.warning_score + 2.0
        if is_yawning:
            span = max(self.yawn_alarm_duration - self.yawn_min_duration, 1e-6)
            progress = min(1.0, (yawn_duration - self.yawn_min_duration) / span)
            top = self.drowsy_score - 2.0
            floor = max(floor, self.warning_score + 2.0 + (top - self.warning_score - 2.0) * progress)
        return floor

    def _blink_contribution(self, blink_rate: float, now: float) -> float:
        """How far the blink rate sits outside the healthy 8-28/min band.

        Silent for the first 30 seconds: with less data the rate estimate is
        meaningless and would produce a false alarm.
        """
        if now - self.session_start < self.BLINK_RATE_WARMUP:
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
        """Map the score to a state, with hysteresis and one safety override.

        Override: if the eyes have been shut longer than microsleep_duration we
        declare DROWSY immediately, no matter what the smoothed score says.
        A microsleep at 100 km/h covers ~33 m per second - we cannot wait for a
        rolling average to catch up.
        """
        if closure_duration >= self.microsleep_duration:
            return DrowsinessState.DROWSY

        last = self._last_state
        drowsy_at = self.drowsy_score
        warning_at = self.warning_score
        # Once in a band, stay there until the score is clearly below it.
        if last == DrowsinessState.DROWSY:
            drowsy_at -= self.STATE_HYSTERESIS
        if last in (DrowsinessState.DROWSY, DrowsinessState.WARNING):
            warning_at -= self.STATE_HYSTERESIS

        if score >= drowsy_at:
            return DrowsinessState.DROWSY
        if score >= warning_at:
            return DrowsinessState.WARNING
        return DrowsinessState.ALERT

    def _rate_window(self, now: float) -> float:
        """Seconds of data to divide by: what we have, but at least 30 s.

        Using the real elapsed time early on would turn 2 blinks in 3 seconds
        into "40 blinks per minute".
        """
        elapsed = now - self.session_start
        return max(min(elapsed, self.perclos_window), self.MIN_RATE_WINDOW)

    def _perclos(self, now: float, ongoing: float = 0.0) -> float:
        """Share of the window spent in closures longer than a blink."""
        cutoff = now - self.perclos_window
        closed = sum(end - max(start, cutoff) for start, end in self._long_closures)
        if ongoing > self.blink_max_duration:
            closed += ongoing
        return min(1.0, closed / self._rate_window(now))

    def _trim_times(self, buffer: Deque[float], now: float) -> None:
        """Drop timestamps that fell out of the rolling window."""
        cutoff = now - self.perclos_window
        while buffer and buffer[0] < cutoff:
            buffer.popleft()

    def _trim_intervals(self, now: float) -> None:
        cutoff = now - self.perclos_window
        while self._long_closures and self._long_closures[0][1] < cutoff:
            self._long_closures.popleft()

    # --------------------------------------------------------------- summary
    def session_summary(self) -> Dict[str, float]:
        """End-of-session statistics, shown when the user stops the camera."""
        elapsed = max(time.time() - self.session_start, 1e-6)
        return {
            "duration_seconds": elapsed,
            "total_blinks": self.blink_count,
            "total_yawns": self.yawn_count,
            "microsleeps": self.microsleep_count,
            "head_nods": self.head_nod_count,
            "longest_closure": self.longest_closure,
            "avg_blink_rate": self.blink_count / (elapsed / 60.0),
            "peak_score": max((s for _, s in self.history), default=0.0),
        }
