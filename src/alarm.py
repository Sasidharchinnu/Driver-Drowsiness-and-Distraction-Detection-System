"""
The alarm: generate a warning tone and play it without blocking the video loop.

Two problems this module solves
-------------------------------
1. WHAT to play. Rather than ship a binary .wav, we *synthesise* one with numpy
   the first time the app runs. That keeps the repo free of binary blobs and
   lets you change the tone by editing numbers.

2. HOW to play it without freezing the video. Playing audio synchronously would
   stall the capture loop for the length of the sound. So every play happens on
   a short-lived background thread, guarded by a cooldown so a sustained DROWSY
   state does not spawn hundreds of overlapping sounds.

Playback backends, tried in order:
   1. macOS `afplay`  - always present on macOS, zero extra dependencies.
   2. pygame.mixer    - cross-platform, used on Windows/Linux if installed.
   3. Windows winsound / Linux `aplay`.
   4. Terminal bell   - last resort, so *something* always happens.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
import threading
import time
import wave
from pathlib import Path
from typing import Optional

import numpy as np

from .config import resolve_path

SAMPLE_RATE = 44100  # CD quality; every backend understands it


def generate_alarm_wav(path: Path, duration: float = 1.2, volume: float = 0.8) -> Path:
    """Create a two-tone 'beep-beep' siren and write it as a 16-bit WAV.

    Design of the sound (chosen to be hard to ignore but not painful):
      * Alternates between 880 Hz and 1200 Hz roughly 6 times a second. Our
        hearing locks onto changing pitch far more than a constant tone.
      * A short fade-in/out on each burst removes the 'click' you get when a
        waveform starts or stops at a non-zero value.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    samples = int(SAMPLE_RATE * duration)
    t = np.linspace(0.0, duration, samples, endpoint=False)

    # A square wave at 6 Hz switches us between the two pitches.
    switch = (np.sin(2 * np.pi * 6.0 * t) > 0).astype(np.float32)
    frequency = 880.0 * switch + 1200.0 * (1.0 - switch)

    # Integrate the frequency to get phase, otherwise the waveform jumps
    # discontinuously at every switch and sounds like static.
    phase = 2 * np.pi * np.cumsum(frequency) / SAMPLE_RATE
    wave_data = np.sin(phase).astype(np.float32)

    # 12 ms fade at each end to kill the start/stop click.
    fade_len = int(SAMPLE_RATE * 0.012)
    envelope = np.ones(samples, dtype=np.float32)
    envelope[:fade_len] = np.linspace(0.0, 1.0, fade_len)
    envelope[-fade_len:] = np.linspace(1.0, 0.0, fade_len)

    audio = wave_data * envelope * float(np.clip(volume, 0.0, 1.0))
    pcm = (audio * 32767).astype(np.int16)  # 16-bit signed PCM

    with wave.open(str(path), "w") as handle:
        handle.setnchannels(1)       # mono
        handle.setsampwidth(2)       # 2 bytes = 16 bit
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(pcm.tobytes())

    return path


class AlarmSystem:
    """Plays the alarm when the driver is drowsy, with rate limiting."""

    def __init__(self, cfg):
        """`cfg` is the `alarm` section of config.yaml."""
        self.enabled = bool(cfg.enabled)
        self.cooldown = float(cfg.cooldown)
        self.min_trigger_duration = float(cfg.min_trigger_duration)
        self.volume = float(cfg.volume)
        self.sound_path = resolve_path(cfg.sound_file)

        self._last_played = 0.0
        self._drowsy_since: Optional[float] = None
        self._is_playing = False
        self._lock = threading.Lock()
        self.trigger_count = 0
        self.backend = "none"
        self.last_error: Optional[str] = None

        self._ensure_sound_file()
        self._mixer = self._init_backend()

    # ----------------------------------------------------------------- setup
    def _ensure_sound_file(self) -> None:
        """Generate the WAV if it is missing or was truncated."""
        try:
            if not self.sound_path.exists() or self.sound_path.stat().st_size < 1000:
                generate_alarm_wav(self.sound_path, volume=self.volume)
        except OSError as exc:
            self.last_error = f"Could not write the alarm sound file: {exc}"

    def _init_backend(self):
        """Pick the best available playback method for this machine."""
        system = platform.system()

        if system == "Darwin" and shutil.which("afplay"):
            self.backend = "afplay"
            return None

        try:
            import pygame

            pygame.mixer.init(frequency=SAMPLE_RATE, size=-16, channels=1, buffer=512)
            sound = pygame.mixer.Sound(str(self.sound_path))
            sound.set_volume(self.volume)
            self.backend = "pygame"
            return sound
        except Exception:
            pass  # pygame is optional; fall through to the OS tools

        if system == "Windows":
            self.backend = "winsound"
            return None
        if shutil.which("aplay"):
            self.backend = "aplay"
            return None
        if shutil.which("paplay"):
            self.backend = "paplay"
            return None

        self.backend = "bell"  # terminal bell - always works, quietly
        return None

    # --------------------------------------------------------------- control
    def update(self, is_drowsy: bool, now: Optional[float] = None) -> bool:
        """Call once per frame with the current DROWSY flag.

        Returns True if an alarm was actually fired on this frame.

        Two guards prevent alarm spam:
          * The state must stay DROWSY for `min_trigger_duration` seconds, so a
            single noisy frame cannot set it off.
          * At least `cooldown` seconds must pass between two sounds.
        """
        now = now if now is not None else time.time()

        if not is_drowsy:
            self._drowsy_since = None   # recovered; reset the sustain timer
            return False

        if self._drowsy_since is None:
            self._drowsy_since = now

        if now - self._drowsy_since < self.min_trigger_duration:
            return False                # drowsy, but not for long enough yet

        if now - self._last_played < self.cooldown:
            return False                # still inside the cooldown

        self._last_played = now
        self.trigger_count += 1
        self.play()
        return True

    def play(self) -> None:
        """Fire the sound on a background thread so the video never stalls."""
        if not self.enabled:
            return
        with self._lock:
            if self._is_playing:
                return                  # a sound is already in flight
            self._is_playing = True

        threading.Thread(target=self._play_blocking, daemon=True).start()

    def _play_blocking(self) -> None:
        """The actual playback. Runs only on the background thread."""
        try:
            if self.backend == "afplay":
                subprocess.run(
                    ["afplay", "-v", str(self.volume), str(self.sound_path)],
                    check=False,
                    timeout=5,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            elif self.backend == "pygame" and self._mixer is not None:
                self._mixer.play()
                time.sleep(1.2)
            elif self.backend == "winsound":
                import winsound

                winsound.PlaySound(str(self.sound_path), winsound.SND_FILENAME)
            elif self.backend in ("aplay", "paplay"):
                subprocess.run(
                    [self.backend, str(self.sound_path)],
                    check=False,
                    timeout=5,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            else:
                # Terminal bell. Not loud, but it is never *nothing*.
                sys.stdout.write("\a")
                sys.stdout.flush()
        except Exception as exc:
            # Audio must never take the application down.
            self.last_error = f"Alarm playback failed ({self.backend}): {exc}"
        finally:
            with self._lock:
                self._is_playing = False

    def test(self) -> bool:
        """Play once immediately, ignoring the cooldown. Used by the Test button."""
        self._last_played = time.time()
        self.play()
        return self.last_error is None

    def reset(self) -> None:
        self._drowsy_since = None
        self._last_played = 0.0
        self.trigger_count = 0

    @property
    def is_playing(self) -> bool:
        with self._lock:
            return self._is_playing

    def describe(self) -> dict:
        return {
            "enabled": self.enabled,
            "backend": self.backend,
            "sound_file": str(self.sound_path),
            "triggers": self.trigger_count,
            "error": self.last_error,
        }
