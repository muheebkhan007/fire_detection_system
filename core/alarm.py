"""
Alarm Manager
--------------
Plays audio alerts in a background thread.
Generates a synthetic alarm tone using NumPy + sounddevice (no file required).
Falls back to a console bell if audio is unavailable.
"""

import time
import logging
import threading
from typing import Optional

logger = logging.getLogger(__name__)


def _generate_alarm_tone(
    frequency: float = 880.0,
    duration: float = 0.3,
    sample_rate: int = 44100,
) -> Optional[object]:
    """Build a numpy waveform for an alarm beep (sine + harmonics)."""
    try:
        import numpy as np
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        wave  = 0.6 * np.sin(2 * np.pi * frequency * t)
        wave += 0.3 * np.sin(2 * np.pi * frequency * 2 * t)   # 2nd harmonic
        wave += 0.1 * np.sin(2 * np.pi * frequency * 3 * t)   # 3rd harmonic
        # Amplitude envelope (avoid clicks)
        fade = int(sample_rate * 0.01)
        wave[:fade]  *= np.linspace(0, 1, fade)
        wave[-fade:] *= np.linspace(1, 0, fade)
        return wave.astype(np.float32)
    except ImportError:
        return None


class AlarmManager:
    """
    Non-blocking alarm controller.

    Levels:
        "warning" – slower, lower-pitch beeps  (smoke detected)
        "fire"    – rapid, high-pitch klaxon   (fire detected)
    """

    BEEP_CONFIGS = {
        "warning": {"freq": 660.0,  "beep": 0.25, "pause": 0.5,  "cycles": 6},
        "fire":    {"freq": 1000.0, "beep": 0.15, "pause": 0.15, "cycles": 20},
    }

    def __init__(self) -> None:
        self._active       = False
        self._level: str   = ""
        self._thread: Optional[threading.Thread] = None
        self._stop_event   = threading.Event()
        self._lock         = threading.Lock()
        self._audio_ok     = self._check_audio()

    # ── Public API ───────────────────────────────────────────────────────────

    def trigger(self, level: str = "fire") -> None:
        """
        Start an alarm at the given level.
        Calling trigger() while already active replaces the level
        only if the new level is higher urgency.
        """
        with self._lock:
            urgency = {"warning": 1, "fire": 2}
            if self._active and urgency.get(level, 0) <= urgency.get(self._level, 0):
                return  # already at same or higher urgency
            self._level  = level
            self._active = True
            self._stop_event.clear()
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(
                    target=self._alarm_loop, daemon=True, name="AlarmThread"
                )
                self._thread.start()

    def silence(self) -> None:
        """Stop all alarms."""
        with self._lock:
            self._active = False
            self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    @property
    def is_active(self) -> bool:
        return self._active

    # ── Internal ─────────────────────────────────────────────────────────────

    def _alarm_loop(self) -> None:
        logger.info("Alarm started – level=%s", self._level)
        while not self._stop_event.is_set() and self._active:
            cfg = self.BEEP_CONFIGS.get(self._level, self.BEEP_CONFIGS["fire"])
            self._play_beep_pattern(cfg)
        logger.info("Alarm silenced.")

    def _play_beep_pattern(self, cfg: dict) -> None:
        for _ in range(cfg["cycles"]):
            if self._stop_event.is_set():
                return
            self._beep(cfg["freq"], cfg["beep"])
            time.sleep(cfg["pause"])

    def _beep(self, freq: float, duration: float) -> None:
        if not self._audio_ok:
            print("\a", end="", flush=True)   # console bell fallback
            return
        try:
            import sounddevice as sd
            wave = _generate_alarm_tone(freq, duration)
            if wave is not None:
                sd.play(wave, samplerate=44100, blocking=True)
        except Exception as exc:
            logger.debug("Audio playback error: %s", exc)
            print("\a", end="", flush=True)

    @staticmethod
    def _check_audio() -> bool:
        try:
            import sounddevice as sd          # noqa: F401
            import numpy as np                # noqa: F401
            # Verify PortAudio is actually available at runtime
            sd.query_devices()
            return True
        except (ImportError, OSError, Exception):
            logger.warning("Audio unavailable (PortAudio/sounddevice) – using console bell.")
            return False
