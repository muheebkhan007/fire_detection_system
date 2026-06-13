"""
Event Logger
-------------
Writes detection events to a rotating CSV log and saves annotated
screenshots on danger transitions.
"""

import cv2
import csv
import time
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional
import threading

from core.detector import FrameResult, ThreatLevel

logger = logging.getLogger(__name__)


class EventLogger:
    """
    Thread-safe logger that:
      - Writes one CSV row per detection event (danger transitions only).
      - Saves annotated frame screenshots to disk.
      - Exposes `recent_events` for the UI sidebar.
    """

    LOG_DIR        = Path("logs")
    SCREENSHOT_DIR = Path("screenshots")
    MAX_RECENT     = 50           # kept in memory for UI display
    SCREENSHOT_COOLDOWN = 5.0    # seconds between auto-screenshots

    def __init__(self) -> None:
        self.LOG_DIR.mkdir(exist_ok=True)
        self.SCREENSHOT_DIR.mkdir(exist_ok=True)

        self._csv_path   = self.LOG_DIR / f"events_{self._date_tag()}.csv"
        self._lock       = threading.Lock()
        self._recent_events: list[dict] = []
        self._last_screenshot_t = 0.0
        self._prev_threat = ThreatLevel.SAFE

        self._init_csv()
        logger.info("EventLogger ready – CSV: %s", self._csv_path)

    # ── Public API ───────────────────────────────────────────────────────────

    def log(self, result: FrameResult) -> None:
        """
        Process a FrameResult.
        Logs an event and saves a screenshot on any threat transition.
        """
        threat_changed = result.threat_level != self._prev_threat

        if threat_changed or result.threat_level != ThreatLevel.SAFE:
            if result.threat_level != ThreatLevel.SAFE:
                self._write_event(result)

            if threat_changed and result.threat_level != ThreatLevel.SAFE:
                self._maybe_screenshot(result)

        self._prev_threat = result.threat_level

    def save_screenshot(self, frame, label: str = "manual") -> Optional[Path]:
        """Manually save a screenshot (e.g. from UI button)."""
        return self._save_image(frame, label)

    @property
    def recent_events(self) -> list[dict]:
        """Return a copy of the most recent events (thread-safe)."""
        with self._lock:
            return list(self._recent_events)

    @property
    def log_path(self) -> Path:
        return self._csv_path

    # ── Internals ────────────────────────────────────────────────────────────

    def _init_csv(self) -> None:
        if not self._csv_path.exists():
            with open(self._csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp", "datetime", "threat_level",
                    "detections_count", "max_confidence",
                    "labels", "frame_number",
                ])

    def _write_event(self, result: FrameResult) -> None:
        now     = time.time()
        dt_str  = datetime.fromtimestamp(now).strftime("%Y-%m-%d %H:%M:%S")
        labels  = "|".join(d.label for d in result.detections)
        row = [
            f"{now:.3f}",
            dt_str,
            result.threat_level.value,
            len(result.detections),
            f"{result.max_confidence:.3f}",
            labels,
            result.frame_count,
        ]
        try:
            with open(self._csv_path, "a", newline="") as f:
                csv.writer(f).writerow(row)
        except OSError as exc:
            logger.error("CSV write failed: %s", exc)

        event = {
            "datetime": dt_str,
            "threat":   result.threat_level.value,
            "confidence": f"{result.max_confidence * 100:.1f}%",
            "labels":   labels,
        }
        with self._lock:
            self._recent_events.append(event)
            if len(self._recent_events) > self.MAX_RECENT:
                self._recent_events.pop(0)

        logger.info("Event: %s  conf=%.1f%%", result.threat_level.value,
                    result.max_confidence * 100)

    def _maybe_screenshot(self, result: FrameResult) -> None:
        if time.time() - self._last_screenshot_t < self.SCREENSHOT_COOLDOWN:
            return
        if result.annotated_frame is None:
            return
        label = result.threat_level.name.lower()
        path  = self._save_image(result.annotated_frame, label)
        if path:
            self._last_screenshot_t = time.time()
            logger.info("Screenshot saved: %s", path)

    def _save_image(
        self, frame, label: str = "event"
    ) -> Optional[Path]:
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        path = self.SCREENSHOT_DIR / f"{label}_{ts}.jpg"
        try:
            cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            return path
        except OSError as exc:
            logger.error("Screenshot write failed: %s", exc)
            return None

    @staticmethod
    def _date_tag() -> str:
        return datetime.now().strftime("%Y%m%d")
