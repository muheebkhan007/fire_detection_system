"""
Camera / Video Capture Thread
------------------------------
Runs capture in a dedicated thread so the UI never blocks.
Supports webcam index, RTSP streams, and local video files.
"""

import cv2
import time
import threading
import logging
from typing import Optional, Callable

logger = logging.getLogger(__name__)


class CaptureThread(threading.Thread):
    """
    Non-blocking video capture thread.

    Usage:
        cap = CaptureThread(source=0, on_frame=my_callback)
        cap.start()
        ...
        cap.stop()

    The `on_frame` callback receives a single ndarray (BGR frame).
    It is invoked from the capture thread – keep it short or dispatch
    further work to a queue.
    """

    def __init__(
        self,
        source: int | str = 0,
        on_frame: Optional[Callable] = None,
        target_fps: int = 30,
        resolution: tuple[int, int] = (640, 480),
    ):
        super().__init__(daemon=True, name="CaptureThread")
        self.source      = source
        self.on_frame    = on_frame
        self.target_fps  = target_fps
        self.resolution  = resolution

        self._stop_event  = threading.Event()
        self._cap: Optional[cv2.VideoCapture] = None
        self._lock        = threading.Lock()

        # Public state – read from UI thread safely
        self.is_running   = False
        self.last_error: Optional[str] = None
        self.frame_count  = 0

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def run(self) -> None:
        self._open_capture()
        if self._cap is None or not self._cap.isOpened():
            self.last_error = f"Cannot open source: {self.source}"
            logger.error(self.last_error)
            return

        self.is_running = True
        frame_interval  = 1.0 / self.target_fps
        logger.info("Capture started – source=%s  fps=%d", self.source, self.target_fps)

        try:
            while not self._stop_event.is_set():
                t0 = time.monotonic()
                ret, frame = self._cap.read()

                if not ret:
                    logger.warning("Frame read failed – attempting reconnect…")
                    if not self._reconnect():
                        break
                    continue

                self.frame_count += 1
                if self.on_frame is not None:
                    try:
                        self.on_frame(frame)
                    except Exception as exc:
                        logger.exception("on_frame callback raised: %s", exc)

                # Throttle to target FPS
                elapsed = time.monotonic() - t0
                sleep_t = frame_interval - elapsed
                if sleep_t > 0:
                    time.sleep(sleep_t)

        finally:
            self._release()
            self.is_running = False
            logger.info("Capture stopped after %d frames.", self.frame_count)

    def stop(self) -> None:
        """Signal the thread to stop and wait for it to finish."""
        self._stop_event.set()
        self.join(timeout=3.0)

    # ── Internals ────────────────────────────────────────────────────────────

    def _open_capture(self) -> None:
        with self._lock:
            cap = cv2.VideoCapture(self.source)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.resolution[0])
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
                cap.set(cv2.CAP_PROP_FPS,          self.target_fps)
                cap.set(cv2.CAP_PROP_BUFFERSIZE,   2)    # low latency
            self._cap = cap

    def _reconnect(self) -> bool:
        """Try to re-open the capture source up to 3 times."""
        self._release()
        for attempt in range(1, 4):
            logger.info("Reconnect attempt %d…", attempt)
            time.sleep(1.0)
            self._open_capture()
            if self._cap and self._cap.isOpened():
                logger.info("Reconnected successfully.")
                return True
        self.last_error = "Reconnect failed after 3 attempts."
        logger.error(self.last_error)
        return False

    def _release(self) -> None:
        with self._lock:
            if self._cap is not None:
                self._cap.release()
                self._cap = None
