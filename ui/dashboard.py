"""
Main Dashboard – PyQt5
-----------------------
Professional dark-theme emergency dashboard with:
  • Live video panel
  • Animated threat banner (SAFE / WARNING / FIRE DETECTED)
  • Confidence meter + FPS counter
  • Event log sidebar
  • Control toolbar (camera select, screenshot, silence alarm)
"""

import sys
import time
import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from PyQt5.QtCore    import Qt, QTimer, QThread, pyqtSignal, QSize, pyqtSlot
from PyQt5.QtGui     import (QImage, QPixmap, QFont, QPalette, QColor,
                              QIcon, QPainter, QBrush, QLinearGradient)
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QGridLayout, QComboBox, QFrame,
    QScrollArea, QSizePolicy, QProgressBar, QStatusBar, QDialog,
    QDialogButtonBox, QMessageBox, QSplitter, QSpacerItem,
    QGroupBox
)

from core.detector   import FireSmokeDetector, FrameResult, ThreatLevel
from core.capture    import CaptureThread
from core.alarm      import AlarmManager
from utils.event_logger import EventLogger

logger = logging.getLogger(__name__)

# ── Colour palette ──────────────────────────────────────────────────────────
PALETTE = {
    "bg_dark":    "#0D0D0D",
    "bg_panel":   "#141414",
    "bg_card":    "#1C1C1C",
    "accent":     "#FF3B1F",
    "accent2":    "#FF8C00",
    "safe":       "#00E676",
    "warning":    "#FF8C00",
    "danger":     "#FF1744",
    "text_primary":   "#F0F0F0",
    "text_secondary": "#888888",
    "border":     "#2A2A2A",
}


def hex_to_qcolor(h: str) -> QColor:
    return QColor(h)


# ── Worker thread (runs detection in background) ─────────────────────────────

class DetectionWorker(QThread):
    """
    Runs the heavy detection loop off the main thread.
    Emits `result_ready` for every processed frame.
    """
    result_ready = pyqtSignal(object)   # FrameResult
    error_signal = pyqtSignal(str)

    def __init__(self, source: int | str = 0):
        super().__init__()
        self.source      = source
        self._detector   = FireSmokeDetector(640, 480)
        self._capture: Optional[CaptureThread] = None
        self._running    = False

    def run(self) -> None:
        self._running = True
        self._capture = CaptureThread(
            source=self.source,
            on_frame=self._on_frame,
            target_fps=25,
            resolution=(640, 480),
        )
        self._capture.start()
        self._capture.join()   # block until capture stops
        self._running = False

    def _on_frame(self, frame: np.ndarray) -> None:
        try:
            result = self._detector.process_frame(frame)
            self.result_ready.emit(result)
        except Exception as exc:
            logger.exception("Detection error: %s", exc)

    def stop(self) -> None:
        if self._capture:
            self._capture.stop()
        self.wait(3000)


# ── Animated threat banner ────────────────────────────────────────────────────

class ThreatBanner(QLabel):
    """Pulsing coloured banner showing current threat level."""

    CONFIGS = {
        ThreatLevel.SAFE:    ("#00E676", "●  SAFE",           "white"),
        ThreatLevel.WARNING: ("#FF8C00", "⚠  WARNING",        "black"),
        ThreatLevel.FIRE:    ("#FF1744", "🔥  FIRE DETECTED",  "white"),
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self._threat = ThreatLevel.SAFE
        self._pulse_alpha = 255
        self._pulse_dir   = -1
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumHeight(56)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._font = QFont("Segoe UI", 18, QFont.Bold)
        self.setFont(self._font)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._pulse)
        self._timer.start(40)

        self.set_threat(ThreatLevel.SAFE)

    def set_threat(self, threat: ThreatLevel) -> None:
        self._threat = threat
        bg, text, fg = self.CONFIGS[threat]
        self.setText(text)
        self.setStyleSheet(
            f"background-color:{bg}; color:{fg}; border-radius:4px;"
            f"padding:6px; font-size:18px; font-weight:bold;"
        )

    def _pulse(self) -> None:
        if self._threat == ThreatLevel.SAFE:
            return
        self._pulse_alpha += self._pulse_dir * 8
        if self._pulse_alpha <= 60:
            self._pulse_dir = 1
        elif self._pulse_alpha >= 255:
            self._pulse_dir = -1
        _, _, fg = self.CONFIGS[self._threat]
        bg, _, _ = self.CONFIGS[self._threat]
        # re-apply opacity via rgba
        r, g, b = (
            int(bg[1:3], 16), int(bg[3:5], 16), int(bg[5:7], 16)
        )
        a = self._pulse_alpha
        self.setStyleSheet(
            f"background-color:rgba({r},{g},{b},{a});"
            f"color:{fg}; border-radius:4px; padding:6px;"
            f"font-size:18px; font-weight:bold;"
        )


# ── Metric card ───────────────────────────────────────────────────────────────

class MetricCard(QFrame):
    """Small KPI card: label + large value."""

    def __init__(self, title: str, value: str = "—", parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet(
            f"background:{PALETTE['bg_card']}; border:1px solid {PALETTE['border']};"
            "border-radius:6px; padding:8px;"
        )

        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 8, 10, 8)
        lay.setSpacing(2)

        self._title_lbl = QLabel(title.upper())
        self._title_lbl.setStyleSheet(
            f"color:{PALETTE['text_secondary']}; font-size:10px;"
            "font-weight:bold; background:transparent; border:none;"
        )

        self._value_lbl = QLabel(value)
        self._value_lbl.setStyleSheet(
            f"color:{PALETTE['text_primary']}; font-size:20px;"
            "font-weight:bold; background:transparent; border:none;"
        )

        lay.addWidget(self._title_lbl)
        lay.addWidget(self._value_lbl)

    def set_value(self, value: str, colour: str | None = None) -> None:
        self._value_lbl.setText(value)
        if colour:
            self._value_lbl.setStyleSheet(
                f"color:{colour}; font-size:20px; font-weight:bold;"
                "background:transparent; border:none;"
            )


# ── Event log sidebar ─────────────────────────────────────────────────────────

class EventLogWidget(QWidget):
    """Scrollable list of recent detection events."""

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)

        hdr = QLabel("RECENT EVENTS")
        hdr.setStyleSheet(
            f"color:{PALETTE['text_secondary']}; font-size:10px;"
            "font-weight:bold; padding:4px;"
        )
        lay.addWidget(hdr)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setStyleSheet(
            f"background:{PALETTE['bg_card']}; border:1px solid {PALETTE['border']};"
            "border-radius:4px;"
        )

        self._container = QWidget()
        self._vlay      = QVBoxLayout(self._container)
        self._vlay.setContentsMargins(4, 4, 4, 4)
        self._vlay.setSpacing(3)
        self._vlay.addStretch()

        self._scroll.setWidget(self._container)
        lay.addWidget(self._scroll)

    def push_event(self, event: dict) -> None:
        colour_map = {
            "FIRE DETECTED": PALETTE["danger"],
            "WARNING":       PALETTE["warning"],
        }
        colour = colour_map.get(event["threat"], PALETTE["safe"])

        row = QFrame()
        row.setStyleSheet(
            f"background:{PALETTE['bg_panel']}; border-left:3px solid {colour};"
            "border-radius:2px; margin:1px;"
        )
        rlay = QVBoxLayout(row)
        rlay.setContentsMargins(6, 4, 6, 4)
        rlay.setSpacing(1)

        t = QLabel(event.get("datetime", ""))
        t.setStyleSheet(f"color:{PALETTE['text_secondary']}; font-size:9px;")

        v = QLabel(f"{event['threat']}  –  {event['labels']}  ({event['confidence']})")
        v.setStyleSheet(f"color:{colour}; font-size:11px; font-weight:bold;")

        rlay.addWidget(t)
        rlay.addWidget(v)

        # Insert before the stretch item
        count = self._vlay.count()
        self._vlay.insertWidget(count - 1, row)

        # Auto-scroll to bottom
        QTimer.singleShot(50, lambda: self._scroll.verticalScrollBar().setValue(
            self._scroll.verticalScrollBar().maximum()
        ))


# ── Main window ───────────────────────────────────────────────────────────────

class MainDashboard(QMainWindow):
    """
    Primary application window.
    Layout:  [toolbar]
             [video_panel | metrics + event_log]
             [status_bar]
    """

    WINDOW_TITLE = "🔥 FireGuard Pro – Real-Time Fire & Smoke Detection"
    MIN_SIZE     = (1100, 680)

    def __init__(self):
        super().__init__()
        self._alarm     = AlarmManager()
        self._ev_logger = EventLogger()
        self._worker: Optional[DetectionWorker] = None
        self._last_result: Optional[FrameResult] = None
        self._alarm_silenced = False
        self._screenshot_dir = Path("screenshots")

        self._setup_ui()
        self._apply_dark_theme()
        self.setWindowTitle(self.WINDOW_TITLE)
        self.setMinimumSize(*self.MIN_SIZE)
        self.resize(1280, 760)

    # ── UI construction ────────────────────────────────────────────────────

    def _setup_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root_lay = QVBoxLayout(central)
        root_lay.setContentsMargins(8, 8, 8, 8)
        root_lay.setSpacing(6)

        # Toolbar
        root_lay.addWidget(self._build_toolbar())

        # Threat banner
        self._banner = ThreatBanner()
        root_lay.addWidget(self._banner)

        # Main content (splitter: video | right panel)
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._build_video_panel())
        splitter.addWidget(self._build_right_panel())
        splitter.setSizes([720, 380])
        splitter.setHandleWidth(4)
        root_lay.addWidget(splitter, stretch=1)

        # Status bar
        self._status = QStatusBar()
        self._status.setStyleSheet(
            f"background:{PALETTE['bg_dark']}; color:{PALETTE['text_secondary']};"
            "font-size:11px; border-top:1px solid #222;"
        )
        self.setStatusBar(self._status)
        self._status.showMessage("Ready  ·  Select a camera source and press START")

    def _build_toolbar(self) -> QWidget:
        bar = QWidget()
        bar.setFixedHeight(52)
        bar.setStyleSheet(
            f"background:{PALETTE['bg_panel']}; border:1px solid {PALETTE['border']};"
            "border-radius:6px;"
        )
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(12, 6, 12, 6)
        lay.setSpacing(8)

        # Logo / title
        logo = QLabel("🔥 FireGuard Pro")
        logo.setStyleSheet(
            f"color:{PALETTE['accent']}; font-size:16px; font-weight:bold;"
        )
        lay.addWidget(logo)

        lay.addStretch()

        # Camera selector
        cam_lbl = QLabel("Camera:")
        cam_lbl.setStyleSheet(f"color:{PALETTE['text_secondary']};")
        self._cam_combo = QComboBox()
        self._cam_combo.addItems(["Webcam 0", "Webcam 1", "Video File…"])
        self._cam_combo.setFixedWidth(130)
        self._cam_combo.setStyleSheet(self._combo_style())
        lay.addWidget(cam_lbl)
        lay.addWidget(self._cam_combo)

        # Buttons
        self._btn_start = self._make_btn("▶  START", PALETTE["safe"],    "black")
        self._btn_stop  = self._make_btn("■  STOP",  "#555",             "white")
        self._btn_shot  = self._make_btn("📷  Screenshot", "#2979FF",    "white")
        self._btn_sil   = self._make_btn("🔇  Silence",    PALETTE["warning"], "black")
        self._btn_stop.setEnabled(False)

        self._btn_start.clicked.connect(self._start_detection)
        self._btn_stop.clicked.connect(self._stop_detection)
        self._btn_shot.clicked.connect(self._manual_screenshot)
        self._btn_sil.clicked.connect(self._silence_alarm)

        for btn in (self._btn_start, self._btn_stop, self._btn_shot, self._btn_sil):
            lay.addWidget(btn)

        return bar

    def _build_video_panel(self) -> QWidget:
        panel = QFrame()
        panel.setStyleSheet(
            f"background:{PALETTE['bg_panel']}; border:2px solid {PALETTE['border']};"
            "border-radius:8px;"
        )
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(6, 6, 6, 6)

        hdr = QLabel("LIVE FEED")
        hdr.setStyleSheet(
            f"color:{PALETTE['text_secondary']}; font-size:10px; font-weight:bold;"
        )
        lay.addWidget(hdr)

        self._video_lbl = QLabel()
        self._video_lbl.setAlignment(Qt.AlignCenter)
        self._video_lbl.setMinimumSize(640, 480)
        self._video_lbl.setStyleSheet("background:#000; border-radius:4px;")
        self._video_lbl.setText("No feed  –  Press START")
        self._video_lbl.setStyleSheet(
            f"background:#050505; color:{PALETTE['text_secondary']};"
            "font-size:14px; border-radius:4px;"
        )
        lay.addWidget(self._video_lbl, stretch=1)

        return panel

    def _build_right_panel(self) -> QWidget:
        panel = QWidget()
        lay   = QVBoxLayout(panel)
        lay.setContentsMargins(4, 0, 0, 0)
        lay.setSpacing(8)

        # Metrics grid
        metrics_grp = QGroupBox("Detection Metrics")
        metrics_grp.setStyleSheet(self._group_box_style())
        mg_lay = QGridLayout(metrics_grp)
        mg_lay.setSpacing(6)

        self._card_status = MetricCard("Status",     "—")
        self._card_conf   = MetricCard("Confidence", "—")
        self._card_fps    = MetricCard("FPS",        "—")
        self._card_events = MetricCard("Alerts",     "0")

        mg_lay.addWidget(self._card_status, 0, 0)
        mg_lay.addWidget(self._card_conf,   0, 1)
        mg_lay.addWidget(self._card_fps,    1, 0)
        mg_lay.addWidget(self._card_events, 1, 1)

        lay.addWidget(metrics_grp)

        # Confidence bar
        conf_grp = QGroupBox("Fire Confidence")
        conf_grp.setStyleSheet(self._group_box_style())
        cg_lay = QVBoxLayout(conf_grp)
        self._conf_bar = QProgressBar()
        self._conf_bar.setRange(0, 100)
        self._conf_bar.setValue(0)
        self._conf_bar.setTextVisible(True)
        self._conf_bar.setFormat("%v%")
        self._conf_bar.setFixedHeight(22)
        self._conf_bar.setStyleSheet(self._progress_style("#FF3B1F"))
        cg_lay.addWidget(self._conf_bar)
        lay.addWidget(conf_grp)

        # Smoke confidence bar
        smoke_grp = QGroupBox("Smoke Confidence")
        smoke_grp.setStyleSheet(self._group_box_style())
        sg_lay = QVBoxLayout(smoke_grp)
        self._smoke_bar = QProgressBar()
        self._smoke_bar.setRange(0, 100)
        self._smoke_bar.setValue(0)
        self._smoke_bar.setTextVisible(True)
        self._smoke_bar.setFormat("%v%")
        self._smoke_bar.setFixedHeight(22)
        self._smoke_bar.setStyleSheet(self._progress_style("#AAAAAA"))
        sg_lay.addWidget(self._smoke_bar)
        lay.addWidget(smoke_grp)

        # Detection indicators row
        ind_lay = QHBoxLayout()
        self._ind_fire  = self._make_indicator("FIRE",  "#333", "white")
        self._ind_smoke = self._make_indicator("SMOKE", "#333", "white")
        ind_lay.addWidget(self._ind_fire)
        ind_lay.addWidget(self._ind_smoke)
        lay.addLayout(ind_lay)

        # Event log
        self._event_log = EventLogWidget()
        lay.addWidget(self._event_log, stretch=1)

        return panel

    # ── Control actions ────────────────────────────────────────────────────

    def _start_detection(self) -> None:
        src_text = self._cam_combo.currentText()
        if src_text.startswith("Webcam"):
            source: int | str = int(src_text.split()[-1])
        else:
            from PyQt5.QtWidgets import QFileDialog
            path, _ = QFileDialog.getOpenFileName(
                self, "Select video file", "",
                "Video files (*.mp4 *.avi *.mov *.mkv *.webm)"
            )
            if not path:
                return
            source = path

        self._worker = DetectionWorker(source=source)
        self._worker.result_ready.connect(self._on_result)
        self._worker.finished.connect(self._on_worker_done)
        self._worker.start()

        self._btn_start.setEnabled(False)
        self._btn_stop.setEnabled(True)
        self._alarm_silenced = False
        self._alert_count    = 0
        self._status.showMessage(f"Detecting  ·  source={source}")
        logger.info("Detection started – source=%s", source)

    def _stop_detection(self) -> None:
        if self._worker:
            self._worker.stop()
        self._alarm.silence()
        self._btn_stop.setEnabled(False)
        self._banner.set_threat(ThreatLevel.SAFE)
        self._status.showMessage("Stopped")

    def _manual_screenshot(self) -> None:
        if self._last_result and self._last_result.annotated_frame is not None:
            path = self._ev_logger.save_screenshot(
                self._last_result.annotated_frame, "manual"
            )
            if path:
                self._status.showMessage(f"Screenshot saved: {path}")

    def _silence_alarm(self) -> None:
        self._alarm.silence()
        self._alarm_silenced = True
        self._status.showMessage("Alarm silenced")

    # ── Detection result handler ───────────────────────────────────────────

    @pyqtSlot(object)
    def _on_result(self, result: FrameResult) -> None:
        self._last_result = result
        self._update_video(result)
        self._update_metrics(result)
        self._ev_logger.log(result)

        if result.threat_level != ThreatLevel.SAFE and not self._alarm_silenced:
            level = "fire" if result.threat_level == ThreatLevel.FIRE else "warning"
            self._alarm.trigger(level)
            self._alarm_silenced = False  # reset only when threat clears
        elif result.threat_level == ThreatLevel.SAFE:
            if self._alarm.is_active:
                self._alarm.silence()
            self._alarm_silenced = False

        # Push events from logger to UI
        events = self._ev_logger.recent_events
        if events:
            last = events[-1]
            # Only push if it's a new event (simple dedup by datetime)
            if not hasattr(self, "_last_event_dt") or self._last_event_dt != last["datetime"]:
                self._last_event_dt = last["datetime"]
                self._event_log.push_event(last)
                self._card_events.set_value(str(len(events)))

    def _update_video(self, result: FrameResult) -> None:
        frame = result.annotated_frame
        if frame is None:
            return
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        img   = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pix   = QPixmap.fromImage(img)
        scaled = pix.scaled(
            self._video_lbl.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self._video_lbl.setPixmap(scaled)

    def _update_metrics(self, result: FrameResult) -> None:
        threat   = result.threat_level
        conf_pct = int(result.max_confidence * 100)

        # Banner
        self._banner.set_threat(threat)

        # Cards
        colours = {
            ThreatLevel.SAFE:    PALETTE["safe"],
            ThreatLevel.WARNING: PALETTE["warning"],
            ThreatLevel.FIRE:    PALETTE["danger"],
        }
        self._card_status.set_value(threat.value, colours[threat])
        self._card_conf.set_value(f"{conf_pct}%")
        self._card_fps.set_value(f"{result.fps:.1f}")

        # Per-label confidence bars
        fire_conf  = max(
            (d.confidence for d in result.detections if d.label == "FIRE"), default=0.0
        )
        smoke_conf = max(
            (d.confidence for d in result.detections if d.label == "SMOKE"), default=0.0
        )
        self._conf_bar.setValue(int(fire_conf * 100))
        self._smoke_bar.setValue(int(smoke_conf * 100))

        # Indicator LEDs
        self._set_indicator(self._ind_fire,  result.has_fire,  "FIRE",  PALETTE["danger"])
        self._set_indicator(self._ind_smoke, result.has_smoke, "SMOKE", "#AAAAAA")

        # Status bar
        det_summary = (
            ", ".join(f"{d.label}({d.confidence_pct})" for d in result.detections)
            or "No detection"
        )
        self._status.showMessage(f"FPS: {result.fps:.1f}  ·  {det_summary}")

    def _on_worker_done(self) -> None:
        self._btn_start.setEnabled(True)
        self._btn_stop.setEnabled(False)
        self._alarm.silence()
        self._status.showMessage("Stream ended")

    # ── Style helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _make_btn(text: str, bg: str, fg: str) -> QPushButton:
        btn = QPushButton(text)
        btn.setFixedHeight(36)
        btn.setCursor(Qt.PointingHandCursor)
        btn.setStyleSheet(
            f"QPushButton{{background:{bg};color:{fg};border:none;"
            f"border-radius:5px;padding:0 14px;font-size:12px;font-weight:bold;}}"
            f"QPushButton:hover{{opacity:0.85;}}"
            f"QPushButton:disabled{{background:#2A2A2A;color:#555;}}"
        )
        return btn

    @staticmethod
    def _make_indicator(text: str, bg: str, fg: str) -> QLabel:
        lbl = QLabel(f"  {text}  ")
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setFixedHeight(30)
        lbl.setStyleSheet(
            f"background:{bg}; color:{fg}; font-size:11px; font-weight:bold;"
            "border-radius:4px; border:1px solid #333;"
        )
        return lbl

    @staticmethod
    def _set_indicator(lbl: QLabel, active: bool, text: str, colour: str) -> None:
        if active:
            lbl.setStyleSheet(
                f"background:{colour}; color:white; font-size:11px; font-weight:bold;"
                "border-radius:4px;"
            )
            lbl.setText(f"● {text}")
        else:
            lbl.setStyleSheet(
                "background:#1A1A1A; color:#555; font-size:11px; font-weight:bold;"
                "border-radius:4px; border:1px solid #333;"
            )
            lbl.setText(f"○ {text}")

    @staticmethod
    def _combo_style() -> str:
        return (
            f"QComboBox{{background:{PALETTE['bg_card']};"
            f"color:{PALETTE['text_primary']}; border:1px solid {PALETTE['border']};"
            "border-radius:4px; padding:4px 8px;}}"
            f"QComboBox QAbstractItemView{{background:{PALETTE['bg_card']};"
            f"color:{PALETTE['text_primary']}; selection-background-color:#333;}}"
        )

    @staticmethod
    def _group_box_style() -> str:
        return (
            f"QGroupBox{{color:{PALETTE['text_secondary']}; font-size:10px;"
            f"font-weight:bold; border:1px solid {PALETTE['border']}; border-radius:6px;"
            "margin-top:10px; padding-top:6px;}}"
            "QGroupBox::title{subcontrol-origin:margin; left:8px; padding:0 4px;}"
        )

    @staticmethod
    def _progress_style(colour: str) -> str:
        return (
            f"QProgressBar{{background:#111; border:1px solid #333; border-radius:4px;"
            f"text-align:center; color:white; font-size:11px;}}"
            f"QProgressBar::chunk{{background:{colour}; border-radius:3px;}}"
        )

    # ── Dark theme ─────────────────────────────────────────────────────────

    def _apply_dark_theme(self) -> None:
        self.setStyleSheet(
            f"QMainWindow, QWidget{{background:{PALETTE['bg_dark']};"
            f"color:{PALETTE['text_primary']};}}"
            "QSplitter::handle{background:#222;}"
        )

    # ── Close event ────────────────────────────────────────────────────────

    def closeEvent(self, event) -> None:
        if self._worker and self._worker.isRunning():
            self._worker.stop()
        self._alarm.silence()
        event.accept()
