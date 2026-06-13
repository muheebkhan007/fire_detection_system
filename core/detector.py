"""
Fire & Smoke Detection Engine
------------------------------
CPU-optimized detection using HSV color analysis + contour validation.
Architecture is future-ready for YOLO/CNN model swap-in.
"""

import cv2
import numpy as np
import time
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from enum import Enum


class ThreatLevel(Enum):
    SAFE = "SAFE"
    WARNING = "WARNING"
    FIRE = "FIRE DETECTED"


@dataclass
class Detection:
    """Represents a single detection event."""
    label: str                          # "FIRE" or "SMOKE"
    confidence: float                   # 0.0 – 1.0
    bbox: Tuple[int, int, int, int]     # x, y, w, h
    area: int                           # pixel area
    timestamp: float = field(default_factory=time.time)

    @property
    def confidence_pct(self) -> str:
        return f"{self.confidence * 100:.1f}%"


@dataclass
class FrameResult:
    """Full analysis result for a single frame."""
    threat_level: ThreatLevel
    detections: List[Detection]
    annotated_frame: Optional[np.ndarray]
    fps: float
    frame_count: int

    @property
    def has_fire(self) -> bool:
        return any(d.label == "FIRE" for d in self.detections)

    @property
    def has_smoke(self) -> bool:
        return any(d.label == "SMOKE" for d in self.detections)

    @property
    def max_confidence(self) -> float:
        if not self.detections:
            return 0.0
        return max(d.confidence for d in self.detections)


class FireSmokeDetector:
    """
    CPU-optimized fire and smoke detection engine.

    Detection strategy:
      - Fire: HSV hue ranges for red/orange/yellow flame colours, validated
              by saturation, value, and region area thresholds.
      - Smoke: Grayscale texture + edge density + contour circularity.

    Architecture is model-agnostic: swap `_detect_fire` / `_detect_smoke`
    for YOLO inference calls without changing any caller code.
    """

    # ── HSV colour bounds for fire ──────────────────────────────────────────
    FIRE_LOWER_1 = np.array([0,   120,  100], dtype=np.uint8)   # red-orange
    FIRE_UPPER_1 = np.array([20,  255,  255], dtype=np.uint8)
    FIRE_LOWER_2 = np.array([160, 120,  100], dtype=np.uint8)   # deep red
    FIRE_UPPER_2 = np.array([180, 255,  255], dtype=np.uint8)
    FIRE_LOWER_3 = np.array([20,   80,  150], dtype=np.uint8)   # yellow-white
    FIRE_UPPER_3 = np.array([40,  255,  255], dtype=np.uint8)

    # ── Geometry thresholds ─────────────────────────────────────────────────
    MIN_FIRE_AREA   = 500
    MAX_FIRE_AREA   = 300_000
    MIN_SMOKE_AREA  = 2_000
    MAX_SMOKE_AREA  = 500_000

    def __init__(self, frame_width: int = 640, frame_height: int = 480):
        self.frame_width  = frame_width
        self.frame_height = frame_height
        self._frame_count = 0
        self._fps_timer   = time.time()
        self._fps_counter = 0
        self._current_fps = 0.0

        # Background subtractor for motion-aware smoke detection
        self._bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=200, varThreshold=40, detectShadows=False
        )

        # Morphological kernels – built once, reused every frame
        self._morph_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        self._morph_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    # ── Public API ───────────────────────────────────────────────────────────

    def process_frame(self, frame: np.ndarray) -> FrameResult:
        """
        Analyse a single BGR frame and return a FrameResult.
        This is the only method callers need to invoke.
        """
        self._frame_count += 1
        self._update_fps()

        resized = self._resize_frame(frame)
        fire_detections  = self._detect_fire(resized)
        smoke_detections = self._detect_smoke(resized)
        all_detections   = fire_detections + smoke_detections

        threat = self._assess_threat(fire_detections, smoke_detections)
        annotated = self._annotate_frame(resized.copy(), all_detections, threat)

        return FrameResult(
            threat_level=threat,
            detections=all_detections,
            annotated_frame=annotated,
            fps=self._current_fps,
            frame_count=self._frame_count,
        )

    # ── Fire detection ───────────────────────────────────────────────────────

    def _detect_fire(self, frame: np.ndarray) -> List[Detection]:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        mask1 = cv2.inRange(hsv, self.FIRE_LOWER_1, self.FIRE_UPPER_1)
        mask2 = cv2.inRange(hsv, self.FIRE_LOWER_2, self.FIRE_UPPER_2)
        mask3 = cv2.inRange(hsv, self.FIRE_LOWER_3, self.FIRE_UPPER_3)
        fire_mask = mask1 | mask2 | mask3

        # Remove noise, close small gaps
        fire_mask = cv2.morphologyEx(fire_mask, cv2.MORPH_OPEN,  self._morph_open)
        fire_mask = cv2.morphologyEx(fire_mask, cv2.MORPH_CLOSE, self._morph_close)

        contours, _ = cv2.findContours(
            fire_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        detections: List[Detection] = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (self.MIN_FIRE_AREA <= area <= self.MAX_FIRE_AREA):
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            roi = frame[y:y+h, x:x+w]
            confidence = self._fire_confidence(roi, area, cnt)

            if confidence > 0.35:
                detections.append(Detection(
                    label="FIRE",
                    confidence=confidence,
                    bbox=(x, y, w, h),
                    area=int(area),
                ))

        return detections

    def _fire_confidence(
        self, roi: np.ndarray, area: float, contour: np.ndarray
    ) -> float:
        """
        Compute fire confidence from multiple visual cues and blend them.
        """
        if roi.size == 0:
            return 0.0

        scores: List[float] = []

        # 1. Area score – larger regions are more likely real fire
        area_score = min(area / 15_000, 1.0)
        scores.append(area_score * 0.20)

        # 2. Colour intensity in HSV
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mean_sat = float(np.mean(hsv_roi[:, :, 1]))
        mean_val = float(np.mean(hsv_roi[:, :, 2]))
        colour_score = (mean_sat / 255) * 0.5 + (mean_val / 255) * 0.5
        scores.append(colour_score * 0.35)

        # 3. Temporal / flickering simulation via edge density
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = float(np.count_nonzero(edges)) / (roi.shape[0] * roi.shape[1] + 1e-6)
        edge_score = min(edge_density * 8, 1.0)
        scores.append(edge_score * 0.25)

        # 4. Convexity – fire blobs are often convex
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        convexity = area / (hull_area + 1e-6)
        scores.append(min(convexity, 1.0) * 0.20)

        return min(sum(scores), 1.0)

    # ── Smoke detection ──────────────────────────────────────────────────────

    def _detect_smoke(self, frame: np.ndarray) -> List[Detection]:
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (21, 21), 0)

        # Motion mask – smoke moves gradually
        fg_mask = self._bg_subtractor.apply(frame, learningRate=0.005)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self._morph_open)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, self._morph_close)

        # Smoke is grey – low saturation, mid value
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        smoke_colour_mask = cv2.inRange(
            hsv,
            np.array([0,   0,  60], dtype=np.uint8),
            np.array([180, 50, 200], dtype=np.uint8),
        )

        combined = cv2.bitwise_and(fg_mask, smoke_colour_mask)
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, self._morph_close)

        contours, _ = cv2.findContours(
            combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        detections: List[Detection] = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (self.MIN_SMOKE_AREA <= area <= self.MAX_SMOKE_AREA):
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            roi_gray = blurred[y:y+h, x:x+w]
            confidence = self._smoke_confidence(roi_gray, area, cnt)

            if confidence > 0.30:
                detections.append(Detection(
                    label="SMOKE",
                    confidence=confidence,
                    bbox=(x, y, w, h),
                    area=int(area),
                ))

        return detections

    def _smoke_confidence(
        self, roi_gray: np.ndarray, area: float, contour: np.ndarray
    ) -> float:
        if roi_gray.size == 0:
            return 0.0

        scores: List[float] = []

        # 1. Area
        scores.append(min(area / 20_000, 1.0) * 0.25)

        # 2. Low-texture (smoke is blurry, featureless)
        laplacian_var = float(cv2.Laplacian(roi_gray, cv2.CV_64F).var())
        texture_score = max(0.0, 1.0 - laplacian_var / 300)
        scores.append(texture_score * 0.35)

        # 3. Circularity – smoke billows in near-circular blobs
        perimeter = cv2.arcLength(contour, True) + 1e-6
        circularity = (4 * np.pi * area) / (perimeter ** 2)
        scores.append(min(circularity, 1.0) * 0.25)

        # 4. Aspect ratio close to 1
        x, y, w, h = cv2.boundingRect(contour)
        aspect = min(w, h) / (max(w, h) + 1e-6)
        scores.append(aspect * 0.15)

        return min(sum(scores), 1.0)

    # ── Threat assessment ────────────────────────────────────────────────────

    @staticmethod
    def _assess_threat(
        fire: List[Detection], smoke: List[Detection]
    ) -> ThreatLevel:
        if fire:
            max_conf = max(d.confidence for d in fire)
            if max_conf > 0.55:
                return ThreatLevel.FIRE
            return ThreatLevel.WARNING
        if smoke:
            max_conf = max(d.confidence for d in smoke)
            if max_conf > 0.50:
                return ThreatLevel.WARNING
        return ThreatLevel.SAFE

    # ── Annotation ───────────────────────────────────────────────────────────

    def _annotate_frame(
        self, frame: np.ndarray, detections: List[Detection], threat: ThreatLevel
    ) -> np.ndarray:
        colours = {"FIRE": (0, 60, 255), "SMOKE": (180, 180, 180)}

        for det in detections:
            x, y, w, h = det.bbox
            colour = colours.get(det.label, (255, 255, 0))
            cv2.rectangle(frame, (x, y), (x + w, y + h), colour, 2)

            label_text = f"{det.label} {det.confidence_pct}"
            (tw, th), _ = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
            )
            cv2.rectangle(frame, (x, y - th - 8), (x + tw + 4, y), colour, -1)
            cv2.putText(
                frame, label_text, (x + 2, y - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2,
            )

        # Overlay border when danger detected
        if threat != ThreatLevel.SAFE:
            border_colour = (0, 0, 255) if threat == ThreatLevel.FIRE else (0, 165, 255)
            cv2.rectangle(frame, (0, 0), (frame.shape[1]-1, frame.shape[0]-1),
                          border_colour, 4)

        return frame

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _resize_frame(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        if w != self.frame_width or h != self.frame_height:
            return cv2.resize(frame, (self.frame_width, self.frame_height),
                              interpolation=cv2.INTER_LINEAR)
        return frame

    def _update_fps(self) -> None:
        self._fps_counter += 1
        elapsed = time.time() - self._fps_timer
        if elapsed >= 1.0:
            self._current_fps = self._fps_counter / elapsed
            self._fps_counter = 0
            self._fps_timer   = time.time()
