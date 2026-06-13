# 🔥 FireGuard Pro
### Real-Time Fire & Smoke Detection System  
**CPU-Optimised | Python | PyQt5 | OpenCV**

---

## Overview

FireGuard Pro is a production-grade fire and smoke detection desktop application. It processes live webcam or video footage in real time, raises audible alarms, logs all events, and auto-saves screenshots—all with zero GPU requirement.

```
┌─────────────────────────────────────────────────────────┐
│  FireGuard Pro  v1.0                                    │
│  ┌─────────────────────────────┐  ┌──────────────────┐ │
│  │    LIVE FEED (annotated)    │  │  METRICS PANEL   │ │
│  │                             │  │  Status: SAFE    │ │
│  │  [FIRE 87.3%]               │  │  FPS: 24.1       │ │
│  │  ┌──────────────────┐       │  │  Confidence: 87% │ │
│  │  │  bounding box    │       │  ├──────────────────┤ │
│  │  └──────────────────┘       │  │  EVENT LOG       │ │
│  │                             │  │  12:04 FIRE 87%  │ │
│  └─────────────────────────────┘  │  11:58 WARNING   │ │
│  ███ FIRE DETECTED ████████████   └──────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

---

## Features

| Feature | Details |
|---|---|
| **Fire Detection** | HSV multi-range colour analysis + contour convexity scoring |
| **Smoke Detection** | Background subtraction + texture analysis + circularity |
| **Threat Levels** | SAFE → WARNING → FIRE DETECTED |
| **Alarm** | Synthetic tone via `sounddevice` (no audio file needed) |
| **Auto-Screenshot** | Saved to `screenshots/` on every threat transition |
| **Event CSV Log** | Timestamped rows in `logs/events_YYYYMMDD.csv` |
| **Dark UI** | Emergency-grade PyQt5 dashboard with pulsing banner |
| **Video Sources** | Webcam index, RTSP stream URL, local video file |
| **CPU Only** | No GPU, no CUDA, runs on any modern laptop |

---

## Project Structure

```
fire_detection_system/
│
├── main.py                    # Entry point
│
├── core/
│   ├── detector.py            # Fire + smoke detection engine
│   ├── capture.py             # Background capture thread
│   └── alarm.py               # Multi-level alarm manager
│
├── ui/
│   └── dashboard.py           # PyQt5 main dashboard
│
├── utils/
│   └── event_logger.py        # CSV logger + screenshot saver
│
├── logs/                      # Auto-created – CSV event logs
├── screenshots/               # Auto-created – alert screenshots
│
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Clone / Extract
```bash
cd fire_detection_system
```

### 2. Create a Virtual Environment (Recommended)
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run
```bash
python main.py
```

---

## Usage

1. **Select source** from the toolbar dropdown (Webcam 0/1 or Video File).
2. Press **▶ START** – detection begins immediately.
3. **Threat banner** pulses red/orange on danger detection.
4. Press **🔇 Silence** to mute the alarm without stopping detection.
5. Press **📷 Screenshot** to save the current frame manually.
6. Press **■ STOP** to end the session.

---

## Detection Architecture

### Fire Detection Pipeline
```
BGR Frame
    ↓
HSV conversion
    ↓
3-range colour mask (red-orange / deep-red / yellow-white)
    ↓
Morphological open → close (noise removal)
    ↓
Contour extraction + area filter
    ↓
Confidence scoring:
    • Area weight          20%
    • Colour saturation    35%
    • Edge density         25%
    • Convexity ratio      20%
    ↓
Detection (threshold > 0.35)
```

### Smoke Detection Pipeline
```
BGR Frame
    ↓
Background subtractor (MOG2, slow learning rate)
    ↓
Low-saturation HSV mask (grey range)
    ↓
Bitwise AND → combined motion+colour mask
    ↓
Confidence scoring:
    • Area weight          25%
    • Low-texture score    35%
    • Circularity          25%
    • Aspect ratio         15%
    ↓
Detection (threshold > 0.30)
```

---

## Configuration

Key constants can be adjusted at the top of `core/detector.py`:

| Constant | Default | Effect |
|---|---|---|
| `MIN_FIRE_AREA` | 500 px² | Ignore tiny fire blobs |
| `MIN_SMOKE_AREA` | 2000 px² | Ignore small smoke puffs |
| `FIRE_LOWER_1/2/3` | HSV arrays | Tune fire colour ranges |
| Confidence thresholds | 0.35 / 0.30 | Detection sensitivity |

---

## Upgrading to YOLO Detection

To replace the CV-based engine with a YOLOv8 model trained on fire/smoke datasets:

1. `pip install ultralytics`
2. In `core/detector.py`, replace `_detect_fire` and `_detect_smoke` with:

```python
from ultralytics import YOLO

class FireSmokeDetector:
    def __init__(self, ...):
        self._model = YOLO("fire_smoke_yolov8.pt")

    def _detect_fire(self, frame):
        results = self._model(frame, classes=[0])  # class 0 = fire
        # convert results to List[Detection]
        ...
```

All other code (UI, alarm, logger) remains unchanged.

---

## Future IoT & Notification Integrations

The architecture is designed for easy extension:

```python
# utils/notifications.py  (example stub)

def send_email_alert(confidence: float) -> None:
    import smtplib, ssl
    # configure SMTP here

def send_sms_alert(confidence: float) -> None:
    from twilio.rest import Client
    # configure Twilio here

def publish_mqtt(topic: str, payload: dict) -> None:
    import paho.mqtt.publish as mqtt
    # push to IoT hub
```

Plug these into `EventLogger.log()` on threat transitions.

---

## Logs & Screenshots

| Path | Contents |
|---|---|
| `logs/fireguard_YYYYMMDD.log` | Application log (debug level to file) |
| `logs/events_YYYYMMDD.csv` | Detection events: timestamp, threat, confidence |
| `screenshots/fire_*.jpg` | Auto-saved on FIRE detection |
| `screenshots/warning_*.jpg` | Auto-saved on WARNING |
| `screenshots/manual_*.jpg` | Saved via 📷 button |

---

## System Requirements

| Component | Minimum |
|---|---|
| Python | 3.9+ |
| CPU | Dual-core 2 GHz |
| RAM | 2 GB |
| Webcam | USB / built-in (OpenCV compatible) |
| OS | Windows 10+ / Ubuntu 20.04+ / macOS 11+ |

---

## License

MIT License – free for personal and commercial use.

---

*Built with Python · OpenCV · PyQt5 · NumPy*
