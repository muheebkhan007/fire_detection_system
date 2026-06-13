# 🔥 FireGuard Pro - Advanced Fire Detection System

**Real-time Fire & Smoke Detection System** built with OpenCV + PyQt5.

## ✨ Features

- Real-time fire detection using HSV color + contour analysis
- Live webcam feed with bounding boxes
- Auto video recording when fire is detected
- Screenshot capture
- Audible alarm with silence option
- Adjustable sensitivity slider
- Dark modern UI
- Detailed logging system
- Ready for multi-camera & IP camera support

## 🚀 How to Run

```bash
# Clone karo
git clone https://github.com/muheebkhan007/fire_detection_system.git
cd fire_detection_system

# Virtual environment
python3 -m venv venv
source venv/bin/activate

# Dependencies install
pip install -r requirements.txt

# Run the app
QT_QPA_PLATFORM=xcb python main.py
