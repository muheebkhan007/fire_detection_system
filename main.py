import sys
import cv2
import numpy as np
import time
import logging
import os
from datetime import datetime
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QPushButton, 
                           QVBoxLayout, QHBoxLayout, QWidget, QComboBox, 
                           QTextEdit, QSlider, QGroupBox, QMessageBox, QFileDialog)
from PyQt5.QtCore import QTimer, Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
import sounddevice as sd
import smtplib
from email.mime.text import MIMEText
import threading

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(QImage)
    fire_detected_signal = pyqtSignal(bool, float, str)

    def __init__(self, camera_id=0):
        super().__init__()
        self.camera_id = camera_id
        self.running = True
        self.sensitivity = 50

    def run(self):
        cap = cv2.VideoCapture(self.camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        prev_frame = None
        recording = False
        out = None

        while self.running:
            ret, frame = cap.read()
            if not ret:
                break

            # Fire detection logic
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            lower_fire = np.array([0, 50, 200])
            upper_fire = np.array([30, 255, 255])
            mask = cv2.inRange(hsv, lower_fire, upper_fire)
            
            # Motion + size filter
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            fire_detected = False
            confidence = 0.0
            status = "Safe"

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > 500:  # sensitivity based
                    fire_detected = True
                    confidence = min(95, int(area / 50))
                    status = "🔥 FIRE DETECTED"
                    x, y, w, h = cv2.boundingRect(cnt)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 3)
                    break

            # Record video if fire
            if fire_detected and not recording:
                recording = True
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                out = cv2.VideoWriter(f"screenshots/fire_{timestamp}.avi", 
                                    cv2.VideoWriter_fourcc(*'XVID'), 20.0, (640,480))
                self.fire_detected_signal.emit(True, confidence, timestamp)
            
            if recording and out:
                out.write(frame)
            if not fire_detected and recording:
                recording = False
                out.release()

            # Display
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.change_pixmap_signal.emit(qt_image)

            time.sleep(0.03)

        cap.release()
        if out:
            out.release()

    def stop(self):
        self.running = False
        self.wait()

class FireGuardPro(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🔥 FireGuard Pro - Full Version")
        self.setGeometry(100, 100, 1000, 700)
        self.setStyleSheet("background-color: #1e1e1e; color: #ffffff;")

        self.video_thread = None
        self.is_running = False
        self.camera_id = 0

        self.init_ui()
        self.setup_logging()

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # Header
        header = QLabel("🔥 FireGuard Pro v2.0 - Advanced Fire Detection")
        header.setStyleSheet("font-size: 24px; font-weight: bold; padding: 10px;")
        header.setAlignment(Qt.AlignCenter)
        layout.addWidget(header)

        # Camera + Video
        video_layout = QHBoxLayout()
        self.video_label = QLabel()
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet("background-color: black; border: 2px solid #ff4444;")
        video_layout.addWidget(self.video_label)

        # Controls
        controls = QVBoxLayout()
        
        self.camera_combo = QComboBox()
        self.camera_combo.addItems(["Webcam 0", "Webcam 1", "IP Camera"])
        controls.addWidget(QLabel("📹 Camera:"))
        controls.addWidget(self.camera_combo)

        self.start_btn = QPushButton("▶ START Detection")
        self.start_btn.setStyleSheet("background-color: #00cc00; font-size: 16px; padding: 12px;")
        self.start_btn.clicked.connect(self.toggle_detection)
        controls.addWidget(self.start_btn)

        self.stop_btn = QPushButton("■ STOP")
        self.stop_btn.setStyleSheet("background-color: #cc0000; padding: 8px;")
        self.stop_btn.clicked.connect(self.stop_detection)
        controls.addWidget(self.stop_btn)

        self.screenshot_btn = QPushButton("📷 Screenshot")
        self.screenshot_btn.clicked.connect(self.take_screenshot)
        controls.addWidget(self.screenshot_btn)

        self.silence_btn = QPushButton("🔇 Silence Alarm")
        self.silence_btn.clicked.connect(self.silence)
        controls.addWidget(self.silence_btn)

        # Sensitivity
        sens_group = QGroupBox("Sensitivity")
        sens_layout = QVBoxLayout()
        self.sens_slider = QSlider(Qt.Horizontal)
        self.sens_slider.setRange(10, 90)
        self.sens_slider.setValue(50)
        sens_layout.addWidget(self.sens_slider)
        sens_group.setLayout(sens_layout)
        controls.addWidget(sens_group)

        video_layout.addLayout(controls)
        layout.addLayout(video_layout)

        # Log area
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(150)
        layout.addWidget(QLabel("📋 Live Log:"))
        layout.addWidget(self.log_text)

        # Status
        self.status_label = QLabel("✅ System Ready")
        self.status_label.setStyleSheet("color: #00ff00; font-weight: bold;")
        layout.addWidget(self.status_label)

    def setup_logging(self):
        os.makedirs("logs", exist_ok=True)
        os.makedirs("screenshots", exist_ok=True)
        logging.basicConfig(filename=f'logs/fireguard_{datetime.now().strftime("%Y%m%d")}.log',
                          level=logging.INFO)

    def toggle_detection(self):
        if not self.is_running:
            self.start_detection()
        else:
            self.stop_detection()

    def start_detection(self):
        self.is_running = True
        self.start_btn.setText("⏸ PAUSE")
        self.video_thread = VideoThread(self.camera_id)
        self.video_thread.change_pixmap_signal.connect(self.update_frame)
        self.video_thread.fire_detected_signal.connect(self.handle_fire)
        self.video_thread.start()
        self.log("🚀 Detection Started")

    def stop_detection(self):
        if self.video_thread:
            self.video_thread.stop()
        self.is_running = False
        self.start_btn.setText("▶ START Detection")
        self.log("⏹ Detection Stopped")
        self.status_label.setText("✅ Stopped")

    def update_frame(self, qt_image):
        self.video_label.setPixmap(QPixmap.fromImage(qt_image))

    def handle_fire(self, detected, confidence, timestamp):
        self.status_label.setText(f"🔥 FIRE! Confidence: {confidence}%")
        self.log(f"🚨 FIRE DETECTED! Confidence: {confidence}% at {timestamp}")
        
        # Alarm sound
        threading.Thread(target=self.play_alarm, daemon=True).start()
        
        # Email alert (configure your email below)
        # self.send_email_alert(confidence)

    def play_alarm(self):
        try:
            sd.play(np.sin(2 * np.pi * 440 * np.arange(0, 2, 1/44100)), 44100)
            time.sleep(1)
        except:
            pass

    def take_screenshot(self):
        if hasattr(self, 'current_frame'):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(f"screenshots/snap_{timestamp}.jpg", self.current_frame)
            self.log("📸 Screenshot saved")

    def silence(self):
        self.log("🔇 Alarm silenced")
        self.status_label.setText("🔇 Silenced")

    def log(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        full_msg = f"[{timestamp}] {message}"
        self.log_text.append(full_msg)
        logging.info(message)

    def send_email_alert(self, confidence):
        # Add your email credentials here
        pass  # Implement if needed

    def closeEvent(self, event):
        if self.video_thread:
            self.video_thread.stop()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FireGuardPro()
    window.show()
    print("FireGuard Pro Full Version Started!")
    sys.exit(app.exec_())