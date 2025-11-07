"""
Abhaya AI - Production Drowsiness Detection v7.1 FIXED
CRITICAL FIX: Initialization order corrected
"""

import os
import sys
os.environ['QT_QPA_PLATFORM'] = 'xcb'

import time
import cv2
import numpy as np
import mediapipe as mp
from collections import deque
from typing import Optional, Tuple
from enum import Enum
from dataclasses import dataclass

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                              QHBoxLayout, QLabel, QPushButton, QProgressBar,
                              QGroupBox, QScrollArea, QGridLayout)
from PyQt6.QtCore import QThread, pyqtSignal, QTimer, Qt
from PyQt6.QtGui import QImage, QPixmap, QFont, QPainter, QPen, QColor


# ============================================================================
# CONFIGURATION
# ============================================================================

class AlertLevel(Enum):
    NORMAL = 0
    CAUTION = 1
    WARNING = 2
    DANGER = 3
    CRITICAL = 4


class HighRiskState(Enum):
    NONE = 0
    POST_MICROSLEEP = 1
    POST_SEVERE_DROWSINESS = 2


@dataclass
class Config:
    # Calibration
    CALIBRATION_DURATION = 15.0
    MIN_CALIBRATION_SAMPLES = 100
    MIN_VALID_BASELINE_EAR = 0.20
    MAX_VALID_BASELINE_EAR = 0.45
    CALIBRATION_OUTLIER_REMOVAL = 0.20

    # Eye detection
    EAR_CLOSED_MULTIPLIER = 0.75
    MICROSLEEP_THRESHOLD = 0.8

    # Blink detection
    BLINK_MIN_DURATION = 0.06
    BLINK_MAX_DURATION = 0.45
    SLOW_BLINK_THRESHOLD = 0.35
    BLINK_COOLDOWN = 0.04

    # PERCLOS
    PERCLOS_WINDOW = 300
    PERCLOS_CAUTION = 12.0
    PERCLOS_WARNING = 20.0
    PERCLOS_DANGER = 30.0

    # Head pose
    HEAD_NOD_THRESHOLD = 18.0
    HEAD_JERK_VELOCITY = 45.0

    # Adaptive threshold
    ADAPTIVE_PERCENTILE = 90
    ADAPTIVE_ALPHA = 0.03

    # Scoring
    SCORE_INCREASE_RATE = 10.0
    SCORE_DECAY_RATE = 5.0
    SCORE_RAPID_DECAY_RATE = 12.0

    # High-risk state
    HIGH_RISK_DURATION = 300.0
    HIGH_RISK_SCORE_FLOOR = 75

    # Symptom synergy
    SYNERGY_SLOW_BLINK_RATIO = 0.4
    SYNERGY_HEAD_INSTABILITY = 8.0
    SYNERGY_LOW_BLINK_RATE = 8.0


# ============================================================================
# DETECTOR (Same as before)
# ============================================================================

class ProductionDrowsinessDetector:
    def __init__(self):
        self.is_calibrated = False
        self.calibrating = False
        self.baseline_ear = None
        self.baseline_head_pitch = None
        self.ear_threshold = 0.20
        self.adaptive_threshold = 0.20
        self.calibration_ear_samples = []
        self.calibration_head_samples = []
        self.calibration_start_time = None

        self.current_ear = 0.25
        self.current_mar = 0.35
        self.current_head_pitch = 0.0
        self.current_head_yaw = 0.0
        self.current_head_roll = 0.0

        self.ear_buffer = deque(maxlen=Config.PERCLOS_WINDOW)
        self.time_buffer = deque(maxlen=Config.PERCLOS_WINDOW)
        self.head_pitch_buffer = deque(maxlen=300)
        self.head_yaw_buffer = deque(maxlen=300)
        self.perclos_history = deque(maxlen=600)
        self.perclos_timestamps = deque(maxlen=600)

        self.eyes_closed = False
        self.eyes_closed_start = 0.0
        self.eyes_closed_duration = 0.0
        self.last_ear = 0.25

        self.blink_count = 0
        self.slow_blink_count = 0
        self.last_blink_time = 0.0

        self.microsleep_count = 0
        self.head_nod_count = 0
        self.head_jerk_count = 0
        self.last_microsleep_time = 0.0
        self.last_head_nod_time = 0.0
        self.last_head_jerk_time = 0.0

        self.last_head_pitch = 0.0
        self.last_head_yaw = 0.0
        self.last_head_roll = 0.0
        self.head_velocity = 0.0

        self.perclos = 0.0
        self.drowsiness_score = 0.0
        self.target_score = 0.0
        self.alert_level = AlertLevel.NORMAL
        self.confidence = 0.0

        self.high_risk_state = HighRiskState.NONE
        self.high_risk_start_time = 0.0

        self.session_start = time.time()
        self.frame_count = 0
        self.fps = 0.0
        self.last_update_time = time.time()
        self.last_perclos_record_time = time.time()

        self.status_message = "System starting..."


    def start_calibration(self):
        self.calibration_ear_samples.clear()
        self.calibration_head_samples.clear()
        self.calibrating = True
        self.is_calibrated = False
        self.calibration_start_time = time.time()
        self.status_message = "Calibration started - Stay FULLY ALERT!"
        print("\n[CALIBRATION] Started - User must be FULLY ALERT")


    def add_calibration_sample(self, ear, mar, head_pitch):
        if not self.calibrating:
            return

        if 0.15 < ear < 0.50:
            self.calibration_ear_samples.append(ear)

        if head_pitch is not None and -30 < head_pitch < 30:
            self.calibration_head_samples.append(head_pitch)


    def finalize_calibration(self) -> bool:
        if len(self.calibration_ear_samples) < Config.MIN_CALIBRATION_SAMPLES:
            self.status_message = f"FAILED: {len(self.calibration_ear_samples)}/{Config.MIN_CALIBRATION_SAMPLES} samples"
            print(f"[CALIBRATION] FAILED - Insufficient samples")
            self.calibrating = False
            return False

        sorted_ears = sorted(self.calibration_ear_samples)
        cutoff = int(len(sorted_ears) * Config.CALIBRATION_OUTLIER_REMOVAL)
        clean_ears = sorted_ears[cutoff:]

        baseline = float(np.median(clean_ears))
        baseline_std = float(np.std(clean_ears))

        if baseline < Config.MIN_VALID_BASELINE_EAR:
            self.status_message = f"REJECTED: Baseline too low ({baseline:.3f}). User appears drowsy!"
            print(f"[CALIBRATION] REJECTED - Baseline EAR {baseline:.3f} < {Config.MIN_VALID_BASELINE_EAR}")
            self.calibrating = False
            return False

        if baseline > Config.MAX_VALID_BASELINE_EAR:
            self.status_message = f"REJECTED: Baseline too high ({baseline:.3f}). Check camera!"
            print(f"[CALIBRATION] REJECTED - Baseline EAR {baseline:.3f} > {Config.MAX_VALID_BASELINE_EAR}")
            self.calibrating = False
            return False

        self.baseline_ear = baseline
        self.ear_threshold = baseline * Config.EAR_CLOSED_MULTIPLIER
        self.adaptive_threshold = self.ear_threshold

        if len(self.calibration_head_samples) >= 20:
            self.baseline_head_pitch = float(np.median(self.calibration_head_samples))
        else:
            self.baseline_head_pitch = 0.0

        self.is_calibrated = True
        self.calibrating = False
        self.status_message = f"SUCCESS! Baseline: {self.baseline_ear:.3f}, Threshold: {self.ear_threshold:.3f}"

        print(f"\n[CALIBRATION] SUCCESS!")
        print(f"  Baseline EAR: {self.baseline_ear:.3f} +/- {baseline_std:.3f}")
        print(f"  Threshold: {self.ear_threshold:.3f}")
        print(f"  Head pitch: {self.baseline_head_pitch:.1f} deg")
        print(f"  Samples: {len(self.calibration_ear_samples)}")

        return True


    def update(self, ear, mar, head_pitch, head_yaw, head_roll, timestamp):
        if not self.is_calibrated:
            return

        self.frame_count += 1

        dt = timestamp - self.last_update_time

        if dt <= 0 or dt > 0.5:
            self.last_update_time = timestamp
            return

        self.last_update_time = timestamp

        self.current_ear = float(ear) if 0 < ear < 1 else self.current_ear
        self.current_mar = float(mar) if 0 < mar < 2 else self.current_mar
        self.current_head_pitch = float(head_pitch) if head_pitch is not None else self.current_head_pitch
        self.current_head_yaw = float(head_yaw) if head_yaw is not None else self.current_head_yaw
        self.current_head_roll = float(head_roll) if head_roll is not None else self.current_head_roll

        self.ear_buffer.append(self.current_ear)
        self.time_buffer.append(timestamp)
        self.head_pitch_buffer.append(self.current_head_pitch)
        self.head_yaw_buffer.append(self.current_head_yaw)

        self._update_adaptive_threshold()
        self._analyze_eye_state(timestamp, dt)
        self._analyze_head_movement(timestamp, dt)
        self._calculate_perclos()

        if timestamp - self.last_perclos_record_time >= 1.0:
            self.perclos_history.append(self.perclos)
            self.perclos_timestamps.append(timestamp)
            self.last_perclos_record_time = timestamp

        self._calculate_intelligent_score(timestamp, dt)
        self._update_alert_level(timestamp)

        if self.frame_count % 60 == 0:
            print(f"[UPDATE] Frame {self.frame_count}: EAR={self.current_ear:.3f}, "
                  f"Score={self.drowsiness_score:.1f}, PERCLOS={self.perclos:.1f}%")


    def _update_adaptive_threshold(self):
        if len(self.ear_buffer) < 100:
            return

        recent_ears = list(self.ear_buffer)[-300:]
        open_eyes = [e for e in recent_ears if e > self.adaptive_threshold * 0.8]

        if len(open_eyes) < 30:
            return

        percentile_90 = np.percentile(open_eyes, Config.ADAPTIVE_PERCENTILE)
        target_baseline = percentile_90
        target_threshold = target_baseline * Config.EAR_CLOSED_MULTIPLIER

        alpha = Config.ADAPTIVE_ALPHA
        self.adaptive_threshold = alpha * target_threshold + (1 - alpha) * self.adaptive_threshold

        min_threshold = self.ear_threshold * 0.85
        max_threshold = self.ear_threshold * 1.15
        self.adaptive_threshold = np.clip(self.adaptive_threshold, min_threshold, max_threshold)


    def _analyze_eye_state(self, timestamp, dt):
        is_closed = self.current_ear < self.adaptive_threshold

        if is_closed:
            if not self.eyes_closed:
                self.eyes_closed = True
                self.eyes_closed_start = timestamp
                self.eyes_closed_duration = 0.0
            else:
                self.eyes_closed_duration = timestamp - self.eyes_closed_start

                if self.eyes_closed_duration >= Config.MICROSLEEP_THRESHOLD:
                    if (timestamp - self.last_microsleep_time) > 2.0:
                        self.microsleep_count += 1
                        self.last_microsleep_time = timestamp

                        self.high_risk_state = HighRiskState.POST_MICROSLEEP
                        self.high_risk_start_time = timestamp

                        self.status_message = f"MICROSLEEP! Duration: {self.eyes_closed_duration:.2f}s"
                        print(f"[CRITICAL] MICROSLEEP #{self.microsleep_count} - Duration: {self.eyes_closed_duration:.2f}s")
        else:
            if self.eyes_closed:
                duration = self.eyes_closed_duration
                time_since_last = timestamp - self.last_blink_time

                if Config.BLINK_MIN_DURATION <= duration <= Config.BLINK_MAX_DURATION:
                    if time_since_last >= Config.BLINK_COOLDOWN:
                        self.blink_count += 1
                        self.last_blink_time = timestamp

                        if duration >= Config.SLOW_BLINK_THRESHOLD:
                            self.slow_blink_count += 1

                self.eyes_closed = False
                self.eyes_closed_duration = 0.0


    def _analyze_head_movement(self, timestamp, dt):
        if len(self.head_pitch_buffer) < 2:
            return

        pitch_delta = abs(self.current_head_pitch - self.last_head_pitch)
        yaw_delta = abs(self.current_head_yaw - self.last_head_yaw)
        roll_delta = abs(self.current_head_roll - self.last_head_roll)

        total_movement = np.sqrt(pitch_delta**2 + yaw_delta**2 + roll_delta**2)
        self.head_velocity = total_movement / dt

        if self.head_velocity > Config.HEAD_JERK_VELOCITY:
            if (timestamp - self.last_head_jerk_time) > 2.0:
                self.head_jerk_count += 1
                self.last_head_jerk_time = timestamp
                print(f"[HEAD] JERK - Velocity: {self.head_velocity:.1f} deg/s")

        if self.baseline_head_pitch is not None:
            pitch_deviation = self.current_head_pitch - self.baseline_head_pitch
            if pitch_deviation > Config.HEAD_NOD_THRESHOLD:
                if (timestamp - self.last_head_nod_time) > 3.0:
                    self.head_nod_count += 1
                    self.last_head_nod_time = timestamp
                    print(f"[HEAD] NOD - Pitch: {self.current_head_pitch:.1f} deg")

        self.last_head_pitch = self.current_head_pitch
        self.last_head_yaw = self.current_head_yaw
        self.last_head_roll = self.current_head_roll


    def _calculate_perclos(self):
        if len(self.ear_buffer) < 30:
            self.perclos = 0.0
            return

        closed_count = sum(1 for ear in self.ear_buffer if ear < self.adaptive_threshold)
        self.perclos = (closed_count / len(self.ear_buffer)) * 100.0


    def _calculate_intelligent_score(self, timestamp, dt):
        score = 0.0

        if self.perclos > Config.PERCLOS_CAUTION:
            perclos_score = min((self.perclos / Config.PERCLOS_DANGER) * 40, 40)
            score += perclos_score

        if self.eyes_closed:
            if self.eyes_closed_duration >= Config.MICROSLEEP_THRESHOLD:
                score += 30
            elif self.eyes_closed_duration > 0.5:
                score += 20
            elif self.eyes_closed_duration > 0.3:
                score += 10

        time_since_microsleep = timestamp - self.last_microsleep_time
        if time_since_microsleep < 30:
            score += 20 * (1 - time_since_microsleep / 30)

        time_since_nod = timestamp - self.last_head_nod_time
        if time_since_nod < 10:
            score += 5

        if self.blink_count > 0:
            slow_ratio = self.slow_blink_count / max(self.blink_count, 1)
            if slow_ratio > 0.3:
                score += 5

        synergy_score = self._detect_symptom_synergy()
        if synergy_score > 0:
            score = max(score, synergy_score)

        self.target_score = min(score, 100)

        if self.high_risk_state != HighRiskState.NONE:
            time_in_risk = timestamp - self.high_risk_start_time

            if time_in_risk < Config.HIGH_RISK_DURATION:
                self.target_score = max(self.target_score, Config.HIGH_RISK_SCORE_FLOOR)
            else:
                self.high_risk_state = HighRiskState.NONE

        if self.target_score > self.drowsiness_score:
            rate = Config.SCORE_INCREASE_RATE * dt
            self.drowsiness_score = min(self.target_score, self.drowsiness_score + rate)
        else:
            if self.target_score < 15 and not self.eyes_closed:
                rate = Config.SCORE_RAPID_DECAY_RATE * dt
            else:
                rate = Config.SCORE_DECAY_RATE * dt

            self.drowsiness_score = max(self.target_score, self.drowsiness_score - rate)

        self.drowsiness_score = np.clip(self.drowsiness_score, 0, 100)
        self.confidence = min(len(self.ear_buffer) / 100.0, 1.0)


    def _detect_symptom_synergy(self) -> float:
        synergy_score = 0.0

        if self.blink_count < 5:
            return 0.0

        slow_ratio = self.slow_blink_count / max(self.blink_count, 1)

        if len(self.head_pitch_buffer) >= 60:
            recent_pitches = list(self.head_pitch_buffer)[-60:]
            head_std = np.std(recent_pitches)
        else:
            head_std = 0.0

        if slow_ratio > Config.SYNERGY_SLOW_BLINK_RATIO and head_std > Config.SYNERGY_HEAD_INSTABILITY:
            synergy_score = max(synergy_score, 85)

        if self.perclos > 15:
            synergy_score = max(synergy_score, 75)

        time_since_nod = time.time() - self.last_head_nod_time
        if time_since_nod < 10 and slow_ratio > 0.3:
            synergy_score = max(synergy_score, 80)

        return synergy_score


    def _update_alert_level(self, timestamp):
        score = self.drowsiness_score

        if self.high_risk_state == HighRiskState.POST_MICROSLEEP:
            self.alert_level = AlertLevel.CRITICAL
        elif score >= 75 or self.perclos >= Config.PERCLOS_DANGER:
            self.alert_level = AlertLevel.DANGER
        elif score >= 55 or self.perclos >= Config.PERCLOS_WARNING:
            self.alert_level = AlertLevel.WARNING
        elif score >= 35 or self.perclos >= Config.PERCLOS_CAUTION:
            self.alert_level = AlertLevel.CAUTION
        else:
            self.alert_level = AlertLevel.NORMAL


# ============================================================================
# VIDEO THREAD (Same as before)
# ============================================================================

class VideoThread(QThread):
    frame_ready = pyqtSignal(np.ndarray)

    def __init__(self, detector):
        super().__init__()
        self.detector = detector
        self.running = True
        self.cap = None
        self.face_mesh = None

        self.LEFT_EYE = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE = [362, 385, 387, 263, 373, 380]
        self.MOUTH = [61, 146, 91, 181, 84, 17, 314, 405]


    def run(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        if not self.cap.isOpened():
            print("[ERROR] Failed to open camera!")
            return

        print("[OK] Camera opened")

        mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        print("[OK] MediaPipe initialized")

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue

            current_time = time.time()

            try:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.face_mesh.process(rgb_frame)

                if results and results.multi_face_landmarks:
                    landmarks_obj = results.multi_face_landmarks[0]
                    h, w = frame.shape[:2]

                    landmarks = np.array([[lm.x * w, lm.y * h]
                                         for lm in landmarks_obj.landmark])

                    left_ear = self.calculate_ear(landmarks[self.LEFT_EYE])
                    right_ear = self.calculate_ear(landmarks[self.RIGHT_EYE])
                    ear = (left_ear + right_ear) / 2.0

                    mar = self.calculate_mar(landmarks[self.MOUTH])
                    pitch, yaw, roll = self.calculate_head_pose(landmarks, frame.shape)

                    if self.detector.calibrating:
                        self.detector.add_calibration_sample(ear, mar, pitch)
                    elif self.detector.is_calibrated:
                        self.detector.update(ear, mar, pitch, yaw, roll, current_time)

                self.frame_ready.emit(frame)

            except Exception as e:
                print(f"[ERROR] Video processing: {e}")
                continue


    @staticmethod
    def calculate_ear(eye_points):
        try:
            if len(eye_points) != 6:
                return 0.0
            v1 = np.linalg.norm(eye_points[1] - eye_points[5])
            v2 = np.linalg.norm(eye_points[2] - eye_points[4])
            h = np.linalg.norm(eye_points[0] - eye_points[3])
            if h < 1e-6:
                return 0.0
            return (v1 + v2) / (2.0 * h)
        except:
            return 0.0


    @staticmethod
    def calculate_mar(mouth_points):
        try:
            if len(mouth_points) < 8:
                return 0.0
            v1 = np.linalg.norm(mouth_points[1] - mouth_points[7])
            v2 = np.linalg.norm(mouth_points[2] - mouth_points[6])
            v3 = np.linalg.norm(mouth_points[3] - mouth_points[5])
            h = np.linalg.norm(mouth_points[0] - mouth_points[4])
            if h < 1e-6:
                return 0.0
            return (v1 + v2 + v3) / (2.0 * h)
        except:
            return 0.0


    @staticmethod
    def calculate_head_pose(landmarks, frame_shape):
        try:
            h, w = frame_shape[:2]

            model_points = np.array([
                (0.0, 0.0, 0.0),
                (0.0, -330.0, -65.0),
                (-225.0, 170.0, -135.0),
                (225.0, 170.0, -135.0),
                (-150.0, -150.0, -125.0),
                (150.0, -150.0, -125.0)
            ], dtype=np.float64)

            image_points = np.array([
                landmarks[1], landmarks[152], landmarks[33],
                landmarks[263], landmarks[61], landmarks[291]
            ], dtype=np.float64)

            focal_length = w
            center = (w/2, h/2)
            camera_matrix = np.array([
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1]
            ], dtype=np.float64)

            dist_coeffs = np.zeros((4, 1))

            success, rot_vec, _ = cv2.solvePnP(
                model_points, image_points, camera_matrix, dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )

            if not success:
                return None, None, None

            rot_mat, _ = cv2.Rodrigues(rot_vec)

            pitch = np.degrees(np.arcsin(-rot_mat[2, 0]))
            yaw = np.degrees(np.arctan2(rot_mat[2, 1], rot_mat[2, 2]))
            roll = np.degrees(np.arctan2(rot_mat[1, 0], rot_mat[0, 0]))

            return pitch, yaw, roll
        except:
            return None, None, None


    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()
        if self.face_mesh:
            self.face_mesh.close()


# ============================================================================
# MAIN WINDOW - FIXED INITIALIZATION ORDER
# ============================================================================

class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()

        self.detector = ProductionDrowsinessDetector()

        # CRITICAL FIX: Create pixmap BEFORE init_ui()
        self.perclos_pixmap = QPixmap(500, 150)
        self.perclos_pixmap.fill(QColor(30, 30, 30))

        # NOW call init_ui (which uses self.perclos_pixmap)
        self.init_ui()

        # Start video thread
        self.video_thread = VideoThread(self.detector)
        self.video_thread.frame_ready.connect(self.update_camera)
        self.video_thread.start()

        # Update timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_ui)
        self.timer.start(50)

        # FPS tracking
        self.fps_counter = 0
        self.fps_start = time.time()


    def init_ui(self):
        self.setWindowTitle("Abhaya AI - Production Drowsiness Detection v7.1 FIXED")
        self.setGeometry(100, 100, 1600, 900)

        self.setStyleSheet("""
            QMainWindow { background-color: #2b2b2b; }
            QLabel { color: #ffffff; }
            QGroupBox {
                color: #ffffff;
                border: 2px solid #555555;
                border-radius: 5px;
                margin-top: 10px;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QPushButton {
                background-color: #0d7377;
                color: white;
                border: none;
                padding: 10px;
                font-size: 14px;
                border-radius: 5px;
            }
            QPushButton:hover { background-color: #14a085; }
            QPushButton:pressed { background-color: #0a5d61; }
            QProgressBar {
                border: 2px solid #555555;
                border-radius: 5px;
                text-align: center;
                color: white;
            }
            QProgressBar::chunk { background-color: #0d7377; }
        """)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)

        main_layout = QHBoxLayout()
        main_widget.setLayout(main_layout)

        left_panel = self.create_camera_panel()
        main_layout.addWidget(left_panel, 1)

        right_panel = self.create_stats_panel()
        main_layout.addWidget(right_panel, 1)


    def create_camera_panel(self):
        panel = QWidget()
        layout = QVBoxLayout()
        panel.setLayout(layout)

        self.camera_label = QLabel()
        self.camera_label.setMinimumSize(640, 480)
        self.camera_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.camera_label.setStyleSheet("border: 2px solid #555555; background-color: #1e1e1e;")
        layout.addWidget(self.camera_label)

        button_layout = QHBoxLayout()

        self.calibrate_button = QPushButton("CALIBRATE")
        self.calibrate_button.clicked.connect(self.start_calibration)
        button_layout.addWidget(self.calibrate_button)

        self.quit_button = QPushButton("QUIT")
        self.quit_button.clicked.connect(self.close)
        button_layout.addWidget(self.quit_button)

        layout.addLayout(button_layout)

        return panel


    def create_stats_panel(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; }")

        container = QWidget()
        layout = QVBoxLayout()
        container.setLayout(layout)

        layout.addWidget(self.create_score_group())
        layout.addWidget(self.create_alert_group())
        layout.addWidget(self.create_eye_metrics_group())
        layout.addWidget(self.create_blink_group())
        layout.addWidget(self.create_events_group())
        layout.addWidget(self.create_head_pose_group())
        layout.addWidget(self.create_perclos_graph_group())
        layout.addWidget(self.create_session_group())
        layout.addWidget(self.create_status_group())

        layout.addStretch()

        scroll.setWidget(container)
        return scroll


    def create_score_group(self):
        group = QGroupBox("DROWSINESS SCORE")
        layout = QVBoxLayout()
        group.setLayout(layout)

        self.score_label = QLabel("0")
        self.score_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        font = QFont()
        font.setPointSize(48)
        font.setBold(True)
        self.score_label.setFont(font)
        layout.addWidget(self.score_label)

        self.score_bar = QProgressBar()
        self.score_bar.setMinimum(0)
        self.score_bar.setMaximum(100)
        self.score_bar.setValue(0)
        self.score_bar.setTextVisible(False)
        self.score_bar.setMinimumHeight(30)
        layout.addWidget(self.score_bar)

        self.risk_label = QLabel("")
        self.risk_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        font2 = QFont()
        font2.setPointSize(12)
        font2.setBold(True)
        self.risk_label.setFont(font2)
        layout.addWidget(self.risk_label)

        return group


    def create_alert_group(self):
        group = QGroupBox("ALERT STATUS")
        layout = QVBoxLayout()
        group.setLayout(layout)

        self.alert_label = QLabel("NORMAL - ALERT")
        self.alert_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        font = QFont()
        font.setPointSize(20)
        font.setBold(True)
        self.alert_label.setFont(font)
        layout.addWidget(self.alert_label)

        return group


    def create_eye_metrics_group(self):
        group = QGroupBox("EYE METRICS")
        layout = QGridLayout()
        group.setLayout(layout)

        labels = [
            ("EAR (Current):", "ear_current"),
            ("EAR (Baseline):", "ear_baseline"),
            ("EAR (Adaptive):", "ear_adaptive"),
            ("PERCLOS:", "perclos"),
            ("Eye Status:", "eye_status"),
            ("Closure Duration:", "closure_duration")
        ]

        self.eye_labels = {}
        for i, (text, key) in enumerate(labels):
            label = QLabel(text)
            value = QLabel("--")
            value.setStyleSheet("color: #00ff00;")
            layout.addWidget(label, i, 0)
            layout.addWidget(value, i, 1)
            self.eye_labels[key] = value

        return group


    def create_blink_group(self):
        group = QGroupBox("BLINK ANALYSIS")
        layout = QGridLayout()
        group.setLayout(layout)

        labels = [
            ("Total Blinks:", "blink_total"),
            ("Slow Blinks:", "blink_slow"),
            ("Slow Blink Ratio:", "blink_ratio")
        ]

        self.blink_labels = {}
        for i, (text, key) in enumerate(labels):
            label = QLabel(text)
            value = QLabel("0")
            value.setStyleSheet("color: #00ff00;")
            layout.addWidget(label, i, 0)
            layout.addWidget(value, i, 1)
            self.blink_labels[key] = value

        return group


    def create_events_group(self):
        group = QGroupBox("DROWSINESS EVENTS")
        layout = QGridLayout()
        group.setLayout(layout)

        labels = [
            ("Microsleeps:", "microsleeps"),
            ("Head Nods:", "head_nods"),
            ("Head Jerks:", "head_jerks")
        ]

        self.event_labels = {}
        for i, (text, key) in enumerate(labels):
            label = QLabel(text)
            value = QLabel("0")
            value.setStyleSheet("color: #00ff00;")
            layout.addWidget(label, i, 0)
            layout.addWidget(value, i, 1)
            self.event_labels[key] = value

        return group


    def create_head_pose_group(self):
        group = QGroupBox("HEAD POSE")
        layout = QGridLayout()
        group.setLayout(layout)

        labels = [
            ("Pitch:", "head_pitch"),
            ("Yaw:", "head_yaw"),
            ("Roll:", "head_roll"),
            ("Velocity:", "head_velocity")
        ]

        self.head_labels = {}
        for i, (text, key) in enumerate(labels):
            label = QLabel(text)
            value = QLabel("0.0")
            value.setStyleSheet("color: #00ff00;")
            layout.addWidget(label, i, 0)
            layout.addWidget(value, i, 1)
            self.head_labels[key] = value

        return group


    def create_perclos_graph_group(self):
        """PERCLOS graph - now self.perclos_pixmap exists!"""
        group = QGroupBox("PERCLOS HISTORY (Last 10 minutes)")
        layout = QVBoxLayout()
        group.setLayout(layout)

        self.perclos_graph_label = QLabel()
        self.perclos_graph_label.setMinimumSize(500, 150)
        self.perclos_graph_label.setPixmap(self.perclos_pixmap)  # Now this works!
        layout.addWidget(self.perclos_graph_label)

        return group


    def create_session_group(self):
        group = QGroupBox("SESSION INFO")
        layout = QGridLayout()
        group.setLayout(layout)

        labels = [
            ("Runtime:", "runtime"),
            ("Frames:", "frames"),
            ("FPS:", "fps"),
            ("Confidence:", "confidence")
        ]

        self.session_labels = {}
        for i, (text, key) in enumerate(labels):
            label = QLabel(text)
            value = QLabel("--")
            value.setStyleSheet("color: #00ff00;")
            layout.addWidget(label, i, 0)
            layout.addWidget(value, i, 1)
            self.session_labels[key] = value

        return group


    def create_status_group(self):
        group = QGroupBox("STATUS")
        layout = QVBoxLayout()
        group.setLayout(layout)

        self.status_label = QLabel("System starting...")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("color: #ffff00;")
        layout.addWidget(self.status_label)

        return group


    def update_camera(self, frame):
        try:
            h, w = frame.shape[:2]

            if self.detector.is_calibrated:
                score = self.detector.drowsiness_score

                box_w, box_h = 200, 80
                box_x, box_y = w - box_w - 20, 20

                cv2.rectangle(frame, (box_x, box_y), (box_x+box_w, box_y+box_h), (0, 0, 0), -1)
                cv2.rectangle(frame, (box_x, box_y), (box_x+box_w, box_y+box_h), (255, 255, 255), 2)

                colors = {
                    AlertLevel.NORMAL: (0, 255, 0),
                    AlertLevel.CAUTION: (0, 255, 255),
                    AlertLevel.WARNING: (0, 165, 255),
                    AlertLevel.DANGER: (0, 0, 255),
                    AlertLevel.CRITICAL: (0, 0, 255)
                }
                color = colors[self.detector.alert_level]

                cv2.putText(frame, f"Score: {score:.0f}", (box_x+15, box_y+35),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                alert_names = ["NORMAL", "CAUTION", "WARNING", "DANGER", "CRITICAL"]
                cv2.putText(frame, alert_names[self.detector.alert_level.value], (box_x+15, box_y+65),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

                if self.detector.alert_level == AlertLevel.CRITICAL:
                    pulse = abs(np.sin(time.time() * 5))
                    alpha = 0.2 * pulse

                    overlay = frame.copy()
                    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 255), -1)
                    cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0, frame)

                    cv2.putText(frame, "MICROSLEEP - STOP NOW!", (w//2-200, h//2),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)

            elif self.detector.calibrating:
                cv2.putText(frame, "CALIBRATING - Stay alert!", (w//2-150, h//2),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            else:
                cv2.putText(frame, "PRESS CALIBRATE BUTTON", (w//2-150, h//2),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)

            scaled_pixmap = pixmap.scaled(self.camera_label.size(),
                                         Qt.AspectRatioMode.KeepAspectRatio,
                                         Qt.TransformationMode.SmoothTransformation)
            self.camera_label.setPixmap(scaled_pixmap)

            self.fps_counter += 1
            current_time = time.time()
            if current_time - self.fps_start >= 1.0:
                self.detector.fps = self.fps_counter / (current_time - self.fps_start)
                self.fps_counter = 0
                self.fps_start = current_time

        except Exception as e:
            print(f"[ERROR] Camera update: {e}")


    def update_ui(self):
        try:
            if self.detector.calibrating:
                elapsed = time.time() - self.detector.calibration_start_time
                progress = min((elapsed / Config.CALIBRATION_DURATION) * 100, 100)

                self.status_label.setText(
                    f"Calibrating... {progress:.0f}% ({len(self.detector.calibration_ear_samples)} samples)"
                )

                if elapsed >= Config.CALIBRATION_DURATION:
                    success = self.detector.finalize_calibration()
                    if success:
                        self.calibrate_button.setText("RECALIBRATE")

            if not self.detector.is_calibrated:
                return

            score = int(self.detector.drowsiness_score)
            self.score_label.setText(str(score))
            self.score_bar.setValue(score)

            if score < 35:
                color = "#00ff00"
            elif score < 55:
                color = "#ffff00"
            elif score < 75:
                color = "#ff9900"
            else:
                color = "#ff0000"

            self.score_label.setStyleSheet(f"color: {color};")
            self.score_bar.setStyleSheet(f"QProgressBar::chunk {{ background-color: {color}; }}")

            if self.detector.high_risk_state != HighRiskState.NONE:
                time_in_risk = time.time() - self.detector.high_risk_start_time
                remaining = Config.HIGH_RISK_DURATION - time_in_risk
                self.risk_label.setText(f"HIGH-RISK: {self.detector.high_risk_state.name} ({remaining:.0f}s)")
                self.risk_label.setStyleSheet("color: #ff0000; font-weight: bold;")
            else:
                self.risk_label.setText("")

            alert_names = {
                AlertLevel.NORMAL: "NORMAL - ALERT",
                AlertLevel.CAUTION: "CAUTION - EARLY FATIGUE",
                AlertLevel.WARNING: "WARNING - TAKE BREAK SOON",
                AlertLevel.DANGER: "DANGER - PULL OVER NOW",
                AlertLevel.CRITICAL: "CRITICAL - MICROSLEEP!"
            }

            self.alert_label.setText(alert_names[self.detector.alert_level])
            self.alert_label.setStyleSheet(f"color: {color}; font-size: 20px; font-weight: bold;")

            self.eye_labels['ear_current'].setText(f"{self.detector.current_ear:.3f}")
            self.eye_labels['ear_current'].setStyleSheet(
                f"color: {'#ff0000' if self.detector.current_ear < self.detector.adaptive_threshold else '#00ff00'};"
            )

            self.eye_labels['ear_baseline'].setText(f"{self.detector.baseline_ear:.3f}")
            self.eye_labels['ear_adaptive'].setText(f"{self.detector.adaptive_threshold:.3f}")
            self.eye_labels['perclos'].setText(f"{self.detector.perclos:.1f}%")

            if self.detector.eyes_closed:
                self.eye_labels['eye_status'].setText("CLOSED")
                self.eye_labels['eye_status'].setStyleSheet("color: #ff0000; font-weight: bold;")
                self.eye_labels['closure_duration'].setText(f"{self.detector.eyes_closed_duration:.2f}s")
                self.eye_labels['closure_duration'].setStyleSheet("color: #ff0000;")
            else:
                self.eye_labels['eye_status'].setText("OPEN")
                self.eye_labels['eye_status'].setStyleSheet("color: #00ff00; font-weight: bold;")
                self.eye_labels['closure_duration'].setText("0.00s")
                self.eye_labels['closure_duration'].setStyleSheet("color: #00ff00;")

            self.blink_labels['blink_total'].setText(str(self.detector.blink_count))
            self.blink_labels['blink_slow'].setText(str(self.detector.slow_blink_count))

            if self.detector.blink_count > 0:
                ratio = (self.detector.slow_blink_count / self.detector.blink_count) * 100
                self.blink_labels['blink_ratio'].setText(f"{ratio:.0f}%")
            else:
                self.blink_labels['blink_ratio'].setText("0%")

            self.event_labels['microsleeps'].setText(str(self.detector.microsleep_count))
            self.event_labels['microsleeps'].setStyleSheet(
                f"color: {'#ff0000' if self.detector.microsleep_count > 0 else '#00ff00'}; font-weight: bold;"
            )

            self.event_labels['head_nods'].setText(str(self.detector.head_nod_count))
            self.event_labels['head_jerks'].setText(str(self.detector.head_jerk_count))

            self.head_labels['head_pitch'].setText(f"{self.detector.current_head_pitch:.1f} deg")
            self.head_labels['head_yaw'].setText(f"{self.detector.current_head_yaw:.1f} deg")
            self.head_labels['head_roll'].setText(f"{self.detector.current_head_roll:.1f} deg")
            self.head_labels['head_velocity'].setText(f"{self.detector.head_velocity:.1f} deg/s")

            self.update_perclos_graph()

            runtime = time.time() - self.detector.session_start
            mins, secs = divmod(int(runtime), 60)
            self.session_labels['runtime'].setText(f"{mins:02d}:{secs:02d}")
            self.session_labels['frames'].setText(str(self.detector.frame_count))
            self.session_labels['fps'].setText(f"{self.detector.fps:.1f}")
            self.session_labels['confidence'].setText(f"{self.detector.confidence*100:.0f}%")

            self.status_label.setText(self.detector.status_message)

        except Exception as e:
            print(f"[ERROR] UI update: {e}")


    def update_perclos_graph(self):
        try:
            painter = QPainter(self.perclos_pixmap)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)

            painter.fillRect(0, 0, 500, 150, QColor(30, 30, 30))

            thresholds = [
                (Config.PERCLOS_CAUTION, QColor(0, 255, 0)),
                (Config.PERCLOS_WARNING, QColor(255, 255, 0)),
                (Config.PERCLOS_DANGER, QColor(255, 0, 0))
            ]

            for threshold, color in thresholds:
                y = 150 - int((threshold / 50.0) * 150)
                painter.setPen(QPen(color, 1))
                painter.drawLine(0, y, 500, y)

            if len(self.detector.perclos_history) > 1:
                painter.setPen(QPen(QColor(0, 200, 255), 2))

                max_points = min(600, len(self.detector.perclos_history))

                for i in range(1, max_points):
                    idx1 = len(self.detector.perclos_history) - max_points + i - 1
                    idx2 = len(self.detector.perclos_history) - max_points + i

                    if idx1 < 0 or idx2 < 0:
                        continue

                    x1 = int((i - 1) / max_points * 500)
                    x2 = int(i / max_points * 500)

                    perclos1 = self.detector.perclos_history[idx1]
                    perclos2 = self.detector.perclos_history[idx2]

                    y1 = 150 - int((perclos1 / 50.0) * 150)
                    y2 = 150 - int((perclos2 / 50.0) * 150)

                    painter.drawLine(x1, y1, x2, y2)

            painter.end()

            self.perclos_graph_label.setPixmap(self.perclos_pixmap)

        except Exception as e:
            print(f"[ERROR] PERCLOS graph: {e}")


    def start_calibration(self):
        if not self.detector.calibrating:
            self.detector.start_calibration()
            self.calibrate_button.setText("CALIBRATING...")
            self.calibrate_button.setEnabled(False)

            QTimer.singleShot(int(Config.CALIBRATION_DURATION * 1000) + 500,
                            lambda: self.calibrate_button.setEnabled(True))


    def closeEvent(self, event):
        self.video_thread.stop()
        self.video_thread.wait()

        if self.detector.is_calibrated:
            runtime = time.time() - self.detector.session_start
            mins, secs = divmod(int(runtime), 60)

            print("\n" + "="*70)
            print(" SESSION SUMMARY ".center(70, '='))
            print("="*70)
            print(f"\nDuration: {mins}m {secs}s")
            print(f"Frames:   {self.detector.frame_count}")
            print(f"\nFinal Score: {self.detector.drowsiness_score:.1f}/100")
            print(f"PERCLOS:     {self.detector.perclos:.1f}%")
            print(f"\nCritical Events:")
            print(f"  Microsleeps: {self.detector.microsleep_count}")
            print(f"  Head Nods:   {self.detector.head_nod_count}")
            print(f"  Head Jerks:  {self.detector.head_jerk_count}")
            print(f"\nBlink Stats:")
            print(f"  Total:       {self.detector.blink_count}")
            print(f"  Slow:        {self.detector.slow_blink_count}")

            if self.detector.microsleep_count > 0:
                print(f"\n[CRITICAL] {self.detector.microsleep_count} microsleep(s) detected!")
            else:
                print(f"\n[OK] Stay alert!")
            print("="*70 + "\n")

        event.accept()


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "="*70)
    print(" Abhaya AI v7.1 - INITIALIZATION FIX ".center(70, '='))
    print("="*70)
    print("\nFIXED:")
    print("  [OK] Pixmap initialization order corrected")
    print("\nALL FEATURES:")
    print("  [OK] Calibration validation")
    print("  [OK] Adaptive threshold (90th percentile)")
    print("  [OK] High-risk state memory")
    print("  [OK] Symptom synergy detection")
    print("  [OK] Proper velocity calculations")
    print("  [OK] PERCLOS history graph")
    print("\n" + "="*70 + "\n")

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
