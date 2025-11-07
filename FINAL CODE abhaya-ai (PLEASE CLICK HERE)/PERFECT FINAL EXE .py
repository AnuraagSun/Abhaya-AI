"""
Abhaya AI - Qt6 Drowsiness Detection v8.0 ULTIMATE INTELLIGENT

INTELLIGENT ALGORITHMS:
✓ BLINK: Direct state tracking - every CLOSED→OPEN = 1 blink
✓ YAWN: State machine - discrete yawn events only
✓ HEAD TILT: Window-based sustained detection
✓ SCORING: Multi-factor weighted comprehensive analysis
✓ ALL METRICS: Maximum accuracy with intelligent filtering
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

# Qt6 imports
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


class Config:
    # Eye closure
    EAR_BASELINE_MULTIPLIER = 0.75

    # Blink detection - MINIMAL filtering for maximum accuracy
    BLINK_MIN_DURATION = 0.03       # Minimum 30ms (sensor noise threshold)
    BLINK_MAX_DURATION = 0.6        # Maximum 600ms
    SLOW_BLINK_THRESHOLD = 0.25     # Slow if >= 250ms

    # Microsleep
    MICROSLEEP_THRESHOLD = 1.2      # Eyes closed >= 1.2 seconds

    # Yawn detection
    MAR_YAWN_THRESHOLD = 0.6        # Mouth aspect ratio for yawn
    YAWN_MIN_DURATION = 0.5         # Minimum yawn duration
    YAWN_MAX_DURATION = 4.0         # Maximum yawn duration
    YAWN_COOLDOWN = 2.5             # Cooldown between yawns

    # Head tilt detection (window-based)
    TILT_WINDOW_SIZE = 30           # 1 second at 30fps
    TILT_ROLL_THRESHOLD = 18.0      # Lateral tilt threshold
    TILT_PITCH_THRESHOLD = 22.0     # Forward droop threshold
    TILT_MIN_DURATION = 1.0         # Sustained for 1 second

    # Head jerk detection
    JERK_VELOCITY_THRESHOLD = 35.0  # deg/s
    JERK_DISPLACEMENT_THRESHOLD = 12.0  # degrees
    JERK_COOLDOWN = 2.5             # seconds

    # PERCLOS
    PERCLOS_WINDOW = 300
    PERCLOS_CAUTION = 15.0
    PERCLOS_WARNING = 25.0
    PERCLOS_DANGER = 35.0

    # Blink rate
    NORMAL_BLINK_RATE_MIN = 12      # blinks/min
    NORMAL_BLINK_RATE_MAX = 20      # blinks/min

    # Calibration
    CALIBRATION_DURATION = 15.0
    MIN_CALIBRATION_SAMPLES = 50


# ============================================================================
# INTELLIGENT BLINK DETECTOR
# Direct state tracking - MAXIMUM ACCURACY
# ============================================================================

class IntelligentBlinkDetector:
    """
    ULTRA-SIMPLE BLINK DETECTION

    Logic: Track eye state (OPEN/CLOSED) directly
    Every CLOSED → OPEN transition = 1 blink

    No complex algorithms - just direct state tracking
    """

    def __init__(self):
        # Simple state tracking
        self.was_closed = False
        self.is_currently_closed = False

        # Timing
        self.closure_start_time = 0.0
        self.closure_duration = 0.0

        # Counters
        self.total_blinks = 0
        self.slow_blinks = 0

        # Last blink info
        self.last_blink_time = 0.0
        self.last_blink_duration = 0.0

        print("[BLINK DETECTOR] Initialized - Direct state tracking")


    def update(self, is_closed, timestamp):
        """
        Update with current eye state

        Args:
            is_closed: Boolean - are eyes currently closed?
            timestamp: Current time

        Returns:
            blink_detected: Boolean - was a blink just completed?
        """
        self.is_currently_closed = is_closed
        blink_detected = False

        # ===== EYES JUST CLOSED =====
        if is_closed and not self.was_closed:
            # Transition: OPEN → CLOSED
            self.closure_start_time = timestamp
            self.closure_duration = 0.0
            print(f"[BLINK] Eyes CLOSED")

        # ===== EYES STILL CLOSED =====
        elif is_closed and self.was_closed:
            # Update closure duration
            self.closure_duration = timestamp - self.closure_start_time

        # ===== EYES JUST OPENED =====
        elif not is_closed and self.was_closed:
            # Transition: CLOSED → OPEN
            final_duration = timestamp - self.closure_start_time

            print(f"[BLINK] Eyes OPENED - Duration: {final_duration:.3f}s")

            # Validate duration (filter sensor noise)
            if Config.BLINK_MIN_DURATION <= final_duration <= Config.BLINK_MAX_DURATION:
                # VALID BLINK!
                self.total_blinks += 1
                self.last_blink_time = timestamp
                self.last_blink_duration = final_duration
                blink_detected = True

                # Check if slow blink
                if final_duration >= Config.SLOW_BLINK_THRESHOLD:
                    self.slow_blinks += 1
                    print(f"[BLINK] *** SLOW BLINK #{self.total_blinks} - {final_duration:.3f}s ***")
                else:
                    print(f"[BLINK] *** BLINK #{self.total_blinks} - {final_duration:.3f}s ***")

            elif final_duration < Config.BLINK_MIN_DURATION:
                print(f"[BLINK] Too fast ({final_duration:.3f}s) - Sensor noise, ignored")

            else:  # final_duration > BLINK_MAX_DURATION
                print(f"[BLINK] Too long ({final_duration:.3f}s) - Microsleep, not blink")

            # Reset
            self.closure_duration = 0.0

        # Update state
        self.was_closed = is_closed

        return blink_detected


# ============================================================================
# INTELLIGENT YAWN DETECTOR
# State machine for discrete yawn events
# ============================================================================

class IntelligentYawnDetector:
    """
    YAWN DETECTION - State Machine

    A yawn is:
    1. Mouth CLOSED (MAR < threshold)
    2. Mouth OPENS wide (MAR > threshold)
    3. Stays OPEN for 0.5-4 seconds
    4. Mouth CLOSES again

    Only count complete cycles to avoid exponential counting
    """

    def __init__(self):
        # State machine: CLOSED, OPENING, OPEN, CLOSING
        self.state = "CLOSED"

        # Timing
        self.open_start_time = 0.0
        self.open_duration = 0.0

        # Counters
        self.yawn_count = 0
        self.last_yawn_time = 0.0

        # Current MAR
        self.current_mar = 0.0

        print("[YAWN DETECTOR] Initialized - State machine")


    def update(self, mar, timestamp):
        """
        Update with current MAR (Mouth Aspect Ratio)

        Args:
            mar: Mouth aspect ratio (0.0 - 1.0+)
            timestamp: Current time

        Returns:
            yawn_detected: Boolean - was a yawn just completed?
        """
        self.current_mar = mar
        yawn_detected = False

        is_mouth_open = mar > Config.MAR_YAWN_THRESHOLD

        # ===== STATE MACHINE =====

        if self.state == "CLOSED":
            # Mouth is closed - waiting for yawn to start
            if is_mouth_open:
                # Mouth just opened wide
                self.state = "OPENING"
                self.open_start_time = timestamp
                self.open_duration = 0.0
                print(f"[YAWN] Mouth OPENING - MAR: {mar:.3f}")

        elif self.state == "OPENING":
            # Mouth is opening - wait for it to stabilize
            if is_mouth_open:
                self.open_duration = timestamp - self.open_start_time

                # If open for enough time, transition to OPEN
                if self.open_duration > 0.2:  # 200ms
                    self.state = "OPEN"
                    print(f"[YAWN] Mouth OPEN - MAR: {mar:.3f}")
            else:
                # Mouth closed again too quickly - not a yawn
                self.state = "CLOSED"
                print(f"[YAWN] False start - mouth closed too quickly")

        elif self.state == "OPEN":
            # Mouth is wide open - waiting for it to close
            if is_mouth_open:
                # Still open - update duration
                self.open_duration = timestamp - self.open_start_time

                # Check if open too long (not a yawn anymore)
                if self.open_duration > Config.YAWN_MAX_DURATION:
                    self.state = "CLOSED"
                    print(f"[YAWN] Open too long ({self.open_duration:.1f}s) - Reset")
            else:
                # Mouth just closed - check if valid yawn
                final_duration = timestamp - self.open_start_time

                print(f"[YAWN] Mouth CLOSING - Duration: {final_duration:.3f}s")

                # Validate yawn duration
                if Config.YAWN_MIN_DURATION <= final_duration <= Config.YAWN_MAX_DURATION:
                    # Check cooldown
                    time_since_last = timestamp - self.last_yawn_time

                    if time_since_last >= Config.YAWN_COOLDOWN:
                        # VALID YAWN!
                        self.yawn_count += 1
                        self.last_yawn_time = timestamp
                        yawn_detected = True
                        print(f"[YAWN] *** YAWN #{self.yawn_count} DETECTED - {final_duration:.2f}s ***")
                    else:
                        print(f"[YAWN] Cooldown - {time_since_last:.1f}s since last yawn")

                # Return to closed state
                self.state = "CLOSED"

        return yawn_detected


# ============================================================================
# INTELLIGENT HEAD TILT DETECTOR
# Window-based sustained detection
# ============================================================================

class IntelligentHeadTiltDetector:
    """
    HEAD TILT DETECTION - Window-Based

    A tilt is:
    1. Head deviates from baseline by > threshold
    2. Sustained for > 1 second (not just a momentary spike)

    Uses sliding window of recent samples
    """

    def __init__(self):
        # Baseline (set during calibration)
        self.baseline_pitch = 0.0
        self.baseline_yaw = 0.0
        self.baseline_roll = 0.0

        # Sliding window buffers
        self.pitch_window = deque(maxlen=Config.TILT_WINDOW_SIZE)
        self.yaw_window = deque(maxlen=Config.TILT_WINDOW_SIZE)
        self.roll_window = deque(maxlen=Config.TILT_WINDOW_SIZE)
        self.time_window = deque(maxlen=Config.TILT_WINDOW_SIZE)

        # Tilt state
        self.is_tilted = False
        self.tilt_start_time = 0.0

        self.is_drooped = False
        self.droop_start_time = 0.0

        # Counters
        self.tilt_count = 0
        self.droop_count = 0
        self.last_tilt_time = 0.0
        self.last_droop_time = 0.0

        print("[TILT DETECTOR] Initialized - Window-based sustained detection")


    def set_baseline(self, pitch, yaw, roll):
        """Set baseline head pose"""
        self.baseline_pitch = pitch
        self.baseline_yaw = yaw
        self.baseline_roll = roll
        print(f"[TILT] Baseline set - Pitch: {pitch:.1f}, Yaw: {yaw:.1f}, Roll: {roll:.1f}")


    def update(self, pitch, yaw, roll, timestamp):
        """
        Update with current head pose

        Args:
            pitch, yaw, roll: Head angles (degrees)
            timestamp: Current time

        Returns:
            (tilt_detected, droop_detected): Tuple of booleans
        """
        if pitch is None or yaw is None or roll is None:
            return False, False

        # Add to windows
        self.pitch_window.append(pitch)
        self.yaw_window.append(yaw)
        self.roll_window.append(roll)
        self.time_window.append(timestamp)

        # Need enough samples
        if len(self.pitch_window) < Config.TILT_WINDOW_SIZE:
            return False, False

        # Calculate MEDIAN over window (robust to spikes)
        median_pitch = float(np.median(list(self.pitch_window)))
        median_yaw = float(np.median(list(self.yaw_window)))
        median_roll = float(np.median(list(self.roll_window)))

        # Calculate deviations from baseline
        pitch_deviation = abs(median_pitch - self.baseline_pitch)
        roll_deviation = abs(median_roll - self.baseline_roll)

        # Time span of window
        window_duration = timestamp - self.time_window[0]

        tilt_detected = False
        droop_detected = False

        # ===== LATERAL TILT (ROLL) =====
        if roll_deviation > Config.TILT_ROLL_THRESHOLD:
            if not self.is_tilted:
                # Tilt just started
                self.is_tilted = True
                self.tilt_start_time = timestamp
                print(f"[TILT] Lateral tilt START - Roll deviation: {roll_deviation:.1f} deg")
            else:
                # Tilt continuing - check duration
                tilt_duration = timestamp - self.tilt_start_time

                if tilt_duration >= Config.TILT_MIN_DURATION:
                    # Sustained tilt - check cooldown
                    time_since_last = timestamp - self.last_tilt_time

                    if time_since_last >= 5.0:  # 5 second cooldown
                        self.tilt_count += 1
                        self.last_tilt_time = timestamp
                        tilt_detected = True
                        print(f"[TILT] *** LATERAL TILT #{self.tilt_count} - {roll_deviation:.1f} deg for {tilt_duration:.1f}s ***")
        else:
            # No tilt
            if self.is_tilted:
                print(f"[TILT] Lateral tilt ENDED")
            self.is_tilted = False

        # ===== FORWARD DROOP (PITCH) =====
        # Only check if pitch is ABOVE baseline (head tilting forward)
        pitch_droop = median_pitch - self.baseline_pitch

        if pitch_droop > Config.TILT_PITCH_THRESHOLD:
            if not self.is_drooped:
                # Droop just started
                self.is_drooped = True
                self.droop_start_time = timestamp
                print(f"[TILT] Forward droop START - Pitch deviation: {pitch_droop:.1f} deg")
            else:
                # Droop continuing - check duration
                droop_duration = timestamp - self.droop_start_time

                if droop_duration >= Config.TILT_MIN_DURATION:
                    # Sustained droop - check cooldown
                    time_since_last = timestamp - self.last_droop_time

                    if time_since_last >= 5.0:  # 5 second cooldown
                        self.droop_count += 1
                        self.last_droop_time = timestamp
                        droop_detected = True
                        print(f"[TILT] *** FORWARD DROOP #{self.droop_count} - {pitch_droop:.1f} deg for {droop_duration:.1f}s ***")
        else:
            # No droop
            if self.is_drooped:
                print(f"[TILT] Forward droop ENDED")
            self.is_drooped = False

        return tilt_detected, droop_detected


# ============================================================================
# INTELLIGENT HEAD JERK DETECTOR
# Velocity + displacement based
# ============================================================================

class IntelligentHeadJerkDetector:
    """
    HEAD JERK DETECTION

    A jerk is:
    1. Sudden rapid movement (high velocity)
    2. Significant displacement
    3. Not just camera shake (uses smoothing)
    """

    def __init__(self):
        # Smoothing buffers
        self.pitch_buffer = deque(maxlen=10)
        self.yaw_buffer = deque(maxlen=10)
        self.roll_buffer = deque(maxlen=10)
        self.time_buffer = deque(maxlen=10)

        # Previous smoothed values
        self.prev_pitch = 0.0
        self.prev_yaw = 0.0
        self.prev_roll = 0.0
        self.prev_time = 0.0

        # Counters
        self.jerk_count = 0
        self.last_jerk_time = 0.0

        print("[JERK DETECTOR] Initialized - Velocity + displacement based")


    def update(self, pitch, yaw, roll, timestamp):
        """
        Update with current head pose

        Args:
            pitch, yaw, roll: Head angles (degrees)
            timestamp: Current time

        Returns:
            jerk_detected: Boolean
        """
        if pitch is None or yaw is None or roll is None:
            return False

        # Add to buffers
        self.pitch_buffer.append(pitch)
        self.yaw_buffer.append(yaw)
        self.roll_buffer.append(roll)
        self.time_buffer.append(timestamp)

        # Need enough samples
        if len(self.pitch_buffer) < 5:
            self.prev_pitch = pitch
            self.prev_yaw = yaw
            self.prev_roll = roll
            self.prev_time = timestamp
            return False

        # Smooth using median (last 5 samples)
        smooth_pitch = float(np.median(list(self.pitch_buffer)[-5:]))
        smooth_yaw = float(np.median(list(self.yaw_buffer)[-5:]))
        smooth_roll = float(np.median(list(self.roll_buffer)[-5:]))

        # Calculate time delta
        dt = timestamp - self.prev_time
        if dt < 0.001 or dt > 0.5:
            dt = 0.033

        # Calculate displacement
        pitch_disp = abs(smooth_pitch - self.prev_pitch)
        yaw_disp = abs(smooth_yaw - self.prev_yaw)
        roll_disp = abs(smooth_roll - self.prev_roll)

        total_displacement = np.sqrt(pitch_disp**2 + yaw_disp**2 + roll_disp**2)

        # Calculate velocity (deg/s)
        pitch_vel = pitch_disp / dt
        yaw_vel = yaw_disp / dt
        roll_vel = roll_disp / dt

        total_velocity = np.sqrt(pitch_vel**2 + yaw_vel**2 + roll_vel**2)

        # Detect jerk
        jerk_detected = False

        if (total_velocity > Config.JERK_VELOCITY_THRESHOLD and
            total_displacement > Config.JERK_DISPLACEMENT_THRESHOLD):

            time_since_last = timestamp - self.last_jerk_time

            if time_since_last >= Config.JERK_COOLDOWN:
                self.jerk_count += 1
                self.last_jerk_time = timestamp
                jerk_detected = True
                print(f"[JERK] *** HEAD JERK #{self.jerk_count} ***")
                print(f"       Velocity: {total_velocity:.1f} deg/s")
                print(f"       Displacement: {total_displacement:.1f} deg")

        # Update previous
        self.prev_pitch = smooth_pitch
        self.prev_yaw = smooth_yaw
        self.prev_roll = smooth_roll
        self.prev_time = timestamp

        return jerk_detected


# ============================================================================
# COMPREHENSIVE DROWSINESS SCORER
# Multi-factor weighted analysis
# ============================================================================

class IntelligentDrowsinessScorer:
    """
    COMPREHENSIVE DROWSINESS SCORING

    Factors (weighted):
    1. PERCLOS (25%) - Eye closure percentage
    2. Blink rate (10%) - Too slow = drowsy
    3. Slow blink ratio (10%) - High ratio = drowsy
    4. Microsleeps (20%) - Critical indicator
    5. Head droop (15%) - Forward tilt
    6. Head tilt (5%) - Lateral tilt
    7. Yawning (10%) - Fatigue
    8. Head jerks (5%) - Fighting sleep
    """

    def __init__(self):
        self.score = 0.0

        print("[SCORER] Initialized - Multi-factor weighted analysis")


    def calculate(self,
                  perclos,
                  blink_rate,
                  slow_blink_ratio,
                  microsleep_count,
                  last_microsleep_time,
                  droop_count,
                  last_droop_time,
                  tilt_count,
                  last_tilt_time,
                  yawn_count,
                  last_yawn_time,
                  jerk_count,
                  last_jerk_time,
                  timestamp):
        """
        Calculate comprehensive drowsiness score (0-100)

        Returns: score (0-100)
        """
        score = 0.0

        # 1. PERCLOS (25 points)
        if perclos > Config.PERCLOS_CAUTION:
            perclos_score = min((perclos / Config.PERCLOS_DANGER) * 25, 25)
            score += perclos_score

        # 2. Blink rate (10 points)
        # Normal: 12-20 blinks/min
        # Too slow = drowsy
        if 0 < blink_rate < Config.NORMAL_BLINK_RATE_MIN:
            blink_rate_score = 10 * (1 - blink_rate / Config.NORMAL_BLINK_RATE_MIN)
            score += blink_rate_score

        # 3. Slow blink ratio (10 points)
        # High ratio = many slow blinks = drowsy
        if slow_blink_ratio > 0.3:
            slow_blink_score = min((slow_blink_ratio / 0.7) * 10, 10)
            score += slow_blink_score

        # 4. Recent microsleeps (20 points)
        if microsleep_count > 0:
            time_since_microsleep = timestamp - last_microsleep_time
            if time_since_microsleep < 30:
                microsleep_score = 20 * (1 - time_since_microsleep / 30)
                score += microsleep_score

        # 5. Head droop (15 points)
        if droop_count > 0:
            time_since_droop = timestamp - last_droop_time
            if time_since_droop < 15:
                droop_score = 15 * (1 - time_since_droop / 15)
                score += droop_score

        # 6. Head tilt (5 points)
        if tilt_count > 0:
            time_since_tilt = timestamp - last_tilt_time
            if time_since_tilt < 15:
                tilt_score = 5 * (1 - time_since_tilt / 15)
                score += tilt_score

        # 7. Yawning (10 points)
        if yawn_count > 0:
            time_since_yawn = timestamp - last_yawn_time
            if time_since_yawn < 20:
                yawn_score = 10 * (1 - time_since_yawn / 20)
                score += yawn_score

        # 8. Head jerks (5 points)
        # Jerks = fighting sleep
        if jerk_count > 0:
            time_since_jerk = timestamp - last_jerk_time
            if time_since_jerk < 10:
                jerk_score = 5 * (1 - time_since_jerk / 10)
                score += jerk_score

        return min(score, 100)


# ============================================================================
# MAIN DROWSINESS DETECTOR
# Integrates all intelligent sub-detectors
# ============================================================================

class DrowsinessDetector:
    """
    Main detector integrating all intelligent algorithms
    """

    def __init__(self):
        # Calibration
        self.is_calibrated = False
        self.baseline_ear = None
        self.ear_threshold = 0.20
        self.calibration_ear_samples = []
        self.calibration_pitch_samples = []
        self.calibration_yaw_samples = []
        self.calibration_roll_samples = []

        # Current state
        self.current_ear = 0.25
        self.current_mar = 0.35
        self.current_head_pitch = 0.0
        self.current_head_yaw = 0.0
        self.current_head_roll = 0.0

        # Buffers
        self.ear_buffer = deque(maxlen=Config.PERCLOS_WINDOW)
        self.time_buffer = deque(maxlen=Config.PERCLOS_WINDOW)

        # Eye state
        self.eyes_currently_closed = False
        self.eye_closure_start = 0.0
        self.eye_closure_duration = 0.0

        # Blink tracking (for rate calculation)
        self.blink_timestamps = deque(maxlen=100)
        self.blink_rate = 0.0  # blinks per minute

        # Microsleep tracking
        self.microsleep_count = 0
        self.last_microsleep_time = 0.0
        self.in_microsleep = False

        # ===== INITIALIZE INTELLIGENT SUB-DETECTORS =====
        self.blink_detector = IntelligentBlinkDetector()
        self.yawn_detector = IntelligentYawnDetector()
        self.tilt_detector = IntelligentHeadTiltDetector()
        self.jerk_detector = IntelligentHeadJerkDetector()
        self.scorer = IntelligentDrowsinessScorer()

        # Metrics
        self.perclos = 0.0
        self.drowsiness_score = 0.0
        self.alert_level = AlertLevel.NORMAL
        self.confidence = 0.0

        # Session
        self.session_start = time.time()
        self.frame_count = 0
        self.fps = 0.0
        self.last_update_time = time.time()

        # Status
        self.status_message = "System starting..."

        print("[DETECTOR] Initialized with intelligent sub-detectors")


    def start_calibration(self):
        """Start calibration"""
        self.calibration_ear_samples.clear()
        self.calibration_pitch_samples.clear()
        self.calibration_yaw_samples.clear()
        self.calibration_roll_samples.clear()
        self.is_calibrated = False
        self.status_message = "Calibration started - Stay alert and look straight!"
        print("\n[CALIBRATION] Started")


    def add_calibration_sample(self, ear, mar, head_pitch, head_yaw, head_roll):
        """Add calibration sample"""
        if 0.15 < ear < 0.50:
            self.calibration_ear_samples.append(ear)

        if head_pitch is not None and -30 < head_pitch < 30:
            self.calibration_pitch_samples.append(head_pitch)

        if head_yaw is not None and -30 < head_yaw < 30:
            self.calibration_yaw_samples.append(head_yaw)

        if head_roll is not None and -30 < head_roll < 30:
            self.calibration_roll_samples.append(head_roll)


    def finalize_calibration(self) -> bool:
        """Finalize calibration"""
        if len(self.calibration_ear_samples) < Config.MIN_CALIBRATION_SAMPLES:
            self.status_message = f"FAILED: Only {len(self.calibration_ear_samples)} samples"
            return False

        # EAR baseline
        sorted_ears = sorted(self.calibration_ear_samples)
        cutoff = int(len(sorted_ears) * 0.2)
        clean_ears = sorted_ears[cutoff:]
        self.baseline_ear = float(np.median(clean_ears))
        self.ear_threshold = self.baseline_ear * Config.EAR_BASELINE_MULTIPLIER

        # Head pose baselines
        baseline_pitch = float(np.median(self.calibration_pitch_samples)) if len(self.calibration_pitch_samples) >= 10 else 0.0
        baseline_yaw = float(np.median(self.calibration_yaw_samples)) if len(self.calibration_yaw_samples) >= 10 else 0.0
        baseline_roll = float(np.median(self.calibration_roll_samples)) if len(self.calibration_roll_samples) >= 10 else 0.0

        # Set baseline in tilt detector
        self.tilt_detector.set_baseline(baseline_pitch, baseline_yaw, baseline_roll)

        self.is_calibrated = True
        self.status_message = f"SUCCESS! Baseline EAR: {self.baseline_ear:.3f}"

        print(f"\n[CALIBRATION] SUCCESS")
        print(f"  EAR Baseline: {self.baseline_ear:.3f}")
        print(f"  EAR Threshold: {self.ear_threshold:.3f}")

        return True


    def update(self, ear, mar, head_pitch, head_yaw, head_roll, timestamp):
        """
        MAIN UPDATE - Processes all metrics
        """
        if not self.is_calibrated:
            return

        self.frame_count += 1

        # Calculate dt
        dt = timestamp - self.last_update_time
        if dt <= 0 or dt > 0.5:
            dt = 0.033
        self.last_update_time = timestamp

        # Update current values
        if 0.05 < ear < 1.0:
            self.current_ear = float(ear)

        if 0.1 < mar < 2.0:
            self.current_mar = float(mar)

        if head_pitch is not None and -90 < head_pitch < 90:
            self.current_head_pitch = float(head_pitch)

        if head_yaw is not None and -90 < head_yaw < 90:
            self.current_head_yaw = float(head_yaw)

        if head_roll is not None and -90 < head_roll < 90:
            self.current_head_roll = float(head_roll)

        # Add to buffers
        self.ear_buffer.append(self.current_ear)
        self.time_buffer.append(timestamp)

        # ===== DETERMINE EYE STATE =====
        is_closed = self.current_ear < self.ear_threshold

        # Track closure state
        if is_closed and not self.eyes_currently_closed:
            # Eyes just closed
            self.eye_closure_start = timestamp
            self.eye_closure_duration = 0.0
            self.eyes_currently_closed = True
        elif is_closed and self.eyes_currently_closed:
            # Still closed
            self.eye_closure_duration = timestamp - self.eye_closure_start
        elif not is_closed and self.eyes_currently_closed:
            # Eyes just opened
            self.eyes_currently_closed = False
            self.eye_closure_duration = 0.0

        # Check for microsleep
        if self.eyes_currently_closed and self.eye_closure_duration >= Config.MICROSLEEP_THRESHOLD:
            if not self.in_microsleep:
                time_since_last = timestamp - self.last_microsleep_time
                if time_since_last > 5.0:
                    self.microsleep_count += 1
                    self.last_microsleep_time = timestamp
                    self.in_microsleep = True
                    self.status_message = f"MICROSLEEP! Duration: {self.eye_closure_duration:.2f}s"
                    print(f"[ALERT] *** MICROSLEEP #{self.microsleep_count} - {self.eye_closure_duration:.2f}s ***")
        else:
            self.in_microsleep = False

        # ===== RUN INTELLIGENT DETECTORS =====

        # 1. Blink detection
        blink_detected = self.blink_detector.update(is_closed, timestamp)
        if blink_detected:
            self.blink_timestamps.append(timestamp)

        # 2. Yawn detection
        yawn_detected = self.yawn_detector.update(self.current_mar, timestamp)
        if yawn_detected:
            self.status_message = f"Yawn detected!"

        # 3. Head tilt detection
        tilt_detected, droop_detected = self.tilt_detector.update(
            self.current_head_pitch,
            self.current_head_yaw,
            self.current_head_roll,
            timestamp
        )

        # 4. Head jerk detection
        jerk_detected = self.jerk_detector.update(
            self.current_head_pitch,
            self.current_head_yaw,
            self.current_head_roll,
            timestamp
        )

        # ===== CALCULATE METRICS =====
        self._calculate_blink_rate(timestamp)
        self._calculate_perclos()
        self._calculate_drowsiness_score(timestamp, dt)
        self._update_alert_level()

        # Debug output
        if self.frame_count % 60 == 0:
            print(f"\n[UPDATE] Frame {self.frame_count}")
            print(f"  EAR: {self.current_ear:.3f} | Closed: {self.eyes_currently_closed}")
            print(f"  Blinks: {self.blink_count} (Slow: {self.slow_blink_count})")
            print(f"  Yawns: {self.yawn_count}")
            print(f"  Tilts: {self.tilt_count} | Droops: {self.droop_count}")
            print(f"  Score: {self.drowsiness_score:.1f}")


    def _calculate_blink_rate(self, timestamp):
        """Calculate blinks per minute"""
        if len(self.blink_timestamps) >= 2:
            time_window = timestamp - self.blink_timestamps[0]
            if time_window > 0:
                blinks_in_window = len(self.blink_timestamps)
                self.blink_rate = (blinks_in_window / time_window) * 60.0
            else:
                self.blink_rate = 0.0
        else:
            self.blink_rate = 0.0


    def _calculate_perclos(self):
        """Calculate PERCLOS"""
        if len(self.ear_buffer) < 30:
            self.perclos = 0.0
            return

        closed_count = sum(1 for ear in self.ear_buffer if ear < self.ear_threshold)
        self.perclos = (closed_count / len(self.ear_buffer)) * 100.0


    def _calculate_drowsiness_score(self, timestamp, dt):
        """Calculate comprehensive drowsiness score"""
        # Get slow blink ratio
        if self.blink_count > 0:
            slow_blink_ratio = self.slow_blink_count / self.blink_count
        else:
            slow_blink_ratio = 0.0

        # Calculate raw score using intelligent scorer
        raw_score = self.scorer.calculate(
            perclos=self.perclos,
            blink_rate=self.blink_rate,
            slow_blink_ratio=slow_blink_ratio,
            microsleep_count=self.microsleep_count,
            last_microsleep_time=self.last_microsleep_time,
            droop_count=self.droop_count,
            last_droop_time=self.last_droop_time,
            tilt_count=self.tilt_count,
            last_tilt_time=self.last_tilt_time,
            yawn_count=self.yawn_count,
            last_yawn_time=self.last_yawn_time,
            jerk_count=self.jerk_count,
            last_jerk_time=self.last_jerk_time,
            timestamp=timestamp
        )

        # Smooth score changes
        if raw_score < 10:
            # Fast decay when alert
            decay = 3.0 * dt * 2.0
            self.drowsiness_score = max(0, self.drowsiness_score - decay)
        elif raw_score < self.drowsiness_score:
            # Normal decay
            decay = 3.0 * dt
            self.drowsiness_score = max(raw_score, self.drowsiness_score - decay)
        else:
            # Increase toward target
            increase = 5.0 * dt
            self.drowsiness_score = min(100, self.drowsiness_score + increase)

        # Calculate confidence
        self.confidence = min(len(self.ear_buffer) / 100.0, 1.0)


    def _update_alert_level(self):
        """Update alert level"""
        score = self.drowsiness_score

        # Critical: Recent microsleep
        if self.microsleep_count > 0 and (time.time() - self.last_microsleep_time) < 5:
            self.alert_level = AlertLevel.CRITICAL

        # Danger
        elif score >= 70 or self.perclos >= Config.PERCLOS_DANGER:
            self.alert_level = AlertLevel.DANGER

        # Warning
        elif score >= 50 or self.perclos >= Config.PERCLOS_WARNING:
            self.alert_level = AlertLevel.WARNING

        # Caution
        elif score >= 30 or self.perclos >= Config.PERCLOS_CAUTION:
            self.alert_level = AlertLevel.CAUTION

        # Normal
        else:
            self.alert_level = AlertLevel.NORMAL


    # Properties to access sub-detector metrics
    @property
    def blink_count(self):
        return self.blink_detector.total_blinks

    @property
    def slow_blink_count(self):
        return self.blink_detector.slow_blinks

    @property
    def yawn_count(self):
        return self.yawn_detector.yawn_count

    @property
    def tilt_count(self):
        return self.tilt_detector.tilt_count

    @property
    def droop_count(self):
        return self.tilt_detector.droop_count

    @property
    def jerk_count(self):
        return self.jerk_detector.jerk_count

    @property
    def last_yawn_time(self):
        return self.yawn_detector.last_yawn_time

    @property
    def last_tilt_time(self):
        return self.tilt_detector.last_tilt_time

    @property
    def last_droop_time(self):
        return self.tilt_detector.last_droop_time

    @property
    def last_jerk_time(self):
        return self.jerk_detector.last_jerk_time


# ============================================================================
# VIDEO THREAD (unchanged)
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
            print("[ERROR] Camera failed!")
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

                    self.detector.update(ear, mar, pitch, yaw, roll, current_time)
                    self.draw_overlay(frame)

                self.frame_ready.emit(frame)

            except Exception as e:
                continue


    def draw_overlay(self, frame):
        h, w = frame.shape[:2]

        if not self.detector.is_calibrated:
            cv2.putText(frame, "PRESS CALIBRATE",
                       (w//2 - 120, h//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            return

        score = int(self.detector.drowsiness_score)

        box_w, box_h = 220, 100
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

        cv2.putText(frame, f"Score: {score}", (box_x+15, box_y+40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        alert_names = ["NORMAL", "CAUTION", "WARNING", "DANGER", "CRITICAL"]
        cv2.putText(frame, alert_names[self.detector.alert_level.value],
                   (box_x+15, box_y+75),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

        if self.detector.alert_level == AlertLevel.CRITICAL:
            pulse = abs(np.sin(time.time() * 5))
            alpha = 0.3 * pulse

            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 255), -1)
            cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0, frame)

            cv2.putText(frame, "MICROSLEEP - PULL OVER!", (w//2-220, h//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3)


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
# MAIN WINDOW (same as before, truncated for brevity)
# ============================================================================

class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.detector = DrowsinessDetector()
        self.calibrating = False
        self.calibration_start = None
        self.init_ui()

        self.video_thread = VideoThread(self.detector)
        self.video_thread.frame_ready.connect(self.update_camera)
        self.video_thread.start()

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_ui)
        self.timer.start(50)

        self.fps_counter = 0
        self.fps_start = time.time()


    def init_ui(self):
        self.setWindowTitle("Abhaya AI - v8.0 INTELLIGENT ALGORITHMS")
        self.setGeometry(100, 100, 1600, 900)

        self.setStyleSheet("""
            QMainWindow { background-color: #1a1a1a; }
            QLabel { color: #ffffff; }
            QGroupBox {
                color: #ffffff;
                border: 2px solid #444444;
                border-radius: 8px;
                margin-top: 12px;
                font-weight: bold;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 8px 0 8px;
            }
            QPushButton {
                background-color: #0d7377;
                color: white;
                border: none;
                padding: 12px;
                font-size: 14px;
                font-weight: bold;
                border-radius: 6px;
            }
            QPushButton:hover { background-color: #14a085; }
            QPushButton:pressed { background-color: #0a5d61; }
            QProgressBar {
                border: 2px solid #444444;
                border-radius: 6px;
                text-align: center;
                color: white;
                font-weight: bold;
            }
            QProgressBar::chunk { background-color: #0d7377; border-radius: 4px; }
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
        self.camera_label.setStyleSheet("border: 2px solid #444444; background-color: #0a0a0a;")
        layout.addWidget(self.camera_label)

        button_layout = QHBoxLayout()

        self.calibrate_button = QPushButton("CALIBRATE (15s)")
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
        layout.addWidget(self.create_perclos_graph_group())
        layout.addWidget(self.create_eye_metrics_group())
        layout.addWidget(self.create_blink_group())
        layout.addWidget(self.create_body_language_group())
        layout.addWidget(self.create_events_group())
        layout.addWidget(self.create_head_pose_group())
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
        font.setPointSize(52)
        font.setBold(True)
        self.score_label.setFont(font)
        layout.addWidget(self.score_label)

        self.score_bar = QProgressBar()
        self.score_bar.setMinimum(0)
        self.score_bar.setMaximum(100)
        self.score_bar.setValue(0)
        self.score_bar.setTextVisible(False)
        self.score_bar.setMinimumHeight(35)
        layout.addWidget(self.score_bar)

        return group


    def create_alert_group(self):
        group = QGroupBox("ALERT STATUS")
        layout = QVBoxLayout()
        group.setLayout(layout)

        self.alert_label = QLabel("NORMAL - ALERT")
        self.alert_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        font = QFont()
        font.setPointSize(22)
        font.setBold(True)
        self.alert_label.setFont(font)
        layout.addWidget(self.alert_label)

        return group


    def create_perclos_graph_group(self):
        group = QGroupBox("PERCLOS - Eye Closure Over Time")
        layout = QVBoxLayout()
        group.setLayout(layout)

        self.perclos_graph_label = QLabel()
        self.perclos_graph_label.setMinimumSize(500, 120)
        self.perclos_graph_label.setStyleSheet("border: 1px solid #444444; background-color: #0a0a0a;")
        layout.addWidget(self.perclos_graph_label)

        return group


    def create_eye_metrics_group(self):
        group = QGroupBox("EYE METRICS")
        layout = QGridLayout()
        group.setLayout(layout)

        labels = [
            ("EAR (Current):", "ear_current"),
            ("EAR (Baseline):", "ear_baseline"),
            ("EAR (Threshold):", "ear_threshold"),
            ("PERCLOS:", "perclos"),
            ("Eye Status:", "eye_status"),
            ("Closure Duration:", "closure_duration")
        ]

        self.eye_labels = {}
        for i, (text, key) in enumerate(labels):
            label = QLabel(text)
            value = QLabel("--")
            value.setStyleSheet("color: #00ff88; font-weight: bold;")
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
            ("Slow Blink Ratio:", "blink_ratio"),
            ("Blink Rate:", "blink_rate")
        ]

        self.blink_labels = {}
        for i, (text, key) in enumerate(labels):
            label = QLabel(text)
            value = QLabel("0")
            value.setStyleSheet("color: #00ff88; font-weight: bold;")
            layout.addWidget(label, i, 0)
            layout.addWidget(value, i, 1)
            self.blink_labels[key] = value

        return group


    def create_body_language_group(self):
        group = QGroupBox("BODY LANGUAGE ANALYSIS")
        layout = QGridLayout()
        group.setLayout(layout)

        labels = [
            ("Yawns:", "yawns"),
            ("Head Tilts:", "head_tilts"),
            ("Head Droops:", "head_droops")
        ]

        self.body_labels = {}
        for i, (text, key) in enumerate(labels):
            label = QLabel(text)
            value = QLabel("0")
            value.setStyleSheet("color: #00ff88; font-weight: bold;")
            layout.addWidget(label, i, 0)
            layout.addWidget(value, i, 1)
            self.body_labels[key] = value

        return group


    def create_events_group(self):
        group = QGroupBox("CRITICAL EVENTS")
        layout = QGridLayout()
        group.setLayout(layout)

        labels = [
            ("Microsleeps:", "microsleeps"),
            ("Head Jerks:", "head_jerks")
        ]

        self.event_labels = {}
        for i, (text, key) in enumerate(labels):
            label = QLabel(text)
            value = QLabel("0")
            value.setStyleSheet("color: #00ff88; font-weight: bold;")
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
            ("Roll:", "head_roll")
        ]

        self.head_labels = {}
        for i, (text, key) in enumerate(labels):
            label = QLabel(text)
            value = QLabel("0.0 deg")
            value.setStyleSheet("color: #00ff88; font-weight: bold;")
            layout.addWidget(label, i, 0)
            layout.addWidget(value, i, 1)
            self.head_labels[key] = value

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
            value.setStyleSheet("color: #00ff88; font-weight: bold;")
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
        self.status_label.setStyleSheet("color: #ffff00; font-weight: bold;")
        layout.addWidget(self.status_label)

        return group


    def update_camera(self, frame):
        try:
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
            pass


    def update_ui(self):
        try:
            if self.calibrating:
                elapsed = time.time() - self.calibration_start
                progress = min((elapsed / Config.CALIBRATION_DURATION) * 100, 100)

                self.detector.add_calibration_sample(
                    self.detector.current_ear,
                    self.detector.current_mar,
                    self.detector.current_head_pitch,
                    self.detector.current_head_yaw,
                    self.detector.current_head_roll
                )

                self.status_label.setText(
                    f"Calibrating... {progress:.0f}% ({len(self.detector.calibration_ear_samples)} samples)"
                )

                if elapsed >= Config.CALIBRATION_DURATION:
                    self.calibrating = False
                    success = self.detector.finalize_calibration()
                    if success:
                        self.calibrate_button.setText("RECALIBRATE")

            if not self.detector.is_calibrated:
                return

            score = int(self.detector.drowsiness_score)
            self.score_label.setText(str(score))
            self.score_bar.setValue(score)

            if score < 30:
                color = "#00ff88"
            elif score < 50:
                color = "#ffff00"
            elif score < 70:
                color = "#ff9900"
            else:
                color = "#ff0000"

            self.score_label.setStyleSheet(f"color: {color}; font-weight: bold;")
            self.score_bar.setStyleSheet(f"QProgressBar::chunk {{ background-color: {color}; border-radius: 4px; }}")

            alert_names = {
                AlertLevel.NORMAL: "NORMAL - ALERT",
                AlertLevel.CAUTION: "CAUTION - EARLY FATIGUE",
                AlertLevel.WARNING: "WARNING - TAKE BREAK",
                AlertLevel.DANGER: "DANGER - PULL OVER NOW",
                AlertLevel.CRITICAL: "CRITICAL - MICROSLEEP!"
            }

            self.alert_label.setText(alert_names[self.detector.alert_level])
            self.alert_label.setStyleSheet(f"color: {color}; font-size: 22px; font-weight: bold;")

            self.eye_labels['ear_current'].setText(f"{self.detector.current_ear:.3f}")
            self.eye_labels['ear_current'].setStyleSheet(
                f"color: {'#ff0000' if self.detector.current_ear < self.detector.ear_threshold else '#00ff88'}; font-weight: bold;"
            )

            self.eye_labels['ear_baseline'].setText(f"{self.detector.baseline_ear:.3f}")
            self.eye_labels['ear_threshold'].setText(f"{self.detector.ear_threshold:.3f}")
            self.eye_labels['perclos'].setText(f"{self.detector.perclos:.1f}%")

            if self.detector.eyes_currently_closed:
                self.eye_labels['eye_status'].setText("CLOSED")
                self.eye_labels['eye_status'].setStyleSheet("color: #ff0000; font-weight: bold; font-size: 14px;")
                self.eye_labels['closure_duration'].setText(f"{self.detector.eye_closure_duration:.2f}s")
                self.eye_labels['closure_duration'].setStyleSheet("color: #ff0000; font-weight: bold;")
            else:
                self.eye_labels['eye_status'].setText("OPEN")
                self.eye_labels['eye_status'].setStyleSheet("color: #00ff88; font-weight: bold; font-size: 14px;")
                self.eye_labels['closure_duration'].setText("0.00s")
                self.eye_labels['closure_duration'].setStyleSheet("color: #00ff88; font-weight: bold;")

            self.blink_labels['blink_total'].setText(str(self.detector.blink_count))
            self.blink_labels['blink_slow'].setText(str(self.detector.slow_blink_count))

            if self.detector.blink_count > 0:
                ratio = (self.detector.slow_blink_count / self.detector.blink_count) * 100
                self.blink_labels['blink_ratio'].setText(f"{ratio:.0f}%")
            else:
                self.blink_labels['blink_ratio'].setText("0%")

            self.blink_labels['blink_rate'].setText(f"{self.detector.blink_rate:.1f}/min")

            self.body_labels['yawns'].setText(str(self.detector.yawn_count))
            self.body_labels['head_tilts'].setText(str(self.detector.tilt_count))
            self.body_labels['head_droops'].setText(str(self.detector.droop_count))

            self.event_labels['microsleeps'].setText(str(self.detector.microsleep_count))
            self.event_labels['microsleeps'].setStyleSheet(
                f"color: {'#ff0000' if self.detector.microsleep_count > 0 else '#00ff88'}; font-weight: bold; font-size: 16px;"
            )

            self.event_labels['head_jerks'].setText(str(self.detector.jerk_count))

            self.head_labels['head_pitch'].setText(f"{self.detector.current_head_pitch:.1f} deg")
            self.head_labels['head_yaw'].setText(f"{self.detector.current_head_yaw:.1f} deg")
            self.head_labels['head_roll'].setText(f"{self.detector.current_head_roll:.1f} deg")

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
            width = 500
            height = 120

            image = QImage(width, height, QImage.Format.Format_RGB888)
            image.fill(QColor(10, 10, 10))

            painter = QPainter(image)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)

            thresholds = [
                (Config.PERCLOS_CAUTION, QColor(0, 255, 136)),
                (Config.PERCLOS_WARNING, QColor(255, 255, 0)),
                (Config.PERCLOS_DANGER, QColor(255, 0, 0))
            ]

            for threshold, color in thresholds:
                y = height - int((threshold / 50.0) * height)
                painter.setPen(QPen(color, 1, Qt.PenStyle.DashLine))
                painter.drawLine(0, y, width, y)

            if len(self.detector.ear_buffer) > 1:
                painter.setPen(QPen(QColor(0, 200, 255), 3))

                max_points = min(300, len(self.detector.ear_buffer))

                points = []
                for i in range(len(self.detector.ear_buffer) - max_points, len(self.detector.ear_buffer)):
                    if i < 0:
                        continue

                    x = int(((i - (len(self.detector.ear_buffer) - max_points)) / max_points) * width)

                    is_closed = 1 if self.detector.ear_buffer[i] < self.detector.ear_threshold else 0
                    y = height - int((is_closed * 45 / 50.0) * height)

                    points.append((x, y))

                for i in range(1, len(points)):
                    painter.drawLine(points[i-1][0], points[i-1][1], points[i][0], points[i][1])

            painter.end()

            pixmap = QPixmap.fromImage(image)
            self.perclos_graph_label.setPixmap(pixmap)

        except Exception as e:
            pass


    def start_calibration(self):
        if not self.calibrating:
            self.detector.start_calibration()
            self.calibrating = True
            self.calibration_start = time.time()
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
            print(f"FPS:      {self.detector.fps:.1f}")
            print(f"\nFinal Score: {self.detector.drowsiness_score:.1f}/100")
            print(f"PERCLOS:     {self.detector.perclos:.1f}%")
            print(f"\nEye Events:")
            print(f"  Total Blinks:  {self.detector.blink_count}")
            print(f"  Slow Blinks:   {self.detector.slow_blink_count}")
            print(f"  Blink Rate:    {self.detector.blink_rate:.1f}/min")
            print(f"  Microsleeps:   {self.detector.microsleep_count}")
            print(f"\nBody Language:")
            print(f"  Yawns:         {self.detector.yawn_count}")
            print(f"  Head Tilts:    {self.detector.tilt_count}")
            print(f"  Head Droops:   {self.detector.droop_count}")
            print(f"  Head Jerks:    {self.detector.jerk_count}")

            if self.detector.microsleep_count > 0:
                print(f"\n[CRITICAL] {self.detector.microsleep_count} microsleep event(s)!")
            else:
                print(f"\n[OK] No microsleeps. Stay alert!")
            print("="*70 + "\n")

        event.accept()


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "="*70)
    print(" Abhaya AI - v8.0 INTELLIGENT ALGORITHMS ".center(70, '='))
    print("="*70)
    print("\nINTELLIGENT ALGORITHMS:")
    print("  [1] BLINK DETECTION - Direct state tracking")
    print("      • Every CLOSED→OPEN = 1 blink")
    print("      • Minimal filtering (only sensor noise)")
    print("      • Maximum accuracy")
    print("\n  [2] YAWN DETECTION - State machine")
    print("      • Detects discrete yawn cycles")
    print("      • CLOSED → OPEN → CLOSED")
    print("      • No exponential counting")
    print("\n  [3] HEAD TILT - Window-based sustained detection")
    print("      • Uses sliding window (1 second)")
    print("      • Median filtering (robust to spikes)")
    print("      • Only counts sustained tilts")
    print("\n  [4] HEAD JERK - Velocity + displacement")
    print("      • Smoothed to remove noise")
    print("      • Requires high velocity AND displacement")
    print("\n  [5] SCORING - Multi-factor weighted (0-100)")
    print("      • PERCLOS (25%)")
    print("      • Blink rate (10%)")
    print("      • Slow blink ratio (10%)")
    print("      • Microsleeps (20%)")
    print("      • Head droop (15%)")
    print("      • Head tilt (5%)")
    print("      • Yawning (10%)")
    print("      • Head jerks (5%)")
    print("\n" + "="*70 + "\n")

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
