"""
CORE DROWSINESS DETECTION ENGINE - PURE OPENCV VERSION
No mediapipe or dlib required - works on all systems
"""

import cv2
import numpy as np
from scipy.spatial import distance
from collections import deque
import time
from dataclasses import dataclass
from typing import Tuple, List, Optional
import math
import os


@dataclass
class FacialMetrics:
    """Container for all facial measurements"""
    ear_left: float = 0.0
    ear_right: float = 0.0
    ear_avg: float = 0.0
    mar: float = 0.0
    head_pitch: float = 0.0
    head_yaw: float = 0.0
    head_roll: float = 0.0
    gaze_x: float = 0.0
    gaze_y: float = 0.0
    timestamp: float = 0.0


@dataclass
class DrowsinessIndicators:
    """Current drowsiness state indicators"""
    is_eyes_closed: bool = False
    is_yawning: bool = False
    is_head_nodding: bool = False
    is_microsleep: bool = False
    blink_rate: float = 0.0
    perclos: float = 0.0
    attention_level: float = 100.0


class DrowsinessDetector:
    """
    OpenCV-based drowsiness detection
    Uses Haar Cascades and custom algorithms
    """
    
    def __init__(self, config: dict):
        self.config = config
        
        print("Initializing OpenCV detector...")
        
        # Load Haar Cascade classifiers (built into OpenCV)
        cascade_path = cv2.data.haarcascades
        
        self.face_cascade = cv2.CascadeClassifier(
            cascade_path + 'haarcascade_frontalface_default.xml'
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cascade_path + 'haarcascade_eye.xml'
        )
        self.mouth_cascade = cv2.CascadeClassifier(
            cascade_path + 'haarcascade_smile.xml'
        )
        
        # Verify cascades loaded
        if self.face_cascade.empty():
            raise Exception("Could not load face cascade")
        
        print("✓ OpenCV detector initialized")
        
        # History buffers
        self.ear_history = deque(maxlen=config['thresholds']['perclos_window'])
        self.mar_history = deque(maxlen=30)
        self.blink_timestamps = deque(maxlen=50)
        self.head_pose_history = deque(maxlen=60)
        self.eye_closure_history = deque(maxlen=30)
        
        # State tracking
        self.eye_closed_start = None
        self.yawn_start = None
        self.last_blink_time = 0
        self.blink_counter = 0
        self.prev_eye_count = 0
        self.eye_closed_frames = 0
        
        # Baseline values
        self.baseline_eye_height = 0
        self.baseline_mouth_height = 0
        self.is_calibrated = False
        
        # Detection parameters
        self.min_face_size = (100, 100)
        self.eyes_detected_threshold = 0.3
        
    def calculate_eye_closure_ratio(self, eye_roi: np.ndarray) -> float:
        """
        Calculate eye closure based on eye region
        Returns value between 0 (closed) and 1 (open)
        """
        if eye_roi.size == 0:
            return 0.5
        
        # Convert to grayscale if needed
        if len(eye_roi.shape) == 3:
            eye_gray = cv2.cvtColor(eye_roi, cv2.COLOR_BGR2GRAY)
        else:
            eye_gray = eye_roi
        
        # Apply threshold to find bright areas (sclera/iris)
        _, thresh = cv2.threshold(eye_gray, 30, 255, cv2.THRESH_BINARY)
        
        # Calculate white pixel ratio (open eye has more white)
        white_pixels = cv2.countNonZero(thresh)
        total_pixels = eye_gray.shape[0] * eye_gray.shape[1]
        
        if total_pixels == 0:
            return 0.5
        
        ratio = white_pixels / total_pixels
        return ratio
    
    def detect_eyes_advanced(self, face_roi: np.ndarray, face_rect: tuple) -> Tuple[float, int]:
        """
        Advanced eye detection and analysis
        Returns (eye_aspect_ratio, number_of_eyes_detected)
        """
        fx, fy, fw, fh = face_rect
        
        # Define eye region (upper half of face)
        eye_region_height = int(fh * 0.5)
        eye_region = face_roi[0:eye_region_height, :]
        
        # Detect eyes in the eye region
        gray_eye_region = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
        eyes = self.eye_cascade.detectMultiScale(
            gray_eye_region,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(20, 20)
        )
        
        num_eyes = len(eyes)
        
        # Calculate EAR approximation
        if num_eyes >= 2:
            # Sort eyes by x coordinate
            eyes_sorted = sorted(eyes, key=lambda x: x[0])
            
            # Get left and right eye
            left_eye = eyes_sorted[0]
            right_eye = eyes_sorted[-1]
            
            # Calculate closure ratio for each eye
            lex, ley, lew, leh = left_eye
            rex, rey, rew, reh = right_eye
            
            left_eye_roi = eye_region[ley:ley+leh, lex:lex+lew]
            right_eye_roi = eye_region[rey:rey+reh, rex:rex+rew]
            
            left_closure = self.calculate_eye_closure_ratio(left_eye_roi)
            right_closure = self.calculate_eye_closure_ratio(right_eye_roi)
            
            # Convert closure ratio to EAR-like metric
            # Open eyes have high ratio (lots of white), closed eyes have low ratio
            ear = (left_closure + right_closure) / 2.0
            
            # Normalize to typical EAR range (0.15 - 0.35)
            ear = 0.15 + (ear * 0.20)
            
        elif num_eyes == 1:
            # Only one eye detected
            ex, ey, ew, eh = eyes[0]
            eye_roi = eye_region[ey:ey+eh, ex:ex+ew]
            closure = self.calculate_eye_closure_ratio(eye_roi)
            ear = 0.15 + (closure * 0.20)
            
        else:
            # No eyes detected - likely closed or looking away
            ear = 0.10  # Very low value indicating closure
        
        return ear, num_eyes
    
    def detect_mouth_opening(self, face_roi: np.ndarray, face_rect: tuple) -> float:
        """
        Detect mouth opening (yawning)
        Returns MAR (Mouth Aspect Ratio)
        """
        fx, fy, fw, fh = face_rect
        
        # Define mouth region (lower half of face)
        mouth_start_y = int(fh * 0.5)
        mouth_region = face_roi[mouth_start_y:, :]
        
        # Detect mouth/smile
        gray_mouth_region = cv2.cvtColor(mouth_region, cv2.COLOR_BGR2GRAY)
        mouths = self.mouth_cascade.detectMultiScale(
            gray_mouth_region,
            scaleFactor=1.3,
            minNeighbors=15,
            minSize=(30, 30)
        )
        
        if len(mouths) > 0:
            # Get largest mouth detection
            mouth = max(mouths, key=lambda x: x[2] * x[3])
            mx, my, mw, mh = mouth
            
            # Calculate aspect ratio (height / width)
            # Yawning produces high aspect ratio
            mar = mh / mw if mw > 0 else 0
            
            return mar
        
        return 0.0
    
    def estimate_head_pose_simple(self, face_rect: tuple, frame_shape: tuple) -> Tuple[float, float, float]:
        """
        Simple head pose estimation based on face position
        Returns approximate pitch, yaw, roll
        """
        fx, fy, fw, fh = face_rect
        frame_h, frame_w = frame_shape[:2]
        
        # Calculate face center
        face_center_x = fx + fw // 2
        face_center_y = fy + fh // 2
        
        # Calculate frame center
        frame_center_x = frame_w // 2
        frame_center_y = frame_h // 2
        
        # Estimate yaw (horizontal deviation)
        yaw = ((face_center_x - frame_center_x) / frame_center_x) * 30
        
        # Estimate pitch (vertical deviation)
        pitch = ((face_center_y - frame_center_y) / frame_center_y) * 20
        
        # Roll is harder to estimate without landmarks, set to 0
        roll = 0.0
        
        return pitch, yaw, roll
    
    def calculate_perclos(self) -> float:
        """Calculate PERCLOS"""
        if len(self.ear_history) == 0:
            return 0.0
        
        threshold = self.config['thresholds']['ear_threshold']
        closed_count = sum(1 for ear in self.ear_history if ear < threshold)
        perclos = (closed_count / len(self.ear_history)) * 100
        
        return perclos
    
    def detect_blink(self, ear: float, current_time: float) -> bool:
        """Detect blink"""
        threshold = self.config['thresholds']['ear_threshold']
        
        if ear < threshold:
            if self.eye_closed_start is None:
                self.eye_closed_start = current_time
        else:
            if self.eye_closed_start is not None:
                blink_duration = current_time - self.eye_closed_start
                max_duration = self.config['thresholds']['max_blink_duration']
                
                if 0.05 < blink_duration < max_duration:
                    self.blink_timestamps.append(current_time)
                    self.blink_counter += 1
                    self.eye_closed_start = None
                    return True
                
                self.eye_closed_start = None
        
        return False
    
    def calculate_blink_rate(self, current_time: float) -> float:
        """Calculate blink rate"""
        while self.blink_timestamps and (current_time - self.blink_timestamps[0] > 60):
            self.blink_timestamps.popleft()
        
        if len(self.blink_timestamps) > 0:
            time_span = current_time - self.blink_timestamps[0] if len(self.blink_timestamps) > 1 else 60
            blink_rate = (len(self.blink_timestamps) / time_span) * 60
            return blink_rate
        
        return 0.0
    
    def detect_yawn(self, mar: float, current_time: float) -> bool:
        """Detect yawning"""
        threshold = self.config['thresholds']['mar_threshold']
        
        if mar > threshold:
            if self.yawn_start is None:
                self.yawn_start = current_time
            else:
                yawn_duration = current_time - self.yawn_start
                if yawn_duration > self.config['thresholds']['yawn_duration']:
                    return True
        else:
            self.yawn_start = None
        
        return False
    
    def detect_microsleep(self, ear: float, current_time: float) -> bool:
        """Detect microsleep"""
        threshold = self.config['thresholds']['ear_threshold']
        microsleep_duration = self.config['thresholds']['microsleep_duration']
        
        if ear < threshold:
            if self.eye_closed_start is not None:
                closure_duration = current_time - self.eye_closed_start
                if closure_duration > microsleep_duration:
                    return True
        
        return False
    
    def analyze_head_stability(self) -> float:
        """Analyze head stability"""
        if len(self.head_pose_history) < 10:
            return 100.0
        
        poses = list(self.head_pose_history)
        pitches = [p[0] for p in poses]
        yaws = [p[1] for p in poses]
        
        pitch_std = np.std(pitches)
        yaw_std = np.std(yaws)
        
        instability = (pitch_std + yaw_std) / 2
        stability = max(0, 100 - instability * 5)
        
        return stability
    
    def process_frame(self, frame: np.ndarray) -> Tuple[Optional[FacialMetrics], Optional[DrowsinessIndicators]]:
        """Main processing pipeline"""
        current_time = time.time()
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=self.min_face_size,
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        if len(faces) == 0:
            return None, None
        
        # Use largest face
        face = max(faces, key=lambda rect: rect[2] * rect[3])
        x, y, w, h = face
        
        # Extract face ROI
        face_roi = frame[y:y+h, x:x+w]
        
        # Calculate metrics
        metrics = FacialMetrics(timestamp=current_time)
        
        # Eye analysis
        ear, num_eyes = self.detect_eyes_advanced(face_roi, face)
        metrics.ear_left = ear
        metrics.ear_right = ear
        metrics.ear_avg = ear
        
        # Mouth analysis
        mar = self.detect_mouth_opening(face_roi, face)
        metrics.mar = mar
        
        # Head pose
        pitch, yaw, roll = self.estimate_head_pose_simple(face, frame.shape)
        metrics.head_pitch = pitch
        metrics.head_yaw = yaw
        metrics.head_roll = roll
        
        # Update histories
        self.ear_history.append(metrics.ear_avg)
        self.mar_history.append(metrics.mar)
        self.head_pose_history.append((pitch, yaw, roll))
        
        # Track eye detection
        if num_eyes < 2:
            self.eye_closed_frames += 1
        else:
            self.eye_closed_frames = 0
        
        # Analyze indicators
        indicators = DrowsinessIndicators()
        
        # Eye closure
        ear_threshold = self.config['thresholds']['ear_threshold']
        indicators.is_eyes_closed = metrics.ear_avg < ear_threshold or num_eyes < 2
        
        # Blink detection
        self.detect_blink(metrics.ear_avg, current_time)
        indicators.blink_rate = self.calculate_blink_rate(current_time)
        
        # PERCLOS
        indicators.perclos = self.calculate_perclos()
        
        # Yawn detection
        indicators.is_yawning = self.detect_yawn(metrics.mar, current_time)
        
        # Microsleep
        indicators.is_microsleep = self.detect_microsleep(metrics.ear_avg, current_time)
        
        # Head nodding
        if abs(pitch) > self.config['thresholds']['head_tilt_threshold']:
            indicators.is_head_nodding = True
        
        self.prev_eye_count = num_eyes
        
        return metrics, indicators
    
    def calibrate(self, ear_values: List[float], mar_values: List[float]):
        """Calibrate detector"""
        if len(ear_values) > 0:
            self.baseline_ear = np.mean(ear_values)
        
        if len(mar_values) > 0:
            self.baseline_mar = np.mean(mar_values)
        
        self.is_calibrated = True
    
    def reset(self):
        """Reset state"""
        self.ear_history.clear()
        self.mar_history.clear()
        self.blink_timestamps.clear()
        self.head_pose_history.clear()
        self.eye_closure_history.clear()
        
        self.eye_closed_start = None
        self.yawn_start = None
        self.last_blink_time = 0
        self.blink_counter = 0
        self.eye_closed_frames = 0