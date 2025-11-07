

"""
Abhaya AI - Geometric Metrics & Calculations Module

This module contains all geometric calculations for drowsiness detection:
- Eye Aspect Ratio (EAR)
- Mouth Aspect Ratio (MAR)
- Head pose estimation
- Gaze vector analysis
- PERCLOS calculation
- Blink and yawn detection

All calculations are optimized using vectorized numpy operations
for maximum performance on resource-constrained devices.
"""

import numpy as np
import cv2
from typing import Optional, Tuple, List
from collections import deque
import logging


logger = logging.getLogger(__name__)


# MediaPipe Face Mesh landmark indices (468-point model)
# Reference: https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png

# Left eye landmarks (6 points for EAR calculation)
LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]

# Right eye landmarks (6 points for EAR calculation)
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]


# Left iris landmarks (for gaze estimation)
LEFT_IRIS_INDICES = [468, 469, 470, 471, 472]

# Right iris landmarks (for gaze estimation)
RIGHT_IRIS_INDICES = [473, 474, 475, 476, 477]


# Mouth landmarks (for MAR and yawn detection)
MOUTH_OUTER_INDICES = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308]
MOUTH_INNER_INDICES = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324]


# Face oval landmarks (for head pose estimation)
FACE_OVAL_INDICES = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                     397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
                     172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]


# Nose tip (for head pose reference point)
NOSE_TIP_INDEX = 1

# Chin (for head pose)
CHIN_INDEX = 152


# Face mesh nose bridge points
NOSE_BRIDGE_INDEX = 6



class LandmarkGeometry:
    """
    Core geometric calculations for facial landmarks
    """
    
    @staticmethod
    def calculate_distance(p1, p2):
        """
        Calculate Euclidean distance between two points
        
        Args:
            p1: First point (x, y) or (x, y, z)
            p2: Second point (x, y) or (x, y, z)
            
        Returns:
            Distance as float
        """
        return np.linalg.norm(np.array(p1) - np.array(p2))


    @staticmethod
    def calculate_ear(eye_landmarks):
        """
        Calculate Eye Aspect Ratio (EAR)
        
        EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
        
        Where p1-p6 are the eye landmarks in order:
        p1, p4: Horizontal eye corners
        p2, p3, p5, p6: Vertical eye landmarks
        
        Args:
            eye_landmarks: Array of 6 eye landmark points [(x,y), ...]
            
        Returns:
            EAR value (typically 0.2-0.4 when open, <0.2 when closed)
        """
        if len(eye_landmarks) != 6:
            return 0.0
            
        # Convert to numpy array for vectorized operations
        pts = np.array(eye_landmarks, dtype=np.float16)
        
        # Vertical distances
        vertical1 = np.linalg.norm(pts[1] - pts[5])
        vertical2 = np.linalg.norm(pts[2] - pts[4])
        
        # Horizontal distance
        horizontal = np.linalg.norm(pts[0] - pts[3])
        
        if horizontal < 1e-6:
            return 0.0
            
        ear = (vertical1 + vertical2) / (2.0 * horizontal)
        
        return float(ear)


    @staticmethod
    def calculate_mar(mouth_landmarks):
        """
        Calculate Mouth Aspect Ratio (MAR) for yawn detection
        
        MAR = (||p2-p8|| + ||p3-p7|| + ||p4-p6||) / (2 * ||p1-p5||)
        
        Args:
            mouth_landmarks: Array of mouth landmark points
            
        Returns:
            MAR value (typically >0.6 indicates yawn)
        """
        if len(mouth_landmarks) < 8:
            return 0.0
            
        pts = np.array(mouth_landmarks, dtype=np.float16)
        
        # Vertical distances (mouth height at different points)
        vertical1 = np.linalg.norm(pts[1] - pts[7])
        vertical2 = np.linalg.norm(pts[2] - pts[6])
        vertical3 = np.linalg.norm(pts[3] - pts[5])
        
        # Horizontal distance (mouth width)
        horizontal = np.linalg.norm(pts[0] - pts[4])
        
        if horizontal < 1e-6:
            return 0.0
            
        mar = (vertical1 + vertical2 + vertical3) / (2.0 * horizontal)
        
        return float(mar)



    @staticmethod
    def estimate_head_pose(landmarks, img_width, img_height):
        """
        Estimate head pose (pitch, yaw, roll) using solvePnP
        
        Args:
            landmarks: Face mesh landmarks (468 points)
            img_width: Image width in pixels
            img_height: Image height in pixels
            
        Returns:
            Tuple of (pitch, yaw, roll) in degrees
        """
        
        # 3D model points (generic face model in cm)
        model_points = np.array([
            (0.0, 0.0, 0.0),          # Nose tip
            (0.0, -33.0, -6.5),       # Chin
            (-22.5, 17.0, -3.0),      # Left eye left corner
            (22.5, 17.0, -3.0),       # Right eye right corner
            (-15.0, -15.0, -3.0),     # Left mouth corner
            (15.0, -15.0, -3.0)       # Right mouth corner
        ], dtype=np.float64)
        
        
        # Corresponding 2D image points from landmarks
        image_points = np.array([
            landmarks[NOSE_TIP_INDEX][:2],      # Nose tip
            landmarks[CHIN_INDEX][:2],          # Chin
            landmarks[33][:2],                  # Left eye left corner
            landmarks[263][:2],                 # Right eye right corner
            landmarks[61][:2],                  # Left mouth corner
            landmarks[291][:2]                  # Right mouth corner
        ], dtype=np.float64)
        
        
        # Camera internals (assuming simple pinhole camera)
        focal_length = img_width
        center = (img_width / 2, img_height / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)
        
        
        # Assume no lens distortion
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)
        
        
        # Solve PnP
        success, rotation_vec, translation_vec = cv2.solvePnP(
            model_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        
        if not success:
            return (0.0, 0.0, 0.0)
        
        
        # Convert rotation vector to rotation matrix
        rotation_mat, _ = cv2.Rodrigues(rotation_vec)
        
        
        # Extract Euler angles from rotation matrix
        # Using the convention: pitch (x), yaw (y), roll (z)
        pitch = np.degrees(np.arctan2(rotation_mat[2, 1], rotation_mat[2, 2]))
        yaw = np.degrees(np.arctan2(-rotation_mat[2, 0], 
                                     np.sqrt(rotation_mat[2, 1]**2 + rotation_mat[2, 2]**2)))
        roll = np.degrees(np.arctan2(rotation_mat[1, 0], rotation_mat[0, 0]))
        
        
        return (float(pitch), float(yaw), float(roll))


    @staticmethod
    def calculate_gaze_vector(eye_landmarks, iris_landmarks, img_width, img_height):
        """
        Calculate gaze direction vector from eye and iris positions
        
        Args:
            eye_landmarks: Eye corner landmarks
            iris_landmarks: Iris center landmarks
            img_width: Image width
            img_height: Image height
            
        Returns:
            Tuple of (gaze_x, gaze_y) normalized deviation from center
        """
        if len(iris_landmarks) < 1:
            return (0.0, 0.0)
            
        # Calculate eye center
        eye_pts = np.array(eye_landmarks, dtype=np.float16)
        eye_center = np.mean(eye_pts, axis=0)
        
        # Get iris center
        iris_pts = np.array(iris_landmarks, dtype=np.float16)
        iris_center = np.mean(iris_pts, axis=0)
        
        
        # Calculate gaze deviation
        gaze_offset = iris_center - eye_center
        
        # Normalize by image dimensions
        gaze_x = float(gaze_offset[0] / img_width)
        gaze_y = float(gaze_offset[1] / img_height)
        
        
        return (gaze_x, gaze_y)



class TemporalMetrics:
    """
    Temporal analysis of metrics over time using circular buffers
    """
    
    def __init__(self, max_buffer_size=300):
        """
        Initialize temporal metrics tracker
        
        Args:
            max_buffer_size: Maximum number of frames to store
        """
        self.max_buffer_size = max_buffer_size
        
        # Circular buffers for different metrics
        self.ear_buffer = deque(maxlen=max_buffer_size)
        self.mar_buffer = deque(maxlen=max_buffer_size)
        self.head_pose_buffer = deque(maxlen=max_buffer_size)
        self.gaze_buffer = deque(maxlen=max_buffer_size)
        
        # Blink tracking
        self.blink_timestamps = deque(maxlen=100)
        self.last_ear = 0.3
        self.eye_closed_frames = 0
        
        # Yawn tracking
        self.yawn_timestamps = deque(maxlen=50)
        self.yawn_frames = 0


    def update_ear(self, ear_value, timestamp):
        """
        Update EAR buffer and detect blinks
        
        Args:
            ear_value: Current EAR value
            timestamp: Current timestamp
            
        Returns:
            True if blink detected
        """
        self.ear_buffer.append(ear_value)
        
        blink_detected = False
        
        # Blink detection: EAR drops below threshold
        if ear_value < 0.21:
            self.eye_closed_frames += 1
        else:
            if self.eye_closed_frames >= 2:
                # Blink completed
                self.blink_timestamps.append(timestamp)
                blink_detected = True
            self.eye_closed_frames = 0
            
        self.last_ear = ear_value
        
        return blink_detected


    def update_mar(self, mar_value, timestamp):
        """
        Update MAR buffer and detect yawns
        
        Args:
            mar_value: Current MAR value
            timestamp: Current timestamp
            
        Returns:
            True if yawn detected
        """
        self.mar_buffer.append(mar_value)
        
        yawn_detected = False
        
        # Yawn detection: MAR above threshold for consecutive frames
        if mar_value > 0.6:
            self.yawn_frames += 1
            if self.yawn_frames >= 10:  # ~1 second at 10fps
                self.yawn_timestamps.append(timestamp)
                yawn_detected = True
                self.yawn_frames = 0
        else:
            self.yawn_frames = 0
            
        return yawn_detected


    def update_head_pose(self, pitch, yaw, roll):
        """
        Update head pose buffer
        
        Args:
            pitch: Head pitch in degrees
            yaw: Head yaw in degrees
            roll: Head roll in degrees
        """
        self.head_pose_buffer.append((pitch, yaw, roll))


    def update_gaze(self, gaze_x, gaze_y):
        """
        Update gaze buffer
        
        Args:
            gaze_x: Gaze x deviation
            gaze_y: Gaze y deviation
        """
        self.gaze_buffer.append((gaze_x, gaze_y))


    def calculate_perclos(self, threshold=0.25, window_seconds=60, fps=10):
        """
        Calculate PERCLOS (Percentage of Eye Closure)
        
        Args:
            threshold: EAR threshold for closed eyes
            window_seconds: Time window in seconds
            fps: Frames per second
            
        Returns:
            PERCLOS percentage (0-100)
        """
        window_frames = min(window_seconds * fps, len(self.ear_buffer))
        
        if window_frames == 0:
            return 0.0
            
        recent_ear = list(self.ear_buffer)[-window_frames:]
        closed_frames = sum(1 for ear in recent_ear if ear < threshold)
        
        perclos = (closed_frames / window_frames) * 100.0
        
        return float(perclos)


    def calculate_blink_rate(self, window_seconds=60):
        """
        Calculate blink rate (blinks per minute)
        
        Args:
            window_seconds: Time window in seconds
            
        Returns:
            Blinks per minute
        """
        if len(self.blink_timestamps) < 2:
            return 0.0
            
        current_time = self.blink_timestamps[-1]
        cutoff_time = current_time - window_seconds
        
        recent_blinks = [t for t in self.blink_timestamps if t >= cutoff_time]
        
        if len(recent_blinks) == 0:
            return 0.0
            
        time_span = current_time - recent_blinks[0]
        
        if time_span == 0:
            return 0.0
            
        blinks_per_minute = (len(recent_blinks) / time_span) * 60.0
        
        return float(blinks_per_minute)


    def calculate_yawn_frequency(self, window_seconds=300):
        """
        Calculate yawn frequency (yawns per 5 minutes)
        
        Args:
            window_seconds: Time window in seconds (default 5 minutes)
            
        Returns:
            Number of yawns in window
        """
        if len(self.yawn_timestamps) == 0:
            return 0
            
        current_time = self.yawn_timestamps[-1]
        cutoff_time = current_time - window_seconds
        
        recent_yawns = [t for t in self.yawn_timestamps if t >= cutoff_time]
        
        return len(recent_yawns)


    def detect_head_nod(self, threshold_degrees=15.0, window_frames=30):
        """
        Detect head nodding (drowsy head drops)
        
        Args:
            threshold_degrees: Minimum pitch change for nod detection
            window_frames: Frames to analyze
            
        Returns:
            True if head nod detected
        """
        if len(self.head_pose_buffer) < window_frames:
            return False
            
        recent_poses = list(self.head_pose_buffer)[-window_frames:]
        pitches = [pose[0] for pose in recent_poses]
        
        # Check for significant downward pitch variation
        pitch_range = max(pitches) - min(pitches)
        
        # Check if head is currently tilted down
        current_pitch = pitches[-1]
        
        if pitch_range > threshold_degrees and current_pitch < -threshold_degrees:
            return True
            
        return False


    def calculate_gaze_stability(self, window_frames=15):
        """
        Calculate gaze stability (lower = more stable)
        
        Args:
            window_frames: Frames to analyze
            
        Returns:
            Gaze deviation standard deviation
        """
        if len(self.gaze_buffer) < window_frames:
            return 0.0
            
        recent_gaze = list(self.gaze_buffer)[-window_frames:]
        
        gaze_x = [g[0] for g in recent_gaze]
        gaze_y = [g[1] for g in recent_gaze]
        
        std_x = np.std(gaze_x)
        std_y = np.std(gaze_y)
        
        # Combined stability metric
        stability = float(np.sqrt(std_x**2 + std_y**2))
        
        return stability


    def get_mean_ear(self, window_frames=30):
        """Get mean EAR over window"""
        if len(self.ear_buffer) == 0:
            return 0.0
        recent = list(self.ear_buffer)[-window_frames:]
        return float(np.mean(recent))


    def get_mean_mar(self, window_frames=45):
        """Get mean MAR over window"""
        if len(self.mar_buffer) == 0:
            return 0.0
        recent = list(self.mar_buffer)[-window_frames:]
        return float(np.mean(recent))


    def clear_buffers(self):
        """Clear all temporal buffers"""
        self.ear_buffer.clear()
        self.mar_buffer.clear()
        self.head_pose_buffer.clear()
        self.gaze_buffer.clear()
        self.blink_timestamps.clear()
        self.yawn_timestamps.clear()



class ValenceDetector:
    """
    Simple emotional valence detection from facial expressions
    """
    
    def __init__(self):
        self.valence_buffer = deque(maxlen=30)


    def calculate_valence(self, mouth_landmarks):
        """
        Calculate emotional valence from mouth shape
        
        Positive valence = smile (mouth corners up)
        Negative valence = frown (mouth corners down)
        Neutral = flat mouth
        
        Args:
            mouth_landmarks: Mouth landmark points
            
        Returns:
            Valence score (-1.0 to 1.0)
        """
        if len(mouth_landmarks) < 6:
            return 0.0
            
        pts = np.array(mouth_landmarks, dtype=np.float16)
        
        # Get mouth corners and center
        left_corner = pts[0]
        right_corner = pts[4]
        
        # Get top and bottom center points
        top_center = pts[2]
        bottom_center = pts[6] if len(pts) > 6 else pts[5]
        
        # Calculate mouth center
        mouth_center_y = (top_center[1] + bottom_center[1]) / 2.0
        
        # Calculate corner heights relative to center
        left_height = mouth_center_y - left_corner[1]
        right_height = mouth_center_y - right_corner[1]
        
        # Average height difference (positive = smile, negative = frown)
        avg_height = (left_height + right_height) / 2.0
        
        # Normalize to -1 to 1 range
        valence = np.clip(avg_height / 10.0, -1.0, 1.0)
        
        return float(valence)


    def update_valence(self, valence_value):
        """Add valence value to buffer"""
        self.valence_buffer.append(valence_value)


    def get_sustained_valence(self, threshold_seconds=30, fps=10):
        """
        Check if valence has been sustained in negative/neutral range
        
        Args:
            threshold_seconds: Duration to check
            fps: Frames per second
            
        Returns:
            Mean valence if sustained, else None
        """
        window_frames = threshold_seconds * fps
        
        if len(self.valence_buffer) < window_frames:
            return None
            
        recent_valence = list(self.valence_buffer)[-window_frames:]
        mean_valence = np.mean(recent_valence)
        
        # Check if sustained negative/neutral
        if mean_valence < 0.2:
            return float(mean_valence)
            
        return None
