# ═══════════════════════════════════════════════════════════════
# FILE: face_detector.py
# LOCATION: abhaya_ai/face_detector.py
# ═══════════════════════════════════════════════════════════════

"""
Abhaya AI - Face Detection & Landmark Extraction

MediaPipe Face Mesh integration with platform-optimized settings.
Lazy loading and singleton pattern to minimize memory footprint.
Includes landmark persistence for stable tracking.
"""

import cv2
import numpy as np
import logging
import time
from typing import Optional, List, Tuple
import mediapipe as mp


from config import PLATFORM, PERF
from utils import performance_profiler


logger = logging.getLogger(__name__)


class FaceMeshDetector:
    """
    MediaPipe Face Mesh wrapper with optimization for constrained hardware
    """

    _instance = None
    _initialized = False


    def __new__(cls):
        """Singleton pattern to prevent multiple model loads"""
        if cls._instance is None:
            cls._instance = super(FaceMeshDetector, cls).__new__(cls)
        return cls._instance


    def __init__(self):
        """Initialize face mesh detector (lazy loading)"""
        if FaceMeshDetector._initialized:
            return

        logger.info("Initializing MediaPipe Face Mesh...")

        # MediaPipe components
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # Face mesh model
        self.face_mesh = None

        # Detection parameters based on platform
        self.use_static_mode = False
        self.max_num_faces = 1
        self.refine_landmarks = False

        # Increased confidence for more stable tracking
        self.min_detection_confidence = 0.7
        self.min_tracking_confidence = 0.7

        # Platform-specific optimizations
        if PLATFORM.is_raspberry_pi:
            if PLATFORM.pi_model == 'PI_MODEL_B':
                # Ultra-lightweight for Pi Model B
                self.refine_landmarks = False
                self.min_detection_confidence = 0.6
                self.min_tracking_confidence = 0.6
            else:
                # Pi 3 settings
                self.refine_landmarks = True
                self.min_detection_confidence = 0.65
                self.min_tracking_confidence = 0.65

        else:
            # Laptop/desktop settings - more stable tracking
            self.refine_landmarks = True

        FaceMeshDetector._initialized = True


    def load_model(self):
        """
        Lazy load the Face Mesh model
        """
        if self.face_mesh is not None:
            return True

        try:
            logger.info("Loading Face Mesh model")

            # Try to load with all parameters (newer MediaPipe versions)
            try:
                self.face_mesh = self.mp_face_mesh.FaceMesh(
                    static_image_mode=self.use_static_mode,
                    max_num_faces=self.max_num_faces,
                    refine_landmarks=self.refine_landmarks,
                    min_detection_confidence=self.min_detection_confidence,
                    min_tracking_confidence=self.min_tracking_confidence
                )
                logger.info("Face Mesh model loaded successfully (with refine_landmarks)")

            except TypeError as e:
                # Fallback for older MediaPipe versions without refine_landmarks
                logger.warning(f"refine_landmarks not supported, using basic model: {e}")
                self.face_mesh = self.mp_face_mesh.FaceMesh(
                    static_image_mode=self.use_static_mode,
                    max_num_faces=self.max_num_faces,
                    min_detection_confidence=self.min_detection_confidence,
                    min_tracking_confidence=self.min_tracking_confidence
                )
                self.refine_landmarks = False
                logger.info("Face Mesh model loaded successfully (basic mode)")

            return True

        except Exception as e:
            logger.error(f"Failed to load Face Mesh model: {e}")
            return False


    @performance_profiler.profile
    def detect_landmarks(self, frame):
        """
        Detect face landmarks in frame

        Args:
            frame: BGR image from camera

        Returns:
            Tuple of (landmarks, success)
            landmarks: numpy array of shape (468, 3) or (478, 3) with iris
            success: True if face detected
        """
        if self.face_mesh is None:
            if not self.load_model():
                return None, False

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Mark as not writable to improve performance
        rgb_frame.flags.writeable = False

        # Process frame
        results = self.face_mesh.process(rgb_frame)

        # Mark as writable again
        rgb_frame.flags.writeable = True

        if not results.multi_face_landmarks:
            return None, False

        # Get first face
        face_landmarks = results.multi_face_landmarks[0]

        # Convert to numpy array
        h, w = frame.shape[:2]
        landmarks = []

        for landmark in face_landmarks.landmark:
            x = landmark.x * w
            y = landmark.y * h
            z = landmark.z * w  # Relative depth
            landmarks.append([x, y, z])

        landmarks_array = np.array(landmarks, dtype=np.float32)

        return landmarks_array, True


    def draw_landmarks(self, frame, landmarks, draw_iris=True):
        """
        Draw face landmarks on frame

        Args:
            frame: BGR image
            landmarks: Landmark array from detect_landmarks
            draw_iris: Whether to draw iris landmarks

        Returns:
            Frame with landmarks drawn
        """
        if landmarks is None:
            return frame

        h, w = frame.shape[:2]

        # Draw face oval - sparse sampling for performance
        for idx in range(0, min(468, len(landmarks)), 5):
            x = int(landmarks[idx][0])
            y = int(landmarks[idx][1])
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

        # Draw key landmarks with larger circles
        key_landmarks = [
            (1, (0, 0, 255)),      # Nose tip (red)
            (33, (255, 0, 0)),     # Left eye (blue)
            (263, (255, 0, 0)),    # Right eye (blue)
            (61, (0, 255, 255)),   # Left mouth (yellow)
            (291, (0, 255, 255))   # Right mouth (yellow)
        ]

        for idx, color in key_landmarks:
            if idx < len(landmarks):
                x = int(landmarks[idx][0])
                y = int(landmarks[idx][1])
                cv2.circle(frame, (x, y), 4, color, -1)

        # Draw iris if available and enabled
        if draw_iris and self.refine_landmarks and len(landmarks) > 468:
            for idx in range(468, min(478, len(landmarks))):
                x = int(landmarks[idx][0])
                y = int(landmarks[idx][1])
                cv2.circle(frame, (x, y), 2, (255, 255, 0), -1)

        return frame


    def extract_eye_landmarks(self, landmarks, eye='left'):
        """
        Extract eye landmarks for EAR calculation

        Args:
            landmarks: Full face landmarks
            eye: 'left' or 'right'

        Returns:
            Array of 6 eye landmark points
        """
        from metrics import LEFT_EYE_INDICES, RIGHT_EYE_INDICES

        if landmarks is None:
            return None

        indices = LEFT_EYE_INDICES if eye == 'left' else RIGHT_EYE_INDICES

        eye_points = []
        for idx in indices:
            if idx < len(landmarks):
                eye_points.append(landmarks[idx][:2])  # Only x, y

        return np.array(eye_points, dtype=np.float32)


    def extract_mouth_landmarks(self, landmarks):
        """
        Extract mouth landmarks for MAR calculation

        Args:
            landmarks: Full face landmarks

        Returns:
            Array of mouth landmark points
        """
        from metrics import MOUTH_OUTER_INDICES

        if landmarks is None:
            return None

        mouth_points = []
        for idx in MOUTH_OUTER_INDICES:
            if idx < len(landmarks):
                mouth_points.append(landmarks[idx][:2])

        return np.array(mouth_points, dtype=np.float32)


    def extract_iris_landmarks(self, landmarks, eye='left'):
        """
        Extract iris landmarks for gaze calculation

        Args:
            landmarks: Full face landmarks
            eye: 'left' or 'right'

        Returns:
            Array of iris landmark points or None
        """
        from metrics import LEFT_IRIS_INDICES, RIGHT_IRIS_INDICES

        if landmarks is None or len(landmarks) <= 468:
            return None

        indices = LEFT_IRIS_INDICES if eye == 'left' else RIGHT_IRIS_INDICES

        iris_points = []
        for idx in indices:
            if idx < len(landmarks):
                iris_points.append(landmarks[idx][:2])

        if len(iris_points) == 0:
            return None

        return np.array(iris_points, dtype=np.float32)


    def release(self):
        """Release MediaPipe resources"""
        if self.face_mesh is not None:
            self.face_mesh.close()
            self.face_mesh = None
            logger.info("Face Mesh model released")



class FrameProcessor:
    """
    High-level frame processor combining camera and face detection
    with landmark persistence for stable tracking
    """

    def __init__(self, camera_manager, face_detector):
        """
        Initialize frame processor

        Args:
            camera_manager: CameraManager instance
            face_detector: FaceMeshDetector instance
        """
        self.camera = camera_manager
        self.detector = face_detector

        self.last_landmarks = None
        self.last_detection_time = 0

        # Persistence settings for stable tracking
        self.frames_since_detection = 0
        self.max_frames_without_detection = 5  # Keep landmarks for 5 frames
        self.consecutive_detections = 0
        self.min_consecutive_for_confidence = 3  # Need 3 detections before trusting


    def process_frame(self, frame, detect_face=True):
        """
        Process single frame with stable landmark tracking

        Args:
            frame: Input frame
            detect_face: Whether to run face detection

        Returns:
            Tuple of (processed_frame, landmarks, metrics_dict)
        """
        if frame is None:
            return None, None, {}

        current_time = time.time()
        landmarks = None
        detection_status = "no_face"

        # Always try to detect face
        if detect_face:
            landmarks, success = self.detector.detect_landmarks(frame)

            if success and landmarks is not None:
                # Face detected successfully
                self.consecutive_detections += 1
                self.frames_since_detection = 0

                # Only update landmarks if we have enough confidence
                if self.consecutive_detections >= self.min_consecutive_for_confidence:
                    self.last_landmarks = landmarks.copy()
                    detection_status = "tracking"
                else:
                    detection_status = "initializing"
                    # Use detected landmarks even during initialization
                    self.last_landmarks = landmarks.copy()

                self.last_detection_time = current_time

            else:
                # Face not detected
                self.consecutive_detections = 0
                self.frames_since_detection += 1

                # Use last known landmarks if within grace period
                if self.frames_since_detection <= self.max_frames_without_detection:
                    if self.last_landmarks is not None:
                        landmarks = self.last_landmarks
                        detection_status = "persisting"
                else:
                    # Lost tracking completely
                    self.last_landmarks = None
                    landmarks = None
                    detection_status = "lost"
        else:
            # Use cached landmarks
            landmarks = self.last_landmarks
            if landmarks is not None:
                detection_status = "cached"

        # Create metrics dictionary
        metrics = {
            'has_face': landmarks is not None,
            'num_landmarks': len(landmarks) if landmarks is not None else 0,
            'detection_status': detection_status,
            'consecutive_detections': self.consecutive_detections,
            'frames_since_detection': self.frames_since_detection
        }

        return frame, landmarks, metrics
