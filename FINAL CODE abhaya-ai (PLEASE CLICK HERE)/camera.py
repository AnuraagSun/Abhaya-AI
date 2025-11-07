# ═══════════════════════════════════════════════════════════════
# FILE: camera.py
# LOCATION: abhaya_ai/camera.py
# ═══════════════════════════════════════════════════════════════

"""
Abhaya AI - Camera Abstraction Layer

Platform-agnostic camera interface that works on:
- Laptop/Desktop (OpenCV VideoCapture)
- Raspberry Pi 3/4 (Picamera2)
- Raspberry Pi Model B (legacy picamera)

Automatically detects platform and uses appropriate camera API
with optimized settings for each hardware configuration.
"""

import cv2
import numpy as np
import logging
import time
import os
from typing import Optional, Tuple
from config import PLATFORM, PERF


logger = logging.getLogger(__name__)


# Suppress Qt platform plugin warnings
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
os.environ['QT_LOGGING_RULES'] = '*.debug=false;qt.qpa.*=false'


class CameraInterface:
    """
    Abstract base class for camera interfaces
    """

    def __init__(self, width, height, fps):
        self.width = width
        self.height = height
        self.fps = fps
        self.is_opened = False


    def open(self):
        """Open camera connection"""
        raise NotImplementedError


    def read(self):
        """
        Read frame from camera

        Returns:
            Tuple of (success, frame)
        """
        raise NotImplementedError


    def release(self):
        """Release camera resources"""
        raise NotImplementedError


    def get_properties(self):
        """Get camera properties"""
        return {
            'width': self.width,
            'height': self.height,
            'fps': self.fps
        }



class OpenCVCamera(CameraInterface):
    """
    OpenCV-based camera for laptop/desktop
    """

    def __init__(self, camera_id=0, width=640, height=480, fps=30):
        super().__init__(width, height, fps)
        self.camera_id = camera_id
        self.capture = None


    def open(self):
        """Open camera using OpenCV VideoCapture"""
        logger.info(f"Opening OpenCV camera {self.camera_id}")

        # Try different backends for better compatibility
        backends = [
            cv2.CAP_V4L2,    # Linux
            cv2.CAP_DSHOW,   # Windows
            cv2.CAP_AVFOUNDATION,  # macOS
            cv2.CAP_ANY      # Fallback
        ]

        for backend in backends:
            try:
                self.capture = cv2.VideoCapture(self.camera_id, backend)
                if self.capture.isOpened():
                    logger.info(f"Camera opened with backend: {backend}")
                    break
            except:
                continue

        if not self.capture or not self.capture.isOpened():
            logger.error("Failed to open camera with any backend")
            return False

        # Set camera properties
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.capture.set(cv2.CAP_PROP_FPS, self.fps)

        # Disable autofocus if possible (for stability)
        try:
            self.capture.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        except:
            pass

        # Verify settings
        actual_width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = int(self.capture.get(cv2.CAP_PROP_FPS))

        logger.info(f"Camera opened: {actual_width}x{actual_height} @ {actual_fps} fps")

        # Warm up camera
        for _ in range(5):
            self.capture.read()

        self.is_opened = True
        return True


    def read(self):
        """Read frame from camera"""
        if not self.is_opened or self.capture is None:
            return False, None

        success, frame = self.capture.read()

        if success and frame is not None:
            # Ensure frame is correct size
            if frame.shape[1] != self.width or frame.shape[0] != self.height:
                frame = cv2.resize(frame, (self.width, self.height))

        return success, frame


    def release(self):
        """Release camera"""
        if self.capture is not None:
            self.capture.release()
            self.is_opened = False
            logger.info("Camera released")



class PiCamera2Wrapper(CameraInterface):
    """
    Picamera2 wrapper for Raspberry Pi 3/4
    """

    def __init__(self, width=320, height=240, fps=15):
        super().__init__(width, height, fps)
        self.picam2 = None


    def open(self):
        """Open camera using Picamera2"""
        try:
            from picamera2 import Picamera2

            logger.info("Opening Picamera2")

            self.picam2 = Picamera2()

            # Configure camera
            config = self.picam2.create_preview_configuration(
                main={
                    "size": (self.width, self.height),
                    "format": "RGB888"
                },
                controls={
                    "FrameRate": self.fps
                }
            )

            self.picam2.configure(config)
            self.picam2.start()

            # Allow camera to warm up
            time.sleep(0.5)

            logger.info(f"Picamera2 opened: {self.width}x{self.height} @ {self.fps} fps")

            self.is_opened = True
            return True

        except ImportError:
            logger.error("Picamera2 not available")
            return False
        except Exception as e:
            logger.error(f"Failed to open Picamera2: {e}")
            return False


    def read(self):
        """Read frame from camera"""
        if not self.is_opened or self.picam2 is None:
            return False, None

        try:
            # Capture frame as numpy array
            frame = self.picam2.capture_array()

            # Convert RGB to BGR for OpenCV compatibility
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            return True, frame

        except Exception as e:
            logger.error(f"Failed to read frame: {e}")
            return False, None


    def release(self):
        """Release camera"""
        if self.picam2 is not None:
            self.picam2.stop()
            self.is_opened = False
            logger.info("Picamera2 released")



class LegacyPiCamera(CameraInterface):
    """
    Legacy picamera wrapper for older Raspberry Pi (Model B)
    """

    def __init__(self, width=160, height=120, fps=10):
        super().__init__(width, height, fps)
        self.camera = None
        self.raw_capture = None


    def open(self):
        """Open camera using legacy picamera"""
        try:
            from picamera import PiCamera
            from picamera.array import PiRGBArray

            logger.info("Opening legacy PiCamera")

            self.camera = PiCamera()
            self.camera.resolution = (self.width, self.height)
            self.camera.framerate = self.fps

            # Create raw capture buffer
            self.raw_capture = PiRGBArray(self.camera, size=(self.width, self.height))

            # Allow camera to warm up
            time.sleep(0.5)

            logger.info(f"Legacy PiCamera opened: {self.width}x{self.height} @ {self.fps} fps")

            self.is_opened = True
            return True

        except ImportError:
            logger.error("Legacy picamera not available")
            return False
        except Exception as e:
            logger.error(f"Failed to open legacy PiCamera: {e}")
            return False


    def read(self):
        """Read frame from camera"""
        if not self.is_opened or self.camera is None:
            return False, None

        try:
            # Clear stream
            self.raw_capture.truncate(0)

            # Capture frame
            self.camera.capture(self.raw_capture, format="bgr", use_video_port=True)

            # Get frame as numpy array
            frame = self.raw_capture.array

            return True, frame

        except Exception as e:
            logger.error(f"Failed to read frame: {e}")
            return False, None


    def release(self):
        """Release camera"""
        if self.camera is not None:
            self.camera.close()
            self.is_opened = False
            logger.info("Legacy PiCamera released")



class CameraManager:
    """
    High-level camera manager with automatic platform detection
    """

    def __init__(self, camera_id=0):
        """
        Initialize camera manager

        Args:
            camera_id: Camera device ID (for OpenCV)
        """
        self.camera_id = camera_id
        self.camera = None
        self.frame_count = 0
        self.skip_counter = 0

        # Get platform-optimized settings
        self.width = PERF['camera_width']
        self.height = PERF['camera_height']
        self.fps = PERF['target_fps']
        self.frame_skip = PERF['process_every_n_frames']


    def initialize(self):
        """
        Initialize camera with platform-appropriate driver

        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Initializing camera for platform: {PLATFORM.get_platform_name()}")
        logger.info(f"Target settings: {self.width}x{self.height} @ {self.fps} fps")

        # Try platform-specific cameras first
        if PLATFORM.is_raspberry_pi:

            # Try Picamera2 first (Pi 3/4)
            if PLATFORM.pi_model in ['PI_3_MODEL_B', 'PI_4']:
                logger.info("Attempting Picamera2...")
                camera = PiCamera2Wrapper(self.width, self.height, self.fps)
                if camera.open():
                    self.camera = camera
                    return True

            # Try legacy picamera (Pi Model B)
            if PLATFORM.pi_model in ['PI_MODEL_B', 'PI_B_PLUS']:
                logger.info("Attempting legacy PiCamera...")
                camera = LegacyPiCamera(self.width, self.height, self.fps)
                if camera.open():
                    self.camera = camera
                    return True

        # Fallback to OpenCV (works on all platforms)
        logger.info("Using OpenCV camera...")
        camera = OpenCVCamera(self.camera_id, self.width, self.height, self.fps)
        if camera.open():
            self.camera = camera
            return True

        logger.error("Failed to initialize any camera")
        return False


    def read_frame(self, skip_processing=False):
        """
        Read frame with optional frame skipping

        Args:
            skip_processing: If True, implements frame skip logic

        Returns:
            Tuple of (success, frame, should_process)
        """
        if self.camera is None:
            return False, None, False

        success, frame = self.camera.read()

        if not success:
            return False, None, False

        self.frame_count += 1

        # Frame skip logic
        should_process = True
        if skip_processing:
            self.skip_counter += 1
            if self.skip_counter < self.frame_skip:
                should_process = False
            else:
                self.skip_counter = 0

        return success, frame, should_process


    def get_frame(self):
        """
        Simple frame read without skip logic

        Returns:
            Tuple of (success, frame)
        """
        if self.camera is None:
            return False, None

        return self.camera.read()


    def release(self):
        """Release camera resources"""
        if self.camera is not None:
            self.camera.release()
            self.camera = None


    def get_resolution(self):
        """Get current camera resolution"""
        return (self.width, self.height)


    def get_fps_target(self):
        """Get target FPS"""
        return self.fps
