

"""
Abhaya AI - Centralized Configuration Module
═══════════════════════════════════════════════════════════════
FILE: config.py
LOCATION: abhaya_ai/config.py
═══════════════════════════════════════════════════════════════
This module contains all configuration constants, thresholds, and
platform-specific settings. It automatically detects the hardware
platform and adjusts parameters accordingly for optimal performance.
"""

import os
import platform
import multiprocessing


class PlatformConfig:
    """
    Detects and stores platform-specific information
    """

    def __init__(self):
        self.is_raspberry_pi = False
        self.pi_model = None
        self.total_ram_mb = self._estimate_ram()
        self.cpu_count = multiprocessing.cpu_count()
        self.architecture = platform.machine()

        self._detect_platform()


    def _detect_platform(self):
        """
        Identify if running on Raspberry Pi and which model
        """
        try:
            with open('/proc/cpuinfo', 'r') as f:
                cpuinfo = f.read()

            if 'Raspberry Pi' in cpuinfo or 'BCM' in cpuinfo:
                self.is_raspberry_pi = True

                # Check for Pi model
                if 'BCM2835' in cpuinfo:
                    # Original Pi, Pi Zero
                    if self.total_ram_mb <= 512:
                        self.pi_model = 'PI_MODEL_B'
                    else:
                        self.pi_model = 'PI_B_PLUS'

                elif 'BCM2836' in cpuinfo or 'BCM2837' in cpuinfo:
                    # Pi 2 or Pi 3
                    self.pi_model = 'PI_3_MODEL_B'

                elif 'BCM2711' in cpuinfo:
                    # Pi 4
                    self.pi_model = 'PI_4'

        except FileNotFoundError:
            # Not a Linux system or no cpuinfo available
            self.is_raspberry_pi = False


    def _estimate_ram(self):
        """
        Estimate total system RAM in MB
        """
        try:
            with open('/proc/meminfo', 'r') as f:
                for line in f:
                    if 'MemTotal' in line:
                        # Extract KB value and convert to MB
                        ram_kb = int(line.split()[1])
                        return ram_kb // 1024
        except:
            # Fallback for non-Linux systems
            return 8192  # Assume 8GB for development laptops


    def get_platform_name(self):
        """
        Return human-readable platform name
        """
        if not self.is_raspberry_pi:
            return "Laptop/Desktop"

        model_names = {
            'PI_MODEL_B': 'Raspberry Pi Model B (512MB)',
            'PI_B_PLUS': 'Raspberry Pi B+',
            'PI_3_MODEL_B': 'Raspberry Pi 3 Model B',
            'PI_4': 'Raspberry Pi 4'
        }

        return model_names.get(self.pi_model, 'Raspberry Pi (Unknown Model)')



# Initialize platform detection
PLATFORM = PlatformConfig()


class PerformanceProfile:
    """
    Platform-specific performance configurations
    """

    # Laptop/Desktop configuration
    LAPTOP = {
        'camera_width': 640,
        'camera_height': 480,
        'target_fps': 30,
        'process_every_n_frames': 1,
        'max_memory_mb': 800,
        'use_face_recognition': True,
        'face_mesh_complexity': 1,
        'enable_video_recording': True,
        'use_realtime_tts': True,
        'buffer_size_seconds': 30,
        'gc_interval_seconds': 120,
    }


    # Raspberry Pi 3 Model B configuration
    PI_3 = {
        'camera_width': 320,
        'camera_height': 240,
        'target_fps': 15,
        'process_every_n_frames': 2,
        'max_memory_mb': 600,
        'use_face_recognition': True,
        'face_mesh_complexity': 0,
        'enable_video_recording': False,
        'use_realtime_tts': False,
        'buffer_size_seconds': 30,
        'gc_interval_seconds': 60,
    }


    # Raspberry Pi Model B (512MB) - Ultra-lightweight configuration
    PI_MODEL_B = {
        'camera_width': 160,
        'camera_height': 120,
        'target_fps': 10,
        'process_every_n_frames': 3,
        'max_memory_mb': 350,
        'use_face_recognition': False,
        'face_mesh_complexity': 0,
        'enable_video_recording': False,
        'use_realtime_tts': False,
        'buffer_size_seconds': 20,
        'gc_interval_seconds': 45,
    }


    @staticmethod
    def get_current_profile():
        """
        Return the appropriate performance profile for current platform
        """
        if not PLATFORM.is_raspberry_pi:
            return PerformanceProfile.LAPTOP

        if PLATFORM.pi_model == 'PI_MODEL_B':
            return PerformanceProfile.PI_MODEL_B
        elif PLATFORM.pi_model == 'PI_3_MODEL_B':
            return PerformanceProfile.PI_3
        else:
            # Default to Pi 3 settings for unknown Pi models
            return PerformanceProfile.PI_3



# Get active performance profile
PERF = PerformanceProfile.get_current_profile()


class DrowsinessThresholds:
    """
    Thresholds for drowsiness detection metrics
    These are baseline values that get personalized during calibration
    """

    # Eye Aspect Ratio (EAR) thresholds
    EAR_THRESHOLD = 0.25
    EAR_CONSEC_FRAMES = 3

    # Mouth Aspect Ratio (MAR) for yawn detection
    MAR_THRESHOLD = 0.6
    MAR_CONSEC_FRAMES = 3

    # Head pose thresholds (degrees)
    HEAD_PITCH_THRESHOLD = 15.0
    HEAD_YAW_THRESHOLD = 20.0
    HEAD_ROLL_THRESHOLD = 15.0

    # Blink frequency (blinks per minute)
    BLINK_FREQ_LOW = 8
    BLINK_FREQ_HIGH = 30

    # PERCLOS (Percentage of Eye Closure) threshold
    PERCLOS_THRESHOLD = 0.20

    # Gaze deviation threshold (pixels from center)
    GAZE_DEVIATION_THRESHOLD = 50

    # Drowsiness score levels (0-100)
    SCORE_LEVEL_1 = 40
    SCORE_LEVEL_2 = 60
    SCORE_LEVEL_3 = 80

    # Hysteresis for level de-escalation
    SCORE_HYSTERESIS = 5



class ModeProfiles:
    """
    Different operational modes with adjusted sensitivities
    """

    DRIVING = {
        'name': 'Driving Mode',
        'ear_multiplier': 1.0,
        'mar_multiplier': 1.0,
        'head_pose_multiplier': 1.2,
        'gaze_multiplier': 0.8,
        'alert_aggressiveness': 1.0,
    }


    STUDY = {
        'name': 'Study Mode',
        'ear_multiplier': 1.2,
        'mar_multiplier': 0.7,
        'head_pose_multiplier': 1.5,
        'gaze_multiplier': 0.5,
        'alert_aggressiveness': 0.6,
    }


    @staticmethod
    def get_mode(mode_name):
        """
        Get mode profile by name
        """
        modes = {
            'driving': ModeProfiles.DRIVING,
            'study': ModeProfiles.STUDY,
        }
        return modes.get(mode_name.lower(), ModeProfiles.DRIVING)



class TemporalConfig:
    """
    Configuration for temporal analysis and buffering
    """

    # Calculate buffer sizes based on platform capabilities
    _buffer_duration = PERF['buffer_size_seconds']
    _fps = PERF['target_fps']

    # Maximum frames to store for temporal analysis
    MAX_FRAME_BUFFER = int(_buffer_duration * _fps)

    # Rolling window sizes (in frames)
    EAR_WINDOW = min(30, MAX_FRAME_BUFFER)
    MAR_WINDOW = min(45, MAX_FRAME_BUFFER)
    BLINK_WINDOW = min(60, MAX_FRAME_BUFFER)
    GAZE_WINDOW = min(15, MAX_FRAME_BUFFER)
    HEAD_POSE_WINDOW = min(30, MAX_FRAME_BUFFER)
    VALENCE_WINDOW = min(20, MAX_FRAME_BUFFER)

    # Exponential Moving Average (EMA) decay factor
    EMA_ALPHA = 0.3

    # Score decay rate (points per second during alertness)
    SCORE_DECAY_RATE = 2.0



class CalibrationConfig:
    """
    Configuration for personalized calibration phase
    """

    # Duration of calibration (seconds)
    CALIBRATION_DURATION = 15

    # Minimum samples required for valid calibration
    MIN_SAMPLES = 50

    # Percentile to use for baseline calculation
    EAR_PERCENTILE = 50
    MAR_PERCENTILE = 30

    # Calibration instructions
    INSTRUCTIONS = [
        "Look straight ahead at the camera",
        "Keep your eyes open normally",
        "Maintain a neutral expression",
        "Sit in your typical position",
    ]



class HardwareConfig:
    """
    GPIO pin configuration for Raspberry Pi
    """

    # GPIO pin assignments (BCM numbering)
    LED_ALERT_LEVEL_1 = 17
    LED_ALERT_LEVEL_2 = 27
    LED_ALERT_LEVEL_3 = 22

    BUTTON_DISMISS = 23
    BUTTON_CALIBRATE = 24

    LIGHT_SENSOR_PIN = 25

    BUZZER_PIN = 18

    # PWM frequency for buzzer (Hz)
    BUZZER_FREQ = 2000



class PathConfig:
    """
    File system paths for data storage
    """

    # Base directory
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # Data directories
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    LOGS_DIR = os.path.join(DATA_DIR, 'logs')
    PROFILES_DIR = DATA_DIR

    # Asset directories
    ASSETS_DIR = os.path.join(BASE_DIR, 'assets')
    AUDIO_DIR = os.path.join(ASSETS_DIR, 'audio')

    # File paths
    USER_PROFILES_FILE = os.path.join(PROFILES_DIR, 'user_profiles.json')

    # Audio file paths
    ALERT_LEVEL_1_AUDIO = os.path.join(AUDIO_DIR, 'alert_level1.wav')
    ALERT_LEVEL_2_AUDIO = os.path.join(AUDIO_DIR, 'alert_level2.wav')
    ALERT_LEVEL_3_AUDIO = os.path.join(AUDIO_DIR, 'alert_level3.wav')


    @staticmethod
    def ensure_directories():
        """
        Create necessary directories if they don't exist
        """
        os.makedirs(PathConfig.DATA_DIR, exist_ok=True)
        os.makedirs(PathConfig.LOGS_DIR, exist_ok=True)
        os.makedirs(PathConfig.AUDIO_DIR, exist_ok=True)



class MetricsWeights:
    """
    Default weights for combining multiple drowsiness metrics
    into final score. These can be personalized per user.
    """

    PERCLOS = 0.30
    YAWN_FREQUENCY = 0.20
    HEAD_NOD = 0.15
    BLINK_FREQUENCY = 0.10
    GAZE_STABILITY = 0.10
    VALENCE = 0.10
    DISTRACTION = 0.05


    @staticmethod
    def get_default_weights():
        """
        Return default weights as dictionary
        """
        return {
            'perclos': MetricsWeights.PERCLOS,
            'yawn_frequency': MetricsWeights.YAWN_FREQUENCY,
            'head_nod': MetricsWeights.HEAD_NOD,
            'blink_frequency': MetricsWeights.BLINK_FREQUENCY,
            'gaze_stability': MetricsWeights.GAZE_STABILITY,
            'valence': MetricsWeights.VALENCE,
            'distraction': MetricsWeights.DISTRACTION,
        }



# Version information
VERSION = "1.0.0"
APP_NAME = "Abhaya AI"


# Debug and logging configuration
DEBUG_MODE = os.environ.get('ABHAYA_DEBUG', 'False').lower() == 'true'
LOG_LEVEL = 'DEBUG' if DEBUG_MODE else 'INFO'
