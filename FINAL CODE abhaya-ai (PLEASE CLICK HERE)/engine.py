# ═══════════════════════════════════════════════════════════════
# FILE: engine.py
# LOCATION: abhaya_ai/engine.py
# ═══════════════════════════════════════════════════════════════

"""
Abhaya AI - Core Engine Components

This module contains the core logic classes:
- UserProfileManager: Multi-user profile management
- CalibrationEngine: Personalized baseline calibration
- DrowsinessScorer: Temporal fusion and scoring engine
"""

import json
import os
import time
import numpy as np
import logging
from typing import Optional, Dict, List, Tuple
from datetime import datetime
from collections import deque


from config import (
    PathConfig,
    DrowsinessThresholds,
    TemporalConfig,
    CalibrationConfig,
    MetricsWeights,
    ModeProfiles
)
from metrics import TemporalMetrics, ValenceDetector


logger = logging.getLogger(__name__)


class UserProfile:
    """
    Individual user profile with personalized baselines
    """
    
    def __init__(self, user_id: str, name: str = "Unknown"):
        """
        Initialize user profile
        
        Args:
            user_id: Unique user identifier
            name: User's name
        """
        self.user_id = user_id
        self.name = name
        
        # Personalized baselines (to be calibrated)
        self.baseline_ear = 0.28
        self.baseline_mar = 0.35
        self.baseline_blink_rate = 15.0
        self.baseline_head_pose = {'pitch': 0.0, 'yaw': 0.0, 'roll': 0.0}
        
        # Personalized thresholds (derived from baselines)
        self.ear_threshold = DrowsinessThresholds.EAR_THRESHOLD
        self.mar_threshold = DrowsinessThresholds.MAR_THRESHOLD
        
        # Metric weights for drowsiness scoring
        self.metric_weights = MetricsWeights.get_default_weights()
        
        # Calibration status
        self.is_calibrated = False
        self.calibration_date = None
        self.calibration_samples = 0
        
        # Usage statistics
        self.total_sessions = 0
        self.total_alerts = 0
        self.false_alarms_reported = 0
        self.created_date = datetime.now().isoformat()
        self.last_used = None


    def calibrate(self, ear_samples: List[float], mar_samples: List[float], 
                  blink_rate_samples: List[float], head_pose_samples: List[Dict]):
        """
        Calibrate user baselines from collected samples
        
        Args:
            ear_samples: List of EAR values during calibration
            mar_samples: List of MAR values during calibration
            blink_rate_samples: List of blink rates
            head_pose_samples: List of head pose dictionaries
        """
        if len(ear_samples) < CalibrationConfig.MIN_SAMPLES:
            logger.warning(f"Insufficient EAR samples for calibration: {len(ear_samples)}")
            return False
            
        # Calculate baseline EAR (median of samples)
        self.baseline_ear = float(np.percentile(ear_samples, CalibrationConfig.EAR_PERCENTILE))
        
        # Derive personalized EAR threshold (80% of baseline)
        self.ear_threshold = self.baseline_ear * 0.8
        
        # Calculate baseline MAR
        if len(mar_samples) >= CalibrationConfig.MIN_SAMPLES:
            self.baseline_mar = float(np.percentile(mar_samples, CalibrationConfig.MAR_PERCENTILE))
            self.mar_threshold = self.baseline_mar * 1.5  # Yawn is 150% of baseline
        
        # Calculate baseline blink rate
        if len(blink_rate_samples) > 0:
            self.baseline_blink_rate = float(np.mean(blink_rate_samples))
        
        # Calculate baseline head pose
        if len(head_pose_samples) > 0:
            self.baseline_head_pose = {
                'pitch': float(np.mean([p['pitch'] for p in head_pose_samples])),
                'yaw': float(np.mean([p['yaw'] for p in head_pose_samples])),
                'roll': float(np.mean([p['roll'] for p in head_pose_samples]))
            }
        
        # Mark as calibrated
        self.is_calibrated = True
        self.calibration_date = datetime.now().isoformat()
        self.calibration_samples = len(ear_samples)
        
        logger.info(f"User {self.name} calibrated:")
        logger.info(f"  Baseline EAR: {self.baseline_ear:.3f} (threshold: {self.ear_threshold:.3f})")
        logger.info(f"  Baseline MAR: {self.baseline_mar:.3f} (threshold: {self.mar_threshold:.3f})")
        logger.info(f"  Baseline Blink Rate: {self.baseline_blink_rate:.1f} blinks/min")
        
        return True


    def update_metric_weights(self, new_weights: Dict[str, float]):
        """
        Update metric weights (for adaptive learning)
        
        Args:
            new_weights: Dictionary of metric weights
        """
        # Validate weights sum to approximately 1.0
        total_weight = sum(new_weights.values())
        
        if abs(total_weight - 1.0) > 0.1:
            logger.warning(f"Weights don't sum to 1.0: {total_weight}")
            # Normalize
            new_weights = {k: v/total_weight for k, v in new_weights.items()}
        
        self.metric_weights = new_weights
        logger.info(f"Updated metric weights for user {self.name}")


    def record_session(self, alerts_triggered: int):
        """Record session statistics"""
        self.total_sessions += 1
        self.total_alerts += alerts_triggered
        self.last_used = datetime.now().isoformat()


    def record_false_alarm(self):
        """Record a false alarm report"""
        self.false_alarms_reported += 1


    def to_dict(self) -> Dict:
        """Convert profile to dictionary for JSON serialization"""
        return {
            'user_id': self.user_id,
            'name': self.name,
            'baseline_ear': self.baseline_ear,
            'baseline_mar': self.baseline_mar,
            'baseline_blink_rate': self.baseline_blink_rate,
            'baseline_head_pose': self.baseline_head_pose,
            'ear_threshold': self.ear_threshold,
            'mar_threshold': self.mar_threshold,
            'metric_weights': self.metric_weights,
            'is_calibrated': self.is_calibrated,
            'calibration_date': self.calibration_date,
            'calibration_samples': self.calibration_samples,
            'total_sessions': self.total_sessions,
            'total_alerts': self.total_alerts,
            'false_alarms_reported': self.false_alarms_reported,
            'created_date': self.created_date,
            'last_used': self.last_used
        }


    @classmethod
    def from_dict(cls, data: Dict) -> 'UserProfile':
        """Create profile from dictionary"""
        profile = cls(data['user_id'], data.get('name', 'Unknown'))
        
        profile.baseline_ear = data.get('baseline_ear', 0.28)
        profile.baseline_mar = data.get('baseline_mar', 0.35)
        profile.baseline_blink_rate = data.get('baseline_blink_rate', 15.0)
        profile.baseline_head_pose = data.get('baseline_head_pose', {'pitch': 0.0, 'yaw': 0.0, 'roll': 0.0})
        
        profile.ear_threshold = data.get('ear_threshold', DrowsinessThresholds.EAR_THRESHOLD)
        profile.mar_threshold = data.get('mar_threshold', DrowsinessThresholds.MAR_THRESHOLD)
        
        profile.metric_weights = data.get('metric_weights', MetricsWeights.get_default_weights())
        
        profile.is_calibrated = data.get('is_calibrated', False)
        profile.calibration_date = data.get('calibration_date')
        profile.calibration_samples = data.get('calibration_samples', 0)
        
        profile.total_sessions = data.get('total_sessions', 0)
        profile.total_alerts = data.get('total_alerts', 0)
        profile.false_alarms_reported = data.get('false_alarms_reported', 0)
        
        profile.created_date = data.get('created_date', datetime.now().isoformat())
        profile.last_used = data.get('last_used')
        
        return profile



class UserProfileManager:
    """
    Manages multiple user profiles with persistence
    """
    
    def __init__(self, profiles_file: str = None):
        """
        Initialize profile manager
        
        Args:
            profiles_file: Path to profiles JSON file
        """
        self.profiles_file = profiles_file or PathConfig.USER_PROFILES_FILE
        self.profiles: Dict[str, UserProfile] = {}
        self.current_user: Optional[UserProfile] = None
        
        # Ensure data directory exists
        PathConfig.ensure_directories()
        
        # Load existing profiles
        self.load_profiles()


    def load_profiles(self):
        """Load profiles from JSON file"""
        if not os.path.exists(self.profiles_file):
            logger.info("No existing profiles file found, starting fresh")
            return
        
        try:
            with open(self.profiles_file, 'r') as f:
                data = json.load(f)
            
            for user_id, profile_data in data.items():
                self.profiles[user_id] = UserProfile.from_dict(profile_data)
            
            logger.info(f"Loaded {len(self.profiles)} user profiles")
            
        except Exception as e:
            logger.error(f"Failed to load profiles: {e}")


    def save_profiles(self):
        """Save profiles to JSON file"""
        try:
            data = {user_id: profile.to_dict() 
                   for user_id, profile in self.profiles.items()}
            
            with open(self.profiles_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Saved {len(self.profiles)} user profiles")
            
        except Exception as e:
            logger.error(f"Failed to save profiles: {e}")


    def create_user(self, user_id: str, name: str = "Unknown") -> UserProfile:
        """
        Create new user profile
        
        Args:
            user_id: Unique user identifier
            name: User's name
            
        Returns:
            New UserProfile instance
        """
        if user_id in self.profiles:
            logger.warning(f"User {user_id} already exists")
            return self.profiles[user_id]
        
        profile = UserProfile(user_id, name)
        self.profiles[user_id] = profile
        
        logger.info(f"Created new user profile: {name} ({user_id})")
        
        # Auto-save
        self.save_profiles()
        
        return profile


    def get_user(self, user_id: str) -> Optional[UserProfile]:
        """Get user profile by ID"""
        return self.profiles.get(user_id)


    def set_current_user(self, user_id: str):
        """Set the current active user"""
        if user_id not in self.profiles:
            logger.error(f"User {user_id} not found")
            return False
        
        self.current_user = self.profiles[user_id]
        logger.info(f"Current user set to: {self.current_user.name}")
        return True


    def list_users(self) -> List[Dict]:
        """Get list of all users with summary info"""
        users = []
        for profile in self.profiles.values():
            users.append({
                'user_id': profile.user_id,
                'name': profile.name,
                'calibrated': profile.is_calibrated,
                'sessions': profile.total_sessions,
                'last_used': profile.last_used
            })
        return users



class CalibrationEngine:
    """
    Personalized baseline calibration engine
    """
    
    def __init__(self, duration_seconds: int = CalibrationConfig.CALIBRATION_DURATION):
        """
        Initialize calibration engine
        
        Args:
            duration_seconds: Calibration duration in seconds
        """
        self.duration_seconds = duration_seconds
        self.is_calibrating = False
        self.start_time = None
        
        # Sample buffers
        self.ear_samples = []
        self.mar_samples = []
        self.blink_timestamps = []
        self.head_pose_samples = []
        
        self.last_sample_time = 0


    def start_calibration(self):
        """Start calibration process"""
        self.is_calibrating = True
        self.start_time = time.time()
        
        # Clear buffers
        self.ear_samples.clear()
        self.mar_samples.clear()
        self.blink_timestamps.clear()
        self.head_pose_samples.clear()
        
        logger.info(f"Calibration started (duration: {self.duration_seconds}s)")


    def add_sample(self, ear: float, mar: float, head_pose: Tuple[float, float, float], 
                   blink_detected: bool = False):
        """
        Add calibration sample
        
        Args:
            ear: Eye Aspect Ratio value
            mar: Mouth Aspect Ratio value
            head_pose: Tuple of (pitch, yaw, roll)
            blink_detected: Whether a blink was detected
        """
        if not self.is_calibrating:
            return
        
        current_time = time.time()
        
        # Sample at reasonable rate (max 10 Hz)
        if current_time - self.last_sample_time < 0.1:
            return
        
        self.ear_samples.append(ear)
        self.mar_samples.append(mar)
        
        self.head_pose_samples.append({
            'pitch': head_pose[0],
            'yaw': head_pose[1],
            'roll': head_pose[2]
        })
        
        if blink_detected:
            self.blink_timestamps.append(current_time)
        
        self.last_sample_time = current_time


    def get_progress(self) -> Tuple[float, int]:
        """
        Get calibration progress
        
        Returns:
            Tuple of (progress_percentage, samples_collected)
        """
        if not self.is_calibrating or self.start_time is None:
            return 0.0, 0
        
        elapsed = time.time() - self.start_time
        progress = min((elapsed / self.duration_seconds) * 100.0, 100.0)
        
        return progress, len(self.ear_samples)


    def is_complete(self) -> bool:
        """Check if calibration is complete"""
        if not self.is_calibrating:
            return False
        
        elapsed = time.time() - self.start_time
        return elapsed >= self.duration_seconds


    def finalize_calibration(self, user_profile: UserProfile) -> bool:
        """
        Finalize calibration and update user profile
        
        Args:
            user_profile: UserProfile to calibrate
            
        Returns:
            True if successful, False otherwise
        """
        if not self.is_calibrating:
            logger.error("Calibration not started")
            return False
        
        # Calculate blink rate samples
        blink_rate_samples = []
        if len(self.blink_timestamps) >= 2:
            time_span = self.blink_timestamps[-1] - self.blink_timestamps[0]
            if time_span > 0:
                blinks_per_minute = (len(self.blink_timestamps) / time_span) * 60.0
                blink_rate_samples.append(blinks_per_minute)
        
        # Calibrate user profile
        success = user_profile.calibrate(
            self.ear_samples,
            self.mar_samples,
            blink_rate_samples,
            self.head_pose_samples
        )
        
        self.is_calibrating = False
        
        if success:
            logger.info(f"Calibration completed successfully with {len(self.ear_samples)} samples")
        else:
            logger.error("Calibration failed")
        
        return success



class DrowsinessScorer:
    """
    Temporal fusion engine for drowsiness scoring
    """
    
    def __init__(self, user_profile: UserProfile, mode: str = 'driving'):
        """
        Initialize drowsiness scorer
        
        Args:
            user_profile: User profile with personalized baselines
            mode: Operating mode ('driving' or 'study')
        """
        self.user_profile = user_profile
        self.mode_profile = ModeProfiles.get_mode(mode)
        
        # Current drowsiness score (0-100)
        self.drowsiness_score = 0.0
        
        # Score history for EMA
        self.score_history = deque(maxlen=30)
        
        # Alert level (0-3)
        self.alert_level = 0
        self.last_alert_level = 0
        
        # Contributing factors for explainability
        self.current_factors = {}
        
        # Time tracking
        self.last_update_time = time.time()


    def update_score(self, perclos: float, yawn_frequency: int, head_nod_detected: bool,
                    blink_rate: float, gaze_stability: float, valence: float,
                    distraction_detected: bool):
        """
        Update drowsiness score based on all metrics
        
        Args:
            perclos: Percentage of eye closure (0-100)
            yawn_frequency: Number of yawns in window
            head_nod_detected: Whether head nod detected
            blink_rate: Blinks per minute
            gaze_stability: Gaze deviation std dev
            valence: Emotional valence (-1 to 1)
            distraction_detected: Whether distraction detected
        """
        current_time = time.time()
        time_delta = current_time - self.last_update_time
        
        # Calculate individual metric scores (0-100)
        scores = {}
        
        # PERCLOS score
        perclos_score = min(perclos * 5.0, 100.0)  # >20% PERCLOS = 100 score
        scores['perclos'] = perclos_score
        
        # Yawn frequency score
        yawn_score = min(yawn_frequency * 25.0, 100.0)  # 4+ yawns = 100 score
        scores['yawn_frequency'] = yawn_score
        
        # Head nod score
        head_nod_score = 100.0 if head_nod_detected else 0.0
        scores['head_nod'] = head_nod_score
        
        # Blink frequency score (both too low and too high are concerning)
        baseline_blink = self.user_profile.baseline_blink_rate
        blink_deviation = abs(blink_rate - baseline_blink) / baseline_blink if baseline_blink > 0 else 0
        blink_score = min(blink_deviation * 100.0, 100.0)
        scores['blink_frequency'] = blink_score
        
        # Gaze stability score (higher deviation = higher score)
        gaze_score = min(gaze_stability * 200.0, 100.0)
        scores['gaze_stability'] = gaze_score
        
        # Valence score (negative/neutral valence is concerning)
        valence_score = max(0, -valence * 50.0)  # -1 valence = 50 score
        scores['valence'] = valence_score
        
        # Distraction score
        distraction_score = 50.0 if distraction_detected else 0.0
        scores['distraction'] = distraction_score
        
        # Apply mode-specific multipliers
        if self.mode_profile:
            scores['perclos'] *= self.mode_profile.get('ear_multiplier', 1.0)
            scores['yawn_frequency'] *= self.mode_profile.get('mar_multiplier', 1.0)
            scores['head_nod'] *= self.mode_profile.get('head_pose_multiplier', 1.0)
            scores['gaze_stability'] *= self.mode_profile.get('gaze_multiplier', 1.0)
        
        # Weighted combination
        weights = self.user_profile.metric_weights
        weighted_score = sum(scores[metric] * weights[metric] 
                           for metric in scores.keys() if metric in weights)
        
        # Apply Exponential Moving Average for smoothing
        alpha = TemporalConfig.EMA_ALPHA
        if len(self.score_history) > 0:
            self.drowsiness_score = alpha * weighted_score + (1 - alpha) * self.drowsiness_score
        else:
            self.drowsiness_score = weighted_score
        
        # Gradual decay during alertness
        if weighted_score < 20.0 and time_delta > 0:
            decay = TemporalConfig.SCORE_DECAY_RATE * time_delta
            self.drowsiness_score = max(0, self.drowsiness_score - decay)
        
        # Clamp score
        self.drowsiness_score = np.clip(self.drowsiness_score, 0.0, 100.0)
        
        # Add to history
        self.score_history.append(self.drowsiness_score)
        
        # Store current factors for explainability
        self.current_factors = scores
        
        # Update alert level with hysteresis
        self._update_alert_level()
        
        self.last_update_time = current_time


    def _update_alert_level(self):
        """Update alert level with hysteresis to prevent oscillation"""
        score = self.drowsiness_score
        hysteresis = DrowsinessThresholds.SCORE_HYSTERESIS
        
        # Determine raw level
        if score >= DrowsinessThresholds.SCORE_LEVEL_3:
            raw_level = 3
        elif score >= DrowsinessThresholds.SCORE_LEVEL_2:
            raw_level = 2
        elif score >= DrowsinessThresholds.SCORE_LEVEL_1:
            raw_level = 1
        else:
            raw_level = 0
        
        # Apply hysteresis for de-escalation
        if raw_level < self.alert_level:
            # Require score to drop below (threshold - hysteresis) to de-escalate
            if self.alert_level == 3 and score < (DrowsinessThresholds.SCORE_LEVEL_3 - hysteresis):
                self.alert_level = 2
            elif self.alert_level == 2 and score < (DrowsinessThresholds.SCORE_LEVEL_2 - hysteresis):
                self.alert_level = 1
            elif self.alert_level == 1 and score < (DrowsinessThresholds.SCORE_LEVEL_1 - hysteresis):
                self.alert_level = 0
        else:
            # Escalate immediately
            self.alert_level = raw_level


    def get_score(self) -> float:
        """Get current drowsiness score"""
        return self.drowsiness_score


    def get_alert_level(self) -> int:
        """Get current alert level (0-3)"""
        return self.alert_level


    def get_explanation(self) -> Dict:
        """
        Get explanation of current score
        
        Returns:
            Dictionary with score breakdown and primary factors
        """
        # Sort factors by contribution
        sorted_factors = sorted(self.current_factors.items(), 
                               key=lambda x: x[1], reverse=True)
        
        # Get top 3 contributing factors
        top_factors = [
            {'metric': factor[0], 'score': factor[1], 
             'weight': self.user_profile.metric_weights.get(factor[0], 0)}
            for factor in sorted_factors[:3]
        ]
        
        return {
            'drowsiness_score': self.drowsiness_score,
            'alert_level': self.alert_level,
            'top_factors': top_factors,
            'all_factors': self.current_factors
        }


    def reset_score(self):
        """Reset drowsiness score (e.g., after user interaction)"""
        self.drowsiness_score = max(0, self.drowsiness_score * 0.5)
        logger.info("Drowsiness score reset")