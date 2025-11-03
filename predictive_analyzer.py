"""
PREDICTIVE DROWSINESS ANALYSIS SYSTEM
Analyzes trends and patterns to predict drowsiness before it becomes critical
"""

import numpy as np
from collections import deque
from typing import List, Tuple, Dict
from dataclasses import dataclass
from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler
import time


@dataclass
class PredictionResult:
    """Prediction analysis result"""
    drowsiness_score: float  # 0-100
    alert_level: str  # 'normal', 'attention', 'warning', 'danger', 'critical'
    confidence: float  # 0-1
    contributing_factors: Dict[str, float]
    trend_direction: str  # 'improving', 'stable', 'degrading'
    estimated_time_to_critical: float  # seconds, -1 if not applicable


class PredictiveAnalyzer:
    """
    Advanced predictive analysis using temporal patterns
    Detects drowsiness trends before they become dangerous
    """

    def __init__(self, config: dict):
        self.config = config
        self.prediction_config = config['prediction']

        # History buffers
        history_size = self.prediction_config['history_window']
        self.ear_history = deque(maxlen=history_size)
        self.mar_history = deque(maxlen=history_size)
        self.blink_rate_history = deque(maxlen=history_size)
        self.perclos_history = deque(maxlen=history_size)
        self.head_stability_history = deque(maxlen=history_size)
        self.yawn_timestamps = deque(maxlen=20)

        # Drowsiness score history
        self.drowsiness_score_history = deque(maxlen=history_size)

        # Feature tracking
        self.feature_trends = {
            'ear_trend': deque(maxlen=100),
            'blink_pattern': deque(maxlen=100),
            'yawn_frequency': deque(maxlen=100),
            'head_stability': deque(maxlen=100),
        }

        # Alert level thresholds
        self.alert_thresholds = self.prediction_config['alert_levels']

        # Weights for different features
        self.weights = self.prediction_config['weights']

    def calculate_ear_trend(self) -> float:
        """
        Calculate EAR trend score
        Decreasing EAR over time indicates increasing drowsiness
        Returns score 0-100 (higher = more drowsy)
        """
        if len(self.ear_history) < 30:
            return 0.0

        ear_array = np.array(list(self.ear_history))

        # Apply smoothing
        if len(ear_array) >= 5:
            ear_smooth = savgol_filter(ear_array, min(len(ear_array)//2*2-1, 51), 3)
        else:
            ear_smooth = ear_array

        # Calculate linear regression slope
        x = np.arange(len(ear_smooth))
        coeffs = np.polyfit(x, ear_smooth, 1)
        slope = coeffs[0]

        # Negative slope = decreasing EAR = drowsiness
        # Normalize slope to 0-100 score
        trend_score = max(0, min(100, -slope * 10000))

        # Consider variance (high variance can indicate struggling to stay awake)
        variance_score = np.std(ear_smooth) * 100

        # Combined score
        combined_score = (trend_score * 0.7) + (variance_score * 0.3)

        return min(100, combined_score)

    def analyze_blink_pattern(self) -> float:
        """
        Analyze blink patterns for drowsiness indicators
        Returns score 0-100 (higher = more abnormal pattern)
        """
        if len(self.blink_rate_history) < 10:
            return 0.0

        blink_rates = list(self.blink_rate_history)
        recent_rate = np.mean(blink_rates[-10:])

        min_normal = self.config['thresholds']['min_blink_frequency']
        max_normal = self.config['thresholds']['max_blink_frequency']

        score = 0.0

        # Too few blinks (drowsiness)
        if recent_rate < min_normal:
            score = ((min_normal - recent_rate) / min_normal) * 100

        # Too many blinks (fatigue/irritation)
        elif recent_rate > max_normal:
            score = ((recent_rate - max_normal) / max_normal) * 70

        # Check for irregular patterns
        if len(blink_rates) >= 20:
            variance = np.std(blink_rates[-20:])
            if variance > 10:  # High variance in blink rate
                score += variance * 2

        return min(100, score)

    def calculate_yawn_frequency(self, current_time: float) -> float:
        """
        Calculate yawn frequency score
        Returns score 0-100 (higher = more yawning)
        """
        # Remove old yawns (older than 5 minutes)
        while self.yawn_timestamps and (current_time - self.yawn_timestamps[0] > 300):
            self.yawn_timestamps.popleft()

        if len(self.yawn_timestamps) == 0:
            return 0.0

        # Calculate yawns per minute
        time_span = current_time - self.yawn_timestamps[0] if len(self.yawn_timestamps) > 1 else 60
        yawns_per_minute = (len(self.yawn_timestamps) / time_span) * 60

        # Convert to score
        threshold = self.config['thresholds']['yawns_per_minute_warning']
        score = (yawns_per_minute / threshold) * 100

        return min(100, score)

    def analyze_perclos_trend(self) -> float:
        """
        Analyze PERCLOS trend over time
        PERCLOS is industry-standard drowsiness metric
        Returns score 0-100
        """
        if len(self.perclos_history) < 10:
            return 0.0

        perclos_array = np.array(list(self.perclos_history))
        recent_perclos = np.mean(perclos_array[-10:])

        threshold = self.config['thresholds']['perclos_threshold'] * 100

        # Direct mapping
        score = (recent_perclos / threshold) * 100

        # Check if increasing
        if len(perclos_array) >= 30:
            x = np.arange(len(perclos_array[-30:]))
            coeffs = np.polyfit(x, perclos_array[-30:], 1)
            slope = coeffs[0]

            if slope > 0:  # Increasing PERCLOS
                score += slope * 1000

        return min(100, score)

    def analyze_head_movement_pattern(self) -> float:
        """
        Analyze head movement patterns
        Drowsy drivers show characteristic head movements
        Returns score 0-100
        """
        if len(self.head_stability_history) < 10:
            return 0.0

        stability_array = np.array(list(self.head_stability_history))
        recent_stability = np.mean(stability_array[-10:])

        # Lower stability = higher drowsiness score
        score = 100 - recent_stability

        # Check for sudden drops in stability
        if len(stability_array) >= 30:
            early_stability = np.mean(stability_array[-30:-10])
            if early_stability - recent_stability > 20:
                score += 20  # Rapid degradation

        return min(100, score)

    def calculate_comprehensive_score(self,
                                     ear_trend: float,
                                     blink_pattern: float,
                                     yawn_freq: float,
                                     perclos_trend: float,
                                     head_movement: float) -> float:
        """
        Calculate weighted comprehensive drowsiness score
        Returns score 0-100
        """
        # Apply weights
        weighted_score = (
            ear_trend * self.weights['ear_trend'] +
            blink_pattern * self.weights['blink_pattern'] +
            yawn_freq * self.weights['yawn_frequency'] +
            head_movement * self.weights['head_stability'] +
            perclos_trend * 0.2  # Additional weight for PERCLOS
        )

        return min(100, max(0, weighted_score))

    def determine_alert_level(self, score: float) -> str:
        """Determine alert level based on score"""
        if score >= self.alert_thresholds['critical']:
            return 'critical'
        elif score >= self.alert_thresholds['danger']:
            return 'danger'
        elif score >= self.alert_thresholds['warning']:
            return 'warning'
        elif score >= self.alert_thresholds['attention']:
            return 'attention'
        else:
            return 'normal'

    def calculate_trend_direction(self) -> str:
        """
        Calculate whether drowsiness is improving, stable, or degrading
        """
        if len(self.drowsiness_score_history) < 20:
            return 'stable'

        scores = list(self.drowsiness_score_history)
        recent = scores[-10:]
        previous = scores[-20:-10]

        recent_avg = np.mean(recent)
        previous_avg = np.mean(previous)

        diff = recent_avg - previous_avg

        if diff > 5:
            return 'degrading'
        elif diff < -5:
            return 'improving'
        else:
            return 'stable'

    def estimate_time_to_critical(self) -> float:
        """
        Estimate time until critical drowsiness level
        Returns seconds, or -1 if not applicable
        """
        if len(self.drowsiness_score_history) < 30:
            return -1

        scores = np.array(list(self.drowsiness_score_history))
        current_score = scores[-1]

        if current_score >= self.alert_thresholds['critical']:
            return 0  # Already critical

        # Fit trend line
        x = np.arange(len(scores))
        coeffs = np.polyfit(x, scores, 1)
        slope = coeffs[0]

        if slope <= 0:
            return -1  # Not increasing

        # Calculate frames to critical
        frames_to_critical = (self.alert_thresholds['critical'] - current_score) / slope

        # Convert to seconds (assuming 30 fps)
        seconds_to_critical = frames_to_critical / 30

        return max(0, seconds_to_critical)

    def update(self, ear: float, mar: float, blink_rate: float,
               perclos: float, head_stability: float, is_yawning: bool) -> None:
        """Update analyzer with new data"""
        current_time = time.time()

        self.ear_history.append(ear)
        self.mar_history.append(mar)
        self.blink_rate_history.append(blink_rate)
        self.perclos_history.append(perclos)
        self.head_stability_history.append(head_stability)

        if is_yawning:
            # Prevent duplicate timestamps
            if not self.yawn_timestamps or (current_time - self.yawn_timestamps[-1] > 5):
                self.yawn_timestamps.append(current_time)

    def predict(self) -> PredictionResult:
        """
        Main prediction method
        Returns comprehensive prediction result
        """
        current_time = time.time()

        # Calculate individual component scores
        ear_trend_score = self.calculate_ear_trend()
        blink_pattern_score = self.analyze_blink_pattern()
        yawn_freq_score = self.calculate_yawn_frequency(current_time)
        perclos_trend_score = self.analyze_perclos_trend()
        head_movement_score = self.analyze_head_movement_pattern()

        # Calculate comprehensive drowsiness score
        drowsiness_score = self.calculate_comprehensive_score(
            ear_trend_score,
            blink_pattern_score,
            yawn_freq_score,
            perclos_trend_score,
            head_movement_score
        )

        # Update history
        self.drowsiness_score_history.append(drowsiness_score)

        # Determine alert level
        alert_level = self.determine_alert_level(drowsiness_score)

        # Calculate confidence (based on data availability)
        data_completeness = min(len(self.ear_history) / 100, 1.0)
        confidence = data_completeness * 0.9  # Max 90% confidence

        # Analyze trend
        trend_direction = self.calculate_trend_direction()

        # Estimate time to critical
        time_to_critical = self.estimate_time_to_critical()

        # Contributing factors
        contributing_factors = {
            'ear_trend': ear_trend_score,
            'blink_pattern': blink_pattern_score,
            'yawn_frequency': yawn_freq_score,
            'perclos_trend': perclos_trend_score,
            'head_movement': head_movement_score
        }

        return PredictionResult(
            drowsiness_score=drowsiness_score,
            alert_level=alert_level,
            confidence=confidence,
            contributing_factors=contributing_factors,
            trend_direction=trend_direction,
            estimated_time_to_critical=time_to_critical
        )
