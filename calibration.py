"""
USER CALIBRATION SYSTEM
Establishes baseline metrics for personalized detection
"""

import cv2
import numpy as np
import time
from typing import Tuple, List
import os
import json


class CalibrationSystem:
    """
    Calibrates system to individual user characteristics
    """

    def __init__(self, config: dict):
        self.config = config
        self.calib_config = config['calibration']

        self.profile_dir = self.calib_config['profile_directory']
        os.makedirs(self.profile_dir, exist_ok=True)

    def run_calibration(self, detector, camera) -> Tuple[bool, dict]:
        """
        Run calibration process
        Returns (success, calibration_data)
        """
        print("\n" + "="*60)
        print("CALIBRATION PROCESS")
        print("="*60)
        print("\nPlease follow these instructions:")
        print("1. Sit in your normal driving position")
        print("2. Look straight ahead at the camera")
        print("3. Blink normally, don't force it")
        print("4. Stay alert and focused")
        print(f"\nCalibration duration: {self.calib_config['duration']} seconds")
        print("\nPress SPACE to start calibration...")

        # Wait for user
        while True:
            ret, frame = camera.read()
            if not ret:
                return False, {}

            frame = cv2.flip(frame, 1)

            # Display instructions
            h, w = frame.shape[:2]
            overlay = frame.copy()
            cv2.rectangle(overlay, (50, 50), (w-50, h-50), (0, 255, 0), 3)

            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(overlay, "Position yourself and press SPACE",
                       (w//4, h//2), font, 1, (0, 255, 0), 2)

            cv2.imshow('Calibration', overlay)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                break
            elif key == ord('q'):
                return False, {}

        # Run calibration
        duration = self.calib_config['duration']
        start_time = time.time()

        ear_values = []
        mar_values = []
        blink_count = 0
        frame_count = 0

        print("\nCalibrating...")

        while time.time() - start_time < duration:
            ret, frame = camera.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            frame_count += 1

            # Process frame
            metrics, indicators = detector.process_frame(frame)

            if metrics is not None:
                ear_values.append(metrics.ear_avg)
                mar_values.append(metrics.mar)

                if indicators and indicators.blink_rate > 0:
                    blink_count = int(indicators.blink_rate)

            # Display progress
            elapsed = time.time() - start_time
            remaining = duration - elapsed
            progress = elapsed / duration

            # Draw progress bar
            bar_width = frame.shape[1] - 100
            bar_height = 30
            bar_x = 50
            bar_y = frame.shape[0] - 100

            cv2.rectangle(frame, (bar_x, bar_y),
                         (bar_x + bar_width, bar_y + bar_height),
                         (255, 255, 255), 2)

            fill_width = int(bar_width * progress)
            cv2.rectangle(frame, (bar_x, bar_y),
                         (bar_x + fill_width, bar_y + bar_height),
                         (0, 255, 0), -1)

            # Display info
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, f"Calibrating: {remaining:.1f}s remaining",
                       (50, 50), font, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Samples collected: {len(ear_values)}",
                       (50, 90), font, 0.7, (255, 255, 255), 2)

            cv2.imshow('Calibration', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                return False, {}

        cv2.destroyWindow('Calibration')

        # Calculate calibration values
        if len(ear_values) < 30:
            print("\nCalibration failed: insufficient data")
            return False, {}

        calibration_data = {
            'baseline_ear': float(np.mean(ear_values)),
            'ear_std': float(np.std(ear_values)),
            'baseline_mar': float(np.mean(mar_values)),
            'mar_std': float(np.std(mar_values)),
            'avg_blink_rate': float(blink_count / (duration / 60)),
            'timestamp': time.time()
        }

        # Apply calibration to detector
        detector.calibrate(ear_values, mar_values)

        # Save profile
        if self.calib_config['save_profile']:
            self._save_profile(calibration_data)

        print("\n✓ Calibration successful!")
        print(f"  Baseline EAR: {calibration_data['baseline_ear']:.3f}")
        print(f"  Baseline MAR: {calibration_data['baseline_mar']:.3f}")
        print(f"  Blink rate: {calibration_data['avg_blink_rate']:.1f}/min")

        return True, calibration_data

    def _save_profile(self, calibration_data: dict):
        """Save calibration profile"""
        profile_path = os.path.join(self.profile_dir, 'default_profile.json')

        with open(profile_path, 'w') as f:
            json.dump(calibration_data, f, indent=4)

        print(f"\n✓ Profile saved to {profile_path}")

    def load_profile(self, profile_name: str = 'default_profile.json') -> dict:
        """Load calibration profile"""
        profile_path = os.path.join(self.profile_dir, profile_name)

        if os.path.exists(profile_path):
            with open(profile_path, 'r') as f:
                return json.load(f)

        return {}
