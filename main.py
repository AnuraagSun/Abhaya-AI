#!/usr/bin/env python3
"""
INDUSTRIAL DRIVER DROWSINESS PREDICTION SYSTEM
Main application with full integration
"""

import cv2
import yaml
import sys
import time
import numpy as np
from pathlib import Path

from drowsiness_detector import DrowsinessDetector
from predictive_analyzer import PredictiveAnalyzer
from alert_system import AlertSystem
from calibration import CalibrationSystem
from data_logger import DataLogger


class DrowsinessMonitoringSystem:
    """
    Main system integrating all components
    """

    def __init__(self, config_path: str = 'config.yaml'):
        """Initialize system"""
        print("="*60)
        print("INDUSTRIAL DROWSINESS PREDICTION SYSTEM")
        print("="*60)

        # Load configuration
        print("\n[1/7] Loading configuration...")
        self.config = self._load_config(config_path)
        print("✓ Configuration loaded")

        # Initialize camera
        print("\n[2/7] Initializing camera...")
        self.camera = self._init_camera()
        print("✓ Camera initialized")

        # Initialize detector
        print("\n[3/7] Initializing drowsiness detector...")
        self.detector = DrowsinessDetector(self.config)
        print("✓ Detector ready")

        # Initialize predictive analyzer
        print("\n[4/7] Initializing predictive analyzer...")
        self.analyzer = PredictiveAnalyzer(self.config)
        print("✓ Analyzer ready")

        # Initialize alert system
        print("\n[5/7] Initializing alert system...")
        self.alert_system = AlertSystem(self.config)
        print("✓ Alert system ready")

        # Initialize calibration system
        print("\n[6/7] Initializing calibration system...")
        self.calibration = CalibrationSystem(self.config)
        print("✓ Calibration system ready")

        # Initialize data logger
        print("\n[7/7] Initializing data logger...")
        self.logger = DataLogger(self.config)
        print("✓ Logger ready")

        # System state
        self.running = False
        self.paused = False
        self.show_help = False

        print("\n✓ System initialization complete!")

    def _load_config(self, config_path: str) -> dict:
        """Load YAML configuration"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            print(f"Error: Configuration file '{config_path}' not found")
            sys.exit(1)

    def _init_camera(self) -> cv2.VideoCapture:
        """Initialize camera"""
        camera_config = self.config['camera']
        source = camera_config['source']

        camera = cv2.VideoCapture(source)

        if not camera.isOpened():
            print(f"Error: Cannot open camera {source}")
            sys.exit(1)

        # Set camera properties
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, camera_config['width'])
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_config['height'])
        camera.set(cv2.CAP_PROP_FPS, camera_config['fps'])

        return camera

    def run_calibration(self) -> bool:
        """Run calibration process"""
        if self.config['calibration']['required_on_start']:
            success, calib_data = self.calibration.run_calibration(
                self.detector,
                self.camera
            )
            return success
        return True

    def draw_ui(self, frame: np.ndarray, metrics, indicators, prediction, fps: float) -> np.ndarray:
        """Draw comprehensive UI overlay"""
        overlay = frame.copy()
        h, w = overlay.shape[:2]

        if metrics is None:
            # No face detected
            cv2.putText(overlay, "NO FACE DETECTED", (w//2 - 150, h//2),
                       cv2.FONT_HERSHEY_BOLD, 1, (0, 0, 255), 2)
            return overlay

        # Draw landmarks if enabled
        if self.config['display']['show_landmarks']:
            # This would require storing landmarks from detector
            pass

        # Info panel background
        panel_width = 350
        panel_alpha = self.config['display']['overlay_opacity']

        # Semi-transparent background
        cv2.rectangle(overlay, (0, 0), (panel_width, h), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 1 - panel_alpha, overlay, panel_alpha, 0)
        overlay = frame.copy()

        # Display metrics
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 2
        line_height = 30
        x = 10
        y = 30

        # Title
        cv2.putText(overlay, "DROWSINESS MONITOR", (x, y),
                   cv2.FONT_HERSHEY_BOLD, 0.7, (0, 255, 255), 2)
        y += line_height + 10

        # FPS
        cv2.putText(overlay, f"FPS: {fps:.1f}", (x, y),
                   font, font_scale, (255, 255, 255), font_thickness)
        y += line_height

        # Separator
        cv2.line(overlay, (x, y), (panel_width - 10, y), (100, 100, 100), 1)
        y += 20

        # Facial Metrics
        cv2.putText(overlay, "FACIAL METRICS", (x, y),
                   font, font_scale, (0, 255, 255), font_thickness)
        y += line_height

        # EAR
        ear_color = (0, 255, 0) if metrics.ear_avg > 0.25 else (0, 0, 255)
        cv2.putText(overlay, f"EAR: {metrics.ear_avg:.3f}", (x, y),
                   font, font_scale, ear_color, font_thickness)
        y += line_height

        # MAR
        cv2.putText(overlay, f"MAR: {metrics.mar:.3f}", (x, y),
                   font, font_scale, (255, 255, 255), font_thickness)
        y += line_height

        # Head Pose
        cv2.putText(overlay, f"Pitch: {metrics.head_pitch:.1f}°", (x, y),
                   font, font_scale, (255, 255, 255), font_thickness)
        y += line_height

        cv2.putText(overlay, f"Yaw: {metrics.head_yaw:.1f}°", (x, y),
                   font, font_scale, (255, 255, 255), font_thickness)
        y += line_height

        # Separator
        cv2.line(overlay, (x, y), (panel_width - 10, y), (100, 100, 100), 1)
        y += 20

        if indicators:
            # Indicators
            cv2.putText(overlay, "INDICATORS", (x, y),
                       font, font_scale, (0, 255, 255), font_thickness)
            y += line_height

            # Blink Rate
            blink_color = (0, 255, 0)
            if indicators.blink_rate < 10 or indicators.blink_rate > 35:
                blink_color = (0, 165, 255)

            cv2.putText(overlay, f"Blink: {indicators.blink_rate:.1f}/min", (x, y),
                       font, font_scale, blink_color, font_thickness)
            y += line_height

            # PERCLOS
            perclos_color = (0, 255, 0) if indicators.perclos < 15 else (0, 0, 255)
            cv2.putText(overlay, f"PERCLOS: {indicators.perclos:.1f}%", (x, y),
                       font, font_scale, perclos_color, font_thickness)
            y += line_height

            # Status indicators
            if indicators.is_eyes_closed:
                cv2.putText(overlay, "● Eyes Closed", (x, y),
                           font, font_scale, (0, 0, 255), font_thickness)
                y += line_height

            if indicators.is_yawning:
                cv2.putText(overlay, "● Yawning", (x, y),
                           font, font_scale, (0, 165, 255), font_thickness)
                y += line_height

            if indicators.is_microsleep:
                cv2.putText(overlay, "● MICROSLEEP!", (x, y),
                           font, font_scale, (0, 0, 255), font_thickness)
                y += line_height

        # Separator
        y += 10
        cv2.line(overlay, (x, y), (panel_width - 10, y), (100, 100, 100), 1)
        y += 20

        if prediction:
            # Prediction Results
            cv2.putText(overlay, "PREDICTION", (x, y),
                       font, font_scale, (0, 255, 255), font_thickness)
            y += line_height

            # Drowsiness Score with bar
            score = prediction.drowsiness_score
            score_color = self._get_score_color(score)

            cv2.putText(overlay, f"Score: {score:.0f}%", (x, y),
                       font, font_scale, score_color, font_thickness)
            y += line_height

            # Score bar
            bar_width = panel_width - 20
            bar_height = 20
            bar_x = x
            bar_y = y

            # Background
            cv2.rectangle(overlay, (bar_x, bar_y),
                         (bar_x + bar_width, bar_y + bar_height),
                         (100, 100, 100), -1)

            # Fill
            fill_width = int(bar_width * (score / 100))
            cv2.rectangle(overlay, (bar_x, bar_y),
                         (bar_x + fill_width, bar_y + bar_height),
                         score_color, -1)

            y += bar_height + 15

            # Alert Level
            level_colors = {
                'normal': (0, 255, 0),
                'attention': (0, 255, 255),
                'warning': (0, 255, 255),
                'danger': (0, 165, 255),
                'critical': (0, 0, 255)
            }

            level_text = prediction.alert_level.upper()
            level_color = level_colors.get(prediction.alert_level, (255, 255, 255))

            cv2.putText(overlay, f"Level: {level_text}", (x, y),
                       font, font_scale, level_color, font_thickness)
            y += line_height

            # Trend
            trend_text = prediction.trend_direction.upper()
            trend_symbols = {
                'improving': '↓',
                'stable': '→',
                'degrading': '↑'
            }
            trend_colors = {
                'improving': (0, 255, 0),
                'stable': (255, 255, 0),
                'degrading': (0, 0, 255)
            }

            symbol = trend_symbols.get(prediction.trend_direction, '')
            trend_color = trend_colors.get(prediction.trend_direction, (255, 255, 255))

            cv2.putText(overlay, f"Trend: {symbol} {trend_text}", (x, y),
                       font, font_scale, trend_color, font_thickness)
            y += line_height

            # Time to critical
            if prediction.estimated_time_to_critical > 0:
                time_mins = prediction.estimated_time_to_critical / 60
                cv2.putText(overlay, f"Critical in: {time_mins:.1f}min", (x, y),
                           font, 0.5, (0, 0, 255), 2)
                y += line_height

        # Help text at bottom
        help_y = h - 20
        cv2.putText(overlay, "Press 'H' for help | 'Q' to quit", (x, help_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

        return overlay

    def _get_score_color(self, score: float) -> tuple:
        """Get color based on drowsiness score"""
        if score < 30:
            return (0, 255, 0)  # Green
        elif score < 50:
            return (0, 255, 255)  # Yellow
        elif score < 70:
            return (0, 165, 255)  # Orange
        else:
            return (0, 0, 255)  # Red

    def show_help_screen(self, frame: np.ndarray) -> np.ndarray:
        """Show help overlay"""
        overlay = frame.copy()
        h, w = overlay.shape[:2]

        # Semi-transparent background
        cv2.rectangle(overlay, (50, 50), (w-50, h-50), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.3, overlay, 0.7, 0)

        font = cv2.FONT_HERSHEY_SIMPLEX

        help_text = [
            "KEYBOARD CONTROLS",
            "",
            "Q - Quit application",
            "H - Toggle this help",
            "P - Pause/Resume",
            "R - Reset/Recalibrate",
            "S - Save screenshot",
            "",
            "ALERT LEVELS",
            "",
            "NORMAL - No drowsiness detected",
            "ATTENTION - Early signs detected",
            "WARNING - Drowsiness increasing",
            "DANGER - High drowsiness level",
            "CRITICAL - Immediate action required",
        ]

        y = 100
        for i, line in enumerate(help_text):
            if line == "" or line.endswith("CONTROLS") or line.endswith("LEVELS"):
                color = (0, 255, 255)
                thickness = 2
            else:
                color = (255, 255, 255)
                thickness = 1

            cv2.putText(frame, line, (100, y), font, 0.7, color, thickness)
            y += 35

        return frame

    def run(self):
        """Main application loop"""
        # Run calibration
        if not self.run_calibration():
            print("\nCalibration cancelled. Exiting...")
            return

        print("\n" + "="*60)
        print("STARTING MONITORING")
        print("="*60)
        print("\nPress 'Q' to quit, 'H' for help\n")

        self.running = True

        # FPS calculation
        fps_start_time = time.time()
        fps_frame_count = 0
        fps = 0

        try:
            while self.running:
                # Read frame
                ret, frame = self.camera.read()

                if not ret:
                    print("Error: Cannot read from camera")
                    break

                # Flip frame for mirror effect
                frame = cv2.flip(frame, 1)

                if not self.paused:
                    # Process frame
                    metrics, indicators = self.detector.process_frame(frame)

                    # Predictive analysis
                    prediction = None
                    if metrics and indicators:
                        # Update analyzer
                        self.analyzer.update(
                            metrics.ear_avg,
                            metrics.mar,
                            indicators.blink_rate,
                            indicators.perclos,
                            self.detector.analyze_head_stability(),
                            indicators.is_yawning
                        )

                        # Get prediction
                        prediction = self.analyzer.predict()

                        # Trigger alerts
                        frame = self.alert_system.trigger_alert(
                            prediction.alert_level,
                            frame,
                            prediction.drowsiness_score,
                            prediction.estimated_time_to_critical
                        )

                        # Log data
                        self.logger.log_frame(metrics, indicators, prediction)

                        # Log alert events
                        if prediction.alert_level != 'normal':
                            self.logger.log_event('alert', {
                                'level': prediction.alert_level,
                                'score': prediction.drowsiness_score,
                                'factors': prediction.contributing_factors
                            })

                    # Draw UI
                    frame = self.draw_ui(frame, metrics, indicators, prediction, fps)
                else:
                    # Paused overlay
                    cv2.putText(frame, "PAUSED", (frame.shape[1]//2 - 100, frame.shape[0]//2),
                               cv2.FONT_HERSHEY_BOLD, 2, (0, 255, 255), 3)

                # Show help if requested
                if self.show_help:
                    frame = self.show_help_screen(frame)

                # Calculate FPS
                fps_frame_count += 1
                if time.time() - fps_start_time >= 1.0:
                    fps = fps_frame_count / (time.time() - fps_start_time)
                    fps_frame_count = 0
                    fps_start_time = time.time()

                # Display
                window_name = 'Driver Drowsiness Monitoring System'
                if self.config['display']['fullscreen']:
                    cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
                    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

                cv2.imshow(window_name, frame)

                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    self.running = False
                elif key == ord('h'):
                    self.show_help = not self.show_help
                elif key == ord('p'):
                    self.paused = not self.paused
                elif key == ord('r'):
                    print("\nRecalibrating...")
                    self.run_calibration()
                elif key == ord('s'):
                    # Save screenshot
                    timestamp = time.strftime('%Y%m%d_%H%M%S')
                    filename = f'screenshot_{timestamp}.png'
                    cv2.imwrite(filename, frame)
                    print(f"Screenshot saved: {filename}")

        except KeyboardInterrupt:
            print("\n\nInterrupted by user")

        finally:
            self.cleanup()

    def cleanup(self):
        """Cleanup resources"""
        print("\n" + "="*60)
        print("SHUTTING DOWN")
        print("="*60)

        print("\nGenerating session summary...")
        self.logger.generate_summary()

        print("\nGenerating analysis plots...")
        try:
            self.logger.generate_plots()
        except Exception as e:
            print(f"Could not generate plots: {e}")

        print("\nReleasing resources...")
        self.camera.release()
        cv2.destroyAllWindows()
        self.alert_system.cleanup()

        print("\n✓ System shutdown complete")
        print("="*60)


def main():
    """Entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description='Industrial Driver Drowsiness Prediction System'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file'
    )

    args = parser.parse_args()

    # Create and run system
    system = DrowsinessMonitoringSystem(config_path=args.config)
    system.run()


if __name__ == '__main__':
    main()
