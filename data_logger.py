"""
DATA LOGGING AND ANALYTICS SYSTEM
Records metrics and events for analysis
"""

import os
import json
import csv
import time
from datetime import datetime
from typing import Dict, List
import pandas as pd
import matplotlib.pyplot as plt


class DataLogger:
    """
    Comprehensive data logging system
    """

    def __init__(self, config: dict):
        self.config = config
        self.log_config = config['logging']

        if not self.log_config['enabled']:
            return

        # Create log directory
        self.log_dir = self.log_config['directory']
        os.makedirs(self.log_dir, exist_ok=True)

        # Create session directory
        session_name = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.session_dir = os.path.join(self.log_dir, session_name)
        os.makedirs(self.session_dir, exist_ok=True)

        # Initialize log files
        self.metrics_file = os.path.join(self.session_dir, 'metrics.csv')
        self.events_file = os.path.join(self.session_dir, 'events.json')
        self.summary_file = os.path.join(self.session_dir, 'summary.txt')

        # Initialize CSV
        if self.log_config['save_metrics']:
            self._init_metrics_csv()

        # Event log
        self.events = []

        # Session stats
        self.session_start = time.time()
        self.frame_count = 0
        self.alert_counts = {'attention': 0, 'warning': 0, 'danger': 0, 'critical': 0}

    def _init_metrics_csv(self):
        """Initialize metrics CSV file"""
        headers = [
            'timestamp', 'frame_number', 'ear_left', 'ear_right', 'ear_avg',
            'mar', 'head_pitch', 'head_yaw', 'head_roll',
            'blink_rate', 'perclos', 'drowsiness_score', 'alert_level'
        ]

        with open(self.metrics_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)

    def log_frame(self, metrics, indicators, prediction):
        """Log frame data"""
        if not self.log_config['enabled'] or not self.log_config['save_metrics']:
            return

        self.frame_count += 1

        if metrics is None:
            return

        row = [
            metrics.timestamp,
            self.frame_count,
            metrics.ear_left,
            metrics.ear_right,
            metrics.ear_avg,
            metrics.mar,
            metrics.head_pitch,
            metrics.head_yaw,
            metrics.head_roll,
            indicators.blink_rate if indicators else 0,
            indicators.perclos if indicators else 0,
            prediction.drowsiness_score if prediction else 0,
            prediction.alert_level if prediction else 'normal'
        ]

        with open(self.metrics_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)

    def log_event(self, event_type: str, data: dict):
        """Log significant event"""
        if not self.log_config['enabled'] or not self.log_config['save_events']:
            return

        event = {
            'timestamp': time.time(),
            'type': event_type,
            'data': data
        }

        self.events.append(event)

        # Update alert counts
        if event_type == 'alert' and 'level' in data:
            level = data['level']
            if level in self.alert_counts:
                self.alert_counts[level] += 1

    def generate_summary(self):
        """Generate session summary"""
        if not self.log_config['enabled'] or not self.log_config['session_summary']:
            return

        duration = time.time() - self.session_start

        summary = []
        summary.append("="*60)
        summary.append("SESSION SUMMARY")
        summary.append("="*60)
        summary.append(f"\nSession Duration: {duration/60:.1f} minutes")
        summary.append(f"Frames Processed: {self.frame_count}")
        summary.append(f"Average FPS: {self.frame_count/duration:.1f}")

        summary.append("\nAlert Statistics:")
        summary.append(f"  Attention alerts: {self.alert_counts['attention']}")
        summary.append(f"  Warning alerts: {self.alert_counts['warning']}")
        summary.append(f"  Danger alerts: {self.alert_counts['danger']}")
        summary.append(f"  Critical alerts: {self.alert_counts['critical']}")

        total_alerts = sum(self.alert_counts.values())
        summary.append(f"\nTotal Alerts: {total_alerts}")

        if total_alerts > 0:
            summary.append(f"Alert Rate: {total_alerts/(duration/60):.2f} per minute")

        # Analyze metrics if available
        if os.path.exists(self.metrics_file):
            df = pd.read_csv(self.metrics_file)

            summary.append("\nMetrics Summary:")
            summary.append(f"  Average EAR: {df['ear_avg'].mean():.3f}")
            summary.append(f"  Average MAR: {df['mar'].mean():.3f}")
            summary.append(f"  Average Blink Rate: {df['blink_rate'].mean():.1f}/min")
            summary.append(f"  Average PERCLOS: {df['perclos'].mean():.2f}%")
            summary.append(f"  Average Drowsiness Score: {df['drowsiness_score'].mean():.1f}%")
            summary.append(f"  Max Drowsiness Score: {df['drowsiness_score'].max():.1f}%")

        summary.append(f"\nLog Directory: {self.session_dir}")
        summary.append("="*60)

        summary_text = '\n'.join(summary)

        # Print to console
        print(summary_text)

        # Save to file
        with open(self.summary_file, 'w') as f:
            f.write(summary_text)

        # Save events
        with open(self.events_file, 'w') as f:
            json.dump(self.events, f, indent=4)

    def generate_plots(self):
        """Generate analysis plots"""
        if not os.path.exists(self.metrics_file):
            return

        df = pd.read_csv(self.metrics_file)

        fig, axes = plt.subplots(3, 1, figsize=(12, 10))

        # Plot EAR over time
        axes[0].plot(df['timestamp'] - df['timestamp'].iloc[0], df['ear_avg'])
        axes[0].set_title('Eye Aspect Ratio Over Time')
        axes[0].set_ylabel('EAR')
        axes[0].axhline(y=0.25, color='r', linestyle='--', label='Threshold')
        axes[0].legend()
        axes[0].grid(True)

        # Plot drowsiness score
        axes[1].plot(df['timestamp'] - df['timestamp'].iloc[0], df['drowsiness_score'])
        axes[1].set_title('Drowsiness Score Over Time')
        axes[1].set_ylabel('Score (%)')
        axes[1].axhline(y=50, color='orange', linestyle='--', label='Warning')
        axes[1].axhline(y=70, color='red', linestyle='--', label='Danger')
        axes[1].legend()
        axes[1].grid(True)

        # Plot PERCLOS
        axes[2].plot(df['timestamp'] - df['timestamp'].iloc[0], df['perclos'])
        axes[2].set_title('PERCLOS Over Time')
        axes[2].set_xlabel('Time (seconds)')
        axes[2].set_ylabel('PERCLOS (%)')
        axes[2].grid(True)

        plt.tight_layout()
        plot_file = os.path.join(self.session_dir, 'analysis.png')
        plt.savefig(plot_file)
        print(f"\n✓ Analysis plot saved to {plot_file}")
