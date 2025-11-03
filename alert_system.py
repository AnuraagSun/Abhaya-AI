"""
MULTI-LEVEL ALERT SYSTEM
Handles visual, audio, and voice alerts with escalation
"""

import cv2
import numpy as np
import pygame
from pygame import mixer
import threading
import time
from typing import Optional
from gtts import gTTS
import os
import tempfile


class AlertSystem:
    """
    Comprehensive alert system with multiple notification methods
    """

    def __init__(self, config: dict):
        self.config = config
        self.alert_config = config['alerts']

        # Initialize pygame mixer for audio
        pygame.init()
        mixer.init()

        # Alert state
        self.current_alert_level = 'normal'
        self.last_alert_time = 0
        self.alert_active = False
        self.alert_lock = threading.Lock()

        # Voice synthesis cache
        self.voice_cache = {}
        self.voice_enabled = self.alert_config['voice']['enabled']

        # Pre-generate voice alerts
        if self.voice_enabled:
            self._generate_voice_alerts()

        # Audio settings
        self.audio_enabled = self.alert_config['audio']['enabled']

    def _generate_voice_alerts(self):
        """Pre-generate voice alert files"""
        messages = self.alert_config['voice']['messages']

        for level, message in messages.items():
            try:
                tts = gTTS(text=message, lang='en', slow=False)
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
                tts.save(temp_file.name)
                self.voice_cache[level] = temp_file.name
            except Exception as e:
                print(f"Error generating voice alert for {level}: {e}")

    def _play_sound(self, sound_file: str):
        """Play sound file"""
        try:
            if os.path.exists(sound_file):
                mixer.music.load(sound_file)
                volume = self.alert_config['audio']['volume']
                mixer.music.set_volume(volume)
                mixer.music.play()
        except Exception as e:
            print(f"Error playing sound: {e}")

    def _play_voice_alert(self, level: str):
        """Play voice alert"""
        if level in self.voice_cache:
            self._play_sound(self.voice_cache[level])

    def _create_visual_overlay(self, frame: np.ndarray, level: str, message: str) -> np.ndarray:
        """
        Create visual alert overlay on frame
        """
        overlay = frame.copy()
        h, w = frame.shape[:2]

        # Get color based on level
        colors = {
            'attention': self.alert_config['visual']['color_warning'],
            'warning': self.alert_config['visual']['color_warning'],
            'danger': self.alert_config['visual']['color_danger'],
            'critical': self.alert_config['visual']['color_critical']
        }

        color = colors.get(level, (0, 255, 255))

        # Flash effect for critical
        if level == 'critical':
            # Full screen flash
            flash_overlay = np.full_like(frame, color, dtype=np.uint8)
            alpha = 0.3 * (1 + np.sin(time.time() * 10))  # Pulsing effect
            overlay = cv2.addWeighted(overlay, 1 - alpha, flash_overlay, alpha, 0)

        # Draw border
        border_thickness = 20 if level == 'critical' else 10
        cv2.rectangle(overlay, (0, 0), (w, h), color, border_thickness)

        # Draw warning banner
        banner_height = 100
        banner_overlay = overlay.copy()
        cv2.rectangle(banner_overlay, (0, h - banner_height), (w, h), color, -1)
        overlay = cv2.addWeighted(overlay, 0.7, banner_overlay, 0.3, 0)

        # Add text
        font = cv2.FONT_HERSHEY_BOLD
        font_scale = 1.5 if level == 'critical' else 1.0
        font_thickness = 3 if level == 'critical' else 2

        # Calculate text size and position
        text_size = cv2.getTextSize(message, font, font_scale, font_thickness)[0]
        text_x = (w - text_size[0]) // 2
        text_y = h - (banner_height // 2) + (text_size[1] // 2)

        # Draw text with shadow
        cv2.putText(overlay, message, (text_x + 2, text_y + 2),
                   font, font_scale, (0, 0, 0), font_thickness + 2)
        cv2.putText(overlay, message, (text_x, text_y),
                   font, font_scale, (255, 255, 255), font_thickness)

        return overlay

    def should_trigger_alert(self, level: str) -> bool:
        """
        Determine if alert should be triggered based on timing and escalation
        """
        current_time = time.time()
        time_between_alerts = self.alert_config['escalation']['time_between_alerts']

        # Always alert on level change or upgrade
        level_priority = {'normal': 0, 'attention': 1, 'warning': 2, 'danger': 3, 'critical': 4}

        if level_priority.get(level, 0) > level_priority.get(self.current_alert_level, 0):
            return True

        # Repeat critical alerts
        if level == 'critical' and self.alert_config['escalation']['repeat_critical']:
            if current_time - self.last_alert_time > time_between_alerts:
                return True

        # Time-based triggering for other levels
        if current_time - self.last_alert_time > time_between_alerts:
            return True

        return False

    def trigger_alert(self, level: str, frame: np.ndarray,
                     drowsiness_score: float,
                     time_to_critical: float = -1) -> np.ndarray:
        """
        Main alert triggering method
        Returns frame with visual alerts applied
        """
        if level == 'normal':
            self.current_alert_level = level
            return frame

        with self.alert_lock:
            should_alert = self.should_trigger_alert(level)

            if should_alert:
                # Update state
                self.current_alert_level = level
                self.last_alert_time = time.time()

                # Audio alert
                if self.audio_enabled:
                    threading.Thread(target=self._play_audio_alert, args=(level,), daemon=True).start()

                # Voice alert
                if self.voice_enabled:
                    threading.Thread(target=self._play_voice_alert, args=(level,), daemon=True).start()

            # Visual alert (always show if level is elevated)
            if self.alert_config['visual']['enabled']:
                # Create message
                messages = {
                    'attention': f"ATTENTION: Fatigue signs detected ({drowsiness_score:.0f}%)",
                    'warning': f"WARNING: Drowsiness detected ({drowsiness_score:.0f}%)",
                    'danger': f"DANGER: High drowsiness ({drowsiness_score:.0f}%) - Take a break!",
                    'critical': f"CRITICAL ALERT: Pull over NOW! ({drowsiness_score:.0f}%)"
                }

                message = messages.get(level, "ALERT")

                if time_to_critical > 0 and level in ['warning', 'danger']:
                    message += f" | Critical in {time_to_critical:.0f}s"

                frame = self._create_visual_overlay(frame, level, message)

        return frame

    def _play_audio_alert(self, level: str):
        """Play appropriate audio alert for level"""
        sound_files = {
            'attention': self.alert_config['audio'].get('warning_sound'),
            'warning': self.alert_config['audio'].get('warning_sound'),
            'danger': self.alert_config['audio'].get('danger_sound'),
            'critical': self.alert_config['audio'].get('critical_sound')
        }

        sound_file = sound_files.get(level)
        if sound_file:
            self._play_sound(sound_file)
        else:
            # Use system beep as fallback
            self._system_beep(level)

    def _system_beep(self, level: str):
        """Fallback system beep"""
        beep_counts = {
            'attention': 1,
            'warning': 2,
            'danger': 3,
            'critical': 5
        }

        count = beep_counts.get(level, 1)
        for _ in range(count):
            print('\a', end='', flush=True)  # System beep
            time.sleep(0.2)

    def cleanup(self):
        """Cleanup resources"""
        # Remove temporary voice files
        for temp_file in self.voice_cache.values():
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except:
                pass

        mixer.quit()
        pygame.quit()
