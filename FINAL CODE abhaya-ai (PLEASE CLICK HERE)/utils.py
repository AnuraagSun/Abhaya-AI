

"""
Abhaya AI - Utility Functions

Platform detection, memory monitoring, performance profiling,
and various helper functions for system optimization.
"""

import os
import gc
import time
import psutil
import logging
import functools
import numpy as np
from datetime import datetime
from typing import Optional, Tuple, Any


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


class MemoryMonitor:
    """
    Monitors and manages system memory usage
    """

    def __init__(self, warning_threshold_mb=400, critical_threshold_mb=450):
        self.warning_threshold = warning_threshold_mb * 1024 * 1024
        self.critical_threshold = critical_threshold_mb * 1024 * 1024
        self.process = psutil.Process(os.getpid())
        self.peak_memory = 0
        self.last_gc_time = time.time()


    def get_memory_usage(self):
        """
        Get current memory usage in MB
        """
        mem_info = self.process.memory_info()
        current_mb = mem_info.rss / 1024 / 1024

        if mem_info.rss > self.peak_memory:
            self.peak_memory = mem_info.rss

        return current_mb


    def get_system_memory(self):
        """
        Get system-wide memory statistics
        """
        mem = psutil.virtual_memory()
        return {
            'total_mb': mem.total / 1024 / 1024,
            'available_mb': mem.available / 1024 / 1024,
            'percent_used': mem.percent,
            'process_mb': self.get_memory_usage()
        }


    def check_memory_status(self):
        """
        Check if memory usage is within safe limits
        Returns: 'ok', 'warning', or 'critical'
        """
        current_usage = self.process.memory_info().rss

        if current_usage >= self.critical_threshold:
            return 'critical'
        elif current_usage >= self.warning_threshold:
            return 'warning'
        else:
            return 'ok'


    def force_garbage_collection(self):
        """
        Force garbage collection and log memory freed
        """
        before = self.get_memory_usage()

        gc.collect()

        after = self.get_memory_usage()
        freed = before - after

        self.last_gc_time = time.time()

        if freed > 1.0:
            logger.info(f"GC freed {freed:.2f} MB")

        return freed


    def periodic_gc_check(self, interval_seconds=60):
        """
        Check if it's time for periodic garbage collection
        """
        if time.time() - self.last_gc_time >= interval_seconds:
            return self.force_garbage_collection()
        return 0



class PerformanceProfiler:
    """
    Profiles function execution time and tracks performance metrics
    """

    def __init__(self):
        self.execution_times = {}
        self.call_counts = {}


    def profile(self, func):
        """
        Decorator to profile function execution time
        """
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()

            result = func(*args, **kwargs)

            end_time = time.perf_counter()
            execution_time = (end_time - start_time) * 1000

            func_name = func.__name__

            if func_name not in self.execution_times:
                self.execution_times[func_name] = []
                self.call_counts[func_name] = 0

            self.execution_times[func_name].append(execution_time)
            self.call_counts[func_name] += 1

            # Keep only last 100 measurements
            if len(self.execution_times[func_name]) > 100:
                self.execution_times[func_name].pop(0)

            return result

        return wrapper


    def get_stats(self, func_name):
        """
        Get performance statistics for a specific function
        """
        if func_name not in self.execution_times:
            return None

        times = self.execution_times[func_name]

        return {
            'avg_ms': np.mean(times),
            'min_ms': np.min(times),
            'max_ms': np.max(times),
            'std_ms': np.std(times),
            'call_count': self.call_counts[func_name]
        }


    def print_summary(self):
        """
        Print performance summary for all profiled functions
        """
        logger.info("=" * 60)
        logger.info("Performance Summary")
        logger.info("=" * 60)

        for func_name in sorted(self.execution_times.keys()):
            stats = self.get_stats(func_name)
            logger.info(f"{func_name}:")
            logger.info(f"  Avg: {stats['avg_ms']:.2f}ms | "
                       f"Min: {stats['min_ms']:.2f}ms | "
                       f"Max: {stats['max_ms']:.2f}ms | "
                       f"Calls: {stats['call_count']}")



class FPSCounter:
    """
    Calculate and track frames per second
    """

    def __init__(self, buffer_size=30):
        self.buffer_size = buffer_size
        self.frame_times = []
        self.last_frame_time = time.time()


    def update(self):
        """
        Update FPS calculation with new frame
        """
        current_time = time.time()
        self.frame_times.append(current_time)

        # Keep only recent frames
        if len(self.frame_times) > self.buffer_size:
            self.frame_times.pop(0)

        self.last_frame_time = current_time


    def get_fps(self):
        """
        Calculate current FPS
        """
        if len(self.frame_times) < 2:
            return 0.0

        time_diff = self.frame_times[-1] - self.frame_times[0]

        if time_diff == 0:
            return 0.0

        fps = (len(self.frame_times) - 1) / time_diff
        return fps



class CircularBuffer:
    """
    Memory-efficient circular buffer for storing fixed-size arrays
    """

    def __init__(self, max_size, dtype=np.float16):
        self.max_size = max_size
        self.dtype = dtype
        self.buffer = None
        self.index = 0
        self.size = 0


    def append(self, value):
        """
        Add value to circular buffer
        """
        if self.buffer is None:
            # Initialize buffer on first use
            self.buffer = np.zeros(self.max_size, dtype=self.dtype)

        self.buffer[self.index] = value
        self.index = (self.index + 1) % self.max_size

        if self.size < self.max_size:
            self.size += 1


    def get_array(self):
        """
        Get current buffer contents as numpy array
        """
        if self.buffer is None or self.size == 0:
            return np.array([], dtype=self.dtype)

        if self.size < self.max_size:
            return self.buffer[:self.size].copy()
        else:
            # Reorder to get chronological sequence
            return np.concatenate([
                self.buffer[self.index:],
                self.buffer[:self.index]
            ])


    def get_mean(self):
        """
        Calculate mean of buffer contents
        """
        if self.size == 0:
            return 0.0

        if self.size < self.max_size:
            return np.mean(self.buffer[:self.size])
        else:
            return np.mean(self.buffer)


    def clear(self):
        """
        Clear buffer contents
        """
        self.index = 0
        self.size = 0
        if self.buffer is not None:
            self.buffer.fill(0)



def adaptive_frame_skip(current_fps, target_fps, current_skip):
    """
    Dynamically adjust frame skip ratio based on performance

    Args:
        current_fps: Current frames per second
        target_fps: Target frames per second
        current_skip: Current frame skip value

    Returns:
        New frame skip value
    """
    if current_fps < target_fps * 0.7:
        # Too slow, skip more frames
        return min(current_skip + 1, 5)
    elif current_fps > target_fps * 1.2:
        # Too fast, can process more frames
        return max(current_skip - 1, 1)
    else:
        # Within acceptable range
        return current_skip



def create_timestamp():
    """
    Create timestamp string for logging
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")



def ensure_bgr_format(image):
    """
    Ensure image is in BGR format (OpenCV standard)
    """
    if image is None:
        return None

    if len(image.shape) == 2:
        # Grayscale, convert to BGR
        import cv2
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        # BGRA, convert to BGR
        import cv2
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    else:
        return image



def safe_division(numerator, denominator, default=0.0):
    """
    Safely divide two numbers, returning default if denominator is zero
    """
    if denominator == 0:
        return default
    return numerator / denominator



class SessionLogger:
    """
    Manages buffered logging to minimize I/O operations
    """

    def __init__(self, log_file, buffer_size=50):
        self.log_file = log_file
        self.buffer_size = buffer_size
        self.buffer = []
        self.last_flush = time.time()


    def log(self, message):
        """
        Add message to buffer
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        self.buffer.append(f"[{timestamp}] {message}")

        # Auto-flush if buffer is full
        if len(self.buffer) >= self.buffer_size:
            self.flush()


    def flush(self):
        """
        Write buffered logs to file
        """
        if not self.buffer:
            return

        try:
            with open(self.log_file, 'a') as f:
                f.write('\n'.join(self.buffer) + '\n')
            self.buffer.clear()
            self.last_flush = time.time()
        except Exception as e:
            logger.error(f"Failed to flush logs: {e}")


    def periodic_flush(self, interval_seconds=30):
        """
        Flush logs if interval has passed
        """
        if time.time() - self.last_flush >= interval_seconds:
            self.flush()


    def close(self):
        """
        Flush remaining logs and close
        """
        self.flush()



# Global instances
memory_monitor = MemoryMonitor()
performance_profiler = PerformanceProfiler()
