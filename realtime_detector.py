"""
Real-time Seizure Detection via Bluetooth Serial

Reads sensor data from ESP32 over serial, buffers 200 samples (12.5s at 16Hz),
and triggers inference using CombinedSeizureDetector.

Packet format from ESP32: timestamp_ms,accel_x,accel_y,accel_z,ppg\n
"""

import sys
import time
import threading
import logging
from collections import deque

import serial
import numpy as np

from combined_seizure_detector_final import CombinedSeizureDetector

# Configuration
SAMPLE_RATE_HZ = 16
WINDOW_SAMPLES = 200
WINDOW_DURATION_SEC = WINDOW_SAMPLES / SAMPLE_RATE_HZ  # 12.5 seconds
BAUD_RATE = 115200

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger(__name__)


class SensorBuffer:
    """Thread-safe rolling buffer for sensor data."""

    def __init__(self, maxlen=WINDOW_SAMPLES):
        self.maxlen = maxlen
        self.accel_x = deque(maxlen=maxlen)
        self.accel_y = deque(maxlen=maxlen)
        self.accel_z = deque(maxlen=maxlen)
        self.ppg = deque(maxlen=maxlen)
        self.lock = threading.Lock()
        self.sample_count = 0
        self.last_timestamp = None

    def append(self, timestamp_ms, ax, ay, az, ppg):
        """Add a sample to all buffers atomically."""
        with self.lock:
            self.accel_x.append(ax)
            self.accel_y.append(ay)
            self.accel_z.append(az)
            self.ppg.append(ppg)
            self.sample_count += 1

            # Check for timestamp discontinuity (>100ms gap suggests dropped samples)
            if self.last_timestamp is not None:
                gap_ms = timestamp_ms - self.last_timestamp
                if gap_ms > 100:
                    log.warning(f"Timestamp gap: {gap_ms}ms (expected ~62.5ms)")
            self.last_timestamp = timestamp_ms

    def get_window(self):
        """
        Return current buffer contents as numpy arrays.
        Returns (accel_x, accel_y, accel_z, ppg, sample_count) or None if insufficient data.
        """
        with self.lock:
            count = len(self.accel_x)
            if count < self.maxlen:
                return None, None, None, None, count

            return (
                np.array(self.accel_x, dtype=np.float64),
                np.array(self.accel_y, dtype=np.float64),
                np.array(self.accel_z, dtype=np.float64),
                np.array(self.ppg, dtype=np.float64),
                count
            )

    def clear(self):
        """Clear all buffers."""
        with self.lock:
            self.accel_x.clear()
            self.accel_y.clear()
            self.accel_z.clear()
            self.ppg.clear()
            self.sample_count = 0
            self.last_timestamp = None


def parse_packet(line):
    """
    Parse a CSV packet from ESP32.

    Expected format: timestamp_ms,accel_x,accel_y,accel_z,ppg
    Returns (timestamp_ms, ax, ay, az, ppg) or None on parse failure.
    Lines starting with # are status/debug lines and return None.
    """
    try:
        line = line.strip()

        # Skip status/debug lines (prefixed with #)
        if line.startswith('#'):
            return None

        parts = line.split(',')
        if len(parts) != 5:
            return None

        timestamp_ms = int(parts[0])
        ax = float(parts[1])
        ay = float(parts[2])
        az = float(parts[3])
        ppg = float(parts[4])

        return timestamp_ms, ax, ay, az, ppg

    except (ValueError, IndexError):
        return None


class RealtimeSeizureDetector:
    """
    Main class that coordinates serial reading, buffering, and inference.
    """

    def __init__(self, serial_port, baud_rate=BAUD_RATE):
        self.serial_port = serial_port
        self.baud_rate = baud_rate

        self.buffer = SensorBuffer(maxlen=WINDOW_SAMPLES)
        self.detector = None  # Lazy init to fail fast on missing model files

        self.serial_conn = None
        self.reader_thread = None
        self.inference_thread = None

        self.running = False
        self.inference_count = 0
        self.parse_errors = 0

    def _init_detector(self):
        """Initialize ML detector (separate to catch model loading errors early)."""
        log.info("Loading ML model...")
        self.detector = CombinedSeizureDetector(
            model_path='triaxial_seizure_model.keras',
            encoder_path='triaxial_seizure_encoder.pkl',
            stats_path='axis_normalization_stats.pkl'
        )

    def _serial_reader(self):
        """Thread: continuously read and parse serial data."""
        log.info(f"Serial reader started on {self.serial_port}")

        while self.running:
            try:
                if self.serial_conn.in_waiting > 0:
                    line = self.serial_conn.readline()
                    try:
                        line = line.decode('utf-8', errors='ignore')
                    except UnicodeDecodeError:
                        self.parse_errors += 1
                        continue

                    parsed = parse_packet(line)
                    if parsed:
                        ts, ax, ay, az, ppg = parsed
                        self.buffer.append(ts, ax, ay, az, ppg)
                    else:
                        self.parse_errors += 1
                        if self.parse_errors <= 5:
                            log.warning(f"Parse error on: {line[:50]!r}")
                else:
                    time.sleep(0.01)  # 10ms poll interval

            except serial.SerialException as e:
                log.error(f"Serial error: {e}")
                time.sleep(1.0)  # Back off before retry

            except Exception as e:
                log.error(f"Reader error: {e}")
                time.sleep(0.1)

        log.info("Serial reader stopped")

    def _inference_loop(self):
        """Thread: trigger inference every 12.5 seconds."""
        log.info(f"Inference loop started (every {WINDOW_DURATION_SEC}s)")

        while self.running:
            time.sleep(WINDOW_DURATION_SEC)

            if not self.running:
                break

            ax, ay, az, ppg, count = self.buffer.get_window()

            if ax is None:
                log.warning(f"Insufficient data: {count}/{WINDOW_SAMPLES} samples")
                continue

            try:
                self.inference_count += 1
                log.info(f"--- Inference #{self.inference_count} ---")

                result = self.detector.predict(ax, ay, az, ppg, verbose=False)

                # Output result
                classification = result['classification']
                probability = result['combined_seizure_probability']
                confidence = result['confidence']

                if result['is_seizure']:
                    log.warning(f"SEIZURE DETECTED: {probability:.1f}% (confidence: {confidence:.1f}%)")
                else:
                    log.info(f"Normal: {probability:.1f}% seizure probability")

                # Component breakdown
                accel_prob = result['accelerometer']['seizure_probability']
                hrv_prob = result['hrv']['seizure_probability']
                log.info(f"  Accel: {accel_prob:.1f}% | HRV: {hrv_prob:.1f}%")

                if result['hrv'].get('error'):
                    log.warning(f"  HRV error: {result['hrv']['error']}")

            except Exception as e:
                log.error(f"Inference error: {e}")

        log.info("Inference loop stopped")

    def start(self):
        """Start the real-time detection system."""
        try:
            # Initialize ML model first (fail fast if files missing)
            self._init_detector()

            # Open serial connection
            log.info(f"Opening serial port {self.serial_port} at {self.baud_rate} baud")
            self.serial_conn = serial.Serial(
                port=self.serial_port,
                baudrate=self.baud_rate,
                timeout=1.0
            )

            self.running = True

            # Start reader thread
            self.reader_thread = threading.Thread(target=self._serial_reader, daemon=True)
            self.reader_thread.start()

            # Start inference thread
            self.inference_thread = threading.Thread(target=self._inference_loop, daemon=True)
            self.inference_thread.start()

            log.info("System started. Waiting for data...")
            log.info(f"First inference in {WINDOW_DURATION_SEC}s after buffer fills")

        except serial.SerialException as e:
            log.error(f"Failed to open serial port: {e}")
            raise

        except Exception as e:
            log.error(f"Startup error: {e}")
            raise

    def stop(self):
        """Stop the detection system gracefully."""
        log.info("Shutting down...")
        self.running = False

        if self.reader_thread:
            self.reader_thread.join(timeout=2.0)

        if self.inference_thread:
            self.inference_thread.join(timeout=2.0)

        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()

        log.info(f"Shutdown complete. Inferences: {self.inference_count}, Parse errors: {self.parse_errors}")

    def run(self):
        """Main entry point - start and wait for Ctrl+C."""
        self.start()

        try:
            while self.running:
                time.sleep(0.5)
        except KeyboardInterrupt:
            log.info("Ctrl+C received")
        finally:
            self.stop()


def list_serial_ports():
    """List available serial ports."""
    import serial.tools.list_ports
    ports = serial.tools.list_ports.comports()

    if not ports:
        print("No serial ports found")
        return []

    print("Available serial ports:")
    for port in ports:
        print(f"  {port.device} - {port.description}")

    return [p.device for p in ports]


def main():
    """Entry point with argument handling."""
    if len(sys.argv) < 2:
        print("Usage: python realtime_detector.py <serial_port>")
        print("       python realtime_detector.py --list")
        print()
        list_serial_ports()
        sys.exit(1)

    if sys.argv[1] == '--list':
        list_serial_ports()
        sys.exit(0)

    serial_port = sys.argv[1]

    detector = RealtimeSeizureDetector(serial_port)
    detector.run()


if __name__ == "__main__":
    main()
