import sys
import time
import threading
import logging
from collections import deque
import asyncio

import numpy as np
from bleak import BleakScanner, BleakClient

from combined_seizure_detector_final import CombinedSeizureDetector

# ============================================================================
# Configuration
# ============================================================================

SAMPLE_RATE_HZ = 16
WINDOW_SAMPLES = 200
WINDOW_DURATION_SEC = WINDOW_SAMPLES / SAMPLE_RATE_HZ  # 12.5 seconds

DEVICE_NAME = "LipsyGuard"
CHAR_UUID = "abcdefab-1234-5678-1234-abcdefabcdef"

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger(__name__)


# ============================================================================
# Buffer + Parsing (unchanged)
# ============================================================================

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
        with self.lock:
            self.accel_x.append(ax)
            self.accel_y.append(ay)
            self.accel_z.append(az)
            self.ppg.append(ppg)
            self.sample_count += 1

            if self.last_timestamp is not None:
                gap_ms = timestamp_ms - self.last_timestamp
                if gap_ms > 105:
                    log.warning(f"Timestamp difference: {gap_ms}ms")
            self.last_timestamp = timestamp_ms

    def get_window(self):
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
        with self.lock:
            self.accel_x.clear()
            self.accel_y.clear()
            self.accel_z.clear()
            self.ppg.clear()
            self.sample_count = 0
            self.last_timestamp = None


def parse_packet(line: str):
    """
    Parse a CSV packet from ESP32.

    Expected: timestamp_ms,accel_x,accel_y,accel_z,ppg
    Returns tuple or None.
    """
    try:
        line = line.strip()

        if not line or line.startswith('#'):
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


# ============================================================================
# BLE Realtime Detector
# ============================================================================

class RealtimeSeizureDetectorBLE:
    """
    Coordinates BLE reading (notifications), buffering, and inference.
    BLE notifications arrive on an asyncio loop; inference runs in a normal thread.
    """

    def __init__(self, device_name=DEVICE_NAME, char_uuid=CHAR_UUID):
        self.device_name = device_name
        self.char_uuid = char_uuid

        self.buffer = SensorBuffer(maxlen=WINDOW_SAMPLES)
        self.detector = None

        self.running = False
        self.inference_thread = None

        self.inference_count = 0
        self.parse_errors = 0
        self.lines_ok = 0

        # BLE parsing: notifications may contain partial/multiple lines
        self._rx_text_buffer = ""

    def _init_detector(self):
        log.info("Loading ML model...")
        self.detector = CombinedSeizureDetector(
            model_path='triaxial_seizure_model.keras',
            encoder_path='triaxial_seizure_encoder.pkl',
            stats_path='axis_normalization_stats.pkl'
        )

    def _handle_text_chunk(self, chunk: str):
        """
        Append incoming text chunk, split into lines, parse each full line.
        """
        self._rx_text_buffer += chunk

        # Process complete lines
        while '\n' in self._rx_text_buffer:
            line, self._rx_text_buffer = self._rx_text_buffer.split('\n', 1)
            line = line + '\n'  # restore newline for consistency with original

            parsed = parse_packet(line)
            if parsed:
                ts, ax, ay, az, ppg = parsed
                self.buffer.append(ts, ax, ay, az, ppg)
                self.lines_ok += 1
            else:
                # Ignore empty fragments quietly; count others as errors
                if line.strip():
                    self.parse_errors += 1
                    if self.parse_errors <= 5:
                        log.warning(f"Parse error on: {line[:80]!r}")

    def _on_notify(self, _, data: bytearray):
        """
        Bleak notification callback (runs in asyncio thread context).
        """
        try:
            text = data.decode("utf-8", errors="ignore")
            self._handle_text_chunk(text)
        except Exception as e:
            self.parse_errors += 1
            if self.parse_errors <= 5:
                log.warning(f"Notify decode/parse error: {e}")

    def _inference_loop(self):
        log.info(f"Inference loop started (every {WINDOW_DURATION_SEC}s)")

        while self.running:
            time.sleep(WINDOW_DURATION_SEC)
            if not self.running:
                break

            ax, ay, az, ppg, count = self.buffer.get_window()
            if ax is None:
                log.info(f"\n\n--- Inferences Starting ---\n")
                continue

            try:
                self.inference_count += 1
                log.info(f"--- Inference #{self.inference_count} ---")

                result = self.detector.predict(ax, ay, az, ppg, verbose=False)

                classification = result['classification']
                probability = result['combined_seizure_probability']
                confidence = result['confidence']

                if result['is_seizure']:
                    log.info(f"SEIZURE DETECTED: {probability:.1f}% (confidence: {confidence:.1f}%)")
                else:
                    log.info(f"NORMAL: {probability:.1f}% seizure probability")

                accel_prob = result['accelerometer']['seizure_probability']
                hrv_prob = result['hrv']['seizure_probability']
                log.info(f"  Accelerometer Probability: {accel_prob:.1f}% | Heart Rate Variability: {hrv_prob:.1f}% \n")

            except Exception as e:
                log.error(f"Inference error: {e}")

        log.info("Inference loop stopped")

    async def _ble_main(self):
        """
        Async BLE logic: scan, connect, subscribe, keep alive until stopped.
        """
        log.info("Scanning for BLE device...")
        dev = await BleakScanner.find_device_by_filter(
            lambda d, ad: d.name == self.device_name,
            timeout=10.0
        )

        if not dev:
            raise RuntimeError(f"Could not find BLE device named {self.device_name!r}")

        log.info(f"Connecting to {dev.address} ({self.device_name})")
        async with BleakClient(dev) as client:
            log.info(f"Connected: {client.is_connected}")

            # Subscribe to notifications
            await client.start_notify(self.char_uuid, self._on_notify)
            log.info("Subscribed. Waiting for data...")
            log.info(f"First inference in 25s after buffer fills")

            # Keep alive until stop() flips running off
            while self.running and client.is_connected:
                await asyncio.sleep(0.5)

            # Best effort cleanup
            try:
                await client.stop_notify(self.char_uuid)
            except Exception:
                pass

    def start(self):
        """
        Start detector:
        - load model
        - start inference thread
        - run BLE loop in background thread (asyncio)
        """
        self._init_detector()
        self.running = True

        # Start inference thread
        self.inference_thread = threading.Thread(target=self._inference_loop, daemon=True)
        self.inference_thread.start()

        # Run BLE loop on its own thread so main thread can handle Ctrl+C cleanly
        def _run_ble():
            try:
                asyncio.run(self._ble_main())
            except Exception as e:
                log.error(f"BLE error: {e}")
                self.running = False

        self.ble_thread = threading.Thread(target=_run_ble, daemon=True)
        self.ble_thread.start()

    def stop(self):
        log.info("Shutting down...")
        self.running = False

        if getattr(self, "ble_thread", None):
            self.ble_thread.join(timeout=2.0)

        if self.inference_thread:
            self.inference_thread.join(timeout=2.0)

        log.info(
            f"Shutdown complete. Inferences: {self.inference_count}, "
            f"Samples OK: {self.lines_ok}, Parse errors: {self.parse_errors}"
        )

    def run(self):
        self.start()
        try:
            while self.running:
                time.sleep(0.5)
        except KeyboardInterrupt:
            log.info("Ctrl+C received")
        finally:
            self.stop()


# ============================================================================
# CLI
# ============================================================================

def main():
    """
    Usage:
      python realtime_detector_ble.py
      python realtime_detector_ble.py --name LipsyGuard --uuid abcdefab-...
    """
    name = DEVICE_NAME
    uuid = CHAR_UUID

    args = sys.argv[1:]
    if "--name" in args:
        i = args.index("--name")
        name = args[i + 1]
    if "--uuid" in args:
        i = args.index("--uuid")
        uuid = args[i + 1]

    detector = RealtimeSeizureDetectorBLE(device_name=name, char_uuid=uuid)
    detector.run()


if __name__ == "__main__":
    main()