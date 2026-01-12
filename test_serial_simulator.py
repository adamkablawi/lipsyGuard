"""
Serial Simulator for Testing

Creates a pseudo-terminal pair and sends simulated sensor data.
Use this to test realtime_detector.py without ESP32 hardware.

Usage:
    python test_serial_simulator.py

This will print the slave port path. Use that path with realtime_detector.py:
    python realtime_detector.py /dev/ttys00X
"""

import os
import pty
import time
import sys
import numpy as np


def generate_normal_sample(t, sample_idx):
    """Generate a normal activity sample."""
    # Stable accelerometer with minor drift
    ax = 500 + np.random.normal(0, 30)
    ay = -100 + np.random.normal(0, 50)
    az = 200 + np.random.normal(0, 40)

    # PPG with regular heartbeat (~75 bpm = 800ms intervals)
    # At 16Hz, one beat every ~12.8 samples
    phase = (sample_idx % 13) / 13.0
    ppg_base = 30000 + 2000 * np.sin(2 * np.pi * phase)
    ppg = ppg_base + np.random.normal(0, 100)

    return int(ax), int(ay), int(az), int(ppg)


def generate_seizure_sample(t, sample_idx):
    """Generate a seizure-like sample with high-frequency oscillations."""
    # Rapid oscillations in accelerometer (simulating convulsions)
    freq = 5.0  # 5 Hz tremor
    ax = 500 + 400 * np.sin(2 * np.pi * freq * t) + np.random.normal(0, 100)
    ay = -100 + 800 * np.cos(2 * np.pi * freq * t) + np.random.normal(0, 150)
    az = 200 + 300 * np.sin(2 * np.pi * 3 * t) + np.random.normal(0, 80)

    # Irregular PPG (tachycardia + noise from motion)
    ppg = 30000 + np.random.normal(0, 3000)

    return int(ax), int(ay), int(az), int(ppg)


def main():
    # Create pseudo-terminal pair
    master_fd, slave_fd = pty.openpty()
    slave_path = os.ttyname(slave_fd)

    print(f"Serial simulator started")
    print(f"Connect realtime_detector.py to: {slave_path}")
    print(f"Example: python realtime_detector.py {slave_path}")
    print()
    print("Simulation pattern:")
    print("  0-25s:  Normal activity")
    print("  25-50s: Seizure-like activity")
    print("  50s+:   Normal activity")
    print()
    print("Press Ctrl+C to stop")
    print("-" * 50)

    sample_idx = 0
    start_time = time.time()
    interval = 1.0 / 16  # 16 Hz = 62.5ms

    try:
        while True:
            elapsed = time.time() - start_time
            timestamp_ms = int(elapsed * 1000)

            # Switch between normal and seizure patterns
            if 25 <= elapsed < 50:
                ax, ay, az, ppg = generate_seizure_sample(elapsed, sample_idx)
                mode = "SEIZURE"
            else:
                ax, ay, az, ppg = generate_normal_sample(elapsed, sample_idx)
                mode = "normal"

            # Format packet
            packet = f"{timestamp_ms},{ax},{ay},{az},{ppg}\n"

            # Write to pseudo-terminal
            os.write(master_fd, packet.encode('utf-8'))

            # Progress indicator every 16 samples (1 second)
            if sample_idx % 16 == 0:
                print(f"[{elapsed:6.1f}s] {mode:7s} | samples: {sample_idx:4d} | last: {ax:5d},{ay:6d},{az:5d},{ppg:5d}")

            sample_idx += 1

            # Maintain 16 Hz rate
            next_time = start_time + (sample_idx * interval)
            sleep_time = next_time - time.time()
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\nSimulator stopped")

    finally:
        os.close(master_fd)
        os.close(slave_fd)


if __name__ == "__main__":
    main()
