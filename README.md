# LipsyGuard

Real-time seizure detection wearable using an ESP32-C3, PPG sensor, and IMU with an on-laptop ML inference pipeline.

<img width="457" height="608" alt="Screenshot 2026-01-18 at 11 13 28 AM" src="https://github.com/user-attachments/assets/5b56482d-0721-45ba-a0dc-d768a6c1691f" />


## How It Works

```
[MAX30101 PPG] ──┐
                 ├── I2C ──► [ESP32-C3 Mini] ──► BLE ──► [Laptop Python ML]
[LSM6DS3TR-C]   ┘
```

<img width="668" height="391" alt="Screenshot 2025-12-22 at 8 17 23 AM" src="https://github.com/user-attachments/assets/ea0943b4-c9fa-4b03-8012-568c264e2c3f" />


The ESP32 samples both sensors at 16 Hz and streams CSV packets over BLE. The laptop buffers 200 samples (12.5 s windows) and runs inference every window using a CNN trained on triaxial accelerometer data, fused with an HRV score from the PPG signal.

**Output:** `SEIZURE` or `NORMAL` with a probability score printed to the terminal.

## Files

| File | Purpose |
|------|---------|
| `firmware/firmware.ino` | ESP32-C3 Arduino sketch — samples sensors, streams BLE |
| `combined_seizure_detector_final.py` | Core ML + HRV detector classes |
| `realtime_detector.py` | BLE reader, rolling buffer, inference loop |
| `test_serial_simulator.py` | Pseudo-terminal simulator for testing without hardware |
| `triaxial_seizure_model.keras` | Trained CNN model (required) |
| `triaxial_seizure_encoder.pkl` | Label encoder (required) |
| `axis_normalization_stats.pkl` | Per-axis normalization stats (required) |

## Setup

### Python dependencies

```bash
pip install numpy scipy tensorflow scikit-learn pyserial bleak
```

Requirements: `numpy>=1.24.0,<2.0.0`, `tensorflow>=2.15.0,<2.18.0`

### Flash firmware

```bash
arduino-cli upload -p /dev/tty.usbmodem* --fqbn esp32:esp32:esp32c3 firmware/
```

Required Arduino libraries: `NimBLE-Arduino`, `SparkFun MAX3010x`

## Running

```bash
python realtime_detector.py
```

The script scans for a BLE device named `LipsyGuard`, connects, and starts streaming. The first inference fires after the buffer fills (~25 s).

**Test without hardware:**

```bash
# Terminal 1
python test_serial_simulator.py

# Terminal 2 — use the port path printed by the simulator
python realtime_detector.py
```

The simulator generates 25 s of normal activity, then 25 s of seizure-like motion, then normal again.

## Detection Logic

Each 12.5 s window is evaluated by two components, combined as a weighted sum:

| Component | Weight | Method |
|-----------|--------|--------|
| Accelerometer CNN | 90% | Triaxial CNN model, per-axis z-score normalization |
| HRV analysis | 10% | Rule-based scoring on time/frequency domain HRV features from PPG |

**Seizure threshold:** combined probability ≥ 85%

## Packet Format (ESP32 → Laptop)

```
timestamp_ms,accel_x,accel_y,accel_z,ppg\n
```

- `accel_*` — world-frame milli-g after complementary filter (gyro 98% / accel 2%)
- `ppg` — raw IR ADC value from MAX30101
- Lines starting with `#` are debug messages and are ignored by the parser

## Hardware

- **ESP32-C3 Mini** — BLE transport, I2C master (SDA GPIO 6, SCL GPIO 7, 400 kHz)
- **LSM6DS3TR-C** — 6-DoF IMU, I2C `0x6A`, 52 Hz ODR, ±4g / ±250 dps
- **MAX30101** — PPG, I2C `0x57`, SpO2 mode, 100 Hz sample rate

## BLE UUIDs

| | UUID |
|---|---|
| Service | `12345678-1234-1234-1234-1234567890ab` |
| TX Characteristic | `abcdefab-1234-5678-1234-abcdefabcdef` |
