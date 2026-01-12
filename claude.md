# LipsyGuard — Project Context for Claude

## Overview

LipsyGuard is a real-time seizure detection system that classifies physiological sensor data to determine seizure events.

## Architecture

```
[MAX30101 PPG] ──┐
                 ├── I2C ──► [ESP32-C3 Mini] ──► USB Serial (CDC) ──► [Laptop Python ML]
[LSM6DS3TR-C IMU]┘
```

Note: ESP32-C3 only supports BLE, not Bluetooth Classic. Using USB CDC serial for reliability.

## ML Pipeline

- **Entry point:** `combined_seizure_detector_final.py`
- **Inference frequency:** Every 12.5 seconds
- **Window size:** 200 samples (16 Hz × 12.5 s)

### Input Channels
| Channel | Samples | Rate | Source |
|---------|---------|------|--------|
| Accel X | 200 | 16 Hz | LSM6DS3TR-C |
| Accel Y | 200 | 16 Hz | LSM6DS3TR-C |
| Accel Z | 200 | 16 Hz | LSM6DS3TR-C |
| PPG | 200 | 16 Hz | MAX30101 |

### Output
- Binary label: `SEIZURE` or `NORMAL`
- Confidence score: 0–100%

### Model Files Required
- `triaxial_seizure_model.keras` — CNN classifier
- `triaxial_seizure_encoder.pkl` — Label encoder
- `axis_normalization_stats.pkl` — Per-axis normalization stats

## Dependencies

```
numpy>=1.24.0,<2.0.0
scipy>=1.11.0
tensorflow>=2.15.0,<2.18.0
scikit-learn>=1.3.0
pyserial>=3.5
```

Install: `pip install numpy scipy tensorflow scikit-learn pyserial`

## Data Contract

### ESP32 → Python Packet Format
```
timestamp_ms,accel_x,accel_y,accel_z,ppg\n
```
| Field | Type | Units | Example |
|-------|------|-------|---------|
| timestamp_ms | uint32 | milliseconds | `12345` |
| accel_x | int16 | milli-g | `500` |
| accel_y | int16 | milli-g | `-200` |
| accel_z | int16 | milli-g | `1000` |
| ppg | uint32 | raw IR ADC (18-bit) | `32000` |

Lines starting with `#` are status/debug messages (ignored by parser).

### Sampling
- Rate: 16 Hz (62.5 ms between samples)
- Window: 12.5 seconds = 200 samples
- All channels must be temporally aligned

### Accelerometer Preprocessing (in ML code)
- Per-axis z-score normalization using stored mean/std
- Input shape: `(200, 3)` → normalized → `(1, 200, 3)` for model

### PPG Preprocessing (in ML code)
- Bandpass filter: 0.5–8 Hz (4th order Butterworth)
- Peak detection for RR intervals
- HRV feature extraction (time + frequency domain)

## Classifier Weights
- Accelerometer ML model: 70%
- HRV rule-based scoring: 30%
- Combined threshold: 50% for seizure classification

## Hardware

### ESP32-C3 Mini
- Controller and USB CDC serial transport
- I2C master for sensors (GPIO 8 SDA, GPIO 9 SCL, 400kHz)

### MAX30101 (PPG)
- I2C address: 0x57
- Provides raw PPG signal

### LSM6DS3TR-C (6-DoF IMU)
- I2C address: 0x6A or 0x6B
- Provides accelerometer + gyroscope data

## Development Phases

### Phase 1: ML Dependency Setup ✓
- Enumerate Python dependencies
- Verify model file presence
- Document install commands

### Phase 2: ML Input Pipeline Refactor ✓
- Created `realtime_detector.py` — serial reader + buffer + inference loop
- Rolling buffer using `collections.deque(maxlen=200)`
- 12.5s timer-based inference trigger
- Graceful error handling (parse errors, disconnects)
- Test simulator: `test_serial_simulator.py`

### Phase 3: ESP32 Firmware ✓
- `firmware/firmware.ino` — Arduino sketch for ESP32-C3
- LSM6DS3TR-C: 52Hz ODR, ±4g range, outputs milli-g
- MAX30101: SpO2 mode, 100Hz, IR channel, 18-bit ADC
- Hardware timer: 62.5ms interval (16 Hz output)
- USB CDC serial at 115200 baud

**Upload:** `arduino-cli upload -p /dev/tty.usbmodem* --fqbn esp32:esp32:esp32c3 firmware/`

## Constraints

- No new sensors
- No cloud services
- No ML retraining (unless approved)
- No UI/visualization
- Minimal, reversible changes only
