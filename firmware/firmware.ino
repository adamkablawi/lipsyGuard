/*
 * LipsyGuard Firmware
 * ESP32-C3 + LSM6DS3TR-C (IMU) + MAX30101 (PPG)
 *
 * Outputs: timestamp_ms,accel_x,accel_y,accel_z,ppg\n
 * Rate: 16 Hz (62.5ms interval)
 * Units: Accelerometer in milli-g, PPG as raw IR ADC
 */

#include <Wire.h>
#include "MAX30105.h"

// ============================================================================
// Configuration
// ============================================================================

#define SAMPLE_INTERVAL_US  62500  // 16 Hz = 62.5ms

// I2C
#define I2C_SDA  6
#define I2C_SCL  7

// Sensor addresses
#define IMU_ADDR  0x6A  // LSM6DS3TR-C

// ============================================================================
// Globals
// ============================================================================

MAX30105 ppgSensor;

volatile bool sampleReady = false;
hw_timer_t *timer = NULL;
uint32_t sampleCount = 0;

// ============================================================================
// I2C Helpers
// ============================================================================

void i2cWrite(uint8_t addr, uint8_t reg, uint8_t val) {
  Wire.beginTransmission(addr);
  Wire.write(reg);
  Wire.write(val);
  Wire.endTransmission();
}

uint8_t i2cRead(uint8_t addr, uint8_t reg) {
  Wire.beginTransmission(addr);
  Wire.write(reg);
  Wire.endTransmission();
  Wire.requestFrom(addr, (uint8_t)1);
  return Wire.available() ? Wire.read() : 0;
}

void i2cReadN(uint8_t addr, uint8_t reg, uint8_t *buf, uint8_t n) {
  Wire.beginTransmission(addr);
  Wire.write(reg);
  Wire.endTransmission();
  Wire.requestFrom(addr, n);
  for (uint8_t i = 0; i < n && Wire.available(); i++) {
    buf[i] = Wire.read();
  }
}

void i2cScan() {
  Serial.println("# Scanning I2C bus...");
  uint8_t count = 0;
  for (uint8_t addr = 1; addr < 127; addr++) {
    Wire.beginTransmission(addr);
    if (Wire.endTransmission() == 0) {
      Serial.print("#   Found device at 0x");
      if (addr < 16) Serial.print("0");
      Serial.println(addr, HEX);
      count++;
    }
  }
  if (count == 0) {
    Serial.println("#   No I2C devices found!");
  } else {
    Serial.print("# I2C scan complete. Devices found: ");
    Serial.println(count);
  }
  Serial.println();
}

// ============================================================================
// LSM6DS3TR-C (Accelerometer) - ORIGINAL METHOD
// ============================================================================

bool initIMU() {
  // Check WHO_AM_I (should be 0x6A)
  if (i2cRead(IMU_ADDR, 0x0F) != 0x6A) return false;

  // CTRL3_C: Block data update + auto-increment
  i2cWrite(IMU_ADDR, 0x12, 0x44);

  // CTRL1_XL: 52 Hz, ±4g (0011 10 00 = 0x38)
  i2cWrite(IMU_ADDR, 0x10, 0x38);

  // CTRL2_G: Gyro off
  i2cWrite(IMU_ADDR, 0x11, 0x00);

  return true;
}

void readIMU(int16_t *ax, int16_t *ay, int16_t *az) {
  uint8_t buf[6];
  i2cReadN(IMU_ADDR, 0x28, buf, 6);

  // Raw values (little-endian)
  int16_t rx = (int16_t)(buf[1] << 8 | buf[0]);
  int16_t ry = (int16_t)(buf[3] << 8 | buf[2]);
  int16_t rz = (int16_t)(buf[5] << 8 | buf[4]);

  // Convert to milli-g (±4g = 0.122 mg/LSB)
  *ax = (int16_t)((int32_t)rx * 122 / 1000);
  *ay = (int16_t)((int32_t)ry * 122 / 1000);
  *az = (int16_t)((int32_t)rz * 122 / 1000);
}

// ============================================================================
// MAX30101 (PPG) - SPARKFUN LIBRARY METHOD
// ============================================================================

bool initPPG() {
  Serial.println("# Initializing MAX30101...");
  
  // Initialize MAX30101 with SparkFun library
  if (!ppgSensor.begin(Wire, I2C_SPEED_STANDARD)) {
    Serial.println("# ERROR: MAX30101 init failed (begin() returned false)");
    Serial.println("# Check:");
    Serial.println("#   - Wiring (SDA/SCL)");
    Serial.println("#   - Power (3.3V not 5V)");
    Serial.println("#   - I2C address (should be 0x57)");
    return false;
  }

  // Configure MAX30101 with optimal settings
  // Parameters: ledBrightness, sampleAverage, ledMode, sampleRate, pulseWidth, adcRange
  
  byte ledBrightness = 60;     // Moderate brightness: 0-255 (60 ≈ 12mA)
  byte sampleAverage = 4;      // Average 4 samples (reduces noise)
  byte ledMode = 2;            // Mode 2: Red + IR LEDs
  int sampleRate = 100;        // 100 Hz sampling
  int pulseWidth = 411;        // 411 µs pulse width (18-bit resolution)
  int adcRange = 4096;         // 4096 nA range

  ppgSensor.setup(ledBrightness, sampleAverage, ledMode, sampleRate, pulseWidth, adcRange);

  Serial.println("# MAX30101 configured:");
  Serial.printf("#   LED Brightness: %d (0-255)\n", ledBrightness);
  Serial.printf("#   Sample Average: %d\n", sampleAverage);
  Serial.printf("#   LED Mode: %d (2=Red+IR)\n", ledMode);
  Serial.printf("#   Sample Rate: %d Hz\n", sampleRate);
  Serial.printf("#   Pulse Width: %d µs\n", pulseWidth);
  Serial.printf("#   ADC Range: %d nA\n", adcRange);

  // Test reading to verify LED is on
  delay(300);
  Serial.println("# Testing PPG readings...");
  
  for (int i = 0; i < 5; i++) {
    uint32_t ir = ppgSensor.getIR();
    uint32_t red = ppgSensor.getRed();
    Serial.printf("#   Sample %d - IR: %lu, RED: %lu\n", i, ir, red);
    
    if (ir > 50000 || red > 50000) {
      Serial.println("# PPG LED confirmed working!");
      break;
    }
    delay(100);
  }
  
  Serial.println("# MAX30101 initialized successfully");
  Serial.println("# Note: IR LED visible with phone camera (purple glow)");
  Serial.println("#       RED LED visible to naked eye");
  
  return true;
}

uint32_t readPPG() {
  // Read IR channel (primary for heart rate)
  return ppgSensor.getIR();
}

// ============================================================================
// Timer ISR
// ============================================================================

void IRAM_ATTR onTimer() {
  sampleReady = true;
}

// ============================================================================
// Setup
// ============================================================================

void setup() {
  Serial.begin(115200);
  delay(500);

  Serial.println("\n# ========================================");
  Serial.println("# LipsyGuard Firmware Starting...");
  Serial.println("# ========================================");

  Wire.begin(I2C_SDA, I2C_SCL);
  Wire.setClock(400000);
  
  Serial.printf("# I2C: SDA=%d, SCL=%d\n", I2C_SDA, I2C_SCL);
  Serial.println();

  // Scan I2C bus
  i2cScan();

  // Initialize sensors
  if (!initIMU()) {
    Serial.println("# ERROR: IMU not found");
    while (1) delay(1000);
  }
  Serial.println("# IMU initialized successfully");

  if (!initPPG()) {
    Serial.println("# ERROR: PPG initialization failed");
    while (1) delay(1000);
  }

  Serial.println("\n# ========================================");
  Serial.println("# LipsyGuard Ready");
  Serial.println("# ========================================");
  Serial.println("# Format: timestamp_ms,accel_x,accel_y,accel_z,ppg_ir");
  Serial.println("# Units: time(ms), accel(milli-g), ppg(ADC counts)");
  Serial.println("# Sampling rate: 16 Hz");
  Serial.println("#");
  Serial.println("# PLACE FINGER ON PPG SENSOR");
  Serial.println("# ========================================\n");

  delay(1000);

  // Start 16 Hz timer
  timer = timerBegin(0, 80, true);  // 1 MHz tick
  timerAttachInterrupt(timer, &onTimer, true);
  timerAlarmWrite(timer, SAMPLE_INTERVAL_US, true);
  timerAlarmEnable(timer);
}

// ============================================================================
// Main Loop
// ============================================================================

void loop() {
  if (!sampleReady) return;
  sampleReady = false;

  // Read sensors
  int16_t ax, ay, az;
  readIMU(&ax, &ay, &az);
  uint32_t ppg = readPPG();

  // Output: timestamp,ax,ay,az,ppg
  Serial.printf("%lu,%d,%d,%d,%lu\n", millis(), ax, ay, az, ppg);

  sampleCount++;
}