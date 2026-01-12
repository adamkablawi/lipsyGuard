/*
 * LipsyGuard Firmware
 * ESP32-C3 + LSM6DS3TR-C (IMU) + MAX30101 (PPG)
 *
 * Outputs: timestamp_ms,accel_x,accel_y,accel_z,ppg\n
 * Rate: 16 Hz (62.5ms interval)
 * Units: Accelerometer in milli-g, PPG as raw IR ADC
 */

#include <Wire.h>

// ============================================================================
// Configuration
// ============================================================================

#define SAMPLE_INTERVAL_US  62500  // 16 Hz = 62.5ms

// I2C
#define I2C_SDA  8
#define I2C_SCL  9

// Sensor addresses
#define IMU_ADDR  0x6A  // LSM6DS3TR-C
#define PPG_ADDR  0x57  // MAX30101

// ============================================================================
// Globals
// ============================================================================

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

// ============================================================================
// LSM6DS3TR-C (Accelerometer)
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
// MAX30101 (PPG - IR Channel)
// ============================================================================

bool initPPG() {
  // Check Part ID (should be 0x15)
  if (i2cRead(PPG_ADDR, 0xFF) != 0x15) return false;

  // Reset
  i2cWrite(PPG_ADDR, 0x09, 0x40);
  delay(100);

  // FIFO config: 4-sample average, rollover enabled
  i2cWrite(PPG_ADDR, 0x08, 0x50);

  // Mode: Heart Rate only (IR LED)
  i2cWrite(PPG_ADDR, 0x09, 0x02);

  // SpO2 config: 100 sps, 411µs pulse, 4096 range
  i2cWrite(PPG_ADDR, 0x0A, 0x27);

  // IR LED current (~16mA)
  i2cWrite(PPG_ADDR, 0x0D, 0x50);

  // Clear FIFO
  i2cWrite(PPG_ADDR, 0x04, 0x00);
  i2cWrite(PPG_ADDR, 0x05, 0x00);
  i2cWrite(PPG_ADDR, 0x06, 0x00);

  return true;
}

uint32_t readPPG() {
  // Check if data available
  uint8_t wr = i2cRead(PPG_ADDR, 0x04);
  uint8_t rd = i2cRead(PPG_ADDR, 0x06);
  if (wr == rd) return 0;  // No new data

  // Read one sample (3 bytes for IR in HR mode)
  uint8_t buf[3];
  i2cReadN(PPG_ADDR, 0x07, buf, 3);

  // 18-bit value
  return ((uint32_t)(buf[0] & 0x03) << 16) | ((uint32_t)buf[1] << 8) | buf[2];
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

  Wire.begin(I2C_SDA, I2C_SCL);
  Wire.setClock(400000);

  // Initialize sensors
  if (!initIMU()) {
    Serial.println("# ERROR: IMU not found");
    while (1) delay(1000);
  }

  if (!initPPG()) {
    Serial.println("# ERROR: PPG not found");
    while (1) delay(1000);
  }

  Serial.println("# LipsyGuard ready");

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
