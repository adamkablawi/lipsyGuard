/*
 * LipsyGuard Firmware - NORMALIZED VERSION (BLE STREAM)
 * ESP32-C3 + LSM6DS3TR-C (IMU) + MAX30101 (PPG)
 *
 * BLE stream output (notify):
 *   timestamp_ms,accel_x,accel_y,accel_z,ppg\n
 *
 * Rate: 16 Hz (62.5ms interval)
 * Units: Accelerometer in milli-g (NORMALIZED via gyro), PPG as raw IR ADC
 *
 * BLE logic: same "simple hello world" approach:
 * - NimBLE
 * - one custom service + one notify characteristic
 * - advertise
 * - every sample: notify the exact CSV line (same as Serial.printf would)
 *
 * Library: NimBLE-Arduino (Arduino Library Manager)
 */

 #include <Wire.h>
 #include "MAX30105.h"
 #include <NimBLEDevice.h>
 
 // ============================================================================
 // Configuration
 // ============================================================================
 
 #define SAMPLE_INTERVAL_US  62500  // 16 Hz = 62.5ms
 
 // I2C
 #define I2C_SDA  6
 #define I2C_SCL  7
 
 // Sensor addresses
 #define IMU_ADDR  0x6A  // LSM6DS3TR-C
 
 // Sensor fusion parameters
 #define GYRO_SENSITIVITY      8.75f     // mdps/LSB for ±250dps range
 #define COMPLEMENTARY_ALPHA   0.98f     // 98% gyro, 2% accel
 #define RAD_TO_DEG            57.29577951f
 #define DEG_TO_RAD            0.01745329251f
 
 // BLE UUIDs (same simplicity as your hello-world)
 static NimBLEUUID BLE_SVC_UUID("12345678-1234-1234-1234-1234567890ab");
 static NimBLEUUID BLE_TX_UUID ("abcdefab-1234-5678-1234-abcdefabcdef");
 
 // ============================================================================
 // Globals
 // ============================================================================
 
 MAX30105 ppgSensor;
 
 volatile bool sampleReady = false;
 hw_timer_t *timer = NULL;
 uint32_t sampleCount = 0;
 
 // Orientation tracking (in degrees)
 float roll = 0.0f;
 float pitch = 0.0f;
 float yaw = 0.0f;
 
 uint32_t lastUpdateTime = 0;
 
 // BLE
 static NimBLECharacteristic* txChar = nullptr;
 
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
   if (count == 0) Serial.println("#   No I2C devices found!");
   else {
     Serial.print("# I2C scan complete. Devices found: ");
     Serial.println(count);
   }
   Serial.println();
 }
 
 // ============================================================================
 // LSM6DS3TR-C (Accelerometer + Gyroscope)
 // ============================================================================
 
 bool initIMU() {
   if (i2cRead(IMU_ADDR, 0x0F) != 0x6A) return false;
 
   i2cWrite(IMU_ADDR, 0x12, 0x44); // CTRL3_C: BDU + auto-increment
   i2cWrite(IMU_ADDR, 0x10, 0x38); // CTRL1_XL: 52 Hz, ±4g
   i2cWrite(IMU_ADDR, 0x11, 0x30); // CTRL2_G:  52 Hz, ±250 dps
 
   Serial.println("# IMU Configuration:");
   Serial.println("#   Accelerometer: 52 Hz, ±4g");
   Serial.println("#   Gyroscope: 52 Hz, ±250 dps");
   Serial.println("#   Orientation tracking: ENABLED");
   return true;
 }
 
 void readIMURaw(int16_t *ax, int16_t *ay, int16_t *az,
                 int16_t *gx, int16_t *gy, int16_t *gz) {
   uint8_t buf[12];
   i2cReadN(IMU_ADDR, 0x22, buf, 12);
 
   *gx = (int16_t)(buf[1] << 8 | buf[0]);
   *gy = (int16_t)(buf[3] << 8 | buf[2]);
   *gz = (int16_t)(buf[5] << 8 | buf[4]);
 
   *ax = (int16_t)(buf[7] << 8 | buf[6]);
   *ay = (int16_t)(buf[9] << 8 | buf[8]);
   *az = (int16_t)(buf[11] << 8 | buf[10]);
 }
 
 // ============================================================================
 // Sensor Fusion - Complementary Filter
 // ============================================================================
 
 void updateOrientation(float ax_g, float ay_g, float az_g,
                        float gx_dps, float gy_dps, float gz_dps, float dt) {
   float accelRoll  = atan2(ay_g, az_g) * RAD_TO_DEG;
   float accelPitch = atan2(-ax_g, sqrt(ay_g * ay_g + az_g * az_g)) * RAD_TO_DEG;
 
   roll  += gx_dps * dt;
   pitch += gy_dps * dt;
   yaw   += gz_dps * dt;
 
   roll  = COMPLEMENTARY_ALPHA * roll  + (1.0f - COMPLEMENTARY_ALPHA) * accelRoll;
   pitch = COMPLEMENTARY_ALPHA * pitch + (1.0f - COMPLEMENTARY_ALPHA) * accelPitch;
 }
 
 void rotateAccelToWorldFrame(float ax, float ay, float az,
                              float *wx, float *wy, float *wz) {
   float rollRad  = roll  * DEG_TO_RAD;
   float pitchRad = pitch * DEG_TO_RAD;
 
   float cosRoll = cos(rollRad);
   float sinRoll = sin(rollRad);
   float cosPitch = cos(pitchRad);
   float sinPitch = sin(pitchRad);
 
   *wx = ax * cosPitch + ay * sinRoll * sinPitch + az * cosRoll * sinPitch;
   *wy = ay * cosRoll - az * sinRoll;
   *wz = -ax * sinPitch + ay * sinRoll * cosPitch + az * cosRoll * cosPitch;
 
   *wz -= 1000.0f; // remove gravity in milli-g
 }
 
 // ============================================================================
 // Unified Sensor Reading with Normalization
 // ============================================================================
 
 void readAndNormalizeIMU(int16_t *norm_ax, int16_t *norm_ay, int16_t *norm_az) {
   int16_t raw_ax, raw_ay, raw_az, raw_gx, raw_gy, raw_gz;
   readIMURaw(&raw_ax, &raw_ay, &raw_az, &raw_gx, &raw_gy, &raw_gz);
 
   float ax_mg = (float)raw_ax * 0.122f;
   float ay_mg = (float)raw_ay * 0.122f;
   float az_mg = (float)raw_az * 0.122f;
 
   float gx_dps = (float)raw_gx * GYRO_SENSITIVITY / 1000.0f;
   float gy_dps = (float)raw_gy * GYRO_SENSITIVITY / 1000.0f;
   float gz_dps = (float)raw_gz * GYRO_SENSITIVITY / 1000.0f;
 
   uint32_t currentTime = millis();
   float dt = (currentTime - lastUpdateTime) / 1000.0f;
   lastUpdateTime = currentTime;
 
   if (dt > 0.001f && dt < 1.0f) {
     updateOrientation(ax_mg / 1000.0f, ay_mg / 1000.0f, az_mg / 1000.0f,
                       gx_dps, gy_dps, gz_dps, dt);
   }
 
   float world_ax, world_ay, world_az;
   rotateAccelToWorldFrame(ax_mg, ay_mg, az_mg, &world_ax, &world_ay, &world_az);
 
   *norm_ax = (int16_t)world_ax;
   *norm_ay = (int16_t)world_ay;
   *norm_az = (int16_t)world_az;
 }
 
 // ============================================================================
 // MAX30101 (PPG)
 // ============================================================================
 
 bool initPPG() {
   Serial.println("# Initializing MAX30101...");
 
   if (!ppgSensor.begin(Wire, I2C_SPEED_STANDARD)) {
     Serial.println("# ERROR: MAX30101 init failed (begin() returned false)");
     Serial.println("# Check:");
     Serial.println("#   - Wiring (SDA/SCL)");
     Serial.println("#   - Power (3.3V not 5V)");
     Serial.println("#   - I2C address (should be 0x57)");
     return false;
   }
 
   byte ledBrightness = 60;
   byte sampleAverage = 4;
   byte ledMode = 2;
   int sampleRate = 100;
   int pulseWidth = 411;
   int adcRange = 4096;
 
   ppgSensor.setup(ledBrightness, sampleAverage, ledMode, sampleRate, pulseWidth, adcRange);
 
   Serial.println("# MAX30101 configured:");
   Serial.printf("#   LED Brightness: %d (0-255)\n", ledBrightness);
   Serial.printf("#   Sample Average: %d\n", sampleAverage);
   Serial.printf("#   LED Mode: %d (2=Red+IR)\n", ledMode);
   Serial.printf("#   Sample Rate: %d Hz\n", sampleRate);
   Serial.printf("#   Pulse Width: %d us\n", pulseWidth);
   Serial.printf("#   ADC Range: %d nA\n", adcRange);
 
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
   return true;
 }
 
 uint32_t readPPG() {
   return ppgSensor.getIR();
 }
 
 // ============================================================================
 // Timer ISR
 // ============================================================================
 
 void IRAM_ATTR onTimer() {
   sampleReady = true;
 }
 
 // ============================================================================
 // BLE setup (simple)
 // ============================================================================
 
 void setupBLE() { 
   NimBLEDevice::init("LipsyGuard"); // name shown on Mac scanners
 
   NimBLEServer* server = NimBLEDevice::createServer();
   NimBLEService* svc = server->createService(BLE_SVC_UUID);
 
   txChar = svc->createCharacteristic(BLE_TX_UUID, NIMBLE_PROPERTY::NOTIFY);
   svc->start();
 
   NimBLEAdvertising* adv = NimBLEDevice::getAdvertising();
   adv->addServiceUUID(BLE_SVC_UUID);
   adv->start();
 
   Serial.println("# BLE advertising started");
 }
 
 // ============================================================================
 // Setup
 // ============================================================================
 
 void setup() {
   Serial.begin(115200);
   delay(500);
 
   Wire.begin(I2C_SDA, I2C_SCL);
   Wire.setClock(400000);
 
   i2cScan();
 
   if (!initIMU()) {
     Serial.println("# ERROR: IMU not found");
     while (1) delay(1000);
   }
 
   if (!initPPG()) {
     Serial.println("# ERROR: PPG initialization failed");
     while (1) delay(1000);
   }
 
   lastUpdateTime = millis();
 
   setupBLE();
 
   // Start 16 Hz timer (ESP32 core 3.x API)
   timer = timerBegin(1000000);            // 1 MHz -> 1 tick = 1 us
   timerAttachInterrupt(timer, &onTimer);  // new signature
   timerAlarm(timer, SAMPLE_INTERVAL_US, true, 0);
   timerStart(timer);
 }
 
 // ============================================================================
 // Main Loop
 // ============================================================================
 
 void loop() {
   if (!sampleReady) return;
   sampleReady = false;
 
   int16_t ax, ay, az;
   readAndNormalizeIMU(&ax, &ay, &az);
 
   uint32_t ppg = readPPG();
 
   // EXACT same line as your Serial.printf would output
   char line[96];
   int n = snprintf(line, sizeof(line), "%lu,%d,%d,%d,%lu\n",
                    millis(), ax, ay, az, ppg);
 
   // Stream the exact bytes over BLE notify
   if (txChar) {
     txChar->setValue((uint8_t*)line, n);
     txChar->notify();
   }
 
   // Optional: keep serial output for debugging (does not affect BLE stream)
   // Serial.print(line);
 
   sampleCount++;
 }