/*
 * LipsyGuard Firmware - NORMALIZED VERSION
 * ESP32-C3 + LSM6DS3TR-C (IMU) + MAX30101 (PPG)
 *
 * Outputs: timestamp_ms,accel_x,accel_y,accel_z,ppg\n
 * Rate: 16 Hz (62.5ms interval)
 * Units: Accelerometer in milli-g (NORMALIZED via gyro), PPG as raw IR ADC
 *
 * CHANGES:
 * - Gyroscope now enabled at 52 Hz (±250 dps)
 * - Complementary filter tracks device orientation
 * - Accelerometer readings rotated to remove gravity component
 * - Output represents body motion independent of device orientation
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
 
 // Sensor fusion parameters
 #define GYRO_SENSITIVITY  8.75f     // mdps/LSB for ±250dps range
 #define COMPLEMENTARY_ALPHA  0.98f  // Filter coefficient (0.98 = 98% gyro, 2% accel)
 #define RAD_TO_DEG  57.29577951f
 #define DEG_TO_RAD  0.01745329251f
 
 // ============================================================================
 // Globals
 // ============================================================================
 
 MAX30105 ppgSensor;
 
 volatile bool sampleReady = false;
 hw_timer_t *timer = NULL;
 uint32_t sampleCount = 0;
 
 // Orientation tracking (in degrees)
 float roll = 0.0f;   // Rotation around X-axis
 float pitch = 0.0f;  // Rotation around Y-axis
 float yaw = 0.0f;    // Rotation around Z-axis (not used for gravity compensation)
 
 uint32_t lastUpdateTime = 0;
 
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
 // LSM6DS3TR-C (Accelerometer + Gyroscope) - ENHANCED WITH GYRO
 // ============================================================================
 
 bool initIMU() {
   // Check WHO_AM_I (should be 0x6A)
   if (i2cRead(IMU_ADDR, 0x0F) != 0x6A) return false;
 
   // CTRL3_C: Block data update + auto-increment
   i2cWrite(IMU_ADDR, 0x12, 0x44);
 
   // CTRL1_XL: 52 Hz, ±4g (0011 10 00 = 0x38)
   i2cWrite(IMU_ADDR, 0x10, 0x38);
 
   // CTRL2_G: 52 Hz, ±250 dps (0011 00 00 = 0x30)
   // Previously was 0x00 (gyro off), now enabled at 52 Hz
   i2cWrite(IMU_ADDR, 0x11, 0x30);
 
   Serial.println("# IMU Configuration:");
   Serial.println("#   Accelerometer: 52 Hz, ±4g");
   Serial.println("#   Gyroscope: 52 Hz, ±250 dps");
   Serial.println("#   Orientation tracking: ENABLED");
 
   return true;
 }
 
 void readIMURaw(int16_t *ax, int16_t *ay, int16_t *az, int16_t *gx, int16_t *gy, int16_t *gz) {
   uint8_t buf[12];
   
   // Read gyro (0x22-0x27) and accel (0x28-0x2D) in one burst
   i2cReadN(IMU_ADDR, 0x22, buf, 12);
 
   // Gyroscope (little-endian)
   *gx = (int16_t)(buf[1] << 8 | buf[0]);
   *gy = (int16_t)(buf[3] << 8 | buf[2]);
   *gz = (int16_t)(buf[5] << 8 | buf[4]);
 
   // Accelerometer (little-endian)
   *ax = (int16_t)(buf[7] << 8 | buf[6]);
   *ay = (int16_t)(buf[9] << 8 | buf[8]);
   *az = (int16_t)(buf[11] << 8 | buf[10]);
 }
 
 // ============================================================================
 // Sensor Fusion - Complementary Filter
 // ============================================================================
 
 void updateOrientation(float ax_g, float ay_g, float az_g, float gx_dps, float gy_dps, float gz_dps, float dt) {
   // Step 1: Calculate roll and pitch from accelerometer (assuming static or slow motion)
   // These represent the "true" orientation if device is stationary
   float accelRoll = atan2(ay_g, az_g) * RAD_TO_DEG;
   float accelPitch = atan2(-ax_g, sqrt(ay_g * ay_g + az_g * az_g)) * RAD_TO_DEG;
 
   // Step 2: Integrate gyroscope to get orientation change
   // Gyroscope drift accumulates but is accurate short-term
   roll += gx_dps * dt;
   pitch += gy_dps * dt;
   yaw += gz_dps * dt;
 
   // Step 3: Complementary filter - blend gyro (short-term accurate) with accel (long-term stable)
   // Alpha = 0.98 means: 98% trust gyro integration, 2% trust accelerometer
   roll = COMPLEMENTARY_ALPHA * roll + (1.0f - COMPLEMENTARY_ALPHA) * accelRoll;
   pitch = COMPLEMENTARY_ALPHA * pitch + (1.0f - COMPLEMENTARY_ALPHA) * accelPitch;
   
   // Note: Yaw cannot be corrected by accelerometer (no magnetometer), so it drifts
   // For mouth guard application, yaw drift is acceptable as we care about linear motion
 }
 
 void rotateAccelToWorldFrame(float ax, float ay, float az, float *wx, float *wy, float *wz) {
   // Convert orientation to radians
   float rollRad = roll * DEG_TO_RAD;
   float pitchRad = pitch * DEG_TO_RAD;
   
   // Rotation matrix to transform from body frame to world frame
   // This removes the gravity component and gives true linear acceleration
   
   float cosRoll = cos(rollRad);
   float sinRoll = sin(rollRad);
   float cosPitch = cos(pitchRad);
   float sinPitch = sin(pitchRad);
   
   // Simplified rotation (roll and pitch only, ignoring yaw)
   // World frame has gravity pointing down in Z
   *wx = ax * cosPitch + ay * sinRoll * sinPitch + az * cosRoll * sinPitch;
   *wy = ay * cosRoll - az * sinRoll;
   *wz = -ax * sinPitch + ay * sinRoll * cosPitch + az * cosRoll * cosPitch;
   
   // Remove gravity (1000 milli-g) from Z-axis
   *wz -= 1000.0f;
 }
 
 // ============================================================================
 // Unified Sensor Reading with Normalization
 // ============================================================================
 
 void readAndNormalizeIMU(int16_t *norm_ax, int16_t *norm_ay, int16_t *norm_az) {
   // Read raw sensor data
   int16_t raw_ax, raw_ay, raw_az, raw_gx, raw_gy, raw_gz;
   readIMURaw(&raw_ax, &raw_ay, &raw_az, &raw_gx, &raw_gy, &raw_gz);
   
   // Convert accelerometer to milli-g (±4g = 0.122 mg/LSB)
   float ax_mg = (float)raw_ax * 0.122f;
   float ay_mg = (float)raw_ay * 0.122f;
   float az_mg = (float)raw_az * 0.122f;
   
   // Convert gyroscope to degrees per second (±250 dps = 8.75 mdps/LSB)
   float gx_dps = (float)raw_gx * GYRO_SENSITIVITY / 1000.0f;
   float gy_dps = (float)raw_gy * GYRO_SENSITIVITY / 1000.0f;
   float gz_dps = (float)raw_gz * GYRO_SENSITIVITY / 1000.0f;
   
   // Calculate time delta
   uint32_t currentTime = millis();
   float dt = (currentTime - lastUpdateTime) / 1000.0f; // Convert to seconds
   lastUpdateTime = currentTime;
   
   // Update orientation estimate (complementary filter)
   if (dt > 0.001f && dt < 1.0f) {  // Sanity check on dt
     updateOrientation(ax_mg / 1000.0f, ay_mg / 1000.0f, az_mg / 1000.0f, 
                       gx_dps, gy_dps, gz_dps, dt);
   }
   
   // Rotate accelerometer to world frame and remove gravity
   float world_ax, world_ay, world_az;
   rotateAccelToWorldFrame(ax_mg, ay_mg, az_mg, &world_ax, &world_ay, &world_az);
   
   // Convert back to int16_t (milli-g)
   *norm_ax = (int16_t)world_ax;
   *norm_ay = (int16_t)world_ay;
   *norm_az = (int16_t)world_az;
 }
 
 // ============================================================================
 // MAX30101 (PPG) - SPARKFUN LIBRARY METHOD (UNCHANGED)
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
 
   Wire.begin(I2C_SDA, I2C_SCL);
   Wire.setClock(400000);
 
   // Scan I2C bus
   i2cScan();
 
   // Initialize sensors
   if (!initIMU()) {
     Serial.println("# ERROR: IMU not found");
     while (1) delay(1000);
   }
 
   if (!initPPG()) {
     Serial.println("# ERROR: PPG initialization failed");
     while (1) delay(1000);
   }
 
   // Initialize time tracking
   lastUpdateTime = millis();
   
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
 
   // Read and normalize accelerometer (gyro-based)
   int16_t ax, ay, az;
   readAndNormalizeIMU(&ax, &ay, &az);
   
   // Read PPG (unchanged)
   uint32_t ppg = readPPG();
   
   // Output: timestamp,ax,ay,az,ppg
   Serial.printf("%lu,%d,%d,%d,%lu\n", millis(), ax, ay, az, ppg);
 
   sampleCount++;
 }
 