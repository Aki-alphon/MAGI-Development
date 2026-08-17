// ============================================================
//  TEST 04 — MPU-6050 IMU Read
//  Purpose: Verify MPU-6050 is responsive on I²C,
//           print live accel/gyro data, run self-calibration
//
//  Expected output at rest on flat surface:
//    accelZ ≈ 1.0g (gravity)
//    accelX, accelY ≈ 0.0g
//    gyroX/Y/Z ≈ 0.0 °/s (slight noise ±0.5 normal)
//    pitch, roll ≈ 0° (if robot is level)
// ============================================================
#include <Arduino.h>
#include <Wire.h>
#include <math.h>

#define SDA_PIN   21
#define SCL_PIN   22
#define MPU_ADDR  0x68

// MPU registers
#define REG_PWR    0x6B
#define REG_ACCEL  0x3B
#define REG_GYRO   0x43
#define REG_WHO    0x75

void writeReg(uint8_t reg, uint8_t val) {
    Wire.beginTransmission(MPU_ADDR);
    Wire.write(reg); Wire.write(val);
    Wire.endTransmission();
}

void readRegs(uint8_t reg, uint8_t* buf, uint8_t len) {
    Wire.beginTransmission(MPU_ADDR);
    Wire.write(reg);
    Wire.endTransmission(false);
    Wire.requestFrom(MPU_ADDR, (int)len);
    for (int i = 0; i < len && Wire.available(); i++) buf[i] = Wire.read();
}

int16_t read16(uint8_t* buf, int offset) {
    return ((int16_t)buf[offset] << 8) | buf[offset + 1];
}

// Calibration offsets
float gxOff = 0, gyOff = 0, gzOff = 0;
float axOff = 0, ayOff = 0;
float pitch = 0, roll = 0;

void calibrate(int samples = 500) {
    Serial.print("Calibrating (keep robot flat and still)");
    double gxS=0, gyS=0, gzS=0, axS=0, ayS=0;
    uint8_t buf[14];
    for (int i = 0; i < samples; i++) {
        readRegs(REG_ACCEL, buf, 14);
        axS += read16(buf, 0); ayS += read16(buf, 2);
        readRegs(REG_GYRO, buf, 6);
        gxS += read16(buf, 0); gyS += read16(buf, 2); gzS += read16(buf, 4);
        delay(4);
        if (i % 50 == 0) Serial.print('.');
    }
    Serial.println(" done!");
    gxOff = gxS/samples/131.0f; gyOff = gyS/samples/131.0f; gzOff = gzS/samples/131.0f;
    axOff = axS/samples/16384.0f; ayOff = ayS/samples/16384.0f;
    Serial.printf("Offsets: gx=%.3f gy=%.3f gz=%.3f | ax=%.3f ay=%.3f\n",
                  gxOff, gyOff, gzOff, axOff, ayOff);
}

void setup() {
    Serial.begin(115200);
    delay(1000);
    Serial.println("=== TEST 04: MPU-6050 IMU ===");

    Wire.begin(SDA_PIN, SCL_PIN);
    Wire.setClock(400000);

    // WHO_AM_I check
    uint8_t who;
    readRegs(REG_WHO, &who, 1);
    Serial.printf("WHO_AM_I = 0x%02X (expected 0x68)\n", who);
    if (who != 0x68) {
        Serial.println("ERROR: MPU-6050 not found! Check wiring.");
        while (true) delay(1000);
    }

    // Init
    writeReg(REG_PWR, 0x00);       // wake up
    delay(100);
    writeReg(0x19, 9);             // sample rate = 1kHz / 10 = 100 Hz
    writeReg(0x1A, 0x03);          // DLPF ~44 Hz
    writeReg(0x1B, 0x00);          // gyro ±250 dps
    writeReg(0x1C, 0x00);          // accel ±2g
    Serial.println("MPU-6050 initialized OK\n");

    calibrate(500);
    Serial.println("\nLive data (press reset to re-calibrate):\n");
}

void loop() {
    uint8_t buf[14];
    readRegs(REG_ACCEL, buf, 14);
    float ax = read16(buf, 0) / 16384.0f - axOff;
    float ay = read16(buf, 2) / 16384.0f - ayOff;
    float az = read16(buf, 4) / 16384.0f;
    float temp = (read16(buf, 6) / 340.0f) + 36.53f;

    readRegs(REG_GYRO, buf, 6);
    float gx = read16(buf, 0) / 131.0f - gxOff;
    float gy = read16(buf, 2) / 131.0f - gyOff;
    float gz = read16(buf, 4) / 131.0f - gzOff;

    float dt = 0.01f;  // 100 Hz assumed
    float accelPitch = atan2f(ax, sqrtf(ay*ay + az*az)) * 180.0f / M_PI;
    float accelRoll  = atan2f(ay, az) * 180.0f / M_PI;
    pitch = 0.96f * (pitch + gy * dt) + 0.04f * accelPitch;
    roll  = 0.96f * (roll  + gx * dt) + 0.04f * accelRoll;

    Serial.printf("ax=% .3fg ay=% .3fg az=% .3fg | gx=% 6.1f gy=% 6.1f gz=% 6.1f | P=% 5.1f° R=% 5.1f° T=%.1f°C\n",
                  ax, ay, az, gx, gy, gz, pitch, roll, temp);

    delay(100);  // 10 Hz print rate
}
