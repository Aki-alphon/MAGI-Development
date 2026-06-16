// ============================================================
//  MAGI ESP32 — imu_driver.cpp
//  MPU-6050 I²C driver with complementary filter
// ============================================================
#include "imu_driver.h"
#include <Arduino.h>
#include <math.h>

// MPU-6050 register addresses
#define MPU_PWR_MGMT_1   0x6B
#define MPU_SMPLRT_DIV   0x19
#define MPU_CONFIG       0x1A
#define MPU_GYRO_CFG     0x1B   // ±250 dps = 0x00
#define MPU_ACCEL_CFG    0x1C   // ±2g     = 0x00
#define MPU_ACCEL_XOUT_H 0x3B
#define MPU_TEMP_OUT_H   0x41
#define MPU_GYRO_XOUT_H  0x43
#define MPU_WHO_AM_I     0x75

// Scale factors
#define ACCEL_SCALE  16384.0f    // LSB/g  for ±2g
#define GYRO_SCALE     131.0f   // LSB/(°/s) for ±250°/s

ImuDriver::ImuDriver()
    : _gyroOffX(0), _gyroOffY(0), _gyroOffZ(0)
    , _accelOffX(0), _accelOffY(0) {
    _data = {};
}

static void writeReg(uint8_t reg, uint8_t val) {
    Wire.beginTransmission(MPU6050_ADDR);
    Wire.write(reg);
    Wire.write(val);
    Wire.endTransmission();
}

bool ImuDriver::begin() {
    // Check WHO_AM_I
    Wire.beginTransmission(MPU6050_ADDR);
    Wire.write(MPU_WHO_AM_I);
    Wire.endTransmission(false);
    Wire.requestFrom(MPU6050_ADDR, 1);
    if (!Wire.available()) {
        Serial.println("[IMU] MPU-6050 not found on I²C bus!");
        _data.valid = false;
        return false;
    }
    uint8_t whoami = Wire.read();
    if (whoami != 0x68) {
        Serial.printf("[IMU] WHO_AM_I = 0x%02X (expected 0x68)\n", whoami);
        _data.valid = false;
        return false;
    }

    // Wake up (clear sleep bit)
    writeReg(MPU_PWR_MGMT_1, 0x00);
    delay(100);

    // Sample rate divider → 100 Hz (1 kHz / (9+1))
    writeReg(MPU_SMPLRT_DIV, 9);

    // DLPF config: ~44 Hz bandwidth
    writeReg(MPU_CONFIG, 0x03);

    // Gyro: ±250 dps
    writeReg(MPU_GYRO_CFG, 0x00);

    // Accel: ±2g
    writeReg(MPU_ACCEL_CFG, 0x00);

    _data.valid = true;
    Serial.println("[IMU] MPU-6050 initialized OK");
    return true;
}

void ImuDriver::readRaw(int16_t& ax, int16_t& ay, int16_t& az,
                         int16_t& gx, int16_t& gy, int16_t& gz) {
    Wire.beginTransmission(MPU6050_ADDR);
    Wire.write(MPU_ACCEL_XOUT_H);
    Wire.endTransmission(false);
    Wire.requestFrom(MPU6050_ADDR, 14);  // 6 accel + 2 temp + 6 gyro

    ax = (Wire.read() << 8) | Wire.read();
    ay = (Wire.read() << 8) | Wire.read();
    az = (Wire.read() << 8) | Wire.read();

    int16_t tRaw = (Wire.read() << 8) | Wire.read();
    _data.temp = tRaw / 340.0f + 36.53f;

    gx = (Wire.read() << 8) | Wire.read();
    gy = (Wire.read() << 8) | Wire.read();
    gz = (Wire.read() << 8) | Wire.read();
}

bool ImuDriver::update(float dt) {
    if (!_data.valid) return false;

    int16_t ax, ay, az, gx, gy, gz;
    readRaw(ax, ay, az, gx, gy, gz);

    // Convert to physical units with zero-offset correction
    _data.accelX = (ax / ACCEL_SCALE) - _accelOffX;
    _data.accelY = (ay / ACCEL_SCALE) - _accelOffY;
    _data.accelZ =  az / ACCEL_SCALE;

    _data.gyroX = (gx / GYRO_SCALE) - _gyroOffX;
    _data.gyroY = (gy / GYRO_SCALE) - _gyroOffY;
    _data.gyroZ = (gz / GYRO_SCALE) - _gyroOffZ;

    // Accelerometer angles
    float accelPitch = atan2f(_data.accelX, sqrtf(_data.accelY * _data.accelY + _data.accelZ * _data.accelZ)) * 180.0f / M_PI;
    float accelRoll  = atan2f(_data.accelY, _data.accelZ) * 180.0f / M_PI;

    // Complementary filter
    _data.pitch = IMU_COMP_ALPHA * (_data.pitch + _data.gyroY * dt) + (1.0f - IMU_COMP_ALPHA) * accelPitch;
    _data.roll  = IMU_COMP_ALPHA * (_data.roll  + _data.gyroX * dt) + (1.0f - IMU_COMP_ALPHA) * accelRoll;

    return true;
}

void ImuDriver::calibrate(int samples) {
    Serial.print("[IMU] Calibrating (keep robot still)");
    double gxSum = 0, gySum = 0, gzSum = 0;
    double axSum = 0, aySum = 0;

    for (int i = 0; i < samples; i++) {
        int16_t ax, ay, az, gx, gy, gz;
        readRaw(ax, ay, az, gx, gy, gz);
        gxSum += gx; gySum += gy; gzSum += gz;
        axSum += ax; aySum += ay;
        delay(4);
        if (i % 50 == 0) Serial.print('.');
    }
    Serial.println(" done");

    _gyroOffX = gxSum / samples / GYRO_SCALE;
    _gyroOffY = gySum / samples / GYRO_SCALE;
    _gyroOffZ = gzSum / samples / GYRO_SCALE;
    _accelOffX = axSum / samples / ACCEL_SCALE;
    _accelOffY = aySum / samples / ACCEL_SCALE;

    Serial.printf("[IMU] Offsets gx=%.3f gy=%.3f gz=%.3f ax=%.3f ay=%.3f\n",
                  _gyroOffX, _gyroOffY, _gyroOffZ, _accelOffX, _accelOffY);
}

void ImuDriver::printDebug() const {
    Serial.printf("[IMU] ax=%.3fg ay=%.3fg az=%.3fg | gx=%.1f gy=%.1f gz=%.1f | pitch=%.1f° roll=%.1f°\n",
                  _data.accelX, _data.accelY, _data.accelZ,
                  _data.gyroX, _data.gyroY, _data.gyroZ,
                  _data.pitch, _data.roll);
}
