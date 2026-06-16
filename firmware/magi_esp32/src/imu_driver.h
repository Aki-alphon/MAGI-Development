// ============================================================
//  MAGI ESP32 — imu_driver.h
//  MPU-6050 6-axis IMU driver with complementary filter
// ============================================================
#pragma once
#include <Wire.h>
#include "config.h"

struct ImuData {
    float accelX, accelY, accelZ;  // g
    float gyroX,  gyroY,  gyroZ;  // deg/s
    float pitch;                    // filtered pitch (deg)
    float roll;                     // filtered roll  (deg)
    float temp;                     // die temperature (°C)
    bool  valid;
};

class ImuDriver {
public:
    ImuDriver();

    // Initialize — returns false if device not found
    bool begin();

    // Read and filter — call at IMU_SAMPLE_HZ rate
    bool update(float dt);

    const ImuData& data() const { return _data; }

    // Calibrate zero offsets (robot must be stationary on flat ground)
    void calibrate(int samples = 500);

    // Print raw + filtered to Serial
    void printDebug() const;

private:
    ImuData _data;
    float _gyroOffX, _gyroOffY, _gyroOffZ;
    float _accelOffX, _accelOffY;  // level offsets

    // Read raw register values
    void readRaw(int16_t& ax, int16_t& ay, int16_t& az,
                 int16_t& gx, int16_t& gy, int16_t& gz);
};
