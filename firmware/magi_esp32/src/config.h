// ============================================================
//  MAGI ESP32 Firmware — config.h
//  All hardware constants and tunable parameters
//  Hardware: ESP32 + PCA9685 + 12x MG996R + MPU-6050
// ============================================================
#pragma once

// ---- I2C Bus ------------------------------------------------
#define SDA_PIN          21
#define SCL_PIN          22
#define I2C_FREQ         400000UL   // 400 kHz fast mode

// ---- PCA9685 ------------------------------------------------
#define PCA9685_ADDR     0x40
#define SERVO_FREQ_HZ    50         // 50 Hz = 20 ms period

// ---- MG996R Servo Pulse Widths (microseconds) ---------------
// Calibrate these for your specific batch of servos
#define SERVO_US_MIN     500        // 0 degrees
#define SERVO_US_MID     1500       // 90 degrees
#define SERVO_US_MAX     2500       // 180 degrees

// ---- Leg / Channel Map (PCA9685 channels) -------------------
// Channel order: leg-major (coxa, femur, tibia per leg)
// Leg 0 = FR (Front Right)  Leg 1 = FL (Front Left)
// Leg 2 = BR (Back Right)   Leg 3 = BL (Back Left)
#define NUM_LEGS         4
#define DOF_PER_LEG      3         // coxa, femur, tibia
#define NUM_SERVOS       12        // 4 × 3

// Channel 0-based: leg * 3 + joint (0=coxa, 1=femur, 2=tibia)
// FR: ch 0,1,2 | FL: ch 3,4,5 | BR: ch 6,7,8 | BL: ch 9,10,11

// ---- Neutral (home) joint angles (degrees) ------------------
// Spider/crab stance: coxa points outward, femur level, tibia down
#define HOME_COXA        90.0f     // 90° = straight out
#define HOME_FEMUR       45.0f     // 45° = slight down angle
#define HOME_TIBIA       90.0f     // 90° = knee bent for stance

// ---- Robot Physical Dimensions (mm) -------------------------
#define LINK_COXA_MM     30.0f
#define LINK_FEMUR_MM    90.0f
#define LINK_TIBIA_MM    90.0f

// Shoulder mount offsets from body centre (mm)
#define SHOULDER_X_FRONT  80.0f
#define SHOULDER_X_BACK  -80.0f
#define SHOULDER_Z_RIGHT  40.0f
#define SHOULDER_Z_LEFT  -40.0f
#define BODY_HEIGHT_MM   100.0f    // target ground clearance

// ---- MPU-6050 -----------------------------------------------
#define MPU6050_ADDR     0x68
#define IMU_SAMPLE_HZ    100       // 100 Hz IMU read rate

// Complementary filter coefficient (0 = all gyro, 1 = all accel)
#define IMU_COMP_ALPHA   0.96f

// ---- UART Protocol (ESP32 ↔ Raspberry Pi) -------------------
// Serial2 on UART2: RX=GPIO16, TX=GPIO17
#define UART_PI_BAUD     115200
#define UART_RX_PIN      16
#define UART_TX_PIN      17
#define UART_BUF_SIZE    256

// ---- Gait Parameters ----------------------------------------
#define GAIT_STEP_HZ_DEFAULT   1.5f    // gait cycles per second
#define GAIT_STRIDE_MM_DEFAULT 60.0f   // foot stride length
#define GAIT_STEP_HEIGHT_MM    30.0f   // foot lift height

// ---- Safety -------------------------------------------------
#define SERVO_JOINT_LIMIT_DEG  30.0f   // max deviation from neutral
#define WATCHDOG_TIMEOUT_MS    3000    // UART silence → safe stop
#define ESTOP_BLINK_MS         200     // emergency LED blink rate

// ---- Debug LED ----------------------------------------------
#define LED_BUILTIN_PIN  2
