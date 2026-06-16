// ============================================================
//  MAGI ESP32 — uart_protocol.h
//  ASCII command protocol for ESP32 ↔ Raspberry Pi 4B
// ============================================================
#pragma once
#include <Arduino.h>
#include "config.h"
#include "gait_engine.h"
#include "imu_driver.h"

// Parser result
enum ParseResult {
    CMD_NONE = 0,
    CMD_MOVE,
    CMD_GAIT,
    CMD_SPEED,
    CMD_STOP,
    CMD_HOME,
    CMD_IMU_REQ,
    CMD_STATUS,
    CMD_UNKNOWN
};

struct MoveCmd {
    float angles[NUM_SERVOS];  // all 12 joint angles in degrees
};

struct ParsedCommand {
    ParseResult type;
    MoveCmd     move;
    GaitType    gaitType;
    float       speed;
};

class UartProtocol {
public:
    UartProtocol(GaitEngine& gait, ImuDriver& imu);

    // Call once in setup()
    void begin();

    // Call in loop() — processes any incoming bytes, returns parsed command
    ParseResult update();

    // Send IMU data response to Pi
    void sendImuData(const ImuData& data);

    // Send status response
    void sendStatus(int generation, float velocity);

    // Send acknowledgement
    void sendAck(const char* msg);

    // Send error
    void sendError(const char* msg);

private:
    GaitEngine& _gait;
    ImuDriver&  _imu;

    char   _buf[UART_BUF_SIZE];
    int    _bufLen;
    uint32_t _lastRxMs;

    // Parse a complete newline-terminated line
    ParseResult parseLine(const char* line, ParsedCommand& out);

    // Apply a parsed command to the gait engine
    void applyCommand(const ParsedCommand& cmd);

    // Tokenize helper
    static int splitCSV(const char* str, float* out, int maxCount);
};
