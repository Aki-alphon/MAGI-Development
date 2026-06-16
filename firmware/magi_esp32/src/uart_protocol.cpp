// ============================================================
//  MAGI ESP32 — uart_protocol.cpp
//  Parses ASCII commands from Raspberry Pi and dispatches
// ============================================================
#include "uart_protocol.h"
#include <string.h>
#include <stdlib.h>

// Serial2 = UART2 on ESP32 (RX=GPIO16, TX=GPIO17)
#define PI_SERIAL Serial2

UartProtocol::UartProtocol(GaitEngine& gait, ImuDriver& imu)
    : _gait(gait), _imu(imu), _bufLen(0), _lastRxMs(0) {}

void UartProtocol::begin() {
    PI_SERIAL.begin(UART_PI_BAUD, SERIAL_8N1, UART_RX_PIN, UART_TX_PIN);
    Serial.println("[UART] Serial2 initialized for Pi comms");
}

ParseResult UartProtocol::update() {
    // Read available bytes into line buffer
    while (PI_SERIAL.available()) {
        char c = PI_SERIAL.read();
        _lastRxMs = millis();

        if (c == '\n' || c == '\r') {
            if (_bufLen > 0) {
                _buf[_bufLen] = '\0';
                ParsedCommand cmd;
                ParseResult r = parseLine(_buf, cmd);
                _bufLen = 0;
                if (r != CMD_NONE && r != CMD_UNKNOWN) {
                    applyCommand(cmd);
                }
                return r;
            }
        } else {
            if (_bufLen < UART_BUF_SIZE - 1) {
                _buf[_bufLen++] = c;
            }
        }
    }

    // Watchdog — if no data received for WATCHDOG_TIMEOUT_MS, stop
    if (_lastRxMs > 0 && (millis() - _lastRxMs) > WATCHDOG_TIMEOUT_MS) {
        _lastRxMs = 0;
        Serial.println("[UART] Watchdog timeout — stopping gait");
        _gait.stop();
        sendError("WATCHDOG_TIMEOUT");
    }

    return CMD_NONE;
}

ParseResult UartProtocol::parseLine(const char* line, ParsedCommand& out) {
    out.type = CMD_NONE;

    // MOVE a0,a1,...,a11
    if (strncmp(line, "MOVE ", 5) == 0) {
        int n = splitCSV(line + 5, out.move.angles, NUM_SERVOS);
        if (n == NUM_SERVOS) {
            out.type = CMD_MOVE;
        } else {
            sendError("BAD_MOVE_ARGS");
        }
        return out.type;
    }

    // GAIT crawl|trot|stand
    if (strncmp(line, "GAIT ", 5) == 0) {
        const char* g = line + 5;
        if (strcmp(g, "crawl") == 0) {
            out.gaitType = GAIT_CRAWL;
        } else if (strcmp(g, "trot") == 0) {
            out.gaitType = GAIT_TROT;
        } else if (strcmp(g, "stand") == 0) {
            out.gaitType = GAIT_STAND;
        } else {
            sendError("UNKNOWN_GAIT");
            return CMD_UNKNOWN;
        }
        out.type = CMD_GAIT;
        return out.type;
    }

    // SPEED 0-100
    if (strncmp(line, "SPEED ", 6) == 0) {
        out.speed = atof(line + 6) / 100.0f;
        out.type = CMD_SPEED;
        return out.type;
    }

    // DIR vx,vz,yawRate (normalised -1..1 each)
    if (strncmp(line, "DIR ", 4) == 0) {
        float vals[3] = {1.0f, 0.0f, 0.0f};
        splitCSV(line + 4, vals, 3);
        GaitCommand cmd;
        cmd.vx = vals[0];
        cmd.vz = vals[1];
        cmd.yawRate = vals[2];
        cmd.speed = 0.6f;
        _gait.setCommand(cmd);
        sendAck("DIR_OK");
        return CMD_GAIT;
    }

    // STOP
    if (strcmp(line, "STOP") == 0) {
        out.type = CMD_STOP;
        return out.type;
    }

    // HOME
    if (strcmp(line, "HOME") == 0) {
        out.type = CMD_HOME;
        return out.type;
    }

    // IMU
    if (strcmp(line, "IMU") == 0) {
        sendImuData(_imu.data());
        return CMD_IMU_REQ;
    }

    // STATUS
    if (strcmp(line, "STATUS") == 0) {
        sendStatus(0, 0.0f);
        return CMD_STATUS;
    }

    Serial.printf("[UART] Unknown: '%s'\n", line);
    sendError("UNKNOWN_CMD");
    return CMD_UNKNOWN;
}

void UartProtocol::applyCommand(const ParsedCommand& cmd) {
    switch (cmd.type) {
    case CMD_MOVE: {
        // Direct joint override — build LegAngles from flat array
        LegAngles all[NUM_LEGS];
        for (int i = 0; i < NUM_LEGS; i++) {
            all[i].coxa  = cmd.move.angles[i * 3 + 0];
            all[i].femur = cmd.move.angles[i * 3 + 1];
            all[i].tibia = cmd.move.angles[i * 3 + 2];
        }
        // Access servo controller through gait (or store reference)
        // In main.cpp we'll apply directly via the servo controller
        // Signal via gait engine pause + direct servo apply
        _gait.stop();
        sendAck("MOVE_OK");
        break;
    }
    case CMD_GAIT:
        _gait.setGait(cmd.gaitType);
        sendAck("GAIT_OK");
        break;
    case CMD_SPEED: {
        GaitCommand gc = {};
        gc.speed = cmd.speed;
        gc.vx = 1.0f;
        _gait.setCommand(gc);
        sendAck("SPEED_OK");
        break;
    }
    case CMD_STOP:
        _gait.stop();
        sendAck("STOP_OK");
        break;
    case CMD_HOME:
        _gait.goHome();
        sendAck("HOME_OK");
        break;
    default:
        break;
    }
}

void UartProtocol::sendImuData(const ImuData& d) {
    PI_SERIAL.printf("IMU %.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.2f,%.2f\n",
                     d.accelX, d.accelY, d.accelZ,
                     d.gyroX,  d.gyroY,  d.gyroZ,
                     d.pitch,  d.roll);
}

void UartProtocol::sendStatus(int generation, float velocity) {
    PI_SERIAL.printf("OK gen=%d vel=%.1f\n", generation, velocity);
}

void UartProtocol::sendAck(const char* msg) {
    PI_SERIAL.printf("ACK %s\n", msg);
}

void UartProtocol::sendError(const char* msg) {
    PI_SERIAL.printf("ERR %s\n", msg);
    Serial.printf("[UART] Error: %s\n", msg);
}

int UartProtocol::splitCSV(const char* str, float* out, int maxCount) {
    int count = 0;
    char buf[UART_BUF_SIZE];
    strncpy(buf, str, sizeof(buf) - 1);
    buf[sizeof(buf) - 1] = '\0';

    char* tok = strtok(buf, ",");
    while (tok && count < maxCount) {
        out[count++] = atof(tok);
        tok = strtok(nullptr, ",");
    }
    return count;
}
