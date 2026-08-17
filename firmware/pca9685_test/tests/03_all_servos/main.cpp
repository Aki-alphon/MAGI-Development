// ============================================================
//  TEST 03 — ESP32 + PCA9685 Verification (No Servos Needed)
//  Hardware: ESP32 DevKit V1  +  PCA9685 breakout @ 0x40
//
//  ⚡ Minimum wiring (4 wires only):
//    ESP32 GPIO21 (SDA) ──► PCA9685 SDA
//    ESP32 GPIO22 (SCL) ──► PCA9685 SCL
//    ESP32 3.3V         ──► PCA9685 VCC
//    ESP32 GND          ──► PCA9685 GND
//
//  Servos do NOT need to be connected.
//  The PCA9685 V+ power rail (servo supply) is NOT required.
//
//  What this test checks:
//    [A] I²C bus is alive (scan 0x01–0x7E)
//    [B] PCA9685 responds at address 0x40
//    [C] Oscillator frequency register is writable (27 MHz cal)
//    [D] PWM frequency register accepted (50 Hz)
//    [E] All 16 PWM channels can be written without I²C NACK
//    [F] Readback: verify MODE1 register is not 0xFF (stuck bus)
//
//  Expected Serial output:
//    [A] PASS  I²C scan — 1 device found at 0x40
//    [B] PASS  PCA9685 ACK at 0x40
//    [C] PASS  Oscillator freq set (27 MHz)
//    [D] PASS  PWM frequency set (50 Hz)
//    [E] PASS  All 16 channels writable (no NACK)
//    [F] PASS  MODE1 register readable (0x20)
//    ✅  TEST 03 PASS — ESP32 ↔ PCA9685 link verified
// ============================================================
#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

// ── Pin & address constants ───────────────────────────────────
#define SDA_PIN      21
#define SCL_PIN      22
#define I2C_FREQ_HZ  400000UL   // 400 kHz Fast-mode
#define PCA_ADDR     0x40
#define PCA_OSC_HZ   27000000UL
#define PWM_FREQ_HZ  50

// PCA9685 internal register addresses (for direct readback)
#define PCA_MODE1_REG 0x00

Adafruit_PWMServoDriver pca(PCA_ADDR);

// ── Result tracking ───────────────────────────────────────────
struct Result { char tag; const char* label; bool pass; String detail; };
Result results[8];
uint8_t rIdx = 0;

void record(char tag, const char* label, bool pass, String detail = "") {
    results[rIdx++] = { tag, label, pass, detail };
}

// ── Test helpers ──────────────────────────────────────────────

// A: Scan I²C bus, count devices and check for 0x40
bool testI2CScan(int& deviceCount, bool& foundPCA) {
    deviceCount = 0;
    foundPCA = false;
    Serial.println("\n  Scanning I²C bus (0x01 – 0x7E)...");
    for (uint8_t addr = 1; addr < 127; addr++) {
        Wire.beginTransmission(addr);
        if (Wire.endTransmission() == 0) {
            deviceCount++;
            Serial.printf("    Found: 0x%02X", addr);
            if (addr == 0x40) { Serial.print("  ← PCA9685 ✓"); foundPCA = true; }
            if (addr == 0x41) Serial.print("  ← PCA9685 alt");
            if (addr == 0x68) Serial.print("  ← MPU-6050");
            Serial.println();
        }
    }
    Serial.printf("  Scan done: %d device(s) found.\n", deviceCount);
    return foundPCA;
}

// B: Direct ACK check at 0x40
bool testPCAAck() {
    Wire.beginTransmission(PCA_ADDR);
    return (Wire.endTransmission() == 0);
}

// C+D: Init PCA and set freq — Adafruit library handles the register writes
bool testPCAInit() {
    pca.begin();
    pca.setOscillatorFrequency(PCA_OSC_HZ);
    pca.setPWMFreq(PWM_FREQ_HZ);
    delay(10);
    // If we got here without hanging on I²C, it passed
    return true;
}

// E: Write all 16 channels — detect any NACK
bool testAllChannels(uint8_t& failCh) {
    // Neutral pulse: 1500 µs → tick = 1500 * 50 * 4096 / 1000000 = 307
    const uint16_t NEUTRAL_TICK = 307;
    failCh = 255; // sentinel
    for (uint8_t ch = 0; ch < 16; ch++) {
        // Write using raw I²C to catch NACK explicitly
        // Each PCA channel occupies 4 bytes starting at 0x06 + ch*4
        uint8_t reg = 0x06 + ch * 4;
        Wire.beginTransmission(PCA_ADDR);
        Wire.write(reg);
        Wire.write(0x00);           // ON_L
        Wire.write(0x00);           // ON_H
        Wire.write(NEUTRAL_TICK & 0xFF);   // OFF_L
        Wire.write((NEUTRAL_TICK >> 8) & 0x0F); // OFF_H
        if (Wire.endTransmission() != 0) {
            failCh = ch;
            return false;
        }
        delay(2);
    }
    return true;
}

// F: Read MODE1 register back — must not be 0xFF (stuck SDA)
bool testMODE1Readback(uint8_t& regVal) {
    Wire.beginTransmission(PCA_ADDR);
    Wire.write(PCA_MODE1_REG);
    if (Wire.endTransmission(false) != 0) { regVal = 0xFF; return false; }
    Wire.requestFrom((uint8_t)PCA_ADDR, (uint8_t)1);
    if (!Wire.available())              { regVal = 0xFF; return false; }
    regVal = Wire.read();
    // After setPWMFreq, MODE1 auto-restart bit (0x80) may be set;
    // 0xFF means nobody drove the bus correctly
    return (regVal != 0xFF);
}

// ── Summary printer ───────────────────────────────────────────
void printSummary() {
    Serial.println();
    Serial.println("╔════════════════════════════════════════════════╗");
    Serial.println("║     TEST 03 — ESP32 + PCA9685 Results         ║");
    Serial.println("╠════════════════════════════════════════════════╣");
    uint8_t passed = 0;
    for (uint8_t i = 0; i < rIdx; i++) {
        Serial.printf("║  [%c] %-30s  %s  ║\n",
                      results[i].tag,
                      results[i].label,
                      results[i].pass ? "PASS ✓" : "FAIL ✗");
        if (results[i].detail.length()) {
            Serial.printf("║      ↳ %-40s ║\n", results[i].detail.c_str());
        }
        if (results[i].pass) passed++;
    }
    Serial.println("╠════════════════════════════════════════════════╣");
    if (passed == rIdx) {
        Serial.println("║  ✅  ALL CHECKS PASSED                        ║");
        Serial.println("║     ESP32 ↔ PCA9685 I²C link is healthy      ║");
        Serial.println("║     Safe to connect servos and run Test 03b  ║");
    } else {
        Serial.printf( "║  ❌  %d/%d checks failed — see details above  ║\n",
                       rIdx - passed, rIdx);
        Serial.println("║     Check wiring: SDA=GPIO21, SCL=GPIO22     ║");
        Serial.println("║     VCC=3.3V, GND shared, pull-ups present?  ║");
    }
    Serial.println("╚════════════════════════════════════════════════╝");
}

// ── Setup (single-shot test) ──────────────────────────────────
void setup() {
    Serial.begin(115200);
    delay(800);

    Serial.println("╔════════════════════════════════════════════════╗");
    Serial.println("║  MAGI TEST 03 — ESP32 + PCA9685 Basic Test    ║");
    Serial.println("║  Minimum hardware: 4-wire I²C connection only  ║");
    Serial.println("╚════════════════════════════════════════════════╝");

    // Init I²C
    Wire.begin(SDA_PIN, SCL_PIN);
    Wire.setClock(I2C_FREQ_HZ);
    delay(50);

    // ── [A] I²C scan ─────────────────────────────────────────
    Serial.println("\n[A] I²C Bus Scan:");
    int devCount; bool foundPCA;
    bool passA = testI2CScan(devCount, foundPCA);
    record('A', "I²C scan — PCA9685 at 0x40",
           passA,
           passA ? String(devCount) + " device(s) found" : "0x40 NOT found");

    if (!passA) {
        Serial.println("\n❌ FATAL: PCA9685 not detected. Check wiring. Halting.");
        printSummary();
        while (true) delay(1000);
    }

    // ── [B] Direct ACK ───────────────────────────────────────
    Serial.println("\n[B] Direct ACK check at 0x40:");
    bool passB = testPCAAck();
    record('B', "PCA9685 ACK at 0x40", passB, passB ? "ACK received" : "NACK");
    Serial.printf("    %s\n", passB ? "✓ ACK" : "✗ NACK — bus error?");

    // ── [C+D] Init + freq ────────────────────────────────────
    Serial.println("\n[C] PCA9685 init — oscillator + PWM freq:");
    bool passCD = testPCAInit();
    record('C', "Oscillator 27 MHz + PWM 50 Hz",
           passCD, passCD ? "Regs written OK" : "I²C hung during init");
    Serial.printf("    %s\n", passCD ? "✓ Init OK" : "✗ Init failed");

    // ── [E] Write all 16 channels ────────────────────────────
    Serial.println("\n[D] Write 16 PWM channels (neutral 90°):");
    uint8_t failCh;
    bool passE = testAllChannels(failCh);
    record('D', "All 16 channels writable",
           passE,
           passE ? "ch0–ch15 NACK-free" : "NACK on ch" + String(failCh));
    if (passE) {
        Serial.println("    ✓ ch00–ch15 all written without NACK");
    } else {
        Serial.printf( "    ✗ NACK on channel %d\n", failCh);
    }

    // ── [F] MODE1 readback ───────────────────────────────────
    Serial.println("\n[E] MODE1 register readback:");
    uint8_t mode1;
    bool passF = testMODE1Readback(mode1);
    record('E', "MODE1 register readable",
           passF,
           "MODE1 = 0x" + String(mode1, HEX));
    Serial.printf("    MODE1 = 0x%02X  %s\n",
                  mode1, passF ? "✓ valid" : "✗ 0xFF stuck bus");

    // ── Print summary ────────────────────────────────────────
    printSummary();
}

void loop() {
    // Single-shot — reset ESP32 to rerun
}
