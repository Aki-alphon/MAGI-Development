#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

#define SDA_PIN 21
#define SCL_PIN 22

#define SERVO_CHANNEL 0
#define SERVO_FREQ 50

Adafruit_PWMServoDriver pca(0x40);

// Convert angle (0-180°) to PCA9685 ticks
uint16_t angleToTick(int angle) {

  angle = constrain(angle, 0, 180);

  // MG996R safe range
  int tick = map(angle, 0, 180, 150, 460);

  return tick;
}

void moveServo(int angle) {

  uint16_t tick = angleToTick(angle);

  pca.setPWM(SERVO_CHANNEL, 0, tick);

  Serial.print("Angle: ");
  Serial.println(angle);
}

void setup() {

  Serial.begin(115200);

  Wire.begin(SDA_PIN, SCL_PIN);

  pca.begin();

  pca.setPWMFreq(SERVO_FREQ);

  delay(500);

  Serial.println("MG996R Sweep Test");
}

void loop() {

  // 0 -> 180
  for (int angle = 0; angle <= 180; angle += 10) {

    moveServo(angle);

    delay(300);
  }

  // 180 -> 0
  for (int angle = 180; angle >= 0; angle -= 10) {

    moveServo(angle);

    delay(300);
  }
}