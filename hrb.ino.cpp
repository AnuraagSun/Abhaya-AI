// hrv_earlobe.ino
#include <Wire.h>
#include <SparkFun_MAX30102.h>

MAX30102 sensor;
const byte SAMPLING_RATE = 25; // Hz
unsigned long lastSample = 0;
float hrv = 60.0; // Simulated HRV (ms between beats)

void setup() {
  Serial.begin(115200);
  Wire.begin(I2C_SPEED_FAST);

  if (!sensor.begin(Wire, I2C_SPEED_FAST)) {
    Serial.println("MAX30102 not found!");
    while (1) delay(10);
  }

  sensor.setup();
  sensor.setPulseAmplitudeRed(0x0A);
  sensor.setPulseAmplitudeIR(0x0A);
  Serial.println("HRV Sensor Ready");
}

void loop() {
  if (millis() - lastSample >= (1000 / SAMPLING_RATE)) {
    sensor.check(); // Read FIFO

    // In a real system: use PPG peak detection → R-R intervals → HRV
    // For demo: simulate stable HRV with small noise
    hrv = 65.0 + (sin(millis() / 1000.0) * 5.0);

    Serial.print("{\"hrv\":");
    Serial.print(hrv, 1);
    Serial.println("}");

    lastSample = millis();
  }
}