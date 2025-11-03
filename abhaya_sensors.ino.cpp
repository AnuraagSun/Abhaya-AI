// Abhaya AI - Arduino Sensor Hub
// Reads: MPU6050 (IMU), MH-Z19B (CO2), MAX30102 (HRV)
// Sends JSON over Serial to Python

#include <Wire.h>

// --- MPU6050 (IMU) ---
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
Adafruit_MPU6050 mpu;
bool mpu_found = false;

// --- MH-Z19B (CO2) ---
#define CO2_RX_PIN 10  // Use SoftwareSerial to avoid conflict
#define CO2_TX_PIN 11
#include <SoftwareSerial.h>
SoftwareSerial co2Serial(CO2_RX_PIN, CO2_TX_PIN);
bool co2_found = false;
uint16_t co2_ppm = 400;

// --- MAX30102 (HRV) ---
#include <SparkFun_MAX30102.h>
MAX30102 particleSensor;
bool hrv_found = false;
float hrv_value = 60.0; // Simulated if not found

void setup() {
  Serial.begin(115200); // To Python
  Wire.begin();

  // Initialize MPU6050
  if (mpu.begin()) {
    mpu.setAccelerometerRange(MPU6050_RANGE_4_G);
    mpu.setGyroRange(MPU6050_RANGE_500_DEG);
    mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);
    mpu_found = true;
    Serial.println("MPU6050 OK");
  } else {
    Serial.println("MPU6050 NOT FOUND");
  }

  // Initialize MH-Z19B
  co2Serial.begin(9600);
  co2_found = true; // Assume present; we'll validate on read

  // Initialize MAX30102
  if (particleSensor.begin(Wire, I2C_SPEED_FAST) == true) {
    particleSensor.setup(); // Configure sensor with default settings
    particleSensor.setPulseAmplitudeRed(0x0A); // Turn Red LED on
    hrv_found = true;
    Serial.println("MAX30102 OK");
  } else {
    Serial.println("MAX30102 NOT FOUND");
  }

  delay(1000);
  Serial.println("Abhaya Sensor Hub Ready");
}

// Read CO2 from MH-Z19B
uint16_t readCO2() {
  byte cmd[9] = {0xFF, 0x01, 0x86, 0x00, 0x00, 0x00, 0x00, 0x00, 0x79};
  byte response[9];
  
  co2Serial.write(cmd, 9);
  delay(10);
  
  if (co2Serial.available() >= 9) {
    for (int i = 0; i < 9; i++) {
      response[i] = co2Serial.read();
    }
    if (response[0] == 0xFF && response[1] == 0x86) {
      return (response[2] * 256) + response[3];
    }
  }
  return 400; // Default if failed
}

void loop() {
  sensors_event_t a, g, temp;
  float jerk_metric = 0.0;

  // --- Read IMU ---
  if (mpu_found) {
    mpu.getEvent(&a, &g, &temp);
    // Compute "jerk": magnitude of acceleration change (simplified)
    static float last_acc = 0;
    float acc_mag = sqrt(a.acceleration.x * a.acceleration.x +
                         a.acceleration.y * a.acceleration.y +
                         a.acceleration.z * a.acceleration.z);
    jerk_metric = abs(acc_mag - last_acc);
    last_acc = acc_mag;
    jerk_metric = min(jerk_metric, 2.0f); // Cap
  } else {
    // Simulate if not present
    jerk_metric = random(0, 100) / 100.0;
  }

  // --- Read CO2 ---
  if (co2_found) {
    co2_ppm = readCO2();
    if (co2_ppm < 400 || co2_ppm > 2000) co2_ppm = 800; // Sanity
  } else {
    co2_ppm = 800 + random(-100, 200);
  }

  // --- Read HRV (simplified as BPM for demo) ---
  if (hrv_found) {
    long irValue = particleSensor.getIR();
    // In real use: use algorithm (e.g., pulse oximetry + time between beats)
    // For demo: simulate stable HRV
    hrv_value = 60 + (irValue % 10); // Fake variation
  } else {
    hrv_value = 65 + (millis() / 1000) % 10;
  }

  // --- Send JSON to Python ---
  Serial.print("{\"steering_jerk\":");
  Serial.print(jerk_metric, 2);
  Serial.print(",\"co2_ppm\":");
  Serial.print(co2_ppm);
  Serial.print(",\"hrv\":");
  Serial.print(hrv_value, 1);
  Serial.println("}");

  delay(2000); // Match Python update rate
}