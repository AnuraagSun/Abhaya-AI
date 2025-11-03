// steering_jerk_imu.ino
#include <Wire.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>

Adafruit_MPU6050 mpu;

void setup() {
  Serial.begin(115200);
  Wire.begin();
  
  if (!mpu.begin()) {
    Serial.println("MPU6050 failed!");
    while (1) delay(10);
  }

  mpu.setAccelerometerRange(MPU6050_RANGE_2_G);
  mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);
  Serial.println("Steering IMU Ready");
}

void loop() {
  sensors_event_t acc, gyro, temp;
  mpu.getEvent(&acc, &gyro, &temp);

  // Compute jerk = derivative of acceleration magnitude
  static float lastAccMag = 0;
  float accMag = sqrt(
    sq(acc.acceleration.x) +
    sq(acc.acceleration.y) +
    sq(acc.acceleration.z)
  );
  float jerk = abs(accMag - lastAccMag);
  lastAccMag = accMag;

  // Normalize to 0–1 range (tune thresholds as needed)
  jerk = min(jerk / 0.5, 1.0);

  Serial.print("{\"steering_jerk\":");
  Serial.print(jerk, 3);
  Serial.println("}");

  delay(100); // 10 Hz update
}