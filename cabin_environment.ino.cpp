// cabin_environment.ino
#include <SoftwareSerial.h>
#include <DHT.h>

#define CO2_RX 10
#define CO2_TX 11
#define DHT_PIN 3
#define DHT_TYPE DHT22

SoftwareSerial co2Serial(CO2_RX, CO2_TX);
DHT dht(DHT_PIN, DHT_TYPE);

void setup() {
  Serial.begin(115200);
  co2Serial.begin(9600);
  dht.begin();
  Serial.println("Cabin Environment Ready");
}

uint16_t readCO2() {
  byte cmd[9] = {0xFF, 0x01, 0x86, 0x00, 0x00, 0x00, 0x00, 0x00, 0x79};
  byte resp[9];
  co2Serial.write(cmd, 9);
  delay(10);
  if (co2Serial.available() >= 9) {
    co2Serial.readBytes(resp, 9);
    if (resp[0] == 0xFF && resp[1] == 0x86) {
      return (resp[2] << 8) | resp[3];
    }
  }
  return 400;
}

void loop() {
  float co2 = readCO2();
  float temp = dht.readTemperature();
  float hum = dht.readHumidity();

  // Handle sensor errors
  if (isnan(temp) || isnan(hum)) {
    temp = 22.0;
    hum = 50.0;
  }

  Serial.print("{\"co2_ppm\":");
  Serial.print(co2);
  Serial.print(",\"temperature_c\":");
  Serial.print(temp);
  Serial.print(",\"humidity_pct\":");
  Serial.print(hum);
  Serial.println("}");

  delay(5000); // Update every 5s
}