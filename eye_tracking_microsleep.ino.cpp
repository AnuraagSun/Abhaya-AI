// eye_tracking_microsleep.ino
// Button on pin 2 simulates eyelid closure
// Hold >2s = microsleep event

const int EYE_BUTTON_PIN = 2;
unsigned long eyeClosedStart = 0;
bool isClosed = false;
bool microsleepDetected = false;

void setup() {
  pinMode(EYE_BUTTON_PIN, INPUT_PULLUP);
  Serial.begin(115200);
  Serial.println("Eye Tracking Ready");
}

void loop() {
  bool buttonPressed = !digitalRead(EYE_BUTTON_PIN); // Active LOW

  if (buttonPressed && !isClosed) {
    // Eyes just closed
    eyeClosedStart = millis();
    isClosed = true;
  } else if (!buttonPressed && isClosed) {
    // Eyes opened
    unsigned long closedDuration = millis() - eyeClosedStart;
    if (closedDuration >= 2000) {
      microsleepDetected = true;
      Serial.println("{\"microsleep\":true}");
      delay(1000); // Debounce
    }
    isClosed = false;
  }

  // Send heartbeat even when no event
  static unsigned long lastPing = 0;
  if (millis() - lastPing > 5000) {
    Serial.println("{\"microsleep\":false}");
    lastPing = millis();
  }

  delay(100);
}