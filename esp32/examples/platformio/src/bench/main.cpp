// SteerTTS Nano benchmark (PlatformIO + Arduino-ESP32). Prints RESULT| lines on the serial console.
#include <Arduino.h>

#include "nano_bench.h"

void setup() {
  Serial.begin(115200);
  delay(1000);
  disableCore0WDT();    // the benchmark keeps core 0 busy for seconds at a time (an app would yield between chunks)
  nano_bench_start(0);  // TTS on core 0; Arduino's loop() runs on core 1
}

void loop() { delay(1000); }
