

#include <Servo.h>
int angle = 90;
bool platTest = false;
String Comp;
bool rev = false;
Servo Servo1;

void setup() {
  Servo1.attach(3);
  Serial.begin(115200); // set the baud rate
  delay(100);
}

void loop() {
  if(Serial.available()){ // only send data back if data has been sent
    char inByte1 = Serial.read();
    delay(10);
    char inByte2 = Serial.read();
    delay(10);
    char inByte3 = Serial.read();
    Serial.println(inByte1);
    int pos = ((int)inByte1-48)*100 + ((int)inByte2-48)*10 + ((int)inByte3-48);
    int diff = abs(pos-320);
    float kp = 4.98/320;
    int speed = int(round(diff*kp));
    speed = 1;
    if(pos>350 && angle < 180){
      angle = angle - speed;
    }
    if (pos<290 && angle > 0){
      angle = angle + speed;
    }
  }
  Servo1.write(angle);
}
  
