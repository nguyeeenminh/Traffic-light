int led_red = 8;
int led_green = 9;
int led_yellow = 10;

void setup() {
  pinMode(led_red, OUTPUT);
  pinMode(led_green, OUTPUT);
  pinMode(led_yellow, OUTPUT);
  Serial.begin(9600);

  // Mặc định: bật đèn đỏ
  digitalWrite(led_red, HIGH);
  digitalWrite(led_green, LOW);
  digitalWrite(led_yellow, LOW);
}

void loop() {
  if (Serial.available()) {
    char cmd = Serial.read();

    if (cmd == 'R') {  // Red
      digitalWrite(led_red, HIGH);
      digitalWrite(led_green, LOW);
    }

    if (cmd == 'G') {  // Green
      digitalWrite(led_red, LOW);
      digitalWrite(led_green, HIGH);
    }
  }
}
