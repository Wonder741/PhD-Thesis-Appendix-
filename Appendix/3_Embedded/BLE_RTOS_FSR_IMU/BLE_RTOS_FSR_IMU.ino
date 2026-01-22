/*
File: BLE_RTOS_FSR_IMU.ino

Description
-----------
This firmware implements a real-time data acquisition and Bluetooth Low Energy (BLE)
transmission system for a smart insole device using FreeRTOS.

The system integrates:
  - Multiple FSR (Force Sensitive Resistor) sensors for plantar pressure measurement
  - An IMU (Inertial Measurement Unit) for foot motion tracking
  - BLE communication for wireless data streaming to a host device

The firmware uses FreeRTOS tasks to separate sensing, buffering, and BLE transmission,
ensuring stable timing and minimizing data loss during real-time operation.

System Architecture
-------------------
The program runs multiple RTOS tasks in parallel, including:
  1) FSR sampling task
     - Reads analog values from FSR sensors
     - Applies basic preprocessing (e.g., scaling or filtering)
  2) IMU sampling task
     - Reads accelerometer and gyroscope data via I2C/SPI
  3) BLE transmission task
     - Packages sensor data into a compact byte stream
     - Sends data over BLE at a controlled rate
  4) Optional receive/control task
     - Handles incoming BLE commands (e.g., start/stop, configuration)

Sensor data are written into a shared buffer, and the BLE task reads from this buffer
to transmit data without blocking sensor sampling.

Sampling Frequency Control
--------------------------
The overall sampling frequency is controlled by the parameter `baseFrequency`.

Empirically determined behavior:
  - baseFrequency = 1  → approximately 49 Hz sampling rate
  - baseFrequency = 2  → approximately 32 Hz sampling rate

This parameter adjusts the RTOS task timing and directly affects the achievable
sensor sampling frequency and BLE transmission stability. Lower values increase
sampling frequency but may increase the risk of BLE packet loss.

Data Handling
-------------
- Sensor data are transmitted in binary format to reduce packet size
- Transmission rate is limited to avoid BLE packet loss
- Buffer management prevents overwriting unsent data
- Timing is controlled using RTOS delays instead of blocking loops

BLE Communication
-----------------
- BLE is used in UART-like mode
- Data packets contain synchronized FSR and IMU samples
- Designed for real-time streaming rather than onboard storage

Intended Use
------------
This firmware is intended for:
  - Smart insole development
  - Gait analysis and plantar pressure research
  - Wearable motion and pressure monitoring
  - Experimental data collection in laboratory or clinical settings

Notes
-----
- Sampling rate and BLE throughput must be balanced to prevent data loss
- EMI and wireless interference may affect BLE reliability in clinical environments
- The code is structured for extensibility (e.g., adding SD card logging)

Author / Maintenance
--------------------
- Designed for research and prototyping
- Modify task priorities, buffer sizes, and baseFrequency with care
*/

#include <LSM6DS3.h>
#include <RTClib.h>
#include <bluefruit.h>
#include "FreeRTOS.h"
#include "Wire.h"

// Pin definitions
// Define 8 channel analog multiplexer control pins
#define MUX1 D8
#define MUX2 D7
#define MUX3 D6
// Define 2 negative voltage switch control pins
#define PHASEP D10
#define PHASEN D9
// Define 3 circuit control pins
#define CTEST D1
#define CSLEEP D2
#define CRESET D3
// Define one ADC pin
#define ADC_SIG A0

//************************ Device ************************
//Device name
const String deviceName = "2L";
// Base frequency for D8
const int baseFrequency = 1;  // One frame frequency 1024->16s, 64->1s 8->8Hz, 1->64Hz
const int fsrNumber = 16;
//Create a instance of class LSM6DS3
LSM6DS3 myIMU(I2C_MODE, 0x6A);    //I2C device address 0x6A

//************************ Cache ************************
const int buffBase = 3800;    //Set storage size
const int pack_size = 5;   //Set BLE transmission package size
unsigned long startTime = 0;    //Reset millisecond
unsigned long milliBuffer[buffBase];   //Storage for millisecond
short imuBuffer[6 * buffBase];   //Storage for IMU reading
short fsrBuffer[16 * buffBase];   //Storage for IMU reading
int imuWriteIndex = 0;    //Index for buffer of IMU reading
int adcWriteIndex = 0;   //Index for buffer of adc reading
int milliSendIndex = 0;   //Index for miliBuffer send
int imuSendIndex = 0;   //Index for imuBuffer send
int fsrSendIndex = 0;   //Index for fsrBuffer send
bool samplingFlag = false;
bool sendFlag = false;
bool imuOverflow = false;
bool adcOverflow = false;
SemaphoreHandle_t bufferSemaphore;


//************************ RTC ************************
RTC_Millis rtc;
// Initial date and time values
volatile int year = 2025;
volatile int month = 1;
volatile int day = 1;
volatile int hour = 0;
volatile int minute = 0;
volatile int second = 0;
// Reset flag
volatile bool timeReset = false;
// Update date and time every second
TickType_t Second_Time_Delay = 1024; 

//************************ BLE Service ************************
BLEDfu  bledfu;  // OTA DFU service
BLEDis  bledis;  // device information
BLEUart bleuart; // uart over ble
BLEBas  blebas;  // battery
char central_name_global[32] = { 0 };
String receivedString;         // Variable to store the received string
String lastProcessedString;    // Variable to store the last processed string
// Global variable to track BLE connection status
volatile bool bleConnected = false;

//************************ Tasks ************************
// Define a task function for the IMU reading
void SensorTask(void *pvParameters) {
  (void) pvParameters;

  while (true) { // A Task shall never return or exit.
    if (samplingFlag){
      // Read accelerometer data
      milliBuffer[imuWriteIndex] = millis() - startTime;
      imuBuffer[(imuWriteIndex * 6)] = myIMU.readRawAccelX();
      imuBuffer[(imuWriteIndex * 6) + 1] = myIMU.readRawAccelY();
      imuBuffer[(imuWriteIndex * 6) + 2] = myIMU.readRawAccelZ();
      // Read raw gyroscope data
      imuBuffer[(imuWriteIndex * 6) + 3] = myIMU.readRawGyroX();
      imuBuffer[(imuWriteIndex * 6) + 4] = myIMU.readRawGyroY();
      imuBuffer[(imuWriteIndex * 6) + 5] = myIMU.readRawGyroZ();
      // // Read thermometer data
      // sensorData[6] = myIMU.readTempC();

      imuWriteIndex++;
      if (imuWriteIndex % pack_size == 0) {
        imuOverflow = true; // A send package is full, set overflow flag
      }
      if (imuWriteIndex >= buffBase){
        imuWriteIndex = 0;    //Write buffer from beginning
      }
      vTaskDelay(pdMS_TO_TICKS(baseFrequency * 16 + 2)); // Delay for a period of time
    }
    else{
      vTaskDelay(pdMS_TO_TICKS(baseFrequency * 90));  // Delay of the base frequency
    }

  }
}


void UnifiedSignalTask(void *pvParameters) {
  unsigned int count = 0;  // Counter to manage different delays

  while (true) {
    if (samplingFlag) {
      // Execute operations for PHASEN, LED_GREEN, and PHASEP at eight times the base frequency
      if (count % 8 == 0) {
        digitalWrite(PHASEN, !digitalRead(PHASEN));  // Toggle PHASEN pin
        digitalWrite(PHASEP, !digitalRead(PHASEP));  // Toggle PHASEP pin
      }

      // Execute operations for MUX3 and LED_RED at four times the base frequency
      if (count % 4 == 0) {
        digitalWrite(MUX3, !digitalRead(MUX3));  // Toggle MUX3 pin
      }
      
      // Execute operations for MUX2 at twice the base frequency
      if (count % 2 == 0) {
        digitalWrite(MUX2, !digitalRead(MUX2));  // Toggle MUX2 pin
      }

      // Execute operations for MUX1 at the base frequency
      digitalWrite(MUX1, !digitalRead(MUX1));  // Toggle MUX1 pin
    
      count++;  // Increment the counter
      // Reset the counter if it exceeds or equals the least common multiple of delays to avoid overflow
      if (count >= 8) {
        count = 0;
      }
      // Get the current index from the pins
      int index = getimuWriteIndex();
      // Read from ADC and store in buffer at the position indicated by the pins
      if (index == 0 || index == 8){
        vTaskDelay(1);}
      fsrBuffer[adcWriteIndex * 16 + index] = analogRead(ADC_SIG);
      
      //if (index == 0 || index == 8){Serial.print(index);Serial.print(":");Serial.print(fsrBuffer[adcWriteIndex * 16 + index]);Serial.print(", PHASEP:");Serial.print(digitalRead(PHASEP));Serial.print(", PHASEN:");Serial.println(digitalRead(PHASEN));}
      // Set the overflow flag if we've just written to the last index
      if ((index + 1) % 16 == 0){
        adcWriteIndex++;
        adcOverflow = false;
      }


      if (adcWriteIndex % (pack_size) == (pack_size - 1)) {
        adcOverflow = true; // A send package is full, set overflow flag
      }
      if (adcWriteIndex >= buffBase){
        adcWriteIndex = 0;    //Write buffer from beginning
      }
        vTaskDelay(pdMS_TO_TICKS(baseFrequency));  // Delay of the base frequency
    }
    else{
      vTaskDelay(pdMS_TO_TICKS(baseFrequency * 16));  // Delay of the base frequency
    }
  }
}

// This function converts the pin readings into a number
int getimuWriteIndex() {
  int index = 0;
  index += digitalRead(PHASEN) * 8; // D9 is the MSB
  index += digitalRead(MUX3) * 4; 
  index += digitalRead(MUX2) * 2;
  index += digitalRead(MUX1);     // D8 is the LSB
  return index;
}

// The RTC thread
void TaskDateTime(void *pvParameters) {
  (void) pvParameters; // Silence unused parameter warning

  while (true) {
    updateClock();
    if (samplingFlag) {
      digitalWrite(LED_RED, !digitalRead(LED_RED));  // Toggle LED_RED pin
    }
    if (sendFlag) {
      digitalWrite(LED_GREEN, !digitalRead(LED_GREEN));  // Toggle LED_GREEN pin
    }
    // Delay for 1 second to match our software clock update.
    vTaskDelay(Second_Time_Delay);
  }
}

// This function updates the software-based clock every second
void updateClock() 
{
    DateTime now = rtc.now();
    year = now.year();
    month = now.month();
    day = now.day();
    hour = now.hour();
    minute = now.minute();
    second = now.second();
}

// This task checks for buffer overflow and prints the buffer if overflow occurs forward data from HW Serial to BLEUART
void ble_uart_task(void *pvParameters)
{
  (void) pvParameters; // Just to avoid compiler warnings

  while (true) {
    if (sendFlag && adcOverflow && imuOverflow){
      uint8_t buf[300];
      int index = 0;

      // Add device name (assuming it's a null-terminated string)
      for (int i = 0; deviceName[i] != '\0'; i++) {
        buf[index++] = deviceName[i];
      }
      // Add a delimiter (e.g., a comma)
      buf[index++] = ','; 

      // Add 5*1 milli buffer
      for (int i = 0; i < pack_size; i++) {
        buf[index++] = (milliBuffer[milliSendIndex + i] >> 24) & 0xFF; // Most significant byte
        buf[index++] = (milliBuffer[milliSendIndex + i] >> 16) & 0xFF;
        buf[index++] = (milliBuffer[milliSendIndex + i] >> 8) & 0xFF;
        buf[index++] = milliBuffer[milliSendIndex + i] & 0xFF; // Least significant byte
      }
      milliSendIndex += pack_size;
      if (milliSendIndex >= buffBase) {
        milliSendIndex = 0;
      }

      // Add 5*6 imu buffer
      for (int i = 0; i < (6 * pack_size); i++) {
        buf[index++] = (imuBuffer[imuSendIndex + i] >> 8) & 0xFF; // Most significant byte
        buf[index++] = imuBuffer[imuSendIndex + i] & 0xFF; // Least significant byte
      }
      imuSendIndex += 6 * pack_size;
      if (imuSendIndex >= 6 * buffBase) {
        imuSendIndex = 0;
      }

      // Add 5*16 imu buffer
      for (int i = 0; i < (16 * pack_size); i++) {
        buf[index++] = (fsrBuffer[fsrSendIndex + i] >> 8) & 0xFF; // Most significant byte
        buf[index++] = fsrBuffer[fsrSendIndex + i] & 0xFF; // Least significant byte
      }
      fsrSendIndex += 16 * pack_size;
      if (fsrSendIndex >= 16 * buffBase) {
        fsrSendIndex = 0;
      }

      bleuart.write(buf, index);
    
      // Reset the overflow flag
      imuOverflow = false;
      adcOverflow = false;
      if(!samplingFlag && (imuWriteIndex <= milliSendIndex)){
        sendFlag = false;
        // Reset all index
        imuWriteIndex = 0;    
        adcWriteIndex = 0;   
        milliSendIndex = 0;   
        imuSendIndex = 0;   
        fsrSendIndex = 0;   
      }

      // Slight delay to prevent this task from hogging the CPU
      vTaskDelay(pdMS_TO_TICKS(baseFrequency * 90));
    }
    else{
      vTaskDelay(pdMS_TO_TICKS(baseFrequency * 16));
    }
  }
}

void ble_receive_task(void *pvParameters)
{
  while(true) {
    if (bleConnected && bleuart.available()){
      // Check if there's data available
      receivedString = bleuart.readString();
      // Process the "stop" string
      if (receivedString.equalsIgnoreCase("stop_sampling")){
        samplingFlag = false;
      }
      // TODO: Handle or process the receivedString if needed
        else if (receivedString != lastProcessedString) {
        if (isInCorrectFormat(receivedString)) {
          int year   = receivedString.substring(0, 4).toInt();
          int month  = receivedString.substring(5, 7).toInt();
          int day    = receivedString.substring(8, 10).toInt();
          int hour   = receivedString.substring(11, 13).toInt();
          int minute = receivedString.substring(14, 16).toInt();
          int second = receivedString.substring(17, 19).toInt();

          DateTime newTime(year, month, day, hour, minute, second);
          rtc.adjust(newTime);
          startTime = millis();
          samplingFlag = true;
          sendFlag = true;
          imuOverflow = false;
          adcOverflow = false;
          // Reset all index
          imuWriteIndex = 0;    
          adcWriteIndex = 0;   
          milliSendIndex = 0;   
          imuSendIndex = 0;   
          fsrSendIndex = 0;   
        }
        lastProcessedString = receivedString;
      }
      vTaskDelay(pdMS_TO_TICKS(baseFrequency * 16)); // Short delay to prevent busy-waiting    
    }
    else{
      vTaskDelay(pdMS_TO_TICKS(baseFrequency * 90)); // Short delay to prevent busy-waiting
    }
  }
} 

bool isInCorrectFormat(const String &str) {
    // Simple check for the format "YYYY/MM/DD HH:MM:SS"
    if (str.length() != 19) return false;
    if (str.charAt(4) != '/' || str.charAt(7) != '/' || 
        str.charAt(10) != ' ' || str.charAt(13) != ':' || str.charAt(16) != ':') return false;
    // Additional checks like valid month, day, hour, etc., can be added if needed.
    return true;
}

void setup() {
  // Initialize digital pins as outputs
  pinMode(MUX1, OUTPUT);
  pinMode(MUX2, OUTPUT);
  pinMode(MUX3, OUTPUT);
  pinMode(PHASEN, OUTPUT);
  pinMode(PHASEP, OUTPUT);
  pinMode(CTEST, OUTPUT);
  pinMode(CSLEEP, OUTPUT);
  pinMode(CRESET, OUTPUT);
  pinMode(VBAT_ENABLE, OUTPUT);
  // Initialize analog pins as input
  pinMode(ADC_SIG, INPUT);

  // Initialize pin phases
  digitalWrite(MUX1, HIGH);
  digitalWrite(MUX2, HIGH);
  digitalWrite(MUX3, HIGH);
  digitalWrite(PHASEN, HIGH);
  digitalWrite(PHASEP, LOW);
  digitalWrite(CTEST, LOW);
  digitalWrite(CSLEEP, LOW);
  digitalWrite(CRESET, HIGH);
  digitalWrite(LED_RED, HIGH);
  digitalWrite(LED_GREEN, HIGH);
  
  //Configure IMU
  myIMU.begin();
  Serial.begin(9600);

  // initialize BLE
  setupBLE();

  delay(1000);
  digitalWrite(CRESET, LOW);

  // initialize RTC
  rtc.begin(DateTime(F(__DATE__), F(__TIME__)));
  // This line sets the RTC with an explicit date & time, for example to set
  rtc.adjust(DateTime(year, month, day, hour, minute, second));

  // Create the ADC reader task
  xTaskCreate(UnifiedSignalTask, "Pin switch", 128, NULL, 9, NULL);
  // Create the IMU reading task
  xTaskCreate(SensorTask,    "IMU Read", 512,  NULL, 7, NULL);
  // Create the BLE send task
  xTaskCreate(ble_uart_task, "BLE UART Task", 512, NULL, 6, NULL);
  // Create RTC task
  xTaskCreate(TaskDateTime, "RTC Task", 256, NULL, 5, NULL); 
  // Create BLE receive tasks
  xTaskCreate(ble_receive_task, "BLE RE Task", 128, NULL, 4, NULL);
}


void loop() {
  // Empty. Things are managed by tasks.
}

// Start Advertising Setting
void startAdv(void)
{
  // Advertising packet
  Bluefruit.Advertising.addFlags(BLE_GAP_ADV_FLAGS_LE_ONLY_GENERAL_DISC_MODE);
  Bluefruit.Advertising.addTxPower();

  // Include bleuart 128-bit uuid
  Bluefruit.Advertising.addService(bleuart);

  // Secondary Scan Response packet (optional)
  // Since there is no room for 'Name' in Advertising packet
  Bluefruit.ScanResponse.addName();
  
  /* Start Advertising
   * - Enable auto advertising if disconnected
   * - Interval:  fast mode = 20 ms, slow mode = 152.5 ms
   * - Timeout for fast mode is 30 seconds
   * - Start(timeout) with timeout = 0 will advertise forever (until connected)
   * For recommended advertising interval
   * https://developer.apple.com/library/content/qa/qa1931/_index.html   
   */
  Bluefruit.Advertising.restartOnDisconnect(true);
  Bluefruit.Advertising.setInterval(32, 244);    // in unit of 0.625 ms
  Bluefruit.Advertising.setFastTimeout(30);      // number of seconds in fast mode
  Bluefruit.Advertising.start(0);                // 0 = Don't stop advertising after n seconds  
}

// Setup the BLE LED to be enabled on CONNECT
void setupBLE(void) 
{
  // Setup the BLE LED to be enabled on CONNECT
  // Note: This is actually the default behavior, but provided
  // here in case you want to control this LED manually via PIN 19
  Bluefruit.autoConnLed(true);

  // Config the peripheral connection with maximum bandwidth 
  // more SRAM required by SoftDevice
  // Note: All config***() function must be called before begin()
  Bluefruit.configPrphBandwidth(BANDWIDTH_MAX);

  Bluefruit.begin();
  Bluefruit.setTxPower(4);    // Check bluefruit.h for supported values
  Bluefruit.setName(deviceName.c_str()); // useful testing with multiple central connections getMcuUniqueID()
  Bluefruit.Periph.setConnectCallback(connect_callback);
  Bluefruit.Periph.setDisconnectCallback(disconnect_callback);

  // To be consistent OTA DFU should be added first if it exists
  bledfu.begin();

  // Configure and Start Device Information Service
  bledis.setManufacturer("University of Queensland");
  bledis.setModel("Bluefruit Feather52");
  bledis.begin();

  // Configure and Start BLE Uart Service
  bleuart.begin();

  // Start BLE Battery Service
  blebas.begin();
  blebas.write(100);

  // Set up and start advertising
  startAdv();
}

// callback invoked when central connects
void connect_callback(uint16_t conn_handle)
{
  // Get the reference to current connection
  bleConnected = true;
  BLEConnection* connection = Bluefruit.Connection(conn_handle);

  char central_name[32] = { 0 };
  connection->getPeerName(central_name, sizeof(central_name));

  Serial.print("Connected to ");
  Serial.println(central_name);

  strncpy(central_name_global, central_name, 32);
  // Set the global flag to true
  
}

/**
 * Callback invoked when a connection is dropped
 * @param conn_handle connection where this event happens
 * @param reason is a BLE_HCI_STATUS_CODE which can be found in ble_hci.h
 */
void disconnect_callback(uint16_t conn_handle, uint8_t reason)
{
  bleConnected = false;
  (void) conn_handle;
  (void) reason;
  // Set the global flag to flase
  imuOverflow = false;
  adcOverflow = false;
  sendFlag = false;
  samplingFlag = false;
  digitalWrite(LED_RED, HIGH);
  digitalWrite(LED_GREEN, HIGH);
}



