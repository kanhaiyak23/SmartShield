# Comprehensive WiFi Network Packet Capture Logging

This document describes all the logging features added for WiFi network packet capture.

## 📋 Overview

The system now includes comprehensive logging for all WiFi network packet capture activities, including:
- Detailed packet information
- Real-time statistics
- Performance metrics
- Error handling
- Protocol distribution
- Capture session information

---

## 🎯 Logging Features

### 1. **Capture Session Start Logging**

When packet capture starts, you'll see:
```
============================================================
Starting WiFi Network Packet Capture
============================================================
Interface: en0
Available interfaces: en0, en1, lo0, awdl0
BPF Filter: None (capturing all)
Packet Buffer Size: 1000
Packet Logging: Enabled (every 100 packets)
Start Time: 2024-11-02 21:45:00
============================================================
```

### 2. **Individual Packet Logging**

**INFO Level (Every N packets - default: 100):**
```
2024-11-02 21:45:05 - SmartShield - INFO - Packet #100: TCP | 192.168.1.100:52345 -> 8.8.8.8:53 | Size: 64 bytes
2024-11-02 21:45:06 - SmartShield - INFO - Packet #200: UDP | 192.168.1.50:49152 -> 192.168.1.1:1900 | Size: 128 bytes
```

**DEBUG Level (Every packet - for detailed analysis):**
To see every packet, set log level to DEBUG:
```python
logger = setup_logging(log_level=logging.DEBUG)
```

### 3. **Periodic Statistics (Every 10 seconds)**

```
============================================================
Capture Statistics (last 10.0s)
  Total Packets: 1,234
  Total Bytes: 456,789 (0.44 MB)
  Packet Rate: 123.40 packets/sec
  Data Rate: 0.04 MB/s (45678 bytes/sec)
  Protocol Distribution:
    - TCP:  856 (69.4%)
    - UDP:  312 (25.3%)
    - ICMP: 42 (3.4%)
    - ARP:  24 (1.9%)
  Average Rate: 123.40 packets/sec, 0.04 MB/s
  Capture Duration: 10.0s (0.2 minutes)
============================================================
```

### 4. **Capture Session Stop Logging**

```
============================================================
Stopping Packet Capture
============================================================
Stop Time: 2024-11-02 22:15:30
Total Duration: 1830.5s (30.5 minutes)
Total Packets Captured: 226,543
Total Bytes Captured: 83,456,789 (79.62 MB)
Average Rate: 123.65 packets/sec, 0.04 MB/s
Final Statistics:
  TCP Packets:  157,234
  UDP Packets:  57,421
  ICMP Packets: 8,234
  ARP Packets:  3,654
============================================================
Packet capture stopped successfully
```

### 5. **Error Logging**

**Permission Errors:**
```
2024-11-02 21:45:00 - SmartShield - ERROR - Permission denied: [Errno 1] Operation not permitted
2024-11-02 21:45:00 - SmartShield - ERROR - Packet capture requires root/administrator privileges!
2024-11-02 21:45:00 - SmartShield - ERROR - Please run with: sudo python live_detection.py --interface <interface>
```

**Interface Errors:**
```
2024-11-02 21:45:00 - SmartShield - ERROR - Interface error: [Errno 19] No such device
2024-11-02 21:45:00 - SmartShield - ERROR - Interface 'eth0' may not exist or be accessible
2024-11-02 21:45:00 - SmartShield - ERROR - Available interfaces: en0, en1, lo0
```

**Packet Processing Errors:**
```
2024-11-02 21:45:05 - SmartShield - ERROR - Error processing packet #1234: Invalid packet format
```

---

## 📊 Logged Information

### Packet Details
- **Protocol**: TCP, UDP, ICMP, ARP, or OTHER
- **Source IP**: Packet source IP address
- **Destination IP**: Packet destination IP address
- **Source Port**: Source port (if applicable)
- **Destination Port**: Destination port (if applicable)
- **Packet Size**: Size in bytes

### Statistics
- **Total Packets**: Cumulative count
- **Total Bytes**: Cumulative data captured
- **Packet Rate**: Packets per second (interval and average)
- **Data Rate**: Bytes/MB per second (interval and average)
- **Protocol Distribution**: Count and percentage per protocol
- **Capture Duration**: Total time capturing

### Performance Metrics
- **Packets per Second**: Real-time capture rate
- **Bytes per Second**: Data throughput
- **MB per Second**: Data throughput in MB
- **Average Rates**: Overall average since capture started

---

## ⚙️ Configuration

### Log Levels

**INFO (Default):**
- Shows session start/stop
- Periodic statistics (every 10 seconds)
- Sample packets (every 100 packets)
- Errors and warnings

**DEBUG:**
- All INFO level logs
- Every single packet captured
- Detailed error stack traces

To enable DEBUG logging:
```python
import logging
from src.utils import setup_logging

logger = setup_logging(log_level=logging.DEBUG)
```

### Logging Intervals

**Packet Logging Interval:**
Default: Log every 100 packets
- Change in `packet_capture.py` initialization:
  ```python
  PacketCapture(log_interval=50)  # Log every 50 packets
  ```

**Statistics Logging Interval:**
Default: Every 10 seconds
- Change in `packet_capture.py` `_log_statistics()` method:
  ```python
  time.sleep(10)  # Change to desired interval
  ```

---

## 📁 Log File Location

Logs are saved to:
```
python/logs/smartshield_YYYYMMDD.log
```

Example:
```
python/logs/smartshield_20241102.log
```

Log files are:
- Rotated daily (new file each day)
- Include detailed format with file:line numbers
- Contain all log levels (based on configuration)

---

## 🔍 Log File Format

**File Handler (Detailed):**
```
2024-11-02 21:45:05 - SmartShield - INFO - [packet_capture.py:166] - Packet #100: TCP | 192.168.1.100:52345 -> 8.8.8.8:53 | Size: 64 bytes
```

**Console Handler (Simple):**
```
2024-11-02 21:45:05 - SmartShield - INFO - Packet #100: TCP | 192.168.1.100:52345 -> 8.8.8.8:53 | Size: 64 bytes
```

---

## 📈 Example Log Output

### Start of Capture Session
```
2024-11-02 21:45:00 - SmartShield - INFO - Logging initialized - Level: INFO, File: python/logs/smartshield_20241102.log
============================================================
Starting WiFi Network Packet Capture
============================================================
Interface: en0
Available interfaces: en0, en1, lo0, awdl0
BPF Filter: None (capturing all)
Packet Buffer Size: 1000
Packet Logging: Enabled (every 100 packets)
Start Time: 2024-11-02 21:45:00
============================================================
2024-11-02 21:45:00 - SmartShield - INFO - Packet capture thread started. Beginning to sniff packets...
2024-11-02 21:45:00 - SmartShield - INFO - Capture initialized. Waiting for packets...
```

### During Capture
```
2024-11-02 21:45:05 - SmartShield - INFO - Packet #100: TCP | 192.168.1.100:52345 -> 8.8.8.8:53 | Size: 64 bytes
2024-11-02 21:45:10 - SmartShield - INFO - Packet #200: UDP | 192.168.1.50:49152 -> 192.168.1.1:1900 | Size: 128 bytes
...
============================================================
Capture Statistics (last 10.0s)
  Total Packets: 1,234
  Total Bytes: 456,789 (0.44 MB)
  Packet Rate: 123.40 packets/sec
  Data Rate: 0.04 MB/s (45678 bytes/sec)
  Protocol Distribution:
    - TCP:  856 (69.4%)
    - UDP:  312 (25.3%)
    - ICMP: 42 (3.4%)
    - ARP:  24 (1.9%)
  Average Rate: 123.40 packets/sec, 0.04 MB/s
  Capture Duration: 10.0s (0.2 minutes)
============================================================
```

### End of Capture
```
2024-11-02 22:15:30 - SmartShield - INFO - ============================================================
2024-11-02 22:15:30 - SmartShield - INFO - Stopping Packet Capture
2024-11-02 22:15:30 - SmartShield - INFO - ============================================================
2024-11-02 22:15:30 - SmartShield - INFO - Stop Time: 2024-11-02 22:15:30
2024-11-02 22:15:30 - SmartShield - INFO - Total Duration: 1830.5s (30.5 minutes)
2024-11-02 22:15:30 - SmartShield - INFO - Total Packets Captured: 226,543
2024-11-02 22:15:30 - SmartShield - INFO - Total Bytes Captured: 83,456,789 (79.62 MB)
2024-11-02 22:15:30 - SmartShield - INFO - Average Rate: 123.65 packets/sec, 0.04 MB/s
2024-11-02 22:15:30 - SmartShield - INFO - Final Statistics:
2024-11-02 22:15:30 - SmartShield - INFO -   TCP Packets:  157,234
2024-11-02 22:15:30 - SmartShield - INFO -   UDP Packets:  57,421
2024-11-02 22:15:30 - SmartShield - INFO -   ICMP Packets: 8,234
2024-11-02 22:15:30 - SmartShield - INFO -   ARP Packets:  3,654
2024-11-02 22:15:30 - SmartShield - INFO - ============================================================
2024-11-02 22:15:30 - SmartShield - INFO - Packet capture stopped successfully
```

---

## 🎛️ Customization

### Disable Packet Logging
```python
packet_capture = PacketCapture(log_packets=False)
```

### Change Log Interval
```python
packet_capture = PacketCapture(log_interval=50)  # Log every 50 packets
```

### Enable DEBUG Level
```python
from src.utils import setup_logging
import logging

logger = setup_logging(log_level=logging.DEBUG)
```

---

## ✅ Benefits

1. **Complete Audit Trail**: Every packet capture session is fully logged
2. **Performance Monitoring**: Real-time and average rates
3. **Troubleshooting**: Detailed error messages with context
4. **Analysis**: Protocol distribution and traffic patterns
5. **Compliance**: Full logging for security audits
6. **Debugging**: DEBUG level shows every single packet

---

## 📝 Notes

- Logs are automatically rotated daily
- File logs include file:line numbers for debugging
- Console logs use simpler format for readability
- All timestamps are in ISO format
- Statistics are logged every 10 seconds during capture
- Packet samples logged every 100 packets (configurable)

---

**All WiFi network packet capture activities are now fully logged!** 📊✨


