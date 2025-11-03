# Live Packet Capture Guide - Campus WiFi

Complete guide to capture live packets from your connected campus WiFi and send them to the backend.

## 🔍 Step 1: Find Your WiFi Interface Name

**On macOS (your system):**
```bash
# Method 1: List all network interfaces
ifconfig -a

# Method 2: Show only active interfaces
ifconfig | grep -E "^[a-z]" | awk '{print $1}' | sed 's/://'

# Method 3: List network services (macOS specific)
networksetup -listallnetworkservices

# Method 4: Find WiFi interface specifically
ifconfig | grep -B 5 "inet " | grep -E "^[a-z]" | awk '{print $1}' | sed 's/://'
```

**Common macOS WiFi interface names:**
- `en0` - Usually Ethernet/WiFi (primary)
- `en1` - Secondary network interface (often WiFi)
- `awdl0` - Apple Wireless Direct Link
- `llw0` - Low Latency WiFi

**Quick check:**
```bash
# Show IP address and interface name
ifconfig | grep -A 2 "inet " | grep -B 1 "inet "

# Or use this one-liner to find WiFi interface
ipconfig getifaddr en0 && echo "WiFi likely on: en0" || ipconfig getifaddr en1 && echo "WiFi likely on: en1"
```

**Example output:**
```
en0: flags=8863<UP,BROADCAST,SMART,RUNNING,SIMPLEX,MULTICAST> mtu 1500
	inet 192.168.1.100 netmask 0xffffff00 broadcast 192.168.1.255
```
Your WiFi interface is likely **`en0`** or **`en1`**

---

## 🚀 Step 2: Complete Setup Flow

### **Prerequisites Check**
```bash
# Verify Python environment is activated
cd /Users/kk/cn_project/python
source venv/bin/activate

# Verify models are trained (required before capture)
ls -la models/
# Should show: random_forest.pkl, isolation_forest.pkl, autoencoder.h5, preprocessor.pkl

# If models don't exist, train them first:
python train_models.py --train-all
```

---

## 📦 Step 3: Start All Services

### **Option A: Start Services Manually (Recommended for Testing)**

**Terminal 1 - Backend Server (Express + WebSocket):**
```bash
cd /Users/kk/cn_project/backend
npm start
# Backend runs on: http://localhost:3001
# Keep this terminal open!
```

**Terminal 2 - Live Packet Detection (THIS IS WHERE CAPTURE HAPPENS):**
```bash
cd /Users/kk/cn_project/python
source venv/bin/activate

# Find your WiFi interface first (usually en0 or en1)
INTERFACE=$(ifconfig | grep -B 5 "inet " | grep -E "^[a-z]" | head -1 | awk '{print $1}' | sed 's/://')
echo "Using interface: $INTERFACE"

# Start packet capture (requires sudo for WiFi capture on macOS)
sudo python live_detection.py --interface $INTERFACE --backend-url http://localhost:3001

# OR specify interface directly:
sudo python live_detection.py --interface en0 --backend-url http://localhost:3001

# OR capture from all interfaces:
sudo python live_detection.py --interface any --backend-url http://localhost:3001
```

**Terminal 3 - Frontend Dashboard (Optional - to view results):**
```bash
cd /Users/kk/cn_project/frontend
npm run dev
# Frontend runs on: http://localhost:3000
# Open in browser to see live alerts and statistics
```

---

### **Option B: Use Automated Start Script**
```bash
cd /Users/kk/cn_project
chmod +x start.sh
sudo ./start.sh
```

---

## 🔄 Complete Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    LIVE PACKET CAPTURE FLOW                 │
└─────────────────────────────────────────────────────────────┘

1. WiFi Interface (en0/en1)
   │
   │  [Packets flowing through network]
   │
   ▼
2. Scapy Packet Capture (packet_capture.py)
   │  - Captures packets in real-time using sniff()
   │  - Extracts features: IP, ports, protocols, packet size
   │  - Stores in buffer (last 1000 packets)
   │
   ▼
3. Feature Extraction (packet_capture.py)
   │  - src_ip, dst_ip, src_port, dst_port
   │  - protocol (TCP/UDP/ICMP/ARP)
   │  - packet_length, flags, TTL
   │  - timestamp
   │
   ▼
4. Preprocessing (preprocessing.py)
   │  - Converts packet features to ML format
   │  - Normalizes and encodes data
   │  - Matches UNSW-NB15 dataset format
   │
   ▼
5. ML Model Prediction (models.py)
   │  - Random Forest: Attack type classification
   │  - Isolation Forest: Anomaly detection
   │  - Autoencoder: Deep learning anomaly score
   │  - Combines predictions for final score
   │
   ▼
6. Alert Generation (live_detection.py)
   │  - If anomaly detected (score > 0.5):
   │    • Creates alert object
   │    • Calculates severity (Critical/High/Medium/Low)
   │    • Extracts attack type
   │
   ▼
7. HTTP POST to Backend (live_detection.py)
   │  POST http://localhost:3001/api/alerts
   │  {
   │    "timestamp": "2024-01-15T10:30:00",
   │    "src_ip": "192.168.1.100",
   │    "dst_ip": "192.168.1.1",
   │    "protocol": "tcp",
   │    "attack_type": "Fuzzers",
   │    "severity": "Medium",
   │    "confidence": 0.75,
   │    "is_anomaly": true
   │  }
   │
   ▼
8. Backend API Server (server.js)
   │  - Receives alert via POST /api/alerts
   │  - Stores in memory (alerts array)
   │  - Emits WebSocket event: 'new_alert'
   │
   ▼
9. Statistics Update (every 10 seconds)
   │  POST http://localhost:3001/api/statistics
   │  {
   │    "total_packets": 15234,
   │    "tcp_packets": 8934,
   │    "udp_packets": 5421,
   │    "alerts": { "Total": 23, "High": 5, "Medium": 12 }
   │  }
   │
   ▼
10. Frontend Dashboard (Optional)
    │  - Connects via WebSocket
    │  - Receives real-time alerts
    │  - Displays charts and alert table
    │
    └──────────────────────────────────────────────────────────┘
```

---

## 📝 Detailed Commands Reference

### **1. Find WiFi Interface (macOS)**
```bash
# Best method - shows active interface with IP
ifconfig | grep -B 1 "inet " | grep "^en" | awk '{print $1}' | sed 's/://'

# Alternative - show all interfaces
networksetup -listallhardwareports

# Check which interface has active connection
route get default | grep interface
```

### **2. Start Packet Capture with Specific Options**

**Basic capture:**
```bash
cd /Users/kk/cn_project/python
source venv/bin/activate
sudo python live_detection.py --interface en0
```

**Capture with BPF filter (e.g., only TCP traffic):**
```bash
sudo python live_detection.py --interface en0 --filter "tcp"
```

**Capture specific ports only:**
```bash
sudo python live_detection.py --interface en0 --filter "tcp port 80 or tcp port 443"
```

**Capture excluding your own IP:**
```bash
MY_IP=$(ipconfig getifaddr en0)
sudo python live_detection.py --interface en0 --filter "not host $MY_IP"
```

**Capture with custom backend URL:**
```bash
sudo python live_detection.py --interface en0 --backend-url http://localhost:3001
```

### **3. Test Backend Connection**

**Check if backend is running:**
```bash
curl http://localhost:3001/api/health
# Expected: {"status":"ok","timestamp":"..."}
```

**Test alert endpoint:**
```bash
curl -X POST http://localhost:3001/api/alerts \
  -H "Content-Type: application/json" \
  -d '{
    "src_ip": "192.168.1.100",
    "dst_ip": "192.168.1.1",
    "protocol": "tcp",
    "attack_type": "Test",
    "severity": "Low"
  }'
```

**View all alerts:**
```bash
curl http://localhost:3001/api/alerts | jq
```

**View statistics:**
```bash
curl http://localhost:3001/api/statistics | jq
```

---

## 🔐 Permission Requirements

### **macOS Packet Capture Permissions**

**Important:** macOS requires special permissions for packet capture:

1. **Grant Terminal/VS Code Full Disk Access:**
   - System Settings → Privacy & Security → Full Disk Access
   - Add Terminal.app or VS Code
   
2. **Run with sudo:**
   ```bash
   # Packet capture requires root privileges
   sudo python live_detection.py --interface en0
   ```

3. **Alternative - Monitor Mode (Advanced):**
   ```bash
   # macOS doesn't support monitor mode like Linux
   # But promiscuous mode works with sudo
   sudo python live_detection.py --interface en0
   ```

---

## 📊 Monitoring & Verification

### **Check if packets are being captured:**
```bash
# The live_detection.py will show logs like:
# INFO: Packets: 1234, Alerts: 5
```

### **View logs:**
```bash
# Backend logs
tail -f /Users/kk/cn_project/logs/backend.log

# Detection logs
tail -f /Users/kk/cn_project/logs/detection.log

# Or if running manually, logs appear in terminal
```

### **Verify packets reaching backend:**
```bash
# Check backend terminal for POST requests
# You should see: POST /api/alerts 200
# And: POST /api/statistics 200
```

### **Monitor network traffic:**
```bash
# Quick packet count check
sudo tcpdump -i en0 -c 10

# Or use built-in network utility
netstat -i
```

---

## 💾 Saving Packets (Optional)

If you want to **save packets to disk** instead of (or in addition to) sending to backend:

### **Option 1: Modify packet_capture.py**
```python
# In packet_capture.py, add file writing:
def _packet_handler(self, packet):
    # ... existing code ...
    
    # Save to file
    if self.save_file:
        with open(self.save_file, 'a') as f:
            f.write(f"{datetime.now()},{features}\n")
```

### **Option 2: Use tcpdump to capture simultaneously:**
```bash
# Capture to file while running live detection
sudo tcpdump -i en0 -w /tmp/campus_wifi_capture.pcap

# Read the file later
tcpdump -r /tmp/campus_wifi_capture.pcap
```

### **Option 3: Save alerts to file (backend does this via API)**
The backend receives all alerts via API. You can query them:
```bash
# Get all alerts as JSON
curl http://localhost:3001/api/alerts > alerts.json

# Or add database to backend to persist alerts
```

---

## 🎯 Quick Start Checklist

```bash
# ✅ 1. Find WiFi interface
ifconfig | grep -B 1 "inet " | grep "^en"

# ✅ 2. Activate Python environment
cd /Users/kk/cn_project/python
source venv/bin/activate

# ✅ 3. Verify models exist
ls models/*.pkl models/*.h5

# ✅ 4. Start backend (Terminal 1)
cd ../backend && npm start

# ✅ 5. Start packet capture (Terminal 2 - requires sudo)
cd ../python && sudo python live_detection.py --interface en0

# ✅ 6. Open dashboard (optional)
# http://localhost:3000
```

---

## 🐛 Troubleshooting

### **Issue: "No such device" or "Interface not found"**
```bash
# List all interfaces
ifconfig -a

# Use 'any' to capture from all interfaces
sudo python live_detection.py --interface any
```

### **Issue: "Permission denied"**
```bash
# Must use sudo for packet capture
sudo python live_detection.py --interface en0
```

### **Issue: "Models not found"**
```bash
# Train models first
cd python
python train_models.py --train-all
```

### **Issue: "Backend connection failed"**
```bash
# Check if backend is running
curl http://localhost:3001/api/health

# Start backend if not running
cd backend && npm start
```

### **Issue: "No packets captured"**
```bash
# Verify interface is active
ifconfig en0

# Test with tcpdump
sudo tcpdump -i en0 -c 5

# Try capturing from 'any' interface
sudo python live_detection.py --interface any
```

### **Issue: "Too many packets / Performance issues"**
```bash
# Use BPF filter to reduce traffic
sudo python live_detection.py --interface en0 --filter "tcp and not port 443"
```

---

## 📈 Performance Tips

1. **Use BPF filters** to reduce packet volume
2. **Process packets in batches** (already done - processes 20 packets every 2 seconds)
3. **Increase packet buffer** if needed (edit `packet_buffer_size` in `packet_capture.py`)
4. **Monitor CPU/Memory** usage with `top` or `htop`

---

## 🔒 Security Notes

⚠️ **IMPORTANT:**
- Only capture on networks you have permission to monitor
- Campus WiFi: Check with IT department before capturing
- Don't capture sensitive/personal data
- Use filters to exclude specific IPs/protocols
- Consider rate limiting for production

---

## 📚 Additional Resources

- Scapy documentation: https://scapy.readthedocs.io/
- BPF filter syntax: https://bpf.readthedocs.io/
- macOS network interfaces: `man ifconfig`

---

## ✅ Success Indicators

When everything is working, you should see:

1. **In detection terminal:**
   ```
   INFO: Starting packet capture on interface: en0
   INFO: Packets: 1234, Alerts: 5
   ```

2. **In backend terminal:**
   ```
   POST /api/alerts 200
   POST /api/statistics 200
   ```

3. **In dashboard (if running):**
   - Live packet count increasing
   - Alerts appearing in table
   - Charts updating in real-time

---

**You're all set! Start capturing packets from your campus WiFi! 🚀**


