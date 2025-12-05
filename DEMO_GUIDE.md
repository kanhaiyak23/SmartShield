# SmartShield Real-Time Attack Detection Demo Guide

## 🎯 Purpose

This guide shows you how to demonstrate SmartShield's real-time attack detection capabilities using the attack simulator.

## 📋 Prerequisites

1. **SmartShield Backend Running**
   ```bash
   sudo python3 server.py
   ```
   - Should show: `✅ Loaded pretrained Random Forest model`
   - Backend running on `http://127.0.0.1:5000`

2. **Dashboard Open**
   - Hosted link: https://smart-shield-gamma.vercel.app/
   - Or local: `npm run dev` → http://localhost:3000
   - Should show: `LOCAL_SERVER_CONNECTED` (green status)

3. **Root/Admin Access**
   - Attack simulator requires `sudo` for packet generation

## 🚀 Quick Demo (5 minutes)

### Step 1: Start Backend
```bash
# Terminal 1
cd /path/to/smartshield
source venv/bin/activate
sudo python3 server.py
```

### Step 2: Open Dashboard
- Open browser to your hosted link or local frontend
- Verify connection status shows "LOCAL_SERVER_CONNECTED"

### Step 3: Run Attack Simulator
```bash
# Terminal 2
cd /path/to/smartshield
source venv/bin/activate
sudo python3 attack_simulator.py
# Choose option 5 (Full Demo Sequence)
```

### Step 4: Watch Real-Time Detection
- **Dashboard will show:**
  - Packets appearing in real-time
  - Attacks flagged as **CRITICAL** (red) or **WARNING** (yellow)
  - Attack probabilities displayed
  - Risk levels updating live

## 🎬 Demo Sequence Breakdown

When you run the full demo sequence, you'll see:

1. **Port Scan (10s)**
   - Rapid connections to multiple ports
   - **Expected Detection:** CRITICAL
   - **Why:** High packet rate, unusual port patterns

2. **DDoS Flood (5s)**
   - Very high packet rate flooding
   - **Expected Detection:** CRITICAL
   - **Why:** Extremely high packet rate, unusual patterns

3. **SQL Injection (8s)**
   - Malicious HTTP payloads
   - **Expected Detection:** WARNING/CRITICAL
   - **Why:** Malicious payload patterns detected

4. **Suspicious Port Scan (7s)**
   - Scanning unusual/privileged ports
   - **Expected Detection:** WARNING
   - **Why:** Unusual port combinations, suspicious behavior

5. **ICMP Flood (5s)**
   - Ping flood attack
   - **Expected Detection:** WARNING/CRITICAL
   - **Why:** High ICMP packet rate

## 📊 What to Show During Demo

### 1. **Real-Time Packet Stream**
- Point out packets appearing in the table
- Show timestamps updating live
- Highlight different protocols (TCP, UDP, ICMP)

### 2. **Attack Detection**
- Show packets turning **RED** (CRITICAL) or **YELLOW** (WARNING)
- Point out attack probability scores
- Explain risk levels (SAFE/WARNING/CRITICAL)

### 3. **Packet Details**
- Click on a flagged packet
- Show the Random Forest assessment
- Display feature vector (20 features)
- Show attack probability score

### 4. **Statistics**
- Point out "ANOMALIES" counter increasing
- Show total packets captured
- Highlight detection rate

## 🎯 Key Talking Points

1. **99.62% Attack Detection Rate**
   - Model trained on UNSW-NB15 dataset
   - 200 trees, 20 enhanced features
   - Real-time inference (< 100ms latency)

2. **Real-Time Analysis**
   - Packets analyzed as they're captured
   - No delay in detection
   - Live risk assessment

3. **Multiple Attack Types**
   - Port scanning
   - DDoS flooding
   - SQL injection
   - Suspicious patterns

4. **Feature-Rich Detection**
   - 20 features analyzed per packet
   - Flow statistics
   - Protocol analysis
   - Behavioral patterns

## 🔧 Advanced Demo Options

### Continuous Random Attacks
```bash
sudo python3 attack_simulator.py --attack 6
```
- Runs random attacks continuously
- Great for extended demos
- Shows system handling various attack patterns

### Specific Attack Type
```bash
# Port scan only
sudo python3 attack_simulator.py --attack 1 --duration 15

# DDoS flood
sudo python3 attack_simulator.py --attack 2 --duration 10
```

### Custom Target
```bash
# Test on different IP (if authorized)
sudo python3 attack_simulator.py --target 192.168.1.100 --attack 5
```

## ⚠️ Safety & Ethics

- **Default target:** `127.0.0.1` (localhost only)
- **Only use on authorized networks**
- **For educational/demonstration purposes**
- **Never use on production systems without permission**

## 🐛 Troubleshooting

### Attacks Not Detected?

1. **Check backend is running:**
   ```bash
   curl http://127.0.0.1:5000/packets
   ```

2. **Verify model is loaded:**
   - Backend should show: `✅ Loaded pretrained Random Forest model`
   - Check for model files: `random_forest_model.joblib`

3. **Check packet capture:**
   - Backend needs root privileges
   - Network interface must be accessible
   - Firewall not blocking packets

### No Packets Appearing?

1. **Verify connection:**
   - Dashboard should show "LOCAL_SERVER_CONNECTED"
   - Check browser console for errors

2. **Check backend logs:**
   - Look for "Capture error" messages
   - Verify Scapy is working

3. **Test packet capture:**
   ```bash
   # In Python
   from scapy.all import sniff
   sniff(count=1, timeout=2)
   ```

## 📈 Expected Results

Based on model evaluation:
- **Attack Detection Rate:** 99.62%
- **False Positive Rate:** 14.11%
- **Most attacks flagged as:** CRITICAL (probability ≥ 0.7)
- **Detection latency:** < 100ms

## 🎓 Educational Value

This demo demonstrates:
- Real-time network monitoring
- ML-based threat detection
- Feature extraction from network traffic
- Risk assessment and classification
- Interactive visualization

Perfect for:
- Security demonstrations
- ML/AI showcases
- Network security education
- System capabilities presentation

