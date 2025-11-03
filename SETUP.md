# SmartShield Setup Guide

Complete setup instructions for SmartShield AI-Powered Campus Intrusion Detection System.

## Prerequisites

- **Python 3.9+** with pip
- **Node.js 18+** with npm
- **Administrative/root privileges** (for packet capture)
- **UNSW-NB15 Dataset** (see DATASET_SETUP.md)

## Installation Steps

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/smartshield.git
cd smartshield
```

### 2. Python Environment Setup

```bash
cd python

# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Dataset Preparation

```bash
# Download UNSW-NB15 dataset
# Place CSV files in python/data/ directory:
# - UNSW_NB15_training-set.csv
# - UNSW_NB15_testing-set.csv

# Verify dataset
ls python/data/
```

### 4. Train ML Models

```bash
# Navigate to python directory
cd python

# Train all models (this may take 30-60 minutes)
python train_models.py --train-all

# Or train individual models
python train_models.py --train-rf     # Random Forest only
python train_models.py --train-if     # Isolation Forest only
python train_models.py --train-ae     # Autoencoder only
```

**Expected Output:**
```
==============================
SmartShield Model Training
==============================
Loading UNSW_NB15_training-set.csv...
Loaded dataset with shape: (175341, 49)
Training Random Forest Classifier...
Random Forest Accuracy: 0.9542
...
Training completed successfully!
```

### 5. Backend Setup

```bash
cd backend

# Install dependencies
npm install

# Create .env file
cp env.example .env

# Edit .env if needed (defaults should work)
```

### 6. Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# No additional configuration needed
```

## Running the System

### Option 1: Development Mode (Recommended for Testing)

**Terminal 1 - Flask ML API:**
```bash
cd python
source venv/bin/activate
python api_server.py
# Running on http://localhost:5000
```

**Terminal 2 - Express Backend:**
```bash
cd backend
npm start
# Running on http://localhost:3001
```

**Terminal 3 - Live Detection:**
```bash
cd python
source venv/bin/activate

# For Linux/Mac (requires root)
sudo python live_detection.py --interface eth0

# For Windows (run as administrator)
python live_detection.py --interface Ethernet

# For loopback/testing
python live_detection.py
```

**Terminal 4 - Frontend Dashboard:**
```bash
cd frontend
npm run dev
# Open http://localhost:3000 in browser
```

### Option 2: Production Mode

Create a startup script to run all services:

```bash
# Create start.sh
chmod +x start.sh
./start.sh
```

## Verification

### 1. Check Models

```bash
cd python
python -c "import joblib; print('RF:', joblib.load('models/random_forest.pkl')); print('IF:', joblib.load('models/isolation_forest.pkl'))"
```

### 2. Test ML API

```bash
curl http://localhost:5000/health

# Expected: {"status":"healthy","models_loaded":true}
```

### 3. Test Backend

```bash
curl http://localhost:3001/api/health

# Expected: {"status":"ok","timestamp":"..."}
```

### 4. Simulate Attack

```bash
cd python
source venv/bin/activate

# Port scan attack
python attack_simulator.py --type port-scan --target 127.0.0.1 --count 50

# DDoS attack
python attack_simulator.py --type ddos --target 127.0.0.1 --duration 30

# Check dashboard for alerts!
```

## Dashboard Features

Open http://localhost:3000 to view:

1. **Statistics Cards**: Total packets, alerts, traffic volume
2. **Protocol Distribution**: Pie chart of network protocols
3. **Top IPs**: Most active/attacking IP addresses
4. **Alerts Table**: Real-time security alerts with filtering
5. **Live Updates**: WebSocket-based real-time data

## Troubleshooting

### Issue: "Permission denied" for packet capture

**Solution:** Run with sudo/administrator privileges:
```bash
sudo python live_detection.py --interface eth0
```

### Issue: "Models not found"

**Solution:** Train models first:
```bash
cd python
python train_models.py --train-all
```

### Issue: "Dataset not found"

**Solution:** Download and place UNSW-NB15 CSV files in `python/data/`

### Issue: "Port already in use"

**Solution:** Kill existing processes:
```bash
# Linux/Mac
lsof -ti:3000,3001,5000 | xargs kill -9

# Windows
netstat -ano | findstr :3000
taskkill /F /PID <PID>
```

### Issue: "Out of memory" during training

**Solution:** Reduce dataset size or use cloud instance with more RAM

### Issue: WebSocket connection fails

**Solution:** Check firewall settings and ensure backend is running

## Network Interfaces

Find available interfaces:

**Linux/Mac:**
```bash
ip link show
ifconfig -a
```

**Windows:**
```bash
ipconfig /all
```

Common interface names:
- `eth0` - Ethernet (Linux)
- `wlan0` - Wireless (Linux)
- `en0` - Ethernet (macOS)
- `en1` - WiFi (macOS)
- `Ethernet` - Wired (Windows)
- `Wi-Fi` - Wireless (Windows)
- `any` or `None` - Capture from all interfaces

## Testing Workflow

1. **Start all services** (ML API, Backend, Frontend)
2. **Start detection** with `live_detection.py`
3. **Simulate attack** with `attack_simulator.py`
4. **Observe dashboard** for real-time alerts
5. **Verify detection** accuracy

## Performance Tuning

### Increase Packet Buffer
Edit `python/src/packet_capture.py`:
```python
self.packet_buffer_size = 5000  # Default: 1000
```

### Adjust Detection Threshold
Edit `python/live_detection.py`:
```python
engine = DetectionEngine(anomaly_threshold=0.6)  # Default: 0.5
```

### Model Parameters
Edit `python/train_models.py` for different model configurations

## Security Notes

⚠️ **IMPORTANT**: 
- Only run on networks you own or have permission to monitor
- Don't capture sensitive data on production networks
- Use filtered capture to exclude certain protocols/IPs
- Implement rate limiting for production deployments

## Next Steps

- [ ] Deploy to campus Wi-Fi network
- [ ] Integrate with SDN controller (POX/Ryu)
- [ ] Add automatic mitigation capabilities
- [ ] Implement logging and alerting
- [ ] Create production Docker containers

## Support

For issues or questions:
1. Check this guide and troubleshooting section
2. Review logs in `python/logs/`
3. Open an issue on GitHub
4. Contact team members

## License

See LICENSE file for details.

