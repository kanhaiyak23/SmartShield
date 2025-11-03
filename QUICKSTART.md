# SmartShield Quick Start Guide

Get SmartShield up and running in 10 minutes!

## Prerequisites Check

```bash
python --version  # Should be 3.9+
node --version    # Should be 18+
pip --version
npm --version
```

## Fast Installation

### 1. Clone and Setup

```bash
# Clone repository
git clone https://github.com/yourusername/smartshield.git
cd smartshield

# Create directories
mkdir -p python/data python/models python/logs backend/logs frontend/.next
```

### 2. Download UNSW-NB15 Dataset

**Option A - From Kaggle (Recommended):**
```bash
# Install kaggle CLI
pip install kaggle

# Download UNSW-NB15
kaggle datasets download -d mrwellsdavid/unsw-nb15

# Extract
unzip unsw-nb15.zip -d python/data/

# Verify
ls python/data/
```

**Option B - Manual Download:**
1. Visit: https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15
2. Download dataset
3. Extract CSV files to `python/data/`

**Required files:**
- `UNSW_NB15_training-set.csv`
- `UNSW_NB15_testing-set.csv`

### 3. Python Setup

```bash
cd python

# Create virtual environment
python -m venv venv

# Activate
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 4. Train Models (One-time, ~30 minutes)

```bash
cd python
source venv/bin/activate

# Train all models
python train_models.py --train-all
```

⏱️ **Expected time**: 20-40 minutes depending on your machine

✅ **Success indicator**: Models saved in `python/models/` directory

### 5. Backend Setup

```bash
cd backend
npm install
```

### 6. Frontend Setup

```bash
cd frontend
npm install
```

## Running SmartShield

### Method 1: Automated Script (Recommended)

```bash
# Start all services
sudo ./start.sh

# Open http://localhost:3000 in browser
```

### Method 2: Manual Start

**Terminal 1 - ML API:**
```bash
cd python
source venv/bin/activate
python api_server.py
```

**Terminal 2 - Backend:**
```bash
cd backend
npm start
```

**Terminal 3 - Frontend:**
```bash
cd frontend
npm run dev
```

**Terminal 4 - Detection (requires root):**
```bash
cd python
source venv/bin/activate
sudo python live_detection.py --interface any
```

## Test Your Installation

### 1. Check Health

```bash
# ML API
curl http://localhost:5000/health

# Backend
curl http://localhost:3001/api/health
```

### 2. Simulate Attack

```bash
cd python
source venv/bin/activate

# Test port scan detection
python attack_simulator.py --type port-scan --target 127.0.0.1 --count 100
```

### 3. Check Dashboard

Open http://localhost:3000 and verify:
- ✅ Statistics cards showing packet counts
- ✅ Protocol distribution chart
- ✅ Alerts appearing in real-time
- ✅ Top IPs updating

## Verification Checklist

- [ ] All services running without errors
- [ ] Dashboard loads at http://localhost:3000
- [ ] Statistics showing packet counts
- [ ] Attack simulation triggers alerts
- [ ] No console errors in browser
- [ ] Logs showing packet capture

## Common Issues

### "Models not found"
```bash
cd python
python train_models.py --train-all
```

### "Permission denied" for packet capture
```bash
sudo python live_detection.py
```

### "Port already in use"
```bash
# Kill existing processes
./stop.sh
# Then restart
./start.sh
```

### "Module not found"
```bash
# Check virtual environment is activated
which python  # Should show venv path

# Reinstall dependencies
pip install -r requirements.txt
```

## Next Steps

1. ✅ Read full documentation in SETUP.md
2. ✅ Review DATASET_SETUP.md for dataset details
3. ✅ Explore attack simulation options
4. ✅ Customize detection thresholds
5. ✅ Deploy to your campus network

## Stopping SmartShield

```bash
# Stop all services
./stop.sh

# Or manually
Ctrl+C in each terminal
```

## Getting Help

- 📖 Read SETUP.md for detailed instructions
- 📖 Check DATASET_SETUP.md for dataset info
- 🐛 Open issue on GitHub
- 📧 Contact team

## Congratulations! 🎉

You now have SmartShield running with:
- ✅ Real-time packet capture
- ✅ AI-powered anomaly detection
- ✅ Beautiful web dashboard
- ✅ Multiple ML models
- ✅ Attack simulation tools

**Start protecting your campus network today!** 🛡️


