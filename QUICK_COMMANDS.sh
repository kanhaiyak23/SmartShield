#!/bin/bash
# Quick Commands for Live Packet Capture
# Copy and paste these commands to start packet capture

# ============================================
# STEP 1: Find Your WiFi Interface
# ============================================
echo "Finding WiFi interface..."
ifconfig | grep -B 1 "inet " | grep "^en" | awk '{print $1}' | sed 's/://'

# Or manually check:
# ifconfig -a
# Look for interface with your IP address (usually en0 or en1)

# ============================================
# STEP 2: Navigate to Project Directory
# ============================================
cd /Users/kk/cn_project/python
source venv/bin/activate

# ============================================
# STEP 3: Verify Models are Trained
# ============================================
echo "Checking for trained models..."
ls -la models/random_forest.pkl models/isolation_forest.pkl models/preprocessor.pkl

# If models don't exist, train them:
# python train_models.py --train-all

# ============================================
# STEP 4: Start Backend Server (Terminal 1)
# ============================================
# Run this in a separate terminal:
# cd /Users/kk/cn_project/backend
# npm start

# ============================================
# STEP 5: Start Packet Capture (Terminal 2)
# ============================================
# Replace 'en0' with your WiFi interface name from Step 1

# Option A: Capture from specific interface (recommended)
sudo python live_detection.py --interface en0 --backend-url http://localhost:3001

# Option B: Capture from all interfaces
sudo python live_detection.py --interface any --backend-url http://localhost:3001

# Option C: Capture with BPF filter (only TCP traffic)
sudo python live_detection.py --interface en0 --filter "tcp" --backend-url http://localhost:3001

# Option D: Capture excluding your own IP
MY_IP=$(ipconfig getifaddr en0)
sudo python live_detection.py --interface en0 --filter "not host $MY_IP" --backend-url http://localhost:3001

# ============================================
# STEP 6: Start Frontend Dashboard (Optional - Terminal 3)
# ============================================
# cd /Users/kk/cn_project/frontend
# npm run dev
# Then open http://localhost:3000 in browser

# ============================================
# STEP 7: Test Backend Connection
# ============================================
echo "Testing backend..."
curl http://localhost:3001/api/health

echo ""
echo "View recent alerts:"
curl http://localhost:3001/api/alerts

echo ""
echo "View statistics:"
curl http://localhost:3001/api/statistics

# ============================================
# STEP 8: Monitor Logs
# ============================================
# View detection logs in real-time:
# tail -f ../logs/detection.log

# View backend logs:
# tail -f ../logs/backend.log

# ============================================
# STOP CAPTURE
# ============================================
# Press Ctrl+C in the detection terminal to stop


