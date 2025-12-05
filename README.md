
# SmartShield - Real-Time Network Intrusion Detection System

A real-time network packet analysis and anomaly detection system powered by **Random Forest Machine Learning**. Captures live network traffic using Scapy and analyzes packets for potential security threats using a trained Random Forest model on the UNSW-NB15 dataset.

## 🎯 Model Performance

**Random Forest Classifier** (200 trees, 20 features)
- ✅ **Attack Detection Rate:** 99.62% (2,081/2,089 attacks detected)
- ✅ **Accuracy:** 86.46%
- ✅ **ROC-AUC:** 92.50%
- ✅ **False Positive Rate:** 14.11%
- ✅ **Model Size:** 76 MB

**Training Dataset:** UNSW-NB15 (500,000+ records)  
**Test Dataset:** UNSW-NB15 (50,000 records)

📊 **Detailed Results:** See [RANDOM_FOREST_RESULTS.md](RANDOM_FOREST_RESULTS.md) for complete evaluation metrics, confusion matrix, and feature importance analysis.

View your app in AI Studio: https://ai.studio/apps/drive/1ERLoG8hK638MU13w1DYJSTEYxXszTMaT

## Run Locally

**Prerequisites:**  
- Node.js (for frontend)
- Python 3.8+ (for backend)
- Root/Admin privileges (for Scapy packet capture)

### Frontend Setup

1. Install dependencies:
   ```bash
   npm install
   ```

2. Run the frontend:
   ```bash
   npm run dev
   ```

### Backend Setup (Scapy Packet Capture)

1. Install Python dependencies:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Train the Random Forest model** (if not already trained):
   ```bash
   source venv/bin/activate
   python3 train_enhanced_rf.py
   ```
   
   This will train on the UNSW-NB15 dataset and create:
   - `random_forest_model.joblib` (76 MB)
   - `rf_feature_scaler.joblib`
   - `service_encoder.joblib`
   - `state_encoder.joblib`

3. Start the Flask backend server:
   ```bash
   # On macOS/Linux (requires sudo for packet capture):
   sudo ./start-server.sh
   
   # Or manually:
   source venv/bin/activate
   sudo python3 server.py
   ```

   **Note:** 
   - The server runs on `http://127.0.0.1:5000` and provides the `/packets` endpoint for real-time packet capture
   - If the Random Forest model is found, it will be loaded automatically
   - If not found, the server will run in simulation mode

### Running Both

1. Start the backend server first (in one terminal):
   ```bash
   sudo ./start-server.sh
   ```

2. Start the frontend (in another terminal):
   ```bash
   npm run dev
   ```

3. Open your browser to `http://localhost:3000` (or the port shown by Vite)

The frontend will automatically connect to the backend when available. If the backend is not running, it will fall back to simulation mode.

## 📊 Features

- **Real-Time Packet Capture:** Uses Scapy to capture live network traffic
- **20 Enhanced Features:** Extracts comprehensive features including flow statistics, protocol info, and network behavior
- **Connection Tracking:** Tracks network flows to calculate duration, packet rates, and other temporal features
- **Risk Classification:** Classifies packets as SAFE, WARNING, or CRITICAL based on attack probability
- **Interactive Dashboard:** Real-time visualization of network traffic and threat detection
- **Packet Inspector:** Detailed view of individual packets with hex dumps and feature vectors

## 🧪 Training on UNSW-NB15 Dataset

To train the Random Forest model on real network data (UNSW-NB15):

1. **Download the UNSW-NB15 dataset:**
   - **Option 1 (Recommended):** Download from Kaggle:
     - Visit: https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15
     - Download the dataset (requires Kaggle account)
     - Extract CSV files to `data/` directory:
       - `UNSW_NB15_training-set.csv`
       - `UNSW_NB15_testing-set.csv` (optional, for evaluation)
   
   - **Option 2:** Download from official source:
     - Visit: https://research.unsw.edu.au/projects/unsw-nb15-dataset
     - Follow their download instructions
     - Place CSV files in `data/` directory

2. **Train the Random Forest model:**
   ```bash
   source venv/bin/activate
   python3 train_enhanced_rf.py
   ```
   
   This will:
   - Load training data from `data/` directory
   - Extract 20 enhanced features
   - Train a Random Forest with 200 trees
   - Save model files (training takes ~45 minutes)

3. **Evaluate the model** (optional):
   ```bash
   source venv/bin/activate
   python3 evaluate_rf_model.py
   ```
   
   This will test the model on the test dataset and display accuracy, precision, recall, and other metrics.

4. **Restart the server** to use the new model:
   ```bash
   # Stop current server (Ctrl+C or pkill)
   sudo python3 server.py
   ```

The model files will be automatically loaded by the server on startup.

## 🧪 Attack Simulation & Testing

To test the system's real-time attack detection capabilities:

### Quick Demo

1. **Start the SmartShield backend:**
   ```bash
   sudo python3 server.py
   ```

2. **Open the dashboard** (hosted link or `npm run dev`)

3. **Run attack simulator** (in another terminal):
   ```bash
   sudo python3 attack_simulator.py
   # Choose option 5 for full demo sequence
   ```

4. **Watch real-time detection** in the dashboard!

### Available Attack Types

- **Port Scan** - Rapid reconnaissance scanning
- **DDoS Flood** - High packet rate flooding
- **SQL Injection** - Malicious HTTP payloads
- **Suspicious Port Scan** - Unusual port combinations
- **ICMP Flood** - Ping flood attacks
- **Full Demo Sequence** - All attacks in sequence
- **Continuous Random** - Random attacks for extended testing

### Command Line Usage

```bash
# Run specific attack
sudo python3 attack_simulator.py --attack 1 --duration 10

# Run demo sequence
sudo python3 attack_simulator.py --attack 5

# Continuous random attacks
sudo python3 attack_simulator.py --attack 6
```

**⚠️ Safety:** The simulator defaults to `127.0.0.1` (localhost) for safe testing. Only use on authorized networks.

## 📁 Project Structure

```
smartshield/
├── server.py                 # Flask backend with Scapy packet capture
├── attack_simulator.py       # Attack simulation for testing/demo
├── train_enhanced_rf.py      # Random Forest training script
├── evaluate_rf_model.py      # Model evaluation script
├── random_forest_model.joblib # Trained model (76 MB)
├── rf_feature_scaler.joblib  # Feature scaler
├── service_encoder.joblib    # Service label encoder
├── state_encoder.joblib      # State label encoder
├── RANDOM_FOREST_RESULTS.md  # Detailed test results and metrics
├── requirements.txt          # Python dependencies
└── src/                      # Frontend React application
```

## 🔧 Technology Stack

- **Frontend:** React + TypeScript + Vite
- **Backend:** Flask + Scapy
- **ML Model:** Random Forest (scikit-learn)
- **Dataset:** UNSW-NB15
- **Visualization:** Recharts

## 📝 License

This project is part of an AI Studio application.
