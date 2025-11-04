# SmartShield: AI-Powered Campus Intrusion Detection System

## Overview

SmartShield is a real-time, AI-driven Intrusion Detection System designed for campus Wi-Fi networks. It monitors live network traffic, detects anomalies using machine learning, and provides an interactive web dashboard for visualization and alerts.

## Features

- **Real-time Packet Capture**: Live monitoring using Scapy for campus Wi-Fi traffic
- **AI-Powered Detection**: Multiple ML models (Random Forest, Isolation Forest, Autoencoder LSTM) for anomaly detection
- **Attack Classification**: Identifies attack types including DDoS, port scanning, ARP spoofing, and more
- **Web Dashboard**: Interactive Next.js dashboard with live alerts, protocol distribution, and top IPs
- **Campus Wi-Fi Optimized**: Trained on UNSW-NB15 dataset and adapted for college network patterns

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Campus Wi-Fi Network                      │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│              Packet Capture (Scapy)                          │
│              Real-time Traffic Sniffing                      │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│              AI Detection Engine (Python)                    │
│  • Random Forest Classifier                                 │
│  • Isolation Forest (Anomaly)                               │
│  • Autoencoder LSTM (Deep Learning)                         │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│              Backend API (Express + Node.js)                │
│  • REST API for traffic data                               │
│  • WebSocket for live alerts                               │
│  • Alert management                                        │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│              Web Dashboard (Next.js)                        │
│  • Real-time visualizations                                │
│  • Protocol distribution                                   │
│  • Top IPs and connections                                 │
│  • Alert management dashboard                              │
└─────────────────────────────────────────────────────────────┘
```

## Tech Stack

### Backend
- **Python 3.9+**: ML models, packet capture, data preprocessing
- **Node.js + Express**: REST API and WebSocket server
- **Scapy**: Real-time packet capture and analysis
- **TensorFlow**: Deep learning models
- **Scikit-learn**: Classical ML algorithms

### Frontend
- **Next.js**: React framework for dashboard
- **Chart.js**: Data visualizations
- **Tailwind CSS**: Styling

### ML Models
1. **Random Forest**: Classification of attack types
2. **Isolation Forest**: Anomaly detection
3. **Autoencoder LSTM**: Deep learning anomaly detection

### Datasets
- **UNSW-NB15**: Primary training dataset for campus Wi-Fi anomalies
- **CICIDS2017**: Benchmark dataset for testing

## Quick Start

Get started in under 10 minutes! See [QUICKSTART.md](QUICKSTART.md) for detailed setup instructions.

```bash
# Clone repository
git clone https://github.com/yourusername/smartshield.git
cd smartshield

# Setup Python environment
cd python && python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Train models (download UNSW-NB15 dataset first!)
python train_models.py --train-all

# Setup and start services
cd .. && ./start.sh
```

For detailed installation instructions, see [SETUP.md](SETUP.md).

## Usage

### 1. Start the AI Detection Engine

```bash
cd python
source venv/bin/activate
python main.py --mode train  # Train models first
python main.py --mode detect --interface eth0  # Start detection
```

### 2. Start the Backend API

```bash
cd backend
npm start
```

The API will be available at `http://localhost:3001`

### 3. Start the Frontend Dashboard

```bash
cd frontend
npm run dev
```

Open `http://localhost:3000` in your browser.

### 4. Simulate Attacks (Testing)

```bash
# Port scanning
nmap -sS target_ip

# ARP spoofing
python python/attack_simulator.py --type arp-spoof

# DDoS (in controlled environment)
python python/attack_simulator.py --type ddos
```

## Project Structure

```
smartshield/
├── python/
│   ├── models/              # Trained ML models
│   ├── data/                # Dataset storage
│   ├── src/
│   │   ├── preprocessing.py # Data preprocessing
│   │   ├── models.py        # ML model definitions
│   │   ├── packet_capture.py # Live packet capture
│   │   ├── detection.py     # Detection engine
│   │   └── utils.py         # Utility functions
│   ├── train_models.py      # Model training script
│   ├── main.py              # Main detection entry point
│   └── requirements.txt
├── backend/
│   ├── src/
│   │   ├── server.js        # Express server
│   │   ├── routes/          # API routes
│   │   └── websocket.js     # WebSocket handler
│   ├── package.json
│   └── .env
├── frontend/
│   ├── src/
│   │   ├── pages/           # Next.js pages
│   │   ├── components/      # UI components
│   │   └── styles/          # CSS/Tailwind
│   ├── package.json
│   └── next.config.js
└── README.md
```

## Features in Detail

### Detection Capabilities

- **DDoS Attacks**: Detects flooding and distributed denial-of-service
- **Port Scanning**: Identifies reconnaissance attempts
- **ARP Spoofing**: Detects man-in-the-middle attacks
- **Malware Traffic**: Recognizes malicious communication patterns
- **Anomaly Detection**: Catches novel or zero-day attacks

### Dashboard Features

- Live traffic monitoring with real-time updates
- Protocol distribution charts
- Top IP addresses and connections
- Alert management with severity levels
- Historical attack trends
- Model performance metrics

## Performance Metrics

Expected performance on campus Wi-Fi:
- **Detection Accuracy**: 95%+ on UNSW-NB15
- **False Positive Rate**: < 5%
- **Latency**: < 100ms for packet analysis
- **Throughput**: Handles up to 10,000 packets/second

## Limitations & Future Work

- Currently in monitor mode (no automatic blocking)
- Performance depends on network interface speed
- Requires labeled data for supervised learning
- Future: SDN integration for automatic mitigation
- Future: Federated learning for privacy-preserving detection




## Acknowledgments

- UNSW-NB15 Dataset: University of New South Wales
- CICIDS2017 Dataset: Canadian Institute for Cybersecurity

