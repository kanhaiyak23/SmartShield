# 🛡️ SmartShield: AI-Powered Network Intrusion Detection System

**SmartShield** is a real-time, academic-grade Network Intrusion Detection System (NIDS) that leverages Machine Learning to classify network traffic as **Safe** or **Malicious** on the fly. It features a modern React-based dashboard for visualization and a robust Python backend for packet capture and inference.

---

## 🚀 Key Features

*   **Real-Time Packet Capture**: Dual-threaded sniffing engine capturing both **Wi-Fi (WAN)** and **Loopback (Local)** traffic simultaneously.
*   **AI Detection Engine**: Pre-trained **Random Forest Classifier** analyzing 20+ distinct packet features (TTL, Entropy, Jitter, Flags, etc.).
*   **Attack Recognition**: Capable of identifying:
    *   🔍 **Port Scanning** (Reconnaissance)
    *   🌊 **DDoS Floods** (Volumetric Attacks)
    *   💉 **SQL Injection** (Application Layer Attacks)
*   **Unified Dashboard**: Live visualization of bandwidth, threat counters, and packet lists with "Risk Level" color coding.

---

## 🛠️ System Architecture

1.  **Packet Capture Layer**: Uses `Scapy` (Python) to sniff raw packets from network interfaces (`en0`, `lo0`).
2.  **Feature Extraction**: Calculates statistical features (Flow Duration, Packet Rate, Byte Rate, Window Size) in real-time.
3.  **Inference Engine**: Passes feature vectors to a serialized `scikit-learn` model (`random_forest_model.joblib`).
4.  **API Server**: Flask-based REST API serving processed packet data to the frontend.
5.  **Visualization**: React (TypeScript/Vite) frontend polling the API for live updates.

---

## 📋 Installation & Setup

### Prerequisites
*   **Python 3.10+**
*   **Node.js 18+**
*   **libpcap** (Usually pre-installed on macOS/Linux)

### 1. Backend Setup (Terminal 1)
```bash
# Navigate to project root
cd SmartShield-1

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies (Flask, Scapy, Pandas, Scikit-learn)
pip install -r requirements.txt

# Start the Server (Requires sudo for packet capture permissions)
sudo ./venv/bin/python3 server.py
```
*The server will listen on `http://127.0.0.1:5000`.*

### 2. Frontend Setup (Terminal 2)
```bash
# Install Node modules
npm install

# Start Development Server
npm run dev
```
*Opens Dashboard at `http://localhost:5173`.*

---

## 🎓 Running the Academic Demo

We have developed a specialized **Demo Controller** to simulate cyber-attacks safely for presentation purposes.

1.  **Ensure Backend & Frontend are running.**
2.  **Open the Dashboard** (`http://localhost:5173`).
    *   Confirm functionality by observing real traffic (e.g., open a YouTube video).
3.  **Launch Demo Script (Terminal 3)**:
    ```bash
    sudo ./venv/bin/python3 academic_demo_controller.py
    ```
4.  **Follow the Script**:
    *   **Phase 1 (Normal Traffic)**: Injects benign HTTP traffic. Dashboard stays **Green**.
    *   **Phase 2 (Attack Sequence)**:
        *   **Port Scan**: Rapid sequential connection attempts.
        *   **DDoS Flood**: High-volume SYN flood (>1000 pps).
        *   **SQL Injection**: Malicious payloads in HTTP packets.
    *   **Result**: Dashboard turns **RED** (Critical Alert), and "Threats Detected" counter spikes.

---

## 🛡️ Traffic Filtering Logic

To ensure a clean demonstration, the system employs a **Smart Filter**:
*   **Real Traffic**: Verified Internet traffic (Wi-Fi) is always shown.
*   **Demo Traffic**: Packets injected containing the cryptographically signed `SMARTSHIELD_ATTACK` payload are highlighted.
*   **System Noise**: Background OS noise (mDNS, internal loopback) is automatically filtered out to reduce clutter.

---

## 👥 Authors & License
Developed for **Academic Research Project**.
*   **Version**: 2.5.0-ALPHA
*   **License**: MIT
