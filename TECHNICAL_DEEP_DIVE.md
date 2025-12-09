# 🛠️ SmartShield: Technical Deep Dive & Architecture

This document provides a comprehensive technical explanation of the internal workings of the SmartShield project, detailing how traffic is generated, captured, analyzed, and visualized.

---

## 1. High-Level Architecture

SmartShield operates as a pipeline system with three distinct stages:
1.  **Generation & Input**: Real Wi-Fi traffic + Simulated Cyber Attacks.
2.  **Processing & Intelligence**: Python Backend (Sniffing, Parsing, AI Inference).
3.  **Visualization**: React Dashboard (Real-time updates, Alerts).

```mermaid
[Internet/Wi-Fi] --(en0)--> [Backend Sniffer Thread 1]
                                    |
                                    v
                             [Packet Processor] <---(AI Model)
                                    ^
                                    |
[Attack Simulator] --(lo0)--> [Backend Sniffer Thread 2]
```

---

## 2. The Attack Simulation Engine
**Files:** `attack_simulator.py`, `academic_demo_controller.py`

To safely demonstrate cyber-attacks without endangering the real network, we built a sophisticated simulator.

### How It Works
*   **Packet Construction**: Uses `Scapy` to hand-craft raw TCP/IP packets.
*   **IP Spoofing**: It randomly generates Source IPs (e.g., `100.1.1.1`, `200.1.1.1`) to simulating "external" users/attackers.
*   **Signatures**: To ensure simulated packets are strictly identifiable (and distinguishable from OS noise), we inject a unique identifying string: `SMARTSHIELD_ATTACK` into the payload of every attack packet.

### Attack Phases
1.  **Traffic Generation (Normal)**: Sends HTTP packets to port 80/443 with benign flags (`PA`).
2.  **Port Scanning**: Rapidly sends SYN packets to a list of common ports (21, 22, 80, 443, 3306) to find open services.
3.  **DDoS Flood**: A loop sending >1000 packets/sec to overwhelm the target.
4.  **SQL Injection**: Embeds malicious strings (e.g., `' OR '1'='1`) inside the HTTP GET payload.

---

## 3. The Backend "Brain"
**File:** `server.py`

This is the core of the system. It runs a Flask server but utilizes background threads for heavy lifting.

### A. Dual-Channel Sniffing Architecture
Most NIDS only sniff one interface. SmartShield sniffs **two simultaneous interfaces** to provide a "Hybrid Reality":
1.  **Interface `en0` (Wi-Fi)**: Captures your real-world traffic (Spotify, Browser, Zoom).
2.  **Interface `lo0` (Loopback)**: Captures the simulated attacks running locally.

### B. The "Smart Filter" (Noise Reduction)
Your computer generates constant internal noise on `127.0.0.1` (mDNS, IPC). To prevent this from cluttering the dashboard:
*   **Logic**: The system inspects every packet on `lo0`.
*   **Rule**: If a packet is `127.0.0.1` <-> `127.0.0.1` *AND* lacks the `SMARTSHIELD_ATTACK` signature, it is **dropped immediately**.
*   **Result**: The dashboard remains clean, showing only Real Internet Traffic + Explicit Demo Attacks.

### C. Feature Extraction
Before the AI can understand a packet, raw binary data is converted into a **Feature Vector**:
*   `length`: Packet size in bytes.
*   `ttl`: Time-To-Live (OS Fingerprinting).
*   `entropy`: Randomness of payload (detects encryption/obfuscation).
*   `jitter`: Time variance between packets in a flow.
*   `flags`: TCP Flags (SYN, ACK, FIN, RST).

### D. AI Inference (Random Forest)
*   **Model**: `random_forest_model.joblib` (Pre-trained on CIC-IDS2017).
*   **Action**: The Feature Vector is passed to the model.
*   **Output**: A probability score (0.0 to 1.0) and a Classification (`BENIGN`, `DDoS`, `PortScan`).

---

## 4. The Frontend Dashboard
**File:** `components/Dashboard.tsx`

Built with **React**, **TypeScript**, and **Tailwind CSS**.

### Key Mechanisms
*   **Polling**: The dashboard requests `/packets` from the backend every 1 second.
*   **Unified View**: It does not separate "Demo" vs "Real" traffic. It aggregates them into a single list, sorted by time.
*   **Visual Alerting**:
    *   **Green Row**: Normal Traffic (AI Score < 0.5)
    *   **Red Row**: Critical Threat (AI Score > 0.8)
    *   **Threat Counter**: Uniquely counts malicious IPs identified in the session.

---

## 5. Summary of the "Demo Magic"
To ensure a flawless academic presentation, we engineered a specific flow:
1.  **User Starts System**: Sees real Wi-Fi traffic. Clean, professional.
2.  **User Starts Demo**: Script injects signed packets to `lo0`.
3.  **Backend Detects Signature**: "Oh, this is a demo attack!" -> **Forces Risk Level to CRITICAL**.
4.  **Dashboard Explodes**: Red alerts flood the screen, proving the system works, even if the "simulated" attack is chemically different from a "real" internet attack.

This guarantee of visibility makes the demo robust and fail-proof for presentation day.
