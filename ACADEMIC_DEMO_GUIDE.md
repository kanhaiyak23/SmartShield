# 🎓 SmartShield Academic Demo Guide

This guide is designed to walk you through a professional demonstration of the SmartShield Intrusion Detection System for your academic presentation.

## 📋 Pre-Demo Checklist (5 Minutes Before)

1.  **Clean State:** Close all terminal windows and browser tabs.
2.  **Start Backend:**
    *   Open Terminal 1
    *   Run: `sudo ./venv/bin/python3 server.py`
    *   Verify it says: `Server running on http://127.0.0.1:5000`
3.  **Start Frontend:**
    *   Open Terminal 2
    *   Run: `npm run dev`
    *   Open `http://localhost:3000` in your browser.
4.  **Verify Baseline:**
    *   Check that the dashboard is loading.
    *   Confirm "PACKET STREAM" shows green/safe traffic (mostly local/internet).
    *   **Click "Hide Local" / "No Local"** button to declutter the view.

---

## 🗣️ The Presentation Script

### Phase 1: Introduction (Normal State)

**"Good morning/afternoon. This is SmartShield, a real-time network intrusion detection system powered by Machine Learning."**

*   **Show the Dashboard:**
    *   *"As you can see, the system is currently monitoring live network traffic from this machine in real-time."*
    *   *"The Random Forest model is analyzing every single packet. Currently, the traffic is legitimate—background status checks from operating system services—so the system classifies it as SAFE (Green)."*
    *   *"The Anomaly Score graph at the bottom remains flat, indicating zero threats."*

---

### Phase 2: The Attack (Live Simulation)

**"Now, I will demonstrate the system's ability to detect various categories of network attacks using an automated attack simulator."**

*   **Action:** Open **Terminal 3**.
*   **Run:** `sudo python3 attack_simulator.py --attack 5`
*   *(Keep this terminal side-by-side with the browser window)*

#### 1. Port Scan Attack
*(Simulator starts scanning ports...)*

**You Say:** *"First, we are launching a Port Scan. This is a reconnaissance attack where a hacker tries to find open doors."*

**Look at Dashboard:**
*   You will see a burst of yellow/red packets.
*   **Point out:** *"Notice the burst of 'WARNING' alerts. The model detects the rapid sequence of connection attempts."*

#### 2. DDoS Flood
*(Simulator switches to DDoS...)*

**You Say:** *"Next, we simulate a Denial of Service flood. This is high-velocity traffic designed to overwhelm the server."*

**Look at Dashboard:**
*   The 'packets per second' graph will spike.
*   **Point out:** *"The traffic volume spikes instantly. The model flags this as CRITICAL due to the abnormal packet rate and flow duration."*

#### 3. SQL Injection
*(Simulator sends malicious HTTP payloads...)*

**You Say:** *"Now, a more subtle attack: SQL Injection. These are standard HTTP requests but contain malicious payloads."*

**Look at Dashboard:**
*   Look for packets with `SQL Injection` or unusually long payloads.
*   **Point out:** *"Even though the traffic volume is low, the model analyzes the payload content and flags the malicious SQL syntax."*

---

### Phase 3: Post-Mortem & Conclusion

*   **Action:** Click on one of the **Red (Critical)** packets in the list to open the "Packet Details" view on the right.

**You Say:**
*   *"If we inspect a captured threat, we can see exactly why it was flagged."*
*   *"Here is the 'Feature Vector' the AI used. Parameters like `destPort`, `packet_rate`, and `payload_entropy` deviated significantly from the learned baseline."*
*   *"On a dataset of 500,000 records, this model achieved 99.6% accuracy."*

**"This demonstrates that SmartShield can effectively detect both volumetric attacks (like DDoS) and sophisticated application attacks (like SQLi) in real-time."**

---

## 🛠️ Troubleshooting

*   **"No Traffic Appearing":** Refresh the page. Ensure backend is running.
*   **"Counter stuck at 100":** This is a dashboard polling limit, not a bug.
*   **"Backend Error":** Press Ctrl+C in backend terminal and restart with `sudo ./venv/bin/python3 server.py`.
