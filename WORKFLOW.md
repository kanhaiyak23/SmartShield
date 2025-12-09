# SmartShield System Workflow

Complete technical overview of SmartShield's architecture and data flow.

---

## 📋 System Architecture

```
┌─────────────┐      ┌──────────────┐      ┌─────────────┐
│   Frontend  │◄────►│   Backend    │◄────►│   Network   │
│  (React)    │ HTTP │   (Flask)    │ Scapy│   Traffic   │
└─────────────┘      └──────────────┘      └─────────────┘
       │                     │
       │                     ▼
       │              ┌──────────────┐
       │              │   ML Model   │
       └─────────────►│  (Random     │
                      │   Forest)    │
                      └──────────────┘
```

---

## 🔄 Complete Workflow

### **Phase 1: Model Training** (One-Time Setup)

**Files:** `train_enhanced_rf.py`

1. **Load Dataset**
   - Reads 5 UNSW-NB15 CSV files (500,000+ records)
   - Combines into single dataset

2. **Feature Extraction**
   - Extracts 20 features per packet:
     - **Basic (1-5):** Length, TTL, Protocol, Source Port, Dest Port
     - **Flow Stats (6-11):** Duration, Packet Rate, Byte Rate, Packet Counts, Packet Ratio
     - **Network Behavior (12-15):** Service, State, Port Categories
     - **Statistics (16-19):** Mean Packet Size, Jitter, Inter-packet Time, TCP RTT
     - **Total (20):** Total Packets

3. **Preprocessing**
   - Encodes categorical features (service, state)
   - Scales numerical features (StandardScaler)
   - Splits: 80% train, 20% validation

4. **Training**
   - Random Forest: 200 trees, max_depth=20
   - Class weights: 'balanced' (handles imbalanced data)
   - Saves: `random_forest_model.joblib`, scaler, encoders

---

### **Phase 2: Backend Server** (Runtime)

**Files:** `server.py`

#### **2.1 Initialization**

1. **Load Model**
   ```python
   - Loads random_forest_model.joblib
   - Loads rf_feature_scaler.joblib
   - Loads service_encoder.joblib, state_encoder.joblib
   - Falls back to simulation mode if missing
   ```

2. **Setup Flask App**
   ```python
   - Creates Flask app with CORS enabled
   - Initializes packet buffer (deque, max 100 packets)
   - Starts connection cache for flow tracking
   ```

3. **Start Packet Capture**
   ```python
   - Spawns background thread for packet sniffing
   - Uses Scapy's sniff() function
   - Filters: IP packets only
   - Runs continuously (daemon thread)
   ```

#### **2.2 Packet Processing Pipeline**

For each captured packet:

1. **Packet Capture** (`sniff()`)
   - Scapy captures IP packets from network interface
   - Calls `packet_handler()` callback

2. **Basic Extraction** (`process_packet()`)
   ```
   - Extract: IP addresses, ports, protocol, TTL, length
   - Parse: TCP flags, window size, sequence numbers
   - Extract: Raw payload (hex + ASCII)
   ```

3. **Connection Tracking** (`update_connection_stats()`)
   ```
   - Group packets by: (src_ip, dst_ip, src_port, dst_port)
   - Calculate flow statistics:
     * Duration, packet rates, byte rates
     * Packet counts (source/dest)
     * Mean packet size, jitter
     * Inter-packet timing
   ```

4. **Feature Extraction** (`extract_enhanced_features()`)
   ```
   - Extracts same 20 features as training
   - Uses connection stats for flow features
   - Encodes service/state using loaded encoders
   ```

5. **ML Inference** (`run_random_forest_inference()`)
   ```
   - Scales features using loaded scaler
   - Runs Random Forest prediction
   - Gets attack probability (0.0 - 1.0)
   - Converts to anomaly score (-0.5 to 0.5)
   ```

6. **Risk Classification**
   ```
   - Probability ≥ 0.7 → CRITICAL (red)
   - Probability ≥ 0.4 → WARNING (yellow)
   - Probability < 0.4  → SAFE (green)
   ```

7. **Pattern Matching** (`detect_malicious_payload()`)
   ```
   - Scans payload for SQL injection patterns
   - Scans for XSS patterns
   - Additional heuristic checks
   ```

8. **Packet Storage**
   ```
   - Creates packet object with all metadata
   - Adds to packet_buffer (thread-safe deque)
   - Buffer keeps last 100 packets
   ```

#### **2.3 API Endpoints**

**GET `/packets`**
- Returns all packets from buffer as JSON array
- Frontend polls this endpoint every 1 second

**GET `/status`**
- Returns server status (capturing, packet count, etc.)

**GET `/health`**
- Health check endpoint

---

### **Phase 3: Frontend Dashboard** (Runtime)

**Files:** `App.tsx`, `components/Dashboard.tsx`, `services/scapyService.ts`

#### **3.1 Application Structure**

```
App.tsx (Root)
  ├── Navbar (Navigation)
  ├── Hero (Landing page)
  ├── Dashboard (Main view) ← Real-time monitoring
  ├── ThreatMap (Geographic visualization)
  ├── Intel (Analysis)
  ├── SystemMonitor (System stats)
  └── Footer
```

#### **3.2 Dashboard Workflow**

1. **Initialization**
   ```typescript
   - Sets up state: packets, isScanning, isConnected
   - Initializes empty packet array
   - Sets up polling interval
   ```

2. **Real-Time Polling** (`useEffect`)
   ```typescript
   Every 1 second:
     - Calls pollPacketStream() (scapyService.ts)
     - Fetches GET http://127.0.0.1:5000/packets
     - Timeout: 1 second
     - Sets connection status
   ```

3. **Packet Processing**
   ```typescript
   - Filters out duplicate packets (by ID)
   - Adds new packets to state
   - Updates statistics:
     * Total packets
     * Threats (CRITICAL count)
     * Total bytes
   - Updates chart data (last 30 data points)
   ```

4. **Display Updates**
   ```typescript
   - Renders packet table (sorted by timestamp)
   - Color codes: RED (CRITICAL), YELLOW (WARNING), GREEN (SAFE)
   - Updates charts (anomaly score, traffic volume)
   - Shows connection status indicator
   ```

5. **Packet Details** (`PacketDetails.tsx`)
   ```
   When user clicks packet:
     - Shows full packet details
     - Displays 20-feature vector
     - Shows hex dump
     - Displays ASCII payload
     - Shows ML assessment
   ```

---

## 📊 Data Flow Diagram

```
Network Traffic
     │
     ▼
[Scapy Sniff] ← Captures packets
     │
     ▼
[Process Packet] ← Extract basic info
     │
     ▼
[Connection Tracking] ← Group by flow, calculate stats
     │
     ▼
[Extract Features] ← 20 features
     │
     ▼
[ML Model Inference] ← Random Forest prediction
     │
     ▼
[Risk Classification] ← SAFE/WARNING/CRITICAL
     │
     ▼
[Store in Buffer] ← deque(maxlen=100)
     │
     ▼
[API Endpoint] ← GET /packets
     │
     ▼
[Frontend Poll] ← Every 1 second
     │
     ▼
[Display Dashboard] ← Real-time visualization
```

---

## 🔧 Key Components

### **Backend (Python)**

| Component | Purpose | File |
|-----------|---------|------|
| Flask Server | HTTP API | `server.py` |
| Packet Capture | Network sniffing | Scapy library |
| Feature Extraction | 20-feature vector | `extract_enhanced_features()` |
| ML Model | Attack detection | Random Forest (scikit-learn) |
| Connection Tracking | Flow statistics | `update_connection_stats()` |

### **Frontend (React/TypeScript)**

| Component | Purpose | File |
|-----------|---------|------|
| Dashboard | Main monitoring view | `components/Dashboard.tsx` |
| Packet Service | API communication | `services/scapyService.ts` |
| Packet Details | Detailed packet view | `components/PacketDetails.tsx` |
| App Router | Navigation | `App.tsx` |

### **Training (Python)**

| Component | Purpose | File |
|-----------|---------|------|
| Data Loader | Load CSV files | `load_unsw_dataset()` |
| Feature Extractor | Extract 20 features | `extract_enhanced_features()` |
| Model Trainer | Train Random Forest | `train_random_forest()` |
| Model Saver | Save model files | joblib |

---

## 🎯 Attack Simulation

**File:** `attack_simulator.py`

Generates attack packets using Scapy:

1. **Port Scan**: Rapid SYN packets to multiple ports
2. **DDoS Flood**: High-rate packet flooding
3. **SQL Injection**: Malicious HTTP payloads
4. **ICMP Flood**: Ping flood attacks
5. **Suspicious Port Scan**: Unusual port combinations

All attacks target `127.0.0.1` (localhost) for safe testing.

---

## 📈 Performance Characteristics

- **Packet Processing**: < 10ms per packet
- **ML Inference**: < 100ms per packet
- **Frontend Polling**: 1 second interval
- **Buffer Size**: 100 packets (last in, first out)
- **Connection Tracking**: Max 10,000 flows
- **Model Size**: 76 MB
- **Attack Detection Rate**: 99.62%

---

## 🔄 State Management

### **Backend State**
- `packet_buffer`: Thread-safe deque (100 packets)
- `connection_cache`: Dictionary of flows
- `packet_id_counter`: Global counter
- `is_capturing`: Boolean flag

### **Frontend State**
- `packets`: Array of Packet objects
- `isScanning`: Boolean (paused/active)
- `isConnected`: Backend connection status
- `selectedPacket`: Currently viewed packet
- `stats`: Aggregated statistics
- `chartData`: Time series data (30 points)

---

## 🛡️ Security Features

1. **Pattern Matching**: Detects SQL injection, XSS in payloads
2. **ML Detection**: 99.62% attack detection rate
3. **Risk Levels**: 3-tier classification (SAFE/WARNING/CRITICAL)
4. **Flow Analysis**: Tracks connections for behavioral patterns
5. **Real-Time**: Instant detection and alerting

---

## 📝 File Summary

### **Core Files**
- `server.py` - Backend server (727 lines)
- `App.tsx` - Frontend root (42 lines)
- `components/Dashboard.tsx` - Main view (346+ lines)
- `train_enhanced_rf.py` - Model training (344 lines)
- `attack_simulator.py` - Attack generation (275 lines)

### **Support Files**
- `services/scapyService.ts` - API client
- `components/PacketDetails.tsx` - Packet inspector
- `evaluate_rf_model.py` - Model evaluation
- `types.ts` - TypeScript definitions

---

## 🚀 Startup Sequence

1. **Backend Starts**
   - Loads ML model (or simulation mode)
   - Starts Flask server on port 5000
   - Begins packet capture in background thread

2. **Frontend Starts**
   - Vite dev server on port 3000
   - React app loads
   - Dashboard begins polling backend

3. **System Ready**
   - Backend capturing packets
   - Frontend displaying real-time data
   - ML model analyzing traffic
   - Ready for attack simulation

---

## 📊 Model Details

- **Algorithm**: Random Forest Classifier
- **Trees**: 200
- **Features**: 20
- **Training Data**: UNSW-NB15 (500,000+ records)
- **Test Performance**: 
  - Accuracy: 86.46%
  - Recall: 99.62%
  - Precision: 23.54%
  - ROC-AUC: 92.50%

---

## 🔍 Key Algorithms

### **Feature Extraction**
- Protocol identification (TCP/UDP/ICMP/HTTP)
- Port categorization (well-known/registered/ephemeral)
- Flow statistics calculation (rates, ratios)
- Time-based features (jitter, inter-packet time)

### **Connection Tracking**
- Groups packets by (src_ip, dst_ip, src_port, dst_port)
- Maintains sliding window of packet history
- Calculates temporal statistics
- Limits cache size (eviction strategy)

### **Risk Classification**
- Maps ML probability → Risk level
- Thresholds: 0.7 (CRITICAL), 0.4 (WARNING)
- Combines ML + pattern matching
- Real-time decision making

---

**End of Workflow Documentation**

