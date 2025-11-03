# SmartShield Architecture

## System Architecture Overview

SmartShield follows a modular, microservices-based architecture that separates concerns for scalability and maintainability.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        SmartShield System                               │
└─────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────┐
│   Campus Wi-Fi Network   │  ← Monitored Network
│  (Students, Devices,     │
│   IoT, Servers)          │
└──────────────┬───────────┘
               │ Network Traffic
               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     Layer 1: Packet Capture                             │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  PacketCapture (Scapy)                                         │   │
│  │  - Real-time packet sniffing                                   │   │
│  │  - Protocol extraction (TCP/UDP/ICMP/ARP)                      │   │
│  │  - Feature extraction                                          │   │
│  │  - Statistics collection                                       │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└──────────────────────┬──────────────────────────────────────────────────┘
                       │ Raw Packet Features
                       ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                   Layer 2: Data Preprocessing                            │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  DataPreprocessor                                               │   │
│  │  - Feature normalization                                        │   │
│  │  - Categorical encoding                                         │   │
│  │  - Missing value handling                                       │   │
│  │  - UNSW-NB15 feature mapping                                    │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└──────────────────────┬──────────────────────────────────────────────────┘
                       │ Processed Features
                       ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                  Layer 3: AI Detection Engine                            │
│  ┌──────────────────┐  ┌──────────────────┐  ┌────────────────────┐   │
│  │ Random Forest    │  │ Isolation Forest │  │ Autoencoder LSTM   │   │
│  │ Classifier       │  │ Anomaly Detector │  │ Deep Learning      │   │
│  │ - Attack Type    │  │ - Novel Attack   │  │ - Pattern Detect   │   │
│  │ - Confidence     │  │ - Score         │  │ - Reconstruction   │   │
│  └────────┬─────────┘  └────────┬─────────┘  └──────────┬─────────┘   │
│           │                     │                        │             │
│           └─────────────────────┼────────────────────────┘             │
│                                 ▼                                       │
│                        ┌─────────────────────┐                         │
│                        │ Combined Scoring    │                         │
│                        │ & Alert Generation  │                         │
│                        └─────────────────────┘                         │
└──────────────────────┬──────────────────────────────────────────────────┘
                       │ Detection Results & Alerts
                       ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                  Layer 4: API Layer (REST + WebSocket)                  │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Flask ML API (Port 5000)                                      │   │
│  │  - /predict (POST)                                             │   │
│  │  - /predict/batch (POST)                                       │   │
│  │  - /health (GET)                                               │   │
│  │  - /model/info (GET)                                           │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Express Backend (Port 3001)                                   │   │
│  │  - REST API for data retrieval                                 │   │
│  │  - WebSocket (Socket.io) for live updates                      │   │
│  │  - Alert storage & management                                  │   │
│  │  - Statistics aggregation                                      │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└──────────────────────┬──────────────────────────────────────────────────┘
                       │ JSON API Responses + WebSocket Events
                       ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     Layer 5: Web Dashboard                               │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Next.js Frontend (Port 3000)                                  │   │
│  │  - Real-time UI updates                                        │   │
│  │  - Interactive visualizations                                  │   │
│  │  - Alert management interface                                  │   │
│  │  - Statistics cards                                            │   │
│  │  - Protocol charts                                             │   │
│  │  - Top IPs list                                                │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. Packet Capture Layer

**File**: `python/src/packet_capture.py`

**Responsibilities:**
- Capture network packets from specified interface
- Extract packet metadata (IP, ports, protocol, etc.)
- Convert packets to feature vectors
- Track statistics (packet counts, byte counts, protocol distribution)

**Key Functions:**
- `start_capture(interface, filter)` - Begin monitoring
- `_extract_packet_features(packet)` - Feature extraction
- `_packet_handler(packet)` - Process each packet
- `get_stats()` - Retrieve statistics

**Dependencies:**
- Scapy for packet capture
- Threading for async capture

### 2. Data Preprocessing Layer

**File**: `python/src/preprocessing.py`

**Responsibilities:**
- Normalize features from raw packet data
- Encode categorical variables
- Scale numerical features
- Handle missing values
- Map to UNSW-NB15 feature space

**Key Components:**
- `StandardScaler` - Feature normalization
- `LabelEncoder` - Categorical encoding
- `preprocess_unsw_nb15()` - Dataset preprocessing
- `preprocess_packet_features()` - Live packet preprocessing

### 3. ML Model Layer

**File**: `python/src/models.py`

**Model 1: Random Forest Classifier**
- **Purpose**: Classify attack types
- **Input**: Preprocessed packet features
- **Output**: Attack category (Normal, DDoS, Port Scan, etc.)
- **Accuracy**: ~95% on UNSW-NB15

**Model 2: Isolation Forest**
- **Purpose**: Detect anomalies
- **Input**: Preprocessed packet features
- **Output**: Anomaly score, binary classification
- **Method**: Unsupervised learning

**Model 3: Autoencoder LSTM**
- **Purpose**: Deep learning anomaly detection
- **Input**: Preprocessed packet features
- **Output**: Reconstruction error, anomaly flag
- **Method**: Neural network-based reconstruction

**Combined Prediction:**
- Weighted ensemble of all models
- Composite confidence score
- Final anomaly decision

### 4. Detection Engine

**File**: `python/src/detection.py`

**Responsibilities:**
- Orchestrate packet capture and ML inference
- Generate alerts with severity levels
- Manage alert history
- Aggregate statistics

**Key Components:**
- `DetectionEngine` - Main orchestrator
- `_create_alert()` - Alert generation
- `_calculate_severity()` - Severity scoring
- `process_packet()` - End-to-end processing

### 5. API Services

**ML API (Flask)**
- **File**: `python/api_server.py`
- **Port**: 5000
- **Purpose**: Expose ML models via REST API
- **Endpoints**: Predict, health checks, model info

**Backend API (Express)**
- **File**: `backend/src/server.js`
- **Port**: 3001
- **Purpose**: Data management and WebSocket server
- **Endpoints**: Statistics, alerts, health
- **WebSocket**: Real-time updates

### 6. Frontend Dashboard

**File**: `frontend/src/pages/index.js`

**Components:**
- `StatisticsCards` - Key metrics display
- `ProtocolChart` - Doughnut chart visualization
- `AlertsTable` - Filterable alert list
- `TopIPs` - Top attacking IPs
- `Layout` - Navigation and structure

**Technologies:**
- Next.js for SSR and routing
- React for UI components
- Chart.js for visualizations
- Tailwind CSS for styling
- Socket.io-client for WebSocket

## Data Flow

### Detection Flow

```
Network Packet
    ↓
PacketCapture extracts features
    ↓
DataPreprocessor normalizes features
    ↓
ModelTrainer.predict()
    ├── Random Forest → Attack Type
    ├── Isolation Forest → Anomaly Score
    └── Autoencoder → Reconstruction Error
    ↓
Combined scoring & alert generation
    ↓
Alert sent to Backend API
    ↓
WebSocket broadcasts to Frontend
    ↓
Dashboard updates in real-time
```

### Training Flow

```
UNSW-NB15 Dataset
    ↓
DataPreprocessor.preprocess_unsw_nb15()
    ├── Categorical encoding
    ├── Feature normalization
    └── Train/test split
    ↓
ModelTrainer trains models
    ├── train_random_forest()
    ├── train_isolation_forest()
    └── train_autoencoder()
    ↓
Models saved to disk
    ↓
Models loaded at runtime
```

## Technology Stack

### Backend

| Component | Technology | Version | Purpose |
|-----------|-----------|---------|---------|
| ML Framework | TensorFlow | 2.13.0 | Deep learning |
| ML Framework | Scikit-learn | 1.3.0 | Classical ML |
| Packet Capture | Scapy | 2.5.0 | Network sniffing |
| API | Flask | 2.3.3 | ML API |
| API | Express | 4.18.2 | Backend API |
| WebSocket | Socket.io | 4.6.1 | Real-time updates |
| Data Processing | Pandas | 2.0.3 | Data manipulation |
| Numerics | NumPy | 1.24.3 | Mathematical operations |

### Frontend

| Component | Technology | Version | Purpose |
|-----------|-----------|---------|---------|
| Framework | Next.js | 14.0.0 | React framework |
| UI Library | React | 18.2.0 | UI components |
| Charts | Chart.js | 4.4.0 | Visualizations |
| Styling | Tailwind CSS | 3.3.5 | CSS framework |
| WebSocket | Socket.io-client | 4.6.1 | Real-time data |
| HTTP Client | Axios | 1.5.0 | API requests |

## Deployment Architecture

### Development Mode

```
┌─────────────┐
│  Developer  │
└──────┬──────┘
       │
       ├──→ Terminal 1: Flask ML API (5000)
       ├──→ Terminal 2: Express Backend (3001)
       ├──→ Terminal 3: Next.js Frontend (3000)
       └──→ Terminal 4: Live Detection
```

### Production Mode

```
┌──────────────────┐
│  Load Balancer   │
└────┬────┬────┬───┘
     │    │    │
     ▼    ▼    ▼
┌────────┐┌────────┐┌────────┐
│ML API  ││Backend ││Frontend│
│Replica ││Replica ││Replica │
└────────┘└────────┘└────────┘
     │         │         │
     └─────────┴─────────┘
               │
               ▼
┌────────────────────────────┐
│    Detection Service       │
│  (Packet Capture + ML)     │
└────────────────────────────┘
```

### Container Architecture (Future)

```
┌────────────────────────────────────┐
│  Docker Compose                    │
│  ┌──────────────┐ ┌──────────────┐│
│  │  ml-api      │ │  backend     ││
│  │  container   │ │  container   ││
│  └──────────────┘ └──────────────┘│
│  ┌──────────────┐ ┌──────────────┐│
│  │  frontend    │ │  detection   ││
│  │  container   │ │  container   ││
│  └──────────────┘ └──────────────┘│
└────────────────────────────────────┘
```

## Security Architecture

### Current Implementation

```
┌─────────────────────────────────────┐
│  Network Traffic                    │
│  (Passive Monitoring)               │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  Detection & Analysis               │
│  (Read-only)                        │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  Alert Generation                   │
│  (Human Review)                     │
└─────────────────────────────────────┘
```

### Future SDN Integration

```
┌─────────────────────────────────────┐
│  Network Traffic                    │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  Detection & Analysis               │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  Automatic Mitigation               │
│  - Block malicious IPs              │
│  - Rate limiting                    │
│  - Traffic shaping                  │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  SDN Controller                     │
│  (POX/Ryu)                          │
└─────────────────────────────────────┘
```

## Scaling Considerations

### Horizontal Scaling

- **ML API**: Stateless, can run multiple instances
- **Backend**: Stateless API, WebSocket clustering
- **Frontend**: Static build, CDN distribution
- **Detection**: Can run multiple capture instances

### Vertical Scaling

- **Memory**: For larger packet buffers
- **CPU**: For faster ML inference
- **Network**: For higher throughput interfaces
- **Storage**: For longer alert history

### Database Integration (Future)

- **Alert Storage**: PostgreSQL/MongoDB
- **Time-series Data**: InfluxDB/TimescaleDB
- **Caching**: Redis
- **Message Queue**: RabbitMQ/Kafka

## Monitoring & Observability

### Current Logging

- Application logs: `python/logs/`
- Backend logs: `backend/logs/`
- Frontend logs: `frontend/logs/`

### Future Additions

- Prometheus metrics
- Grafana dashboards
- Distributed tracing
- Error tracking (Sentry)
- Performance monitoring

## Conclusion

SmartShield uses a well-architected, modular design that:
- Separates concerns across layers
- Enables independent scaling
- Facilitates testing and maintenance
- Supports future enhancements
- Provides clear data flow
- Maintains security boundaries

The architecture is production-ready and can be extended with additional components as needed.


