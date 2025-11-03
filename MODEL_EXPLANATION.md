# How SmartShield Models Work

## 🔍 Understanding Attack Detection: Single Packet vs Training Data

### Can a Single Packet Detect an Attack?

**Short Answer: YES, but with important caveats**

#### How It Works:

1. **Single Packet Analysis:**
   - Each captured packet is **independently analyzed**
   - The model extracts features from that **single packet**
   - These features are compared against **patterns learned from training**
   - The model makes a prediction based on that **one packet**

2. **What Models Learn During Training:**

   **Random Forest:**
   - Learns patterns from **thousands of training examples**
   - Each training row is a single packet/flow with features
   - Model learns: "Packets with these feature combinations are attacks"
   - Example: "Packets with high packet_rate + specific ports + specific flags = Port Scan"
   
   **Isolation Forest:**
   - Learns **normal traffic patterns** from training data
   - Flags anything that **deviates** from learned normal patterns
   - Works on single packets: "Does this packet look normal?"
   
   **Autoencoder:**
   - Learns to **reconstruct normal traffic** from training
   - Calculates reconstruction error for each packet
   - High error = packet doesn't match normal patterns = potential attack

#### Example Flow:

```
College WiFi Packet Captured:
┌─────────────────────────────────────┐
│ Packet: 192.168.1.100 → 8.8.8.8    │
│ Protocol: TCP, Port: 80, Size: 64   │
└─────────────────────────────────────┘
           ↓
   Extract Features:
┌─────────────────────────────────────┐
│ Features: [src_ip, dst_ip, protocol,│
│            src_port, dst_port,      │
│            packet_length, flags,    │
│            rate, duration, ...]     │
└─────────────────────────────────────┘
           ↓
   Compare with Training Patterns:
┌─────────────────────────────────────┐
│ Model Checks:                       │
│ - "Do these features match Normal?"│
│ - "Do these features match Fuzzers?"│
│ - "Do these features match DoS?"   │
│ - etc.                              │
└─────────────────────────────────────┘
           ↓
   Prediction:
┌─────────────────────────────────────┐
│ Attack Type: Fuzzers                │
│ Confidence: 0.85                    │
│ Anomaly: True                       │
└─────────────────────────────────────┘
```

### Why Training Data Matters:

**The model doesn't see your college WiFi during training!**

Instead:
- Model learns **patterns** from UNSW-NB15 training dataset
- UNSW-NB15 has **82,000+ training examples** of different attack types
- Model learns: "Packets with X,Y,Z features = Attack Type A"
- When your college WiFi packet has features X,Y,Z → Model predicts Attack Type A

**Analogy:**
- Like learning to recognize cats from millions of cat photos
- Then you see a new cat photo → you recognize it as a cat
- Model learned attack patterns from training → recognizes attacks in new packets

---

## 🎯 UNSW-NB15 Attack Types

The model is trained to detect these attack categories:

### Attack Categories:

1. **Normal** - Legitimate network traffic
2. **Generic** - Generic attacks
3. **Exploits** - Exploitation attempts
4. **Fuzzers** - Fuzzing attacks (automated testing)
5. **DoS** - Denial of Service attacks
6. **Reconnaissance** - Scanning/probing attacks
7. **Analysis** - Traffic analysis attacks
8. **Backdoor** - Backdoor installations
9. **Shellcode** - Shellcode injection
10. **Worms** - Worm propagation

### Attack Type Mapping:

```
UNSW-NB15 attack_cat → Dashboard Display
─────────────────────────────────────────────
Normal              → Normal / No Attack
Generic             → Generic Attack
Exploits            → Exploits
Fuzzers             → Fuzzers
DoS                 → DoS Attack
Reconnaissance      → Reconnaissance
Analysis            → Analysis Attack
Backdoor            → Backdoor
Shellcode           → Shellcode
Worms               → Worms
```

---

## ⚠️ Current Problem: Why All Packets Show "High" Severity

### Issue #1: Model Training Problem

**Current Training:**
- Random Forest trained on **binary label** (0=Normal, 1=Attack)
- Cannot distinguish between attack types
- Always predicts same class → same confidence

**What Should Happen:**
- Random Forest should be trained on **attack_cat** (10 categories)
- Can classify: Normal, Generic, Exploits, Fuzzers, DoS, etc.

### Issue #2: Composite Score Calculation

**Current Formula (WRONG):**
```python
composite_score = confidence × 0.4 + if_anomaly × 0.3 + ae_anomaly × 0.3
```

**Problem:**
- If confidence = 0.75 (always same)
- IF anomaly = True (flagging all packets)
- AE anomaly = True (flagging all packets)
- Score = 0.75 × 0.4 + 1 × 0.3 + 1 × 0.3 = 0.30 + 0.30 + 0.30 = **0.90**
- 0.90 > 0.8 → Always "High" severity!

**Why Isolation Forest Flags Everything:**
- Trained on UNSW-NB15 normal patterns
- College WiFi traffic might be **different** from UNSW-NB15
- Model thinks: "This doesn't look like UNSW-NB15 normal → ANOMALY"
- But it's actually just **different normal traffic**!

### Issue #3: Severity Threshold Too Low

**Current Thresholds:**
- Score > 0.8 → High
- Score > 0.6 → Medium
- Score > 0.4 → Low

**Problem:**
- Composite score calculation causes most packets to score > 0.8
- Should adjust thresholds or fix score calculation

---

## 🔧 Solutions Needed

1. **Retrain Random Forest on attack_cat** (not just binary label)
2. **Fix composite score calculation** - use anomaly scores, not binary flags
3. **Adjust Isolation Forest contamination** - might be too sensitive
4. **Calibrate thresholds** - college WiFi is different from training data
5. **Add confidence-based severity** - lower confidence = lower severity

---

## 📊 Model Learning Process

### Training Phase (One-time):

```
UNSW-NB15 Training Dataset (82,000+ rows)
├── Each row = One packet/flow with features
├── Features: IPs, ports, protocol, packet size, etc.
├── Label: attack_cat (Normal, Generic, Exploits, ...)
│
↓ Model Training

Random Forest:
├── Learns decision trees from training data
├── Each tree asks: "If feature X > threshold, then..."
├── 100 trees vote on: "What attack type is this?"
└── Result: Can classify new packets into 10 categories

Isolation Forest:
├── Learns what "normal" looks like from training
├── Flags anything that doesn't match normal patterns
└── Returns: anomaly score (lower = more anomalous)

Autoencoder:
├── Learns to reconstruct normal traffic
├── If reconstruction error is high → anomaly
└── Returns: reconstruction error
```

### Detection Phase (Real-time):

```
Live College WiFi Packet
├── Extract features from packet
├── Preprocess (normalize, encode)
│
↓ Model Prediction

Random Forest: "This looks like Fuzzers (confidence: 0.85)"
Isolation Forest: "This is anomalous (score: -0.3)"
Autoencoder: "Can't reconstruct this well (error: 0.15)"
│
↓ Combine Results

Composite Score = 0.85 × 0.4 + (anomaly scores) = 0.65
Attack Type = "Fuzzers"
Severity = "Medium" (score between 0.6-0.8)
```

---

## 🎓 Key Takeaways

1. **Single packets CAN be analyzed** - each packet is independent
2. **Models learn patterns from training** - not your specific WiFi
3. **Training data matters** - UNSW-NB15 has 82,000+ examples
4. **Attack types come from attack_cat** - 10 categories in UNSW-NB15
5. **Current problem** - model trained wrong, scoring wrong, thresholds wrong

---

**Next Steps:** Fix model training and composite score calculation!


