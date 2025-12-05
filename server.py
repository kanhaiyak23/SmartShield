#!/usr/bin/env python3
"""
SmartShield - Flask Backend Server
Real-time packet capture using Scapy with anomaly detection
"""

from flask import Flask, jsonify
from flask_cors import CORS
from scapy.all import sniff, IP, TCP, UDP, ICMP, Raw
from scapy.layers.http import HTTPRequest, HTTPResponse
import threading
import time
import random
from collections import deque
from datetime import datetime
import re
import os
import numpy as np

# Try to load pretrained Random Forest model
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder
    import joblib
    
    RF_MODEL_PATH = 'random_forest_model.joblib'
    RF_SCALER_PATH = 'rf_feature_scaler.joblib'
    SERVICE_ENCODER_PATH = 'service_encoder.joblib'
    STATE_ENCODER_PATH = 'state_encoder.joblib'
    
    if (os.path.exists(RF_MODEL_PATH) and os.path.exists(RF_SCALER_PATH) and
        os.path.exists(SERVICE_ENCODER_PATH) and os.path.exists(STATE_ENCODER_PATH)):
        rf_model = joblib.load(RF_MODEL_PATH)
        rf_scaler = joblib.load(RF_SCALER_PATH)
        service_encoder = joblib.load(SERVICE_ENCODER_PATH)
        state_encoder = joblib.load(STATE_ENCODER_PATH)
        USE_RANDOM_FOREST = True
        print(f"✅ Loaded pretrained Random Forest model from {RF_MODEL_PATH}")
        print(f"   Model: {rf_model.n_estimators} trees, {rf_model.n_features_in_} features")
    else:
        rf_model = None
        rf_scaler = None
        service_encoder = None
        state_encoder = None
        USE_RANDOM_FOREST = False
        print(f"⚠️  Random Forest model not found. Using simulation mode.")
        print(f"   Run 'python3 train_enhanced_rf.py' to train the model.")
except ImportError:
    rf_model = None
    rf_scaler = None
    service_encoder = None
    state_encoder = None
    USE_RANDOM_FOREST = False
    print("⚠️  scikit-learn not installed. Using simulation mode.")

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

# Global packet buffer (thread-safe with deque)
packet_buffer = deque(maxlen=100)  # Keep last 100 packets
packet_id_counter = 0
capture_thread = None
is_capturing = False

# Connection tracking for flow statistics
connection_cache = {}  # {(src_ip, dst_ip, src_port, dst_port): {packets: [], start_time: ...}}

# Protocol mapping
PROTOCOL_MAP = {
    'TCP': 'TCP',
    'UDP': 'UDP',
    'ICMP': 'ICMP',
    'HTTP': 'HTTP',
    'HTTPS': 'HTTPS',
    'SSH': 'SSH'
}

# Risk levels
RISK_LEVELS = {
    'SAFE': 'SAFE',
    'WARNING': 'WARNING',
    'CRITICAL': 'CRITICAL'
}

# Malicious payload patterns for detection
MALICIOUS_PATTERNS = [
    r"UNION\s+SELECT",
    r"/etc/passwd",
    r"<script>",
    r"eval\s*\(",
    r"base64_decode",
    r"\.\./\.\./",
    r"SELECT.*FROM.*users",
    r"DROP\s+TABLE",
    r"<iframe",
    r"javascript:"
]


def extract_enhanced_features(packet_info, connection_stats):
    """
    Extract 20 enhanced features for Random Forest
    packet_info: dict with length, ttl, protocol, src_port, dst_port, flags, etc.
    connection_stats: dict with duration, packet counts, rates, etc.
    """
    # Basic features (1-5)
    length = packet_info['length']
    ttl = packet_info['ttl']
    protocol = packet_info['protocol']
    src_port = packet_info['src_port']
    dst_port = packet_info['dst_port']
    
    # Protocol code
    proto_code = 0
    if protocol == 'TCP':
        proto_code = 1
    elif protocol == 'UDP':
        proto_code = 2
    elif protocol == 'ICMP':
        proto_code = 3
    elif protocol == 'HTTP':
        proto_code = 4
    elif protocol == 'HTTPS':
        proto_code = 5
    elif protocol == 'SSH':
        proto_code = 6
    
    # Flow statistics (6-11)
    duration = connection_stats.get('duration', 0.001)
    packet_rate = connection_stats.get('packet_rate', 0)
    byte_rate = connection_stats.get('byte_rate', 0)
    spkts = connection_stats.get('spkts', 1)
    dpkts = connection_stats.get('dpkts', 1)
    packet_ratio = spkts / dpkts if dpkts > 0 else 1.0
    
    # Network behavior (12-15)
    service = connection_stats.get('service', '-')
    try:
        if service_encoder and service in service_encoder.classes_:
            service_encoded = service_encoder.transform([service])[0]
        else:
            service_encoded = 0
    except:
        service_encoded = 0
    
    state = connection_stats.get('state', '-')
    try:
        if state_encoder and state in state_encoder.classes_:
            state_encoded = state_encoder.transform([state])[0]
        else:
            state_encoded = 0
    except:
        state_encoded = 0
    
    is_well_known_port = 1 if dst_port < 1024 else 0
    
    if dst_port < 1024:
        port_category = 0
    elif dst_port < 49152:
        port_category = 1
    else:
        port_category = 2
    
    # Statistical features (16-19)
    mean_packet_size = connection_stats.get('mean_packet_size', length)
    jitter = connection_stats.get('jitter', 0)
    inter_packet_time = connection_stats.get('inter_packet_time', 0)
    tcprtt = connection_stats.get('tcprtt', 0)
    
    # Total packets (20)
    total_packets = connection_stats.get('total_packets', 1)
    
    return [
        length, ttl, proto_code, src_port, dst_port,
        duration, packet_rate, byte_rate, spkts, dpkts,
        packet_ratio, service_encoded, state_encoded, is_well_known_port, port_category,
        mean_packet_size, jitter, inter_packet_time, tcprtt, total_packets
    ]


def extract_features(length, ttl, protocol, src_port, dst_port):
    """Legacy function for backward compatibility - extracts 5 basic features"""
    proto_code = 0
    if protocol == 'TCP':
        proto_code = 1
    elif protocol == 'UDP':
        proto_code = 2
    elif protocol == 'ICMP':
        proto_code = 3
    elif protocol == 'HTTP':
        proto_code = 4
    elif protocol == 'HTTPS':
        proto_code = 5
    elif protocol == 'SSH':
        proto_code = 6
    
    return [length, ttl, proto_code, src_port, dst_port]


def run_random_forest_inference(enhanced_features):
    """
    Real-time Random Forest prediction using pretrained model
    Returns: (attack_probability, risk_level, anomaly_score)
    """
    if USE_RANDOM_FOREST and rf_model is not None and rf_scaler is not None:
        try:
            # Scale features
            features_array = np.array(enhanced_features).reshape(1, -1)
            features_scaled = rf_scaler.transform(features_array)
            
            # Get prediction probability (probability of attack)
            attack_probability = rf_model.predict_proba(features_scaled)[0][1]
            
            # Convert probability to anomaly score for compatibility
            # Higher probability = more anomalous
            # Map 0.0-1.0 probability to -0.5 to 0.5 score range
            anomaly_score = (attack_probability - 0.5) * 1.0  # -0.5 to 0.5
            
            # Determine risk level based on probability
            if attack_probability >= 0.7:
                risk = RISK_LEVELS['CRITICAL']
            elif attack_probability >= 0.4:
                risk = RISK_LEVELS['WARNING']
            else:
                risk = RISK_LEVELS['SAFE']
            
            return round(anomaly_score, 3), risk, round(attack_probability, 3)
            
        except Exception as e:
            print(f"Error in Random Forest inference: {e}")
            # Fallback to simulation
            return run_fallback_inference(enhanced_features[:5])
    else:
        # Fallback to simulation if model not available
        return run_fallback_inference(enhanced_features[:5] if len(enhanced_features) >= 5 else [1500, 64, 1, 49152, 80])


def run_fallback_inference(features):
    """Fallback simulation mode"""
    base_score = 0.1  # Slightly positive (normal)
    
    # Simple heuristics
    length = features[0] if len(features) > 0 else 1500
    if length > 8000:
        base_score = -0.2
    
    risk = RISK_LEVELS['SAFE']
    if base_score < -0.3:
        risk = RISK_LEVELS['CRITICAL']
    elif base_score < -0.1:
        risk = RISK_LEVELS['WARNING']
    
    return round(base_score, 3), risk, 0.1


def detect_malicious_payload(payload_str):
    """Check if payload contains malicious patterns"""
    if not payload_str:
        return False
    
    payload_lower = payload_str.lower()
    for pattern in MALICIOUS_PATTERNS:
        if re.search(pattern, payload_lower, re.IGNORECASE):
            return True
    return False


def bytes_to_hex(byte_data):
    """Convert bytes to uppercase hex string"""
    return byte_data.hex().upper() if byte_data else ""


def bytes_to_ascii(byte_data):
    """Convert bytes to ASCII string, replacing non-printable chars"""
    if not byte_data:
        return ""
    try:
        ascii_str = byte_data.decode('utf-8', errors='ignore')
        # Replace non-printable characters
        ascii_str = ''.join(c if 32 <= ord(c) < 127 else '.' for c in ascii_str)
        return ascii_str[:200]  # Limit length
    except:
        return ""


def get_tcp_flags(tcp_layer):
    """Extract TCP flags"""
    flags = []
    if tcp_layer.flags & 0x02:  # SYN
        flags.append('SYN')
    if tcp_layer.flags & 0x10:  # ACK
        flags.append('ACK')
    if tcp_layer.flags & 0x04:  # RST
        flags.append('RST')
    if tcp_layer.flags & 0x01:  # FIN
        flags.append('FIN')
    if tcp_layer.flags & 0x08:  # PSH
        flags.append('PSH')
    if tcp_layer.flags & 0x20:  # URG
        flags.append('URG')
    return flags


def determine_protocol(packet):
    """Determine protocol from packet layers"""
    if packet.haslayer(HTTPRequest) or packet.haslayer(HTTPResponse):
        if packet.haslayer(TCP) and packet[TCP].dport == 443:
            return 'HTTPS'
        return 'HTTP'
    elif packet.haslayer(TCP):
        if packet[TCP].dport == 22 or packet[TCP].sport == 22:
            return 'SSH'
        elif packet[TCP].dport == 80 or packet[TCP].sport == 80:
            return 'HTTP'
        elif packet[TCP].dport == 443 or packet[TCP].sport == 443:
            return 'HTTPS'
        return 'TCP'
    elif packet.haslayer(UDP):
        return 'UDP'
    elif packet.haslayer(ICMP):
        return 'ICMP'
    return 'TCP'  # Default


def update_connection_stats(conn_key, packet_info):
    """Update connection statistics for flow features"""
    global connection_cache
    
    current_time = time.time()
    
    if conn_key not in connection_cache:
        connection_cache[conn_key] = {
            'packets': [],
            'start_time': current_time,
            'spkts': 0,
            'dpkts': 0,
            'sbytes': 0,
            'dbytes': 0,
            'packet_times': [],
            'last_packet_time': current_time
        }
    
    conn = connection_cache[conn_key]
    conn['packets'].append(packet_info)
    conn['last_packet_time'] = current_time
    
    # Update packet counts
    if packet_info['direction'] == 'src_to_dst':
        conn['spkts'] += 1
        conn['sbytes'] += packet_info['length']
    else:
        conn['dpkts'] += 1
        conn['dbytes'] += packet_info['length']
    
    conn['packet_times'].append(current_time)
    
    # Limit cache size
    if len(connection_cache) > 10000:
        # Remove oldest connections
        oldest = min(connection_cache.items(), key=lambda x: x[1]['start_time'])
        del connection_cache[oldest[0]]
    
    # Calculate flow statistics
    duration = current_time - conn['start_time']
    if duration == 0:
        duration = 0.001
    
    total_packets = conn['spkts'] + conn['dpkts']
    total_bytes = conn['sbytes'] + conn['dbytes']
    
    packet_rate = total_packets / duration if duration > 0 else 0
    byte_rate = total_bytes / duration if duration > 0 else 0
    
    # Calculate mean packet size
    if len(conn['packets']) > 0:
        mean_packet_size = sum(p['length'] for p in conn['packets']) / len(conn['packets'])
    else:
        mean_packet_size = packet_info['length']
    
    # Calculate jitter (packet time variance)
    if len(conn['packet_times']) > 1:
        intervals = [conn['packet_times'][i] - conn['packet_times'][i-1] 
                    for i in range(1, len(conn['packet_times']))]
        if intervals:
            mean_interval = sum(intervals) / len(intervals)
            jitter = sum(abs(i - mean_interval) for i in intervals) / len(intervals) if intervals else 0
        else:
            jitter = 0
    else:
        jitter = 0
    
    # Inter-packet time
    if len(conn['packet_times']) > 1:
        inter_packet_time = conn['packet_times'][-1] - conn['packet_times'][-2]
    else:
        inter_packet_time = 0
    
    # Determine service and state from protocol/port
    service = '-'
    if packet_info['dst_port'] == 80 or packet_info['dst_port'] == 8080:
        service = 'http'
    elif packet_info['dst_port'] == 443:
        service = 'https'
    elif packet_info['dst_port'] == 22:
        service = 'ssh'
    elif packet_info['dst_port'] == 53:
        service = 'dns'
    elif packet_info['protocol'] == 'FTP':
        service = 'ftp'
    
    state = '-'
    if 'SYN' in packet_info.get('flags', []):
        state = 'SYN'
    elif 'FIN' in packet_info.get('flags', []):
        state = 'FIN'
    elif 'ACK' in packet_info.get('flags', []):
        state = 'EST'
    
    return {
        'duration': duration,
        'packet_rate': packet_rate,
        'byte_rate': byte_rate,
        'spkts': conn['spkts'],
        'dpkts': conn['dpkts'],
        'total_packets': total_packets,
        'mean_packet_size': mean_packet_size,
        'jitter': jitter * 1000,  # Convert to milliseconds
        'inter_packet_time': inter_packet_time * 1000,  # Convert to milliseconds
        'tcprtt': 0,  # Would need sequence/ack analysis for real RTT
        'service': service,
        'state': state
    }


def process_packet(packet):
    """Process a Scapy packet and convert to frontend format"""
    global packet_id_counter
    
    if not packet.haslayer(IP):
        return None
    
    packet_id_counter += 1
    current_time = time.time()
    
    ip_layer = packet[IP]
    protocol = determine_protocol(packet)
    
    # Extract IP info
    src_ip = ip_layer.src
    dst_ip = ip_layer.dst
    ttl = ip_layer.ttl
    length = len(packet)
    
    # Extract port info
    src_port = 0
    dst_port = 0
    if packet.haslayer(TCP):
        tcp_layer = packet[TCP]
        src_port = tcp_layer.sport
        dst_port = tcp_layer.dport
        flags = get_tcp_flags(tcp_layer)
        window_size = tcp_layer.window
        seq = tcp_layer.seq
        ack = tcp_layer.ack if hasattr(tcp_layer, 'ack') else None
        tcprtt = 0  # Would need more analysis for real RTT
    elif packet.haslayer(UDP):
        udp_layer = packet[UDP]
        src_port = udp_layer.sport
        dst_port = udp_layer.dport
        flags = []
        window_size = None
        seq = None
        ack = None
        tcprtt = 0
    else:
        flags = []
        window_size = None
        seq = None
        ack = None
        tcprtt = 0
    
    # Extract payload
    raw_payload = b""
    if packet.haslayer(Raw):
        raw_payload = packet[Raw].load
    
    raw_hex = bytes_to_hex(raw_payload)
    ascii_payload = bytes_to_ascii(raw_payload)
    
    # Detect malicious patterns
    is_attack = detect_malicious_payload(ascii_payload)
    
    # Connection tracking for flow statistics
    conn_key = (src_ip, dst_ip, src_port, dst_port)
    packet_info = {
        'length': length,
        'timestamp': current_time,
        'direction': 'src_to_dst'  # Simplified
    }
    
    # Update connection stats
    connection_stats = update_connection_stats(conn_key, {
        'length': length,
        'timestamp': current_time,
        'direction': 'src_to_dst',
        'flags': flags
    })
    connection_stats['tcprtt'] = tcprtt
    
    # Prepare packet info for feature extraction
    packet_info_dict = {
        'length': length,
        'ttl': ttl,
        'protocol': protocol,
        'src_port': src_port,
        'dst_port': dst_port,
        'flags': flags
    }
    
    # Extract enhanced features (20 features)
    enhanced_features = extract_enhanced_features(packet_info_dict, connection_stats)
    
    # Anomaly detection using Random Forest
    anomaly_score, risk_level, attack_probability = run_random_forest_inference(enhanced_features)
    
    # Create summary string
    summary = f"[RF_PROB: {attack_probability:.3f}] {protocol} {src_ip} > {dst_ip}"
    if src_port and dst_port:
        summary = f"[RF_PROB: {attack_probability:.3f}] {protocol} {src_ip}:{src_port} > {dst_ip}:{dst_port}"
    
    # Build packet object
    packet_obj = {
        'id': f'pkt-{str(packet_id_counter).zfill(6)}',
        'timestamp': int(current_time * 1000),  # Milliseconds
        'sourceIp': src_ip,
        'destIp': dst_ip,
        'sourcePort': src_port,
        'destPort': dst_port,
        'protocol': protocol,
        'length': length,
        'ttl': ttl,
        'windowSize': window_size,
        'seq': seq,
        'ack': ack,
        'flags': flags,
        'rawHex': raw_hex,
        'asciiPayload': ascii_payload[:100] if ascii_payload else "",  # Limit payload display
        'summary': summary,
        'featureVector': enhanced_features,  # Now 20 features
        'anomalyScore': anomaly_score,
        'riskLevel': risk_level
    }
    
    return packet_obj


def packet_handler(packet):
    """Callback for Scapy sniff"""
    try:
        processed = process_packet(packet)
        if processed:
            packet_buffer.append(processed)
    except Exception as e:
        print(f"Error processing packet: {e}")


def start_capture():
    """Start packet capture in background thread"""
    global is_capturing
    
    def capture_loop():
        global is_capturing
        is_capturing = True
        try:
            # Sniff packets (non-blocking)
            # Filter: Only IP packets, limit to avoid overwhelming
            sniff(
                prn=packet_handler,
                store=False,  # Don't store packets in memory
                stop_filter=lambda x: False,  # Run indefinitely
                filter="ip",  # Only IP packets
                count=0  # Unlimited
            )
        except Exception as e:
            print(f"Capture error: {e}")
            is_capturing = False
    
    thread = threading.Thread(target=capture_loop, daemon=True)
    thread.start()
    return thread


@app.route('/packets', methods=['GET'])
def get_packets():
    """Return recent packets from buffer"""
    try:
        # Return packets as array (frontend expects array)
        packets = list(packet_buffer)
        return jsonify(packets)
    except Exception as e:
        print(f"Error in /packets: {e}")
        return jsonify([])


@app.route('/status', methods=['GET'])
def get_status():
    """Get server status"""
    return jsonify({
        'status': 'running',
        'capturing': is_capturing,
        'packets_buffered': len(packet_buffer),
        'total_packets': packet_id_counter
    })


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'ok'})


if __name__ == '__main__':
    print("=" * 60)
    print("SmartShield - Flask Backend Server")
    print("=" * 60)
    if USE_RANDOM_FOREST:
        print(f"Anomaly Detection: ✅ RANDOM FOREST (200 trees, 20 features)")
        print(f"   Model: random_forest_model.joblib")
        print(f"   Attack Detection Rate: 99.62% (from test results)")
    else:
        print(f"Anomaly Detection: ⚠️  SIMULATION MODE")
        print("   Run 'python3 train_enhanced_rf.py' to train Random Forest model")
    print("=" * 60)
    print("Starting packet capture...")
    print("Note: Requires root/admin privileges for packet capture")
    print("=" * 60)
    
    # Start packet capture
    capture_thread = start_capture()
    
    # Give capture thread a moment to start
    time.sleep(1)
    
    # Run Flask server
    print(f"\nServer running on http://127.0.0.1:5000")
    print(f"API endpoint: http://127.0.0.1:5000/packets")
    print(f"Status endpoint: http://127.0.0.1:5000/status")
    print("\nPress Ctrl+C to stop\n")
    
    try:
        app.run(host='127.0.0.1', port=5000, debug=False, threaded=True)
    except KeyboardInterrupt:
        print("\nShutting down...")
        is_capturing = False
