#!/usr/bin/env python3
"""
Train Enhanced Random Forest model on UNSW-NB15 dataset
Uses 15-20 features for better attack detection
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import joblib
import os
import sys

def load_unsw_dataset(file_paths):
    """Load and combine multiple UNSW-NB15 files"""
    print("Loading UNSW-NB15 dataset...")
    dfs = []
    
    for file_path in file_paths:
        if not os.path.exists(file_path):
            print(f"⚠️  File not found: {file_path}")
            continue
            
        try:
            print(f"  Loading {file_path}...")
            # Check if file has headers
            sample = pd.read_csv(file_path, nrows=1)
            if 'id' in sample.columns or 'dur' in sample.columns:
                # Has headers
                df = pd.read_csv(file_path, low_memory=False)
            else:
                # No headers - use column mapping
                df = pd.read_csv(file_path, header=None, low_memory=False)
                column_names = [
                    'srcip', 'sport', 'dstip', 'dsport', 'proto', 'state', 'dur', 'sbytes', 'dbytes',
                    'sttl', 'dttl', 'sloss', 'dloss', 'service', 'sload', 'dload', 'spkts', 'dpkts',
                    'swin', 'dwin', 'stcpb', 'dtcpb', 'smeansz', 'dmeansz', 'trans_depth', 'res_bdy_len',
                    'sjit', 'djit', 'stime', 'ltime', 'sintpkt', 'dintpkt', 'tcprtt', 'synack', 'ackdat',
                    'is_sm_ips_ports', 'ct_state_ttl', 'ct_flw_http_mthd', 'is_ftp_login', 'ct_ftp_cmd',
                    'ct_srv_src', 'ct_srv_dst', 'ct_dst_ltm', 'ct_src_ltm', 'ct_src_dport_ltm',
                    'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'attack_cat', 'label'
                ]
                if len(df.columns) <= len(column_names):
                    df.columns = column_names[:len(df.columns)]
                else:
                    df.columns = column_names + [f'col_{i}' for i in range(len(column_names), len(df.columns))]
            
            dfs.append(df)
            print(f"    ✅ Loaded {len(df)} records")
        except Exception as e:
            print(f"    ❌ Error loading {file_path}: {e}")
            continue
    
    if not dfs:
        return None
    
    # Combine all dataframes
    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"\n✅ Total records loaded: {len(combined_df)}")
    return combined_df


def extract_enhanced_features(df):
    """Extract 15-20 enhanced features from UNSW-NB15 dataset"""
    print("\nExtracting enhanced features...")
    
    features = []
    labels = []
    
    # Protocol mapping
    proto_map = {'tcp': 1, 'udp': 2, 'icmp': 3, 'http': 4, 'https': 5, 'ssl': 5, 'ssh': 6}
    
    # Service encoding
    service_encoder = LabelEncoder()
    if 'service' in df.columns:
        services = df['service'].fillna('-').astype(str)
        service_encoder.fit(services.unique())
    
    # State encoding
    state_encoder = LabelEncoder()
    if 'state' in df.columns:
        states = df['state'].fillna('-').astype(str)
        state_encoder.fit(states.unique())
    
    for idx, row in df.iterrows():
        try:
            # Feature 1-5: Basic features
            sbytes = float(row.get('sbytes', 0) or 0)
            dbytes = float(row.get('dbytes', 0) or 0)
            length = sbytes + dbytes if (sbytes + dbytes) > 0 else 1500
            
            ttl = float(row.get('sttl', 64) or 64)  # Source TTL
            
            proto_str = str(row.get('proto', 'tcp')).lower().strip()
            proto_code = next((v for k, v in proto_map.items() if k in proto_str), 1)
            
            src_port = max(0, min(65535, int(float(row.get('sport', 0) or 0))))
            dst_port = max(0, min(65535, int(float(row.get('dsport', 0) or 0))))
            
            if src_port == 0 and dst_port == 0:
                continue
            
            # Feature 6-11: Flow statistics
            dur = float(row.get('dur', 0) or 0)
            if dur == 0:
                dur = 0.001  # Avoid division by zero
            
            spkts = float(row.get('spkts', 0) or 0)
            dpkts = float(row.get('dpkts', 0) or 0)
            total_packets = spkts + dpkts
            
            packet_rate = total_packets / dur if dur > 0 else 0
            byte_rate = length / dur if dur > 0 else 0
            packet_ratio = spkts / dpkts if dpkts > 0 else 1.0
            
            # Feature 12-15: Network behavior
            service = str(row.get('service', '-')).lower()
            service_encoded = service_encoder.transform([service])[0] if service in service_encoder.classes_ else 0
            
            state = str(row.get('state', '-'))
            state_encoded = state_encoder.transform([state])[0] if state in state_encoder.classes_ else 0
            
            is_well_known_port = 1 if dst_port < 1024 else 0
            
            # Port category: 0=well-known, 1=registered, 2=ephemeral
            if dst_port < 1024:
                port_category = 0
            elif dst_port < 49152:
                port_category = 1
            else:
                port_category = 2
            
            # Feature 16-19: Statistical features
            smeansz = float(row.get('smeansz', 0) or 0)
            dmeansz = float(row.get('dmeansz', 0) or 0)
            mean_packet_size = (smeansz + dmeansz) / 2 if (smeansz + dmeansz) > 0 else length
            
            sjit = float(row.get('sjit', 0) or 0)
            djit = float(row.get('djit', 0) or 0)
            jitter = sjit + djit
            
            sintpkt = float(row.get('sintpkt', 0) or 0)
            dintpkt = float(row.get('dintpkt', 0) or 0)
            inter_packet_time = sintpkt + dintpkt
            
            tcprtt = float(row.get('tcprtt', 0) or 0)
            
            # Build feature vector (20 features)
            feature_vector = [
                length,              # 1. Total bytes
                ttl,                 # 2. TTL
                proto_code,          # 3. Protocol code
                src_port,            # 4. Source port
                dst_port,            # 5. Destination port
                dur,                 # 6. Duration
                packet_rate,         # 7. Packet rate
                byte_rate,           # 8. Byte rate
                spkts,               # 9. Source packets
                dpkts,               # 10. Destination packets
                packet_ratio,        # 11. Packet ratio
                service_encoded,     # 12. Service type
                state_encoded,       # 13. Connection state
                is_well_known_port,  # 14. Well-known port flag
                port_category,       # 15. Port category
                mean_packet_size,    # 16. Mean packet size
                jitter,              # 17. Jitter
                inter_packet_time,   # 18. Inter-packet time
                tcprtt,              # 19. TCP RTT
                total_packets        # 20. Total packets
            ]
            
            features.append(feature_vector)
            
            # Label
            label_val = row.get('label', 0)
            if pd.isna(label_val):
                label = 0
            elif isinstance(label_val, str):
                label = 1 if label_val.lower().strip() in ['attack', '1', 'true', 'yes'] else 0
            else:
                label = int(float(label_val))
            
            labels.append(label)
            
        except Exception as e:
            continue
    
    print(f"✅ Extracted {len(features)} feature vectors")
    normal_count = sum(1 for l in labels if l == 0)
    attack_count = sum(1 for l in labels if l == 1)
    print(f"   Normal samples: {normal_count} ({100*normal_count/len(labels):.1f}%)")
    print(f"   Attack samples: {attack_count} ({100*attack_count/len(labels):.1f}%)")
    
    return np.array(features), np.array(labels), service_encoder, state_encoder


def train_random_forest():
    """Train Random Forest classifier on UNSW-NB15"""
    print("=" * 70)
    print("Enhanced Random Forest Training on UNSW-NB15")
    print("=" * 70)
    
    # Load dataset from multiple files
    data_files = [
        'data/UNSW-NB15_1.csv',
        'data/UNSW-NB15_2.csv',
        'data/UNSW-NB15_3.csv',
        'data/UNSW-NB15_4.csv',
        'data/UNSW_NB15_training-set.csv'
    ]
    
    df = load_unsw_dataset(data_files)
    if df is None:
        print("❌ No data loaded!")
        return None, None, None, None
    
    # Extract features
    X, y, service_encoder, state_encoder = extract_enhanced_features(df)
    
    if len(X) == 0:
        print("❌ No features extracted!")
        return None, None, None, None
    
    print(f"\n{'='*70}")
    print("Training Configuration")
    print(f"{'='*70}")
    print(f"Total samples: {len(X)}")
    print(f"Features: {X.shape[1]}")
    print(f"Normal samples: {(y == 0).sum()}")
    print(f"Attack samples: {(y == 1).sum()}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTrain set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Scale features
    print("\nScaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest
    print("\nTraining Random Forest classifier...")
    print("  Parameters: n_estimators=200, max_depth=20, class_weight='balanced'")
    
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',  # Handle imbalanced data
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    rf_model.fit(X_train_scaled, y_train)
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    y_pred = rf_model.predict(X_test_scaled)
    y_pred_proba = rf_model.predict_proba(X_test_scaled)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    print(f"\n{'='*70}")
    print("TEST SET RESULTS")
    print(f"{'='*70}")
    print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"Recall:    {recall:.4f} ({recall*100:.2f}%)")
    print(f"F1-Score:  {f1:.4f} ({f1*100:.2f}%)")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:")
    print(f"                  Predicted")
    print(f"                Normal  Attack")
    tn, fp, fn, tp = cm.ravel()
    print(f"Actual Normal    {tn:5d}  {fp:5d}")
    print(f"      Attack     {fn:5d}  {tp:5d}")
    
    # Feature importance
    print(f"\nTop 10 Most Important Features:")
    feature_names = [
        'length', 'ttl', 'proto', 'src_port', 'dst_port',
        'duration', 'packet_rate', 'byte_rate', 'spkts', 'dpkts',
        'packet_ratio', 'service', 'state', 'well_known_port', 'port_category',
        'mean_packet_size', 'jitter', 'inter_packet_time', 'tcp_rtt', 'total_packets'
    ]
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1][:10]
    for i, idx in enumerate(indices, 1):
        print(f"  {i:2d}. {feature_names[idx]:20s}: {importances[idx]:.4f}")
    
    # Cross-validation
    print(f"\nPerforming 5-fold cross-validation...")
    cv_scores = cross_val_score(rf_model, X_train_scaled, y_train, cv=5, scoring='f1')
    print(f"  CV F1-Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Save model and encoders
    model_path = 'random_forest_model.joblib'
    scaler_path = 'rf_feature_scaler.joblib'
    service_encoder_path = 'service_encoder.joblib'
    state_encoder_path = 'state_encoder.joblib'
    
    print(f"\nSaving model to {model_path}...")
    joblib.dump(rf_model, model_path)
    
    print(f"Saving scaler to {scaler_path}...")
    joblib.dump(scaler, scaler_path)
    
    print(f"Saving service encoder to {service_encoder_path}...")
    joblib.dump(service_encoder, service_encoder_path)
    
    print(f"Saving state encoder to {state_encoder_path}...")
    joblib.dump(state_encoder, state_encoder_path)
    
    print(f"\n{'='*70}")
    print("✅ Model training complete!")
    print(f"{'='*70}")
    print(f"   Model: {model_path}")
    print(f"   Scaler: {scaler_path}")
    print(f"   Service Encoder: {service_encoder_path}")
    print(f"   State Encoder: {state_encoder_path}")
    print(f"\nThe model is ready for real-time inference.")
    
    return rf_model, scaler, service_encoder, state_encoder


if __name__ == '__main__':
    train_random_forest()

