#!/usr/bin/env python3
"""
Train Isolation Forest model on UNSW-NB15 real network dataset
UNSW-NB15 is a comprehensive network intrusion detection dataset
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
import os
import urllib.request
import sys

def download_unsw_nb15(output_dir='data'):
    """Download UNSW-NB15 dataset"""
    os.makedirs(output_dir, exist_ok=True)
    
    train_file = os.path.join(output_dir, 'UNSW_NB15_training-set.csv')
    
    # Check if files already exist
    if os.path.exists(train_file):
        print(f"✅ Dataset file already exists: {train_file}")
        return train_file
    
    print("=" * 60)
    print("UNSW-NB15 Dataset Download")
    print("=" * 60)
    print("\nThe UNSW-NB15 dataset is large (~500MB+) and requires manual download.")
    print("\nPlease download from one of these sources:")
    print("1. Official: https://research.unsw.edu.au/projects/unsw-nb15-dataset")
    print("2. Kaggle: https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15")
    print("3. GitHub: Search for 'UNSW-NB15' repositories")
    print(f"\nPlace the CSV file(s) in: {os.path.abspath(output_dir)}/")
    print("Required file: UNSW_NB15_training-set.csv")
    print("=" * 60)
    
    # Try to download from a known mirror
    mirrors = [
        'https://github.com/UNSWComputing/rl_nids/raw/master/data/UNSW_NB15_training-set.csv',
    ]
    
    for mirror_url in mirrors:
        try:
            print(f"\nAttempting download from mirror...")
            print(f"URL: {mirror_url}")
            urllib.request.urlretrieve(mirror_url, train_file)
            if os.path.exists(train_file) and os.path.getsize(train_file) > 1000:
                print(f"✅ Downloaded training set to {train_file}")
                return train_file
        except Exception as e:
            print(f"⚠️  Mirror failed: {e}")
            continue
    
    print("\n⚠️  Automatic download failed.")
    print("Please download manually and place in data/ directory")
    return None


def load_unsw_nb15(file_path):
    """Load and process UNSW-NB15 dataset"""
    print(f"\nLoading dataset from {file_path}...")
    
    try:
        # UNSW-NB15 CSV files don't have headers - use column indices
        # Column mapping based on NUSW-NB15_features.csv:
        # 0: srcip, 1: sport, 2: dstip, 3: dsport, 4: proto, 7: sbytes, 8: dbytes, 47: label
        print("Reading CSV file (this may take a moment)...")
        print("Note: UNSW-NB15 files don't have headers, using column indices")
        
        # Read without headers
        df = pd.read_csv(file_path, header=None, low_memory=False, nrows=100000)  # Limit to first 100k rows for speed
        
        # Set column names based on UNSW-NB15 structure
        # Based on features file: srcip(0), sport(1), dstip(2), dsport(3), proto(4), state(5), dur(6), sbytes(7), dbytes(8), ...
        column_names = [
            'srcip', 'sport', 'dstip', 'dsport', 'proto', 'state', 'dur', 'sbytes', 'dbytes',
            'sttl', 'dttl', 'sloss', 'dloss', 'service', 'sload', 'dload', 'spkts', 'dpkts',
            'swin', 'dwin', 'stcpb', 'dtcpb', 'smeansz', 'dmeansz', 'trans_depth', 'res_bdy_len',
            'sjit', 'djit', 'stime', 'ltime', 'sintpkt', 'dintpkt', 'tcprtt', 'synack', 'ackdat',
            'is_sm_ips_ports', 'ct_state_ttl', 'ct_flw_http_mthd', 'is_ftp_login', 'ct_ftp_cmd',
            'ct_srv_src', 'ct_srv_dst', 'ct_dst_ltm', 'ct_src_ltm', 'ct_src_dport_ltm',
            'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'attack_cat', 'label'
        ]
        
        # Use available columns
        num_cols = len(df.columns)
        if num_cols <= len(column_names):
            df.columns = column_names[:num_cols]
        else:
            # If more columns, use first part of names and add numeric for rest
            df.columns = column_names + [f'col_{i}' for i in range(len(column_names), num_cols)]
        
        print(f"✅ Loaded {len(df)} records")
        print(f"   Columns: {len(df.columns)}")
        print(f"   Key columns: sport={1 if 'sport' in df.columns else 'N/A'}, dsport={3 if 'dsport' in df.columns else 'N/A'}, proto={4 if 'proto' in df.columns else 'N/A'}")
        
        return df
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return None


def extract_features_from_unsw(df):
    """
    Extract our 5 features from UNSW-NB15 dataset
    Features: [length, ttl, proto_code, src_port, dst_port]
    """
    print("\nExtracting features from UNSW-NB15...")
    
    features = []
    labels = []
    
    # Common UNSW-NB15 column names (dataset may vary)
    possible_length_cols = ['sbytes', 'dbytes', 'totbytes', 'sbytes+dbytes']
    possible_proto_cols = ['proto', 'protocol', 'service']
    possible_sport_cols = ['sport', 'srcport', 'src_port']
    possible_dport_cols = ['dsport', 'dstport', 'dst_port', 'dport']
    possible_label_cols = ['label', 'Label', 'attack', 'Attack']
    
    # Find actual column names
    length_col = None
    proto_col = None
    sport_col = None
    dport_col = None
    label_col = None
    
    for col in df.columns:
        col_lower = col.lower()
        if not length_col and any(x in col_lower for x in ['sbytes', 'dbytes', 'totbytes']):
            if 'totbytes' in col_lower or 'total' in col_lower:
                length_col = col
        if not proto_col and any(x in col_lower for x in ['proto', 'protocol']):
            proto_col = col
        if not sport_col and any(x in col_lower for x in ['sport', 'srcport']):
            sport_col = col
        if not dport_col and any(x in col_lower for x in ['dsport', 'dstport', 'dport']):
            dport_col = col
        if not label_col and any(x in col_lower for x in ['label', 'attack']):
            label_col = col
    
    print(f"Found columns:")
    print(f"  Length: {length_col or 'sbytes+dbytes (calculated)'}")
    print(f"  Protocol: {proto_col or 'NOT FOUND'}")
    print(f"  Source Port: {sport_col or 'NOT FOUND'}")
    print(f"  Dest Port: {dport_col or 'NOT FOUND'}")
    print(f"  Label: {label_col or 'NOT FOUND'}")
    
    # Protocol mapping
    proto_map = {
        'tcp': 1, 'udp': 2, 'icmp': 3, 'http': 4, 'https': 5, 'ssl': 5, 'ssh': 6
    }
    
    for idx, row in df.iterrows():
        try:
            # Length: sum of source and destination bytes
            if length_col and length_col in df.columns and pd.notna(row.get(length_col)):
                length = int(float(row[length_col]))
            elif 'sbytes' in df.columns and 'dbytes' in df.columns:
                sbytes_val = row.get('sbytes', 0)
                dbytes_val = row.get('dbytes', 0)
                sbytes = int(float(sbytes_val)) if pd.notna(sbytes_val) else 0
                dbytes = int(float(dbytes_val)) if pd.notna(dbytes_val) else 0
                length = sbytes + dbytes
                if length == 0:
                    length = 1500  # Default if both are 0
            else:
                length = 1500  # Default
            
            # TTL: UNSW-NB15 doesn't have TTL, use common values based on protocol
            ttl = 64  # Default TTL
            
            # Protocol: map string to code
            if proto_col and proto_col in df.columns and pd.notna(row.get(proto_col)):
                proto_str = str(row[proto_col]).lower().strip()
                proto_code = 0
                for key, val in proto_map.items():
                    if key in proto_str:
                        proto_code = val
                        break
                if proto_code == 0:
                    proto_code = 1  # Default to TCP
            else:
                proto_code = 1  # Default to TCP
            
            # Ports
            if sport_col and sport_col in df.columns and pd.notna(row.get(sport_col)):
                src_port_val = row[sport_col]
                src_port = int(float(src_port_val)) if pd.notna(src_port_val) else 0
            else:
                src_port = 0
            
            if dport_col and dport_col in df.columns and pd.notna(row.get(dport_col)):
                dst_port_val = row[dport_col]
                dst_port = int(float(dst_port_val)) if pd.notna(dst_port_val) else 0
            else:
                dst_port = 0
            
            # Validate ports
            if src_port < 0 or src_port > 65535:
                src_port = 0
            if dst_port < 0 or dst_port > 65535:
                dst_port = 0
            
            # Skip if no valid ports
            if src_port == 0 and dst_port == 0:
                continue
            
            features.append([length, ttl, proto_code, src_port, dst_port])
            
            # Label: 0 = normal, 1 = attack (column 48, index 48)
            if label_col and label_col in df.columns and pd.notna(row.get(label_col)):
                label_val = row[label_col]
                if isinstance(label_val, str):
                    label = 1 if label_val.lower().strip() in ['attack', '1', 'true', 'yes'] else 0
                else:
                    label = int(float(label_val))
            else:
                label = 0  # Assume normal if no label
            
            labels.append(label)
            
        except Exception as e:
            continue  # Skip problematic rows
    
    print(f"\n✅ Extracted {len(features)} feature vectors")
    if len(labels) > 0:
        normal_count = sum(1 for l in labels if l == 0)
        attack_count = sum(1 for l in labels if l == 1)
        print(f"   Normal samples: {normal_count} ({100*normal_count/len(labels):.1f}%)")
        print(f"   Attack samples: {attack_count} ({100*attack_count/len(labels):.1f}%)")
    else:
        print("   ⚠️  No labels found - assuming all normal traffic")
    
    return np.array(features), np.array(labels)


def train_on_unsw_nb15(train_file=None):
    """Train Isolation Forest on UNSW-NB15 dataset"""
    
    if train_file is None:
        # Download dataset
        train_file = download_unsw_nb15()
        if train_file is None:
            print("\n⚠️  Could not download dataset automatically.")
            print("Please download UNSW-NB15 manually and specify the path.")
            print("\nExample:")
            print("  python3 train_unsw_nb15.py data/UNSW_NB15_training-set.csv")
            return None, None
    
    # Load dataset
    df = load_unsw_nb15(train_file)
    if df is None:
        print("\n❌ Failed to load dataset")
        return None, None
    
    # Extract features
    X_train, y_train = extract_features_from_unsw(df)
    
    if len(X_train) == 0:
        print("❌ No features extracted from dataset")
        return None, None
    
    # Filter to normal traffic only for Isolation Forest training
    # Isolation Forest learns normal patterns, so we train only on normal data
    normal_mask = y_train == 0
    X_normal = X_train[normal_mask]
    
    print(f"\n{'='*60}")
    print("Training Configuration")
    print(f"{'='*60}")
    print(f"Total samples: {len(X_train)}")
    print(f"Normal samples: {len(X_normal)}")
    print(f"Attack samples: {len(X_train) - len(X_normal)}")
    print(f"Training on: {len(X_normal)} normal samples")
    
    if len(X_normal) < 100:
        print("⚠️  Not enough normal samples. Using all data...")
        X_normal = X_train
    
    # Standardize features
    print("\nFitting StandardScaler...")
    scaler = StandardScaler()
    X_normal_scaled = scaler.fit_transform(X_normal)
    
    # Train Isolation Forest
    print("\nTraining Isolation Forest model...")
    print(f"  Samples: {len(X_normal)}")
    print(f"  Features: {X_normal.shape[1]}")
    print(f"  Parameters: n_estimators=100, contamination=0.05")
    
    model = IsolationForest(
        n_estimators=100,
        contamination=0.05,  # Expect ~5% anomalies
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_normal_scaled)
    
    # Test the model
    print("\nTesting model on training data...")
    scores = model.decision_function(X_normal_scaled)
    print(f"  Normal samples - Score range: {scores.min():.3f} to {scores.max():.3f}")
    print(f"  Normal samples - Mean score: {scores.mean():.3f}")
    
    # Evaluate on attack samples if available
    attack_mask = y_train == 1
    if attack_mask.sum() > 0:
        X_attack = X_train[attack_mask]
        X_attack_scaled = scaler.transform(X_attack)
        attack_scores = model.decision_function(X_attack_scaled)
        print(f"\nAttack sample evaluation:")
        print(f"  Attack samples - Score range: {attack_scores.min():.3f} to {attack_scores.max():.3f}")
        print(f"  Attack samples - Mean score: {attack_scores.mean():.3f}")
        
        # Count anomalies detected (score < -0.2)
        anomalies_detected = (attack_scores < -0.2).sum()
        print(f"  Anomalies detected (score < -0.2): {anomalies_detected} / {len(attack_scores)} ({100*anomalies_detected/len(attack_scores):.1f}%)")
        
        # Count critical anomalies (score < -0.4)
        critical_detected = (attack_scores < -0.4).sum()
        print(f"  Critical anomalies (score < -0.4): {critical_detected} / {len(attack_scores)} ({100*critical_detected/len(attack_scores):.1f}%)")
    
    # Save model
    model_path = 'isolation_forest_model.joblib'
    scaler_path = 'feature_scaler.joblib'
    
    print(f"\nSaving model to {model_path}...")
    joblib.dump(model, model_path)
    
    print(f"Saving scaler to {scaler_path}...")
    joblib.dump(scaler, scaler_path)
    
    print(f"\n{'='*60}")
    print("✅ Model training complete on UNSW-NB15 dataset!")
    print(f"{'='*60}")
    print(f"   Model: {model_path}")
    print(f"   Scaler: {scaler_path}")
    print(f"   Trained on: {len(X_normal)} normal samples")
    print(f"\nThe model is ready for real-time inference.")
    
    return model, scaler


if __name__ == '__main__':
    print("=" * 60)
    print("NetGuardian AI - UNSW-NB15 Dataset Training")
    print("=" * 60)
    
    # Allow specifying dataset path as argument
    dataset_path = sys.argv[1] if len(sys.argv) > 1 else None
    
    train_on_unsw_nb15(dataset_path)

