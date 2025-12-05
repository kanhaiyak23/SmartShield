#!/usr/bin/env python3
"""
Evaluate Enhanced Random Forest model on UNSW-NB15 test dataset
Shows comprehensive results and metrics
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import joblib
import os

def load_test_dataset(file_path):
    """Load test dataset"""
    print(f"Loading test dataset from {file_path}...")
    
    try:
        # Check if file has headers
        sample = pd.read_csv(file_path, nrows=1)
        if 'id' in sample.columns or 'dur' in sample.columns:
            # Has headers
            df = pd.read_csv(file_path, low_memory=False)
        else:
            # No headers
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
        
        print(f"✅ Loaded {len(df)} test records")
        return df
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def extract_enhanced_features(df, service_encoder, state_encoder):
    """Extract 20 enhanced features (same as training)"""
    print("\nExtracting enhanced features...")
    
    features = []
    labels = []
    
    proto_map = {'tcp': 1, 'udp': 2, 'icmp': 3, 'http': 4, 'https': 5, 'ssl': 5, 'ssh': 6}
    
    for idx, row in df.iterrows():
        try:
            # Basic features
            sbytes = float(row.get('sbytes', 0) or 0)
            dbytes = float(row.get('dbytes', 0) or 0)
            length = sbytes + dbytes if (sbytes + dbytes) > 0 else 1500
            
            ttl = float(row.get('sttl', 64) or 64)
            
            proto_str = str(row.get('proto', 'tcp')).lower().strip()
            proto_code = next((v for k, v in proto_map.items() if k in proto_str), 1)
            
            # Try different column name variations
            src_port = 0
            dst_port = 0
            
            # Try sport/dsport first
            if 'sport' in df.columns:
                src_port = max(0, min(65535, int(float(row.get('sport', 0) or 0))))
            elif 'srcport' in df.columns:
                src_port = max(0, min(65535, int(float(row.get('srcport', 0) or 0))))
            
            if 'dsport' in df.columns:
                dst_port = max(0, min(65535, int(float(row.get('dsport', 0) or 0))))
            elif 'dstport' in df.columns:
                dst_port = max(0, min(65535, int(float(row.get('dstport', 0) or 0))))
            
            # If still no ports, use defaults based on service
            if src_port == 0 and dst_port == 0:
                service = str(row.get('service', '')).lower()
                if 'http' in service:
                    dst_port = 80
                elif 'https' in service or 'ssl' in service:
                    dst_port = 443
                elif 'dns' in service:
                    dst_port = 53
                else:
                    dst_port = 80  # Default
                src_port = 49152  # Ephemeral port
            
            # Flow statistics
            dur = float(row.get('dur', 0) or 0)
            if dur == 0:
                dur = 0.001
            
            spkts = float(row.get('spkts', 0) or 0)
            dpkts = float(row.get('dpkts', 0) or 0)
            total_packets = spkts + dpkts
            
            packet_rate = total_packets / dur if dur > 0 else 0
            byte_rate = length / dur if dur > 0 else 0
            packet_ratio = spkts / dpkts if dpkts > 0 else 1.0
            
            # Network behavior
            service = str(row.get('service', '-')).lower()
            try:
                service_encoded = service_encoder.transform([service])[0] if service in service_encoder.classes_ else 0
            except:
                service_encoded = 0
            
            state = str(row.get('state', '-'))
            try:
                state_encoded = state_encoder.transform([state])[0] if state in state_encoder.classes_ else 0
            except:
                state_encoded = 0
            
            is_well_known_port = 1 if dst_port < 1024 else 0
            
            if dst_port < 1024:
                port_category = 0
            elif dst_port < 49152:
                port_category = 1
            else:
                port_category = 2
            
            # Statistical features
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
                length, ttl, proto_code, src_port, dst_port,
                dur, packet_rate, byte_rate, spkts, dpkts,
                packet_ratio, service_encoded, state_encoded, is_well_known_port, port_category,
                mean_packet_size, jitter, inter_packet_time, tcprtt, total_packets
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
    return np.array(features), np.array(labels)


def evaluate_model():
    """Evaluate Random Forest model on test set"""
    print("=" * 70)
    print("Enhanced Random Forest Model Evaluation")
    print("=" * 70)
    
    # Load model and components
    print("\nLoading trained model...")
    try:
        model = joblib.load('random_forest_model.joblib')
        scaler = joblib.load('rf_feature_scaler.joblib')
        service_encoder = joblib.load('service_encoder.joblib')
        state_encoder = joblib.load('state_encoder.joblib')
        print("✅ Model and components loaded")
        print(f"   Model: {model.n_estimators} trees, {model.n_features_in_} features")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Load test dataset
    test_file = 'data/UNSW_NB15_testing-set.csv'
    if not os.path.exists(test_file):
        print(f"❌ Test file not found: {test_file}")
        return
    
    df = load_test_dataset(test_file)
    if df is None:
        return
    
    # Limit to reasonable size for evaluation (or use full dataset)
    if len(df) > 50000:
        print(f"\nUsing first 50,000 records for evaluation...")
        df = df.head(50000)
    
    # Extract features
    X_test, y_test = extract_enhanced_features(df, service_encoder, state_encoder)
    
    if len(X_test) == 0:
        print("❌ No features extracted!")
        return
    
    print(f"\nTest Set: {len(X_test)} samples")
    print(f"  Normal: {(y_test == 0).sum()} ({(y_test == 0).sum()/len(y_test)*100:.1f}%)")
    print(f"  Attacks: {(y_test == 1).sum()} ({(y_test == 1).sum()/len(y_test)*100:.1f}%)")
    
    # Scale features
    print("\nScaling features...")
    X_test_scaled = scaler.transform(X_test)
    
    # Make predictions
    print("Making predictions...")
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Detailed rates
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    # ROC-AUC
    try:
        roc_auc = roc_auc_score(y_test, y_pred_proba)
    except:
        roc_auc = 0
    
    # Print results
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    
    print(f"\n📊 PERFORMANCE METRICS")
    print(f"   Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"   Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"   Recall:    {recall:.4f} ({recall*100:.2f}%)")
    print(f"   F1-Score:  {f1:.4f} ({f1*100:.2f}%)")
    print(f"   ROC-AUC:   {roc_auc:.4f}")
    
    print(f"\n📋 CONFUSION MATRIX")
    print(f"                  Predicted")
    print(f"                Normal  Attack")
    print(f"Actual Normal    {tn:5d}  {fp:5d}")
    print(f"      Attack     {fn:5d}  {tp:5d}")
    
    print(f"\n📈 DETAILED RATES")
    print(f"   True Positive Rate (Sensitivity):  {tpr:.4f} ({tpr*100:.2f}%)")
    print(f"   True Negative Rate (Specificity): {tnr:.4f} ({tnr*100:.2f}%)")
    print(f"   False Positive Rate:                {fpr:.4f} ({fpr*100:.2f}%)")
    print(f"   False Negative Rate:                {fnr:.4f} ({fnr*100:.2f}%)")
    
    # Classification report
    print(f"\n📝 CLASSIFICATION REPORT")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Attack']))
    
    # Risk level analysis
    print(f"\n⚠️  RISK LEVEL ANALYSIS")
    risk_levels = []
    for proba in y_pred_proba:
        if proba >= 0.7:
            risk_levels.append('CRITICAL')
        elif proba >= 0.4:
            risk_levels.append('WARNING')
        else:
            risk_levels.append('SAFE')
    
    risk_counts = {'SAFE': 0, 'WARNING': 0, 'CRITICAL': 0}
    for r in risk_levels:
        risk_counts[r] += 1
    
    for risk, count in risk_counts.items():
        pct = (count / len(risk_levels)) * 100
        print(f"   {risk:8s}: {count:6d} ({pct:5.2f}%)")
    
    # Attack detection by risk level
    print(f"\n🔍 ATTACK DETECTION BY RISK LEVEL")
    risk_array = np.array(risk_levels)
    for risk_level in ['SAFE', 'WARNING', 'CRITICAL']:
        mask = risk_array == risk_level
        if mask.sum() > 0:
            attacks_in_risk = (y_test[mask] == 1).sum()
            total_in_risk = mask.sum()
            attack_rate = (attacks_in_risk / total_in_risk) * 100 if total_in_risk > 0 else 0
            print(f"   {risk_level:8s}: {attacks_in_risk:4d} attacks / {total_in_risk:5d} total ({attack_rate:5.2f}%)")
    
    # Feature importance
    print(f"\n🎯 TOP 10 MOST IMPORTANT FEATURES")
    feature_names = [
        'length', 'ttl', 'proto', 'src_port', 'dst_port',
        'duration', 'packet_rate', 'byte_rate', 'spkts', 'dpkts',
        'packet_ratio', 'service', 'state', 'well_known_port', 'port_category',
        'mean_packet_size', 'jitter', 'inter_packet_time', 'tcp_rtt', 'total_packets'
    ]
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:10]
    for i, idx in enumerate(indices, 1):
        print(f"   {i:2d}. {feature_names[idx]:20s}: {importances[idx]:.4f}")
    
    print(f"\n" + "=" * 70)
    print("✅ Evaluation Complete!")
    print("=" * 70)
    
    # Summary
    print(f"\n📊 SUMMARY")
    print(f"   Model successfully detected {tp} out of {(y_test == 1).sum()} attacks")
    print(f"   Detection Rate: {recall*100:.2f}%")
    print(f"   False Alarms: {fp} out of {(y_test == 0).sum()} normal packets")
    print(f"   False Alarm Rate: {fpr*100:.2f}%")
    print()
    
    if accuracy > 0.95 and recall > 0.90:
        print("🎉 EXCELLENT PERFORMANCE! Model is ready for production.")
    elif accuracy > 0.90 and recall > 0.80:
        print("✅ GOOD PERFORMANCE! Model is ready for deployment.")
    else:
        print("⚠️  Model may need further tuning.")


if __name__ == '__main__':
    evaluate_model()

