#!/usr/bin/env python3
"""
Evaluate Isolation Forest model on UNSW-NB15 testing dataset
Calculates accuracy, precision, recall, F1-score, and other metrics
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import joblib
import os
import sys

def load_test_dataset(file_path):
    """Load and process UNSW-NB15 test dataset"""
    print(f"\nLoading test dataset from {file_path}...")
    
    try:
        # Read without headers (same as training)
            df = pd.read_csv(file_path, header=None, low_memory=False)
            
            # Set column names based on UNSW-NB15 structure
        column_names = [
                'srcip', 'sport', 'dstip', 'dsport', 'proto', 'state', 'dur', 'sbytes', 'dbytes',
                'sttl', 'dttl', 'sloss', 'dloss', 'service', 'sload', 'dload', 'spkts', 'dpkts',
                'swin', 'dwin', 'stcpb', 'dtcpb', 'smeansz', 'dmeansz', 'trans_depth', 'res_bdy_len',
                'sjit', 'djit', 'stime', 'ltime', 'sintpkt', 'dintpkt', 'tcprtt', 'synack', 'ackdat',
                'is_sm_ips_ports', 'ct_state_ttl', 'ct_flw_http_mthd', 'is_ftp_login', 'ct_ftp_cmd',
                'ct_srv_src', 'ct_srv_dst', 'ct_dst_ltm', 'ct_src_ltm', 'ct_src_dport_ltm',
                'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'attack_cat', 'label'
            ]
            
            num_cols = len(df.columns)
        if num_cols <= len(column_names):
            df.columns = column_names[:num_cols]
            else:
            df.columns = column_names + [f'col_{i}' for i in range(len(column_names), num_cols)]
        
        print(f"✅ Loaded {len(df)} test records")
        print(f"   Columns: {len(df.columns)}")
        
        return df
    except Exception as e:
        print(f"❌ Error loading test dataset: {e}")
        import traceback
        traceback.print_exc()
        return None


def extract_features_from_test(df):
    """Extract features from test dataset (same as training)"""
    print("\nExtracting features from test dataset...")
    
    features = []
    labels = []
    
    proto_map = {
        'tcp': 1, 'udp': 2, 'icmp': 3, 'http': 4, 'https': 5, 'ssl': 5, 'ssh': 6
    }
    
    for idx, row in df.iterrows():
        try:
            # Length: sum of source and destination bytes
            if 'sbytes' in df.columns and 'dbytes' in df.columns:
                sbytes_val = row.get('sbytes', 0)
                dbytes_val = row.get('dbytes', 0)
                sbytes = int(float(sbytes_val)) if pd.notna(sbytes_val) else 0
                dbytes = int(float(dbytes_val)) if pd.notna(dbytes_val) else 0
                length = sbytes + dbytes
                if length == 0:
                    length = 1500
            else:
                length = 1500
            
            # TTL
            ttl = 64
            
            # Protocol
            if 'proto' in df.columns and pd.notna(row.get('proto')):
                proto_str = str(row['proto']).lower().strip()
                proto_code = 0
                for key, val in proto_map.items():
                    if key in proto_str:
                        proto_code = val
                        break
                if proto_code == 0:
                    proto_code = 1
            else:
                proto_code = 1
            
            # Ports
            if 'sport' in df.columns and pd.notna(row.get('sport')):
                src_port_val = row['sport']
                src_port = int(float(src_port_val)) if pd.notna(src_port_val) else 0
            else:
                src_port = 0
            
            if 'dsport' in df.columns and pd.notna(row.get('dsport')):
                dst_port_val = row['dsport']
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
            
            # Label: 0 = normal, 1 = attack
            if 'label' in df.columns and pd.notna(row.get('label')):
                label_val = row['label']
                if isinstance(label_val, str):
                    label = 1 if label_val.lower().strip() in ['attack', '1', 'true', 'yes'] else 0
                else:
                    label = int(float(label_val))
            else:
                label = 0
            
            labels.append(label)
            
        except Exception as e:
            continue
    
    print(f"✅ Extracted {len(features)} feature vectors")
        normal_count = sum(1 for l in labels if l == 0)
        attack_count = sum(1 for l in labels if l == 1)
        print(f"   Normal samples: {normal_count} ({100*normal_count/len(labels):.1f}%)")
        print(f"   Attack samples: {attack_count} ({100*attack_count/len(labels):.1f}%)")
    
    return np.array(features), np.array(labels)


def evaluate_model(test_file=None):
    """Evaluate the trained Isolation Forest model"""
    
    print("=" * 70)
    print("NetGuardian AI - Model Evaluation on UNSW-NB15 Test Set")
    print("=" * 70)
    
    # Load model and scaler
    model_path = 'isolation_forest_model.joblib'
    scaler_path = 'feature_scaler.joblib'
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        print(f"❌ Model files not found!")
        print(f"   Expected: {model_path}, {scaler_path}")
        print("   Please train the model first using train_unsw_nb15.py")
        return
    
    print("\nLoading trained model...")
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    print("✅ Model and scaler loaded")
    
    # Load test dataset
    if test_file is None:
        test_file = 'data/UNSW_NB15_testing-set.csv'
        if not os.path.exists(test_file):
            # Try other test files
            test_files = [
                'data/UNSW-NB15_2.csv',
                'data/UNSW-NB15_3.csv',
                'data/UNSW-NB15_4.csv'
            ]
            for tf in test_files:
                if os.path.exists(tf):
                    test_file = tf
                    break
    
    if not os.path.exists(test_file):
        print(f"❌ Test file not found: {test_file}")
        print("   Available files:")
        if os.path.exists('data'):
            for f in os.listdir('data'):
                if f.endswith('.csv'):
                    print(f"     - data/{f}")
        return
    
    # Load and extract features
    df = load_test_dataset(test_file)
    if df is None:
        return
    
    X_test, y_test = extract_features_from_test(df)
    
    if len(X_test) == 0:
        print("❌ No features extracted from test dataset")
        return
    
    print(f"\n{'='*70}")
    print("Model Evaluation")
    print(f"{'='*70}")
    print(f"Test samples: {len(X_test)}")
    print(f"Normal samples: {(y_test == 0).sum()}")
    print(f"Attack samples: {(y_test == 1).sum()}")
    
    # Scale features
    print("\nScaling test features...")
    X_test_scaled = scaler.transform(X_test)
    
    # Get anomaly scores
    print("Computing anomaly scores...")
    anomaly_scores = model.decision_function(X_test_scaled)
    
    # Convert scores to predictions
    # Isolation Forest: negative scores = anomaly, positive = normal
    # We'll use thresholds: < -0.2 = suspicious, < -0.4 = critical
    
    # For binary classification, use -0.2 as threshold
    predictions_binary = (anomaly_scores < -0.2).astype(int)
    
    # Risk level predictions
    predictions_risk = []
    for score in anomaly_scores:
        if score < -0.4:
            predictions_risk.append('CRITICAL')
        elif score < -0.2:
            predictions_risk.append('WARNING')
        else:
            predictions_risk.append('SAFE')
    
    # Calculate metrics
    print("\n" + "="*70)
    print("EVALUATION METRICS")
    print("="*70)
    
    # Binary classification metrics
    accuracy = accuracy_score(y_test, predictions_binary)
    precision = precision_score(y_test, predictions_binary, zero_division=0)
    recall = recall_score(y_test, predictions_binary, zero_division=0)
    f1 = f1_score(y_test, predictions_binary, zero_division=0)
    
    print(f"\n📊 Binary Classification (Threshold: -0.2)")
    print(f"   Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"   Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"   Recall:    {recall:.4f} ({recall*100:.2f}%)")
    print(f"   F1-Score:  {f1:.4f} ({f1*100:.2f}%)")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, predictions_binary)
    print(f"\n📋 Confusion Matrix")
    print(f"                  Predicted")
    print(f"                Normal  Attack")
    print(f"   Actual Normal  {cm[0][0]:5d}  {cm[0][1]:5d}")
    print(f"          Attack  {cm[1][0]:5d}  {cm[1][1]:5d}")
    
    # Calculate rates
    tn, fp, fn, tp = cm.ravel()
    true_negative_rate = tn / (tn + fp) if (tn + fp) > 0 else 0
    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
    false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
    true_positive_rate = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    print(f"\n📈 Detailed Rates")
    print(f"   True Positive Rate (Sensitivity):  {true_positive_rate:.4f} ({true_positive_rate*100:.2f}%)")
    print(f"   True Negative Rate (Specificity): {true_negative_rate:.4f} ({true_negative_rate*100:.2f}%)")
    print(f"   False Positive Rate:                {false_positive_rate:.4f} ({false_positive_rate*100:.2f}%)")
    print(f"   False Negative Rate:                {false_negative_rate:.4f} ({false_negative_rate*100:.2f}%)")
    
    # Score distribution analysis
    print(f"\n📊 Anomaly Score Distribution")
    normal_scores = anomaly_scores[y_test == 0]
    attack_scores = anomaly_scores[y_test == 1]
    
    print(f"   Normal Traffic:")
    print(f"     Mean:   {normal_scores.mean():.4f}")
    print(f"     Std:    {normal_scores.std():.4f}")
    print(f"     Min:    {normal_scores.min():.4f}")
    print(f"     Max:    {normal_scores.max():.4f}")
    
    print(f"   Attack Traffic:")
    print(f"     Mean:   {attack_scores.mean():.4f}")
    print(f"     Std:    {attack_scores.std():.4f}")
    print(f"     Min:    {attack_scores.min():.4f}")
    print(f"     Max:    {attack_scores.max():.4f}")
    
    # Detection at different thresholds
    print(f"\n🎯 Detection at Different Thresholds")
    thresholds = [-0.4, -0.3, -0.2, -0.1, 0.0]
    for threshold in thresholds:
        preds = (anomaly_scores < threshold).astype(int)
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)
        f1_t = f1_score(y_test, preds, zero_division=0)
        print(f"   Threshold {threshold:5.1f}: Acc={acc:.3f}, Prec={prec:.3f}, Rec={rec:.3f}, F1={f1_t:.3f}")
    
    # Risk level analysis
    print(f"\n⚠️  Risk Level Distribution")
    risk_counts = {'SAFE': 0, 'WARNING': 0, 'CRITICAL': 0}
    for pred in predictions_risk:
        risk_counts[pred] += 1
    
    for risk, count in risk_counts.items():
        pct = (count / len(predictions_risk)) * 100
        print(f"   {risk:8s}: {count:6d} ({pct:5.2f}%)")
    
    # Attack detection by risk level
    print(f"\n🔍 Attack Detection by Risk Level")
    for risk_level in ['SAFE', 'WARNING', 'CRITICAL']:
        risk_mask = np.array(predictions_risk) == risk_level
        if risk_mask.sum() > 0:
            attacks_in_risk = (y_test[risk_mask] == 1).sum()
            total_in_risk = risk_mask.sum()
            attack_rate = (attacks_in_risk / total_in_risk) * 100 if total_in_risk > 0 else 0
            print(f"   {risk_level:8s}: {attacks_in_risk:4d} attacks out of {total_in_risk:5d} ({attack_rate:5.2f}%)")
    
    # ROC-AUC if possible
    try:
        # For ROC-AUC, we need to invert scores (lower = more anomalous = positive class)
        # So we use -anomaly_scores as the prediction score
        roc_auc = roc_auc_score(y_test, -anomaly_scores)
        print(f"\n📈 ROC-AUC Score: {roc_auc:.4f}")
    except Exception as e:
        print(f"\n⚠️  Could not calculate ROC-AUC: {e}")
    
    print(f"\n{'='*70}")
    print("✅ Evaluation Complete!")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    test_file = sys.argv[1] if len(sys.argv) > 1 else None
    evaluate_model(test_file)
