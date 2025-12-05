#!/usr/bin/env python3
"""
Train Isolation Forest model for network anomaly detection
This script generates training data and trains a model that can be used for real-time inference
"""

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
import os

def generate_training_data(n_samples=10000):
    """
    Generate synthetic training data that represents normal network traffic
    This simulates what you'd get from real packet captures
    """
    np.random.seed(42)
    
    # Normal traffic characteristics
    # [length, ttl, proto_code, src_port, dst_port]
    data = []
    
    for _ in range(n_samples):
        # Normal packet lengths (most common: 64-1500 bytes)
        length = np.random.choice([
            np.random.randint(64, 1500),  # 80% normal MTU
            np.random.randint(1500, 2000),  # 15% slightly larger
            np.random.randint(40, 64),      # 5% small packets
        ], p=[0.8, 0.15, 0.05])
        
        # Normal TTL values (64, 128, 255 are common)
        ttl = np.random.choice([64, 128, 255], p=[0.4, 0.5, 0.1])
        
        # Protocol distribution (TCP/UDP most common)
        proto_code = np.random.choice([1, 2, 3, 4, 5, 6], p=[0.5, 0.3, 0.05, 0.1, 0.04, 0.01])
        
        # Source port (ephemeral ports for clients)
        src_port = np.random.choice([
            np.random.randint(49152, 65535),  # Ephemeral ports
            np.random.randint(1024, 49151),    # Registered ports
        ], p=[0.7, 0.3])
        
        # Destination port (common service ports)
        dst_port = np.random.choice([
            80, 443, 53, 22, 25, 110, 143, 993, 995,  # Common services
            np.random.randint(1024, 65535),  # Other ports
        ], p=[0.3, 0.2, 0.1, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.1])
        
        data.append([length, ttl, proto_code, src_port, dst_port])
    
    return np.array(data)


def train_isolation_forest():
    """Train Isolation Forest model on normal traffic patterns"""
    print("Generating training data...")
    X_train = generate_training_data(n_samples=10000)
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Feature ranges:")
    print(f"  Length: {X_train[:, 0].min():.0f} - {X_train[:, 0].max():.0f}")
    print(f"  TTL: {X_train[:, 1].min():.0f} - {X_train[:, 1].max():.0f}")
    print(f"  Protocol: {X_train[:, 2].min():.0f} - {X_train[:, 2].max():.0f}")
    print(f"  Src Port: {X_train[:, 3].min():.0f} - {X_train[:, 3].max():.0f}")
    print(f"  Dst Port: {X_train[:, 4].min():.0f} - {X_train[:, 4].max():.0f}")
    
    # Standardize features (important for Isolation Forest)
    print("\nFitting StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Train Isolation Forest
    # contamination: expected proportion of anomalies (5% is reasonable)
    # random_state: for reproducibility
    print("\nTraining Isolation Forest model...")
    print("  Parameters: n_estimators=100, contamination=0.05, random_state=42")
    
    model = IsolationForest(
        n_estimators=100,
        contamination=0.05,  # Expect ~5% anomalies
        random_state=42,
        n_jobs=-1  # Use all CPU cores
    )
    
    model.fit(X_train_scaled)
    
    # Test the model
    print("\nTesting model...")
    scores = model.decision_function(X_train_scaled)
    print(f"  Score range: {scores.min():.3f} to {scores.max():.3f}")
    print(f"  Mean score: {scores.mean():.3f}")
    
    # Save model and scaler
    model_path = 'isolation_forest_model.joblib'
    scaler_path = 'feature_scaler.joblib'
    
    print(f"\nSaving model to {model_path}...")
    joblib.dump(model, model_path)
    
    print(f"Saving scaler to {scaler_path}...")
    joblib.dump(scaler, scaler_path)
    
    print("\n✅ Model training complete!")
    print(f"   Model: {model_path}")
    print(f"   Scaler: {scaler_path}")
    print("\nThe model is ready for real-time inference.")
    
    return model, scaler


if __name__ == '__main__':
    print("=" * 60)
    print("NetGuardian AI - Isolation Forest Model Training")
    print("=" * 60)
    train_isolation_forest()

