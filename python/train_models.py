"""Train ML models for SmartShield."""
import os
import sys
import argparse
import pandas as pd
import numpy as np
import logging

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.preprocessing import DataPreprocessor
from src.models import ModelTrainer
from src.utils import setup_logging, ensure_dir

logger = setup_logging()

def load_datasets_separately(data_dir='data'):
    """Load training and testing datasets separately to avoid data leakage."""
    logger.info("Loading UNSW-NB15 datasets separately...")
    
    train_file = os.path.join(data_dir, 'UNSW_NB15_training-set.csv')
    test_file = os.path.join(data_dir, 'UNSW_NB15_testing-set.csv')
    
    # Load training set
    if not os.path.exists(train_file):
        raise FileNotFoundError(f"Training file not found: {train_file}")
    
    logger.info(f"Loading training set: {train_file}")
    df_train = pd.read_csv(train_file)
    logger.info(f"Training set shape: {df_train.shape}")
    
    # Load testing set
    if not os.path.exists(test_file):
        logger.warning(f"Testing file not found: {test_file}. Will use train/validation split only.")
        df_test = None
    else:
        logger.info(f"Loading testing set: {test_file}")
        df_test = pd.read_csv(test_file)
        logger.info(f"Testing set shape: {df_test.shape}")
    
    return df_train, df_test

def main(args):
    """Main training function."""
    logger.info("=" * 60)
    logger.info("SmartShield Model Training (Fixed: No Data Leakage)")
    logger.info("=" * 60)
    
    # Load datasets separately
    df_train_raw, df_test_raw = load_datasets_separately(args.data_dir)
    
    # Extract labels BEFORE preprocessing (they get removed during preprocessing)
    train_attack_cat = df_train_raw['attack_cat'].copy() if 'attack_cat' in df_train_raw.columns else None
    train_label = df_train_raw['label'].copy() if 'label' in df_train_raw.columns else None
    
    if df_test_raw is not None:
        test_attack_cat = df_test_raw['attack_cat'].copy() if 'attack_cat' in df_test_raw.columns else None
        test_label = df_test_raw['label'].copy() if 'label' in df_test_raw.columns else None
    else:
        test_attack_cat = None
        test_label = None
    
    # Log attack category distribution
    if train_attack_cat is not None:
        logger.info("\n" + "=" * 60)
        logger.info("Attack Categories in Training Set:")
        logger.info("=" * 60)
        attack_counts = train_attack_cat.value_counts()
        for attack_type, count in attack_counts.items():
            logger.info(f"  {attack_type}: {count:,} ({count/len(train_attack_cat)*100:.1f}%)")
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # CRITICAL FIX #1: Fit preprocessor ONLY on training data
    logger.info("\n" + "=" * 60)
    logger.info("Fitting Preprocessor on TRAINING DATA ONLY")
    logger.info("=" * 60)
    logger.info("(This prevents data leakage - test data statistics are not seen)")
    preprocessor.fit(df_train_raw)
    
    # Transform training data
    logger.info("\n" + "=" * 60)
    logger.info("Transforming Training Data")
    logger.info("=" * 60)
    df_train_processed = preprocessor.transform(df_train_raw)
    
    # Transform test data (if available)
    if df_test_raw is not None:
        logger.info("\n" + "=" * 60)
        logger.info("Transforming Testing Data")
        logger.info("=" * 60)
        logger.info("(Using scalers/encoders fitted on training data only)")
        df_test_processed = preprocessor.transform(df_test_raw)
    else:
        df_test_processed = None
    
    # Prepare features and labels
    logger.info("\n" + "=" * 60)
    logger.info("Preparing Features and Labels")
    logger.info("=" * 60)
    
    # Training data
    X_train = df_train_processed.copy()
    # Ensure all columns are numeric (after encoding, categorical should be numeric)
    for col in X_train.columns:
        if X_train[col].dtype == 'object':
            logger.warning(f"Column {col} is still object type, attempting to convert to numeric")
            X_train[col] = pd.to_numeric(X_train[col], errors='coerce')
    X_train = X_train.fillna(0)
    X_train = X_train.select_dtypes(include=[np.number])
    X_train = np.array(X_train, dtype=np.float32)
    
    y_train_attack_cat = train_attack_cat.values if train_attack_cat is not None else None
    y_train_label = train_label.values if train_label is not None else None
    
    logger.info(f"Training features shape: {X_train.shape}")
    logger.info(f"Training samples: {len(X_train)}")
    
    # Test data (if available)
    if df_test_processed is not None:
        X_test = df_test_processed.copy()
        for col in X_test.columns:
            if X_test[col].dtype == 'object':
                X_test[col] = pd.to_numeric(X_test[col], errors='coerce')
        X_test = X_test.fillna(0)
        X_test = X_test.select_dtypes(include=[np.number])
        X_test = np.array(X_test, dtype=np.float32)
        
        y_test_attack_cat = test_attack_cat.values if test_attack_cat is not None else None
        y_test_label = test_label.values if test_label is not None else None
        
        logger.info(f"Test features shape: {X_test.shape}")
        logger.info(f"Test samples: {len(X_test)}")
    else:
        # If no test set, split training data into train/validation
        from sklearn.model_selection import train_test_split
        logger.info("No separate test set found. Splitting training data into train/validation...")
        X_train, X_test, y_train_attack_cat, y_test_attack_cat = train_test_split(
            X_train, y_train_attack_cat, test_size=0.2, random_state=42, stratify=y_train_attack_cat
        )
        if y_train_label is not None:
            _, _, y_train_label, y_test_label = train_test_split(
                train_label.values, train_label.values, test_size=0.2, random_state=42, stratify=train_label.values
            )
        else:
            y_test_label = None
    
    # CRITICAL FIX #2: Filter normal data for anomaly detectors
    logger.info("\n" + "=" * 60)
    logger.info("Preparing Normal-Only Data for Anomaly Detectors")
    logger.info("=" * 60)
    logger.info("(Anomaly detectors must learn from normal traffic only)")
    
    # Find normal samples in training data
    if y_train_attack_cat is not None:
        normal_mask = (y_train_attack_cat == 'Normal')
        X_train_normal = X_train[normal_mask]
        logger.info(f"Normal samples in training set: {len(X_train_normal):,} / {len(X_train):,} ({len(X_train_normal)/len(X_train)*100:.1f}%)")
        
        if len(X_train_normal) == 0:
            logger.warning("WARNING: No normal samples found! Using label column instead.")
            if y_train_label is not None:
                normal_mask = (y_train_label == 0)
                X_train_normal = X_train[normal_mask]
                logger.info(f"Normal samples (from label): {len(X_train_normal):,}")
    elif y_train_label is not None:
        normal_mask = (y_train_label == 0)
        X_train_normal = X_train[normal_mask]
        logger.info(f"Normal samples in training set: {len(X_train_normal):,} / {len(X_train):,} ({len(X_train_normal)/len(X_train)*100:.1f}%)")
    else:
        logger.error("No labels found! Cannot filter normal data for anomaly detectors.")
        X_train_normal = X_train  # Fallback (not ideal)
    
    if len(X_train_normal) == 0:
        logger.error("ERROR: No normal samples found! Cannot train anomaly detectors.")
        logger.error("Skipping Isolation Forest and Autoencoder training.")
        args.train_if = False
        args.train_ae = False
    
    # Initialize model trainer
    ensure_dir(args.models_dir)
    trainer = ModelTrainer(models_dir=args.models_dir)
    
    # Train Random Forest for attack classification (multi-class) - uses ALL data
    if args.train_rf or args.train_all:
        logger.info("\n" + "=" * 60)
        logger.info("Training Random Forest Classifier")
        logger.info("=" * 60)
        logger.info("Training on ALL data (normal + attacks) for classification")
        y_train = y_train_attack_cat if y_train_attack_cat is not None else y_train_label
        y_test = y_test_attack_cat if y_test_attack_cat is not None else y_test_label
        trainer.train_random_forest(X_train, y_train, X_test, y_test)
    
    # Train Isolation Forest for anomaly detection - uses NORMAL-ONLY data
    if args.train_if or args.train_all:
        logger.info("\n" + "=" * 60)
        logger.info("Training Isolation Forest (Anomaly Detection)")
        logger.info("=" * 60)
        logger.info(f"Training on NORMAL-ONLY data: {len(X_train_normal):,} samples")
        trainer.train_isolation_forest(X_train_normal)
    
    # Train Autoencoder for anomaly detection - uses NORMAL-ONLY data
    if args.train_ae or args.train_all:
        logger.info("\n" + "=" * 60)
        logger.info("Training Autoencoder (Anomaly Detection)")
        logger.info("=" * 60)
        logger.info(f"Training on NORMAL-ONLY data: {len(X_train_normal):,} samples")
        # For validation, use a subset of normal test data
        if X_test is not None and y_test_attack_cat is not None:
            test_normal_mask = (y_test_attack_cat == 'Normal')
            X_val_normal = X_test[test_normal_mask]
            if len(X_val_normal) == 0 and y_test_label is not None:
                test_normal_mask = (y_test_label == 0)
                X_val_normal = X_test[test_normal_mask]
        elif X_test is not None and y_test_label is not None:
            test_normal_mask = (y_test_label == 0)
            X_val_normal = X_test[test_normal_mask]
        else:
            X_val_normal = None
        
        if X_val_normal is not None and len(X_val_normal) > 0:
            logger.info(f"Using {len(X_val_normal):,} normal samples for validation")
        else:
            logger.info("Using training data for validation (no normal test samples)")
            X_val_normal = X_train_normal[:min(10000, len(X_train_normal))]  # Use subset for validation
        
        trainer.train_autoencoder(X_train_normal, X_val_normal, epochs=args.epochs)
    
    # Save preprocessor
    import joblib
    preprocessor_path = os.path.join(args.models_dir, 'preprocessor.pkl')
    joblib.dump(preprocessor, preprocessor_path)
    logger.info(f"\nSaved preprocessor to {preprocessor_path}")
    
    logger.info("\n" + "=" * 60)
    logger.info("Training completed successfully!")
    logger.info("=" * 60)
    logger.info("\n✅ FIXES APPLIED:")
    logger.info("  1. No data leakage: Preprocessor fitted on training data only")
    logger.info("  2. Proper anomaly detection: Isolation Forest and Autoencoder trained on normal-only data")
    logger.info("=" * 60)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train SmartShield ML models')
    parser.add_argument('--data-dir', type=str, default='data',
                        help='Directory containing UNSW-NB15 dataset')
    parser.add_argument('--models-dir', type=str, default='models',
                        help='Directory to save trained models')
    parser.add_argument('--train-all', action='store_true',
                        help='Train all models')
    parser.add_argument('--train-rf', action='store_true',
                        help='Train Random Forest only')
    parser.add_argument('--train-if', action='store_true',
                        help='Train Isolation Forest only')
    parser.add_argument('--train-ae', action='store_true',
                        help='Train Autoencoder only')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs for Autoencoder training')
    
    args = parser.parse_args()
    
    if not any([args.train_all, args.train_rf, args.train_if, args.train_ae]):
        args.train_all = True
    
    main(args)
