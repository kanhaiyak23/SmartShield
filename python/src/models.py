"""ML model definitions for SmartShield."""
import numpy as np
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import logging
import os

logger = logging.getLogger('SmartShield')

class ModelTrainer:
    """Train and manage ML models for intrusion detection."""
    
    def __init__(self, models_dir='models'):
        self.models_dir = models_dir
        self.random_forest = None
        self.isolation_forest = None
        self.autoencoder = None
        self.attack_label_map = {}
        
        # Ensure models directory exists
        os.makedirs(models_dir, exist_ok=True)
    
    def train_random_forest(self, X_train, y_train, X_val, y_val, n_estimators=100):
        """Train Random Forest classifier for attack type detection."""
        logger.info("Training Random Forest classifier...")
        
        self.random_forest = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=20,
            random_state=42,
            n_jobs=-1,
            verbose=1
        )
        
        self.random_forest.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.random_forest.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        logger.info(f"Random Forest Accuracy: {accuracy:.4f}")
        logger.info("\nClassification Report:\n" + classification_report(y_val, y_pred))
        
        # Save model
        model_path = os.path.join(self.models_dir, 'random_forest.pkl')
        joblib.dump(self.random_forest, model_path)
        logger.info(f"Saved Random Forest model to {model_path}")
        
        return self.random_forest
    
    def train_isolation_forest(self, X_train, contamination=0.1):
        """Train Isolation Forest for anomaly detection."""
        logger.info("Training Isolation Forest for anomaly detection...")
        
        self.isolation_forest = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_jobs=-1
        )
        
        self.isolation_forest.fit(X_train)
        logger.info("Isolation Forest trained successfully")
        
        # Save model
        model_path = os.path.join(self.models_dir, 'isolation_forest.pkl')
        joblib.dump(self.isolation_forest, model_path)
        logger.info(f"Saved Isolation Forest model to {model_path}")
        
        return self.isolation_forest
    
    def build_autoencoder(self, input_dim, encoding_dim=32):
        """Build Autoencoder model for anomaly detection."""
        input_layer = tf.keras.layers.Input(shape=(input_dim,))
        
        # Encoder
        encoded = tf.keras.layers.Dense(64, activation='relu')(input_layer)
        encoded = tf.keras.layers.Dense(encoding_dim, activation='relu')(encoded)
        
        # Decoder
        decoded = tf.keras.layers.Dense(64, activation='relu')(encoded)
        decoded = tf.keras.layers.Dense(input_dim, activation='sigmoid')(decoded)
        
        autoencoder = tf.keras.Model(input_layer, decoded)
        autoencoder.compile(optimizer='adam', loss='mse')
        
        return autoencoder
    
    def train_autoencoder(self, X_train, X_val, epochs=50, batch_size=32, encoding_dim=32):
        """Train Autoencoder for anomaly detection."""
        logger.info("Training Autoencoder...")
        
        input_dim = X_train.shape[1]
        self.autoencoder = self.build_autoencoder(input_dim, encoding_dim)
        
        # Early stopping
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Train
        history = self.autoencoder.fit(
            X_train, X_train,
            validation_data=(X_val, X_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping],
            verbose=1
        )
        
        # Save model
        model_path = os.path.join(self.models_dir, 'autoencoder.h5')
        self.autoencoder.save(model_path)
        logger.info(f"Saved Autoencoder model to {model_path}")
        
        return self.autoencoder
    
    def load_models(self):
        """Load trained models."""
        # Load Random Forest
        rf_path = os.path.join(self.models_dir, 'random_forest.pkl')
        if os.path.exists(rf_path):
            self.random_forest = joblib.load(rf_path)
            logger.info(f"Loaded Random Forest from {rf_path}")
        
        # Load Isolation Forest
        if_path = os.path.join(self.models_dir, 'isolation_forest.pkl')
        if os.path.exists(if_path):
            self.isolation_forest = joblib.load(if_path)
            logger.info(f"Loaded Isolation Forest from {if_path}")
        
        # Load Autoencoder
        ae_path = os.path.join(self.models_dir, 'autoencoder.h5')
        if os.path.exists(ae_path):
            try:
                # Try loading with compile=False to avoid metric deserialization issues
                self.autoencoder = tf.keras.models.load_model(ae_path, compile=False)
                # Recompile if needed (for inference, compile is optional)
                try:
                    self.autoencoder.compile(optimizer='adam', loss='mse')
                except Exception as compile_error:
                    logger.warning(f"Could not recompile autoencoder, but model loaded: {compile_error}")
                logger.info(f"Loaded Autoencoder from {ae_path}")
            except Exception as e:
                logger.error(f"Failed to load Autoencoder: {e}")
                logger.warning("Continuing without autoencoder model")
                self.autoencoder = None
    
    def predict_attack_type(self, features):
        """Predict attack type using Random Forest."""
        if self.random_forest is None:
            raise ValueError("Random Forest model not loaded")
        
        prediction = self.random_forest.predict([features])[0]
        probabilities = self.random_forest.predict_proba([features])[0]
        
        # Convert NumPy types to native Python types
        if hasattr(prediction, 'item'):
            attack_type = prediction.item()
        else:
            attack_type = str(prediction)
        
        return {
            'attack_type': attack_type,
            'confidence': float(np.max(probabilities)),
            'probabilities': dict(zip(
                [str(c) for c in self.random_forest.classes_],
                [float(p) for p in probabilities.tolist()]
            ))
        }
    
    def detect_anomaly_isolation_forest(self, features):
        """Detect anomaly using Isolation Forest."""
        if self.isolation_forest is None:
            raise ValueError("Isolation Forest model not loaded")
        
        prediction = self.isolation_forest.predict([features])[0]
        score = self.isolation_forest.score_samples([features])[0]
        
        # Convert: -1 (anomaly) to True, 1 (normal) to False
        is_anomaly = prediction == -1
        
        return {
            'is_anomaly': bool(is_anomaly),
            'anomaly_score': float(score)
        }
    
    def detect_anomaly_autoencoder(self, features, threshold=0.1):
        """Detect anomaly using Autoencoder."""
        if self.autoencoder is None:
            raise ValueError("Autoencoder model not loaded")
        
        features_reshaped = np.array([features])
        reconstructed = self.autoencoder.predict(features_reshaped, verbose=0)
        
        # Calculate reconstruction error
        reconstruction_error = np.mean(np.abs(features_reshaped - reconstructed))
        
        is_anomaly = reconstruction_error > threshold
        
        return {
            'is_anomaly': bool(is_anomaly),
            'reconstruction_error': float(reconstruction_error),
            'threshold': threshold
        }
    
    def combine_predictions(self, features, rf_weight=0.4, if_weight=0.3, ae_weight=0.3):
        """Combine predictions from multiple models."""
        results = {
            'attack_type': None,
            'is_anomaly': False,
            'confidence': 0.0,
            'scores': {}
        }
        
        # Random Forest prediction
        try:
            rf_result = self.predict_attack_type(features)
            results['attack_type'] = rf_result['attack_type']
            results['confidence'] = rf_result['confidence']
            results['scores']['random_forest'] = rf_result
        except Exception as e:
            logger.error(f"Random Forest prediction failed: {e}")
        
        # Isolation Forest prediction
        try:
            if_result = self.detect_anomaly_isolation_forest(features)
            results['scores']['isolation_forest'] = if_result
            if if_result['is_anomaly']:
                results['is_anomaly'] = True
        except Exception as e:
            logger.error(f"Isolation Forest prediction failed: {e}")
        
        # Autoencoder prediction
        try:
            ae_result = self.detect_anomaly_autoencoder(features)
            results['scores']['autoencoder'] = ae_result
            if ae_result['is_anomaly']:
                results['is_anomaly'] = True
        except Exception as e:
            logger.error(f"Autoencoder prediction failed: {e}")
        
        # Calculate composite score (FIXED: use normalized scores, not binary flags)
        composite_score = 0.0
        score_components = {}
        
        # Random Forest contribution: confidence score (0-1)
        if 'random_forest' in results['scores']:
            rf_confidence = float(results['confidence'])
            # Only count confidence if attack is NOT Normal
            # Normal traffic should have low threat score
            attack_type = str(results.get('attack_type', 'Normal')).lower()
            if attack_type == 'normal':
                rf_threat_score = 0.0  # Normal = no threat
            else:
                # Higher confidence for attack types = higher threat
                rf_threat_score = rf_confidence
            rf_contribution = rf_threat_score * rf_weight
            composite_score += rf_contribution
            score_components['rf_confidence'] = rf_confidence
            score_components['rf_threat_score'] = rf_threat_score
            score_components['rf_contribution'] = rf_contribution
        
        # Isolation Forest contribution: use anomaly score (not binary)
        if 'isolation_forest' in results['scores']:
            if_score = float(results['scores']['isolation_forest'].get('anomaly_score', 0))
            # Anomaly scores are typically negative (lower = more anomalous)
            # Normalize: convert to 0-1 scale where lower scores = higher threat
            # Typical range: -0.5 to 0.5, normalize to 0-1
            if_threat_score = max(0, min(1, (0.5 - if_score) / 1.0))  # Normalize -0.5 to 0.5 -> 0 to 1
            if_contribution = if_threat_score * if_weight
            composite_score += if_contribution
            score_components['if_anomaly_score'] = if_score
            score_components['if_threat_score'] = if_threat_score
            score_components['if_contribution'] = if_contribution
        
        # Autoencoder contribution: use reconstruction error (normalized)
        if 'autoencoder' in results['scores']:
            ae_error = float(results['scores']['autoencoder'].get('reconstruction_error', 0))
            ae_threshold = float(results['scores']['autoencoder'].get('threshold', 0.1))
            # Normalize error: higher error relative to threshold = higher threat
            # If error > threshold, threat = 1, otherwise scale linearly
            if ae_threshold > 0:
                ae_threat_score = min(1.0, ae_error / (ae_threshold * 2))  # Scale so error at 2x threshold = 1.0
            else:
                ae_threat_score = min(1.0, ae_error / 0.2)  # Default scaling
            ae_contribution = ae_threat_score * ae_weight
            composite_score += ae_contribution
            score_components['ae_error'] = ae_error
            score_components['ae_threshold'] = ae_threshold
            score_components['ae_threat_score'] = ae_threat_score
            score_components['ae_contribution'] = ae_contribution
        
        # Ensure composite score is between 0 and 1
        composite_score = max(0.0, min(1.0, composite_score))
        
        results['composite_score'] = float(composite_score)
        results['score_components'] = score_components  # Store for debugging
        
        return results

