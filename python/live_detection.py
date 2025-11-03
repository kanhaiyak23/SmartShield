"""Live packet detection with backend integration."""
import os
import sys
import argparse
import logging
import time
import requests
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.packet_capture import PacketCapture
from src.preprocessing import DataPreprocessor
from src.models import ModelTrainer
from src.utils import setup_logging

logger = setup_logging()

class LiveDetection:
    """Live detection with backend integration."""
    
    def __init__(self, models_dir='models', backend_url='http://localhost:3001', verbose=True, log_every_n=20):
        self.backend_url = backend_url
        self.packet_capture = PacketCapture()
        self.preprocessor = DataPreprocessor()
        self.model_trainer = ModelTrainer(models_dir)
        self.alert_count = 0
        self.verbose = verbose  # Enable/disable detailed logging
        self.log_every_n = log_every_n  # Log detailed info every N packets
        self.packets_processed = 0
        
    def initialize(self):
        """Initialize models and load preprocessor."""
        logger.info("=" * 80)
        logger.info("🔧 INITIALIZING DETECTION SYSTEM")
        logger.info("=" * 80)
        logger.info("Loading ML models...")
        self.model_trainer.load_models()
        
        # Load preprocessor
        import joblib
        preprocessor_path = os.path.join('models', 'preprocessor.pkl')
        if os.path.exists(preprocessor_path):
            self.preprocessor = joblib.load(preprocessor_path)
            logger.info("✅ Preprocessor loaded successfully")
            
            # Log preprocessor configuration
            logger.info("=" * 80)
            logger.info("📊 PREPROCESSOR CONFIGURATION")
            logger.info("=" * 80)
            
            if hasattr(self.preprocessor, 'numeric_cols') and self.preprocessor.numeric_cols:
                logger.info(f"  Numeric Features ({len(self.preprocessor.numeric_cols)}):")
                for col in self.preprocessor.numeric_cols[:20]:  # Show first 20
                    logger.info(f"    - {col}")
                if len(self.preprocessor.numeric_cols) > 20:
                    logger.info(f"    ... and {len(self.preprocessor.numeric_cols) - 20} more")
            
            if hasattr(self.preprocessor, 'categorical_cols') and self.preprocessor.categorical_cols:
                logger.info(f"  Categorical Features ({len(self.preprocessor.categorical_cols)}):")
                for col in self.preprocessor.categorical_cols:
                    logger.info(f"    - {col}")
            
            # Log feature names if available
            if hasattr(self.preprocessor, 'get_feature_names'):
                feature_names = self.preprocessor.get_feature_names()
                logger.info(f"  Total Expected Features: {len(feature_names)}")
                logger.info(f"  Feature Order: {feature_names[:10]}...")  # First 10
        else:
            logger.warning("⚠️  Preprocessor not found, using default")
        
        logger.info("=" * 80)
        logger.info("✅ Initialization complete - Ready for packet detection")
        logger.info("=" * 80)
        logger.info("")
    
    def process_packets(self, num_packets=10):
        """Process captured packets."""
        packets = self.packet_capture.get_recent_packets(num_packets)
        
        packet_count = 0
        for packet_features in packets:
            packet_count += 1
            self.packets_processed += 1
            should_log_detailed = self.verbose and (
                self.packets_processed <= 3 or  # Always log first 3 packets
                self.packets_processed % self.log_every_n == 0 or  # Then every N packets
                packet_count == 1  # Always log first in batch
            )
            
            try:
                if should_log_detailed:
                    # Log raw packet features BEFORE preprocessing
                    logger.info("=" * 80)
                    logger.info(f"📦 PACKET #{self.packets_processed} - RAW FEATURES (Before Preprocessing)")
                    logger.info("=" * 80)
                    for key, value in packet_features.items():
                        logger.info(f"  {key}: {value}")
                
                # Preprocess
                processed_features = self.preprocessor.preprocess_packet_features(packet_features)
                
                if should_log_detailed:
                    # Log preprocessed features (what model sees)
                    logger.info("=" * 80)
                    logger.info(f"🔧 PREPROCESSED FEATURES (What Model Receives)")
                    logger.info("=" * 80)
                    logger.info(f"  Feature columns ({len(processed_features.columns)}): {list(processed_features.columns)[:10]}...")
                    logger.info(f"  Sample feature values (first 10):")
                    for col in processed_features.columns[:10]:
                        value = processed_features[col].iloc[0]
                        logger.info(f"    {col}: {value}")
                    logger.info(f"  Feature statistics:")
                    logger.info(f"    Min: {processed_features.min().min():.4f}, Max: {processed_features.max().max():.4f}")
                    logger.info(f"    Mean: {processed_features.mean().mean():.4f}, Std: {processed_features.std().mean():.4f}")
                
                # Predict
                results = self.model_trainer.combine_predictions(processed_features.iloc[0])
                
                # Always log summary, detailed logs only when enabled
                composite_score = float(results.get('composite_score', 0.0))
                is_anomaly = bool(results.get('is_anomaly', False))
                attack_type = results.get('attack_type', 'N/A')
                confidence = results.get('confidence', 0.0)
                severity = self._calculate_severity(results)
                
                # Quick summary for every packet
                logger.info(
                    f"Packet #{self.packets_processed}: {attack_type} | "
                    f"Confidence: {confidence:.3f} | "
                    f"Score: {composite_score:.3f} | "
                    f"Anomaly: {is_anomaly} | "
                    f"Severity: {severity}"
                )
                
                if should_log_detailed:
                    # Log detailed model predictions
                    logger.info("=" * 80)
                    logger.info(f"🤖 MODEL PREDICTIONS")
                    logger.info("=" * 80)
                    
                    # Random Forest results
                    if 'random_forest' in results.get('scores', {}):
                        rf_scores = results['scores']['random_forest']
                        logger.info(f"  Random Forest:")
                        logger.info(f"    Attack Type: {rf_scores.get('attack_type', 'N/A')}")
                        logger.info(f"    Confidence: {rf_scores.get('confidence', 0):.4f}")
                        if 'probabilities' in rf_scores:
                            logger.info(f"    Top 5 Probabilities:")
                            sorted_probs = sorted(rf_scores['probabilities'].items(), key=lambda x: x[1], reverse=True)[:5]
                            for attack, prob in sorted_probs:
                                logger.info(f"      {attack}: {prob:.4f}")
                    
                    # Isolation Forest results
                    if 'isolation_forest' in results.get('scores', {}):
                        if_scores = results['scores']['isolation_forest']
                        logger.info(f"  Isolation Forest:")
                        logger.info(f"    Is Anomaly: {if_scores.get('is_anomaly', False)}")
                        logger.info(f"    Anomaly Score: {if_scores.get('anomaly_score', 0):.4f}")
                    
                    # Autoencoder results
                    if 'autoencoder' in results.get('scores', {}):
                        ae_scores = results['scores']['autoencoder']
                        logger.info(f"  Autoencoder:")
                        logger.info(f"    Is Anomaly: {ae_scores.get('is_anomaly', False)}")
                        logger.info(f"    Reconstruction Error: {ae_scores.get('reconstruction_error', 0):.4f}")
                        logger.info(f"    Threshold: {ae_scores.get('threshold', 0):.4f}")
                    
                    # Final results
                    logger.info(f"  Final Combined Results:")
                    logger.info(f"    Attack Type: {results.get('attack_type', 'N/A')}")
                    logger.info(f"    Is Anomaly: {results.get('is_anomaly', False)}")
                    logger.info(f"    Confidence: {results.get('confidence', 0):.4f}")
                    logger.info(f"    Composite Score: {results.get('composite_score', 0):.4f}")
                    
                    # Log composite score breakdown (FIXED calculation)
                    if 'score_components' in results:
                        logger.info(f"  Composite Score Breakdown:")
                        components = results['score_components']
                        logger.info(f"    Random Forest:")
                        rf_conf = components.get('rf_confidence', 'N/A')
                        if rf_conf != 'N/A':
                            logger.info(f"      Confidence: {float(rf_conf):.4f}")
                        else:
                            logger.info(f"      Confidence: {rf_conf}")
                        logger.info(f"      Threat Score: {components.get('rf_threat_score', 0):.4f} (Normal=0, Attack=confidence)")
                        logger.info(f"      Contribution: {components.get('rf_threat_score', 0):.4f} × 0.4 = {components.get('rf_contribution', 0):.4f}")
                        
                        logger.info(f"    Isolation Forest:")
                        if_score = components.get('if_anomaly_score', 'N/A')
                        if if_score != 'N/A':
                            logger.info(f"      Anomaly Score: {float(if_score):.4f}")
                        else:
                            logger.info(f"      Anomaly Score: {if_score}")
                        logger.info(f"      Threat Score: {components.get('if_threat_score', 0):.4f} (normalized)")
                        logger.info(f"      Contribution: {components.get('if_threat_score', 0):.4f} × 0.3 = {components.get('if_contribution', 0):.4f}")
                        
                        logger.info(f"    Autoencoder:")
                        ae_error = components.get('ae_error', 'N/A')
                        ae_thresh = components.get('ae_threshold', 'N/A')
                        if ae_error != 'N/A' and ae_thresh != 'N/A':
                            logger.info(f"      Error: {float(ae_error):.4f}, Threshold: {float(ae_thresh):.4f}")
                        else:
                            logger.info(f"      Error: {ae_error}, Threshold: {ae_thresh}")
                        logger.info(f"      Threat Score: {components.get('ae_threat_score', 0):.4f} (normalized)")
                        logger.info(f"      Contribution: {components.get('ae_threat_score', 0):.4f} × 0.3 = {components.get('ae_contribution', 0):.4f}")
                        
                        logger.info(f"    Total Composite Score: {results.get('composite_score', 0):.4f} (0-1 scale)")
                    
                    # Calculate severity
                    logger.info(f"    Calculated Severity: {severity}")
                    logger.info(f"    Severity Calculation Details:")
                    score = float(results.get('composite_score', 0.0))
                    attack_type_str = str(results.get('attack_type', ''))
                    logger.info(f"      - Composite Score: {score:.4f}")
                    logger.info(f"      - Attack Type: {attack_type_str}")
                    logger.info(f"      - Score > 0.8: {score > 0.8} (→ High)")
                    logger.info(f"      - Score > 0.6: {score > 0.6} (→ Medium)")
                    logger.info(f"      - Score > 0.4: {score > 0.4} (→ Low)")
                    
                    logger.info("=" * 80)
                    logger.info(f"🚨 ALERT DECISION")
                    logger.info("=" * 80)
                    logger.info(f"  Is Anomaly: {is_anomaly}")
                    logger.info(f"  Composite Score: {composite_score:.4f}")
                    logger.info(f"  Threshold: 0.3")
                    logger.info(f"  Will Send Alert: {is_anomaly or composite_score > 0.3}")
                    logger.info("=" * 80)
                    logger.info("")  # Blank line for readability
                
                # Check for anomalies (lowered threshold from 0.5 to 0.3 for better detection)
                if is_anomaly or composite_score > 0.3:
                    if should_log_detailed:
                        logger.info("  ✅ Sending alert to backend...")
                    self._send_alert(packet_features, results)
                elif should_log_detailed:
                    logger.info("  ⏭️  Skipping alert (below threshold)")
                    
            except Exception as e:
                logger.error("=" * 80)
                logger.error(f"❌ ERROR PROCESSING PACKET #{packet_count}")
                logger.error("=" * 80)
                logger.error(f"Error: {e}", exc_info=True)
                logger.error("=" * 80)
    
    def _send_alert(self, packet_features, detection_results):
        """Send alert to backend."""
        # Helper function to convert NumPy types to native Python types
        def to_native_type(value):
            """Convert NumPy/pandas types to native Python types for JSON serialization."""
            import numpy as np
            if isinstance(value, (np.integer, np.int_, np.intc, np.intp, np.int8,
                                   np.int16, np.int32, np.int64)):
                return int(value)
            elif isinstance(value, (np.floating, np.float_, np.float16, np.float32, np.float64)):
                return float(value)
            elif isinstance(value, np.bool_):
                return bool(value)
            elif isinstance(value, np.ndarray):
                return value.tolist()
            elif isinstance(value, str):
                # Convert numpy string types
                return str(value)
            else:
                return value
        
        # Convert all values to JSON-serializable types
        attack_type = detection_results.get('attack_type', 'Unknown')
        if attack_type is not None:
            attack_type = to_native_type(attack_type)
            # If it's still not a string, convert it
            if not isinstance(attack_type, str):
                attack_type = str(attack_type)
        
        alert = {
            'timestamp': packet_features.get('timestamp', datetime.now().isoformat()),
            'src_ip': str(packet_features.get('src_ip', 'unknown')),
            'dst_ip': str(packet_features.get('dst_ip', 'unknown')),
            'protocol': str(packet_features.get('protocol', 'unknown')),
            'packet_length': to_native_type(packet_features.get('packet_length', 0)),
            'attack_type': attack_type if attack_type else 'Unknown',
            'severity': self._calculate_severity(detection_results),
            'is_anomaly': to_native_type(detection_results.get('is_anomaly', False)),
            'confidence': to_native_type(detection_results.get('confidence', 0.0)),
            'composite_score': to_native_type(detection_results.get('composite_score', 0.0))
        }
        
        try:
            response = requests.post(f'{self.backend_url}/api/alerts', json=alert, timeout=5)
            if response.status_code == 200:
                self.alert_count += 1
                logger.warning(
                    f"[{alert['severity']}] {alert['attack_type']} "
                    f"from {alert['src_ip']} to {alert['dst_ip']}"
                )
            else:
                logger.error(f"Backend returned status {response.status_code}: {response.text}")
        except Exception as e:
            logger.error(f"Failed to send alert: {e}")
            logger.debug(f"Alert data: {alert}")
    
    def _calculate_severity(self, results):
        """Calculate alert severity with improved thresholds."""
        score = float(results.get('composite_score', 0.0))
        attack_type = str(results.get('attack_type', 'Normal')).lower()
        confidence = float(results.get('confidence', 0.0))
        
        # Log severity calculation
        logger.debug(f"Calculating severity - Score: {score:.4f}, Attack Type: {attack_type}, Confidence: {confidence:.4f}")
        
        # Normal traffic should be Info severity
        if attack_type == 'normal' or attack_type == '0':
            logger.debug(f"  -> Info (Normal traffic)")
            return 'Info'
        
        # Check for critical attack types
        critical_attacks = ['dos', 'ddos', 'backdoor', 'exploits', 'shellcode', 'worms']
        is_critical_attack = any(attack in attack_type for attack in critical_attacks)
        
        # Adjusted thresholds to prevent all packets being "High"
        # Consider both score and confidence
        combined_risk = (score * 0.6 + confidence * 0.4)  # Weighted combination
        
        if is_critical_attack:
            if combined_risk > 0.8:
                logger.debug(f"  -> Critical (critical attack + high risk: {combined_risk:.4f})")
                return 'Critical'
            elif combined_risk > 0.6:
                logger.debug(f"  -> High (critical attack + medium risk: {combined_risk:.4f})")
                return 'High'
            elif combined_risk > 0.4:
                logger.debug(f"  -> Medium (critical attack + low risk: {combined_risk:.4f})")
                return 'Medium'
            else:
                logger.debug(f"  -> Low (critical attack + very low risk: {combined_risk:.4f})")
                return 'Low'
        
        # Score-based severity for non-critical attacks (ADJUSTED THRESHOLDS)
        if combined_risk > 0.7:  # Was 0.8, lowered but still high
            logger.debug(f"  -> High (combined risk > 0.7: {combined_risk:.4f})")
            return 'High'
        elif combined_risk > 0.5:  # Was 0.6
            logger.debug(f"  -> Medium (0.5 < combined risk <= 0.7: {combined_risk:.4f})")
            return 'Medium'
        elif combined_risk > 0.3:  # Was 0.4
            logger.debug(f"  -> Low (0.3 < combined risk <= 0.5: {combined_risk:.4f})")
            return 'Low'
        else:
            logger.debug(f"  -> Info (combined risk <= 0.3: {combined_risk:.4f})")
            return 'Info'
    
    def update_stats(self):
        """Send statistics to backend."""
        # Helper function to convert NumPy types to native Python types
        def to_native_type(value):
            """Convert NumPy/pandas types to native Python types for JSON serialization."""
            import numpy as np
            if isinstance(value, (np.integer, np.int_, np.intc, np.intp, np.int8,
                                   np.int16, np.int32, np.int64)):
                return int(value)
            elif isinstance(value, (np.floating, np.float_, np.float16, np.float32, np.float64)):
                return float(value)
            elif isinstance(value, np.bool_):
                return bool(value)
            elif isinstance(value, np.ndarray):
                return value.tolist()
            else:
                return value
        
        stats = self.packet_capture.get_stats()
        # Convert all NumPy types to native Python types
        stats = {k: to_native_type(v) for k, v in stats.items()}
        stats['alerts'] = {
            'Critical': 0,
            'High': 0,
            'Medium': 0,
            'Low': 0,
            'Total': self.alert_count
        }
        
        try:
            requests.post(f'{self.backend_url}/api/statistics', json=stats, timeout=5)
        except Exception as e:
            logger.error(f"Failed to update stats: {e}")
    
    def run(self, interface=None, filter_str=None):
        """Run live detection."""
        self.initialize()
        
        # Start packet capture
        self.packet_capture.interface = interface
        self.packet_capture.start_capture(filter_str)
        logger.info("Live detection started")
        
        last_stats_time = time.time()
        
        try:
            while True:
                time.sleep(2)
                
                # Process packets
                self.process_packets(20)
                
                # Update statistics
                if time.time() - last_stats_time > 10:
                    self.update_stats()
                    last_stats_time = time.time()
                    
                    logger.info(
                        f"Packets: {self.packet_capture.stats['total_packets']}, "
                        f"Alerts: {self.alert_count}"
                    )
                    
        except KeyboardInterrupt:
            logger.info("\nStopping detection...")
        finally:
            self.packet_capture.stop_capture()
            logger.info("Detection stopped")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Live SmartShield detection')
    parser.add_argument('--interface', type=str, default=None,
                        help='Network interface')
    parser.add_argument('--filter', type=str, default=None,
                        help='BPF filter')
    parser.add_argument('--models-dir', type=str, default='models',
                        help='Models directory')
    parser.add_argument('--backend-url', type=str, default='http://localhost:3001',
                        help='Backend URL')
    
    args = parser.parse_args()
    
    detector = LiveDetection(
        models_dir=args.models_dir,
        backend_url=args.backend_url
    )
    detector.run(interface=args.interface, filter_str=args.filter)

if __name__ == '__main__':
    main()

