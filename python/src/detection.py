"""Detection engine combining ML models and packet analysis."""
import logging
from collections import deque
import time
from .packet_capture import PacketCapture
from .models import ModelTrainer
from .preprocessing import DataPreprocessor

logger = logging.getLogger('SmartShield')

class DetectionEngine:
    """Main detection engine that combines packet capture and ML models."""
    
    def __init__(self, models_dir='models', anomaly_threshold=0.5):
        self.packet_capture = PacketCapture()
        self.model_trainer = ModelTrainer(models_dir)
        self.preprocessor = DataPreprocessor()
        self.anomaly_threshold = anomaly_threshold
        
        # Alert management
        self.alerts = deque(maxlen=1000)
        self.alert_history = []
        
    def initialize_models(self):
        """Initialize and load trained models."""
        logger.info("Initializing ML models...")
        self.model_trainer.load_models()
        
        # Load preprocessor scalers and encoders
        # This would need to be saved/loaded separately in production
        logger.info("Models initialized successfully")
    
    def start_detection(self, interface=None, packet_filter=None):
        """Start real-time detection."""
        self.packet_capture.interface = interface
        self.packet_capture.start_capture(packet_filter)
        logger.info("Detection started")
    
    def stop_detection(self):
        """Stop real-time detection."""
        self.packet_capture.stop_capture()
        logger.info("Detection stopped")
    
    def process_packet(self, packet_features):
        """Process a single packet through ML models."""
        try:
            # Preprocess features
            processed_features = self.preprocessor.preprocess_packet_features(packet_features)
            
            # Get predictions from models
            results = self.model_trainer.combine_predictions(processed_features.iloc[0])
            
            # Generate alert if anomaly detected
            if results['is_anomaly'] or results['composite_score'] > self.anomaly_threshold:
                alert = self._create_alert(packet_features, results)
                self.alerts.append(alert)
                self.alert_history.append(alert)
                logger.warning(f"Alert generated: {alert['type']} - {alert['severity']}")
                
            return results
            
        except Exception as e:
            logger.error(f"Error processing packet: {e}")
            return None
    
    def _create_alert(self, packet_features, detection_results):
        """Create alert from detection results."""
        alert = {
            'timestamp': packet_features.get('timestamp', time.time()),
            'src_ip': packet_features.get('src_ip', 'unknown'),
            'dst_ip': packet_features.get('dst_ip', 'unknown'),
            'protocol': packet_features.get('protocol', 'unknown'),
            'packet_length': packet_features.get('packet_length', 0),
            'attack_type': detection_results.get('attack_type', 'Unknown'),
            'severity': self._calculate_severity(detection_results),
            'is_anomaly': detection_results.get('is_anomaly', False),
            'confidence': detection_results.get('confidence', 0.0),
            'composite_score': detection_results.get('composite_score', 0.0)
        }
        
        return alert
    
    def _calculate_severity(self, results):
        """Calculate alert severity based on detection results."""
        score = results.get('composite_score', 0.0)
        attack_type = results.get('attack_type', '')
        
        # Critical attacks
        critical_attacks = ['ddos', 'backdoor', 'exploits']
        if any(attack in str(attack_type).lower() for attack in critical_attacks):
            if score > 0.8:
                return 'Critical'
        
        # Severity based on composite score
        if score > 0.8:
            return 'High'
        elif score > 0.6:
            return 'Medium'
        elif score > 0.4:
            return 'Low'
        else:
            return 'Info'
    
    def get_recent_alerts(self, n=10):
        """Get recent n alerts."""
        return list(self.alerts)[-n:]
    
    def get_alerts_by_severity(self, severity=None):
        """Get alerts filtered by severity."""
        if severity:
            return [alert for alert in self.alert_history if alert['severity'] == severity]
        return list(self.alert_history)
    
    def get_statistics(self):
        """Get detection statistics."""
        stats = self.packet_capture.get_stats()
        
        # Add alert statistics
        alerts_by_severity = {
            'Critical': len(self.get_alerts_by_severity('Critical')),
            'High': len(self.get_alerts_by_severity('High')),
            'Medium': len(self.get_alerts_by_severity('Medium')),
            'Low': len(self.get_alerts_by_severity('Low')),
            'Total': len(self.alert_history)
        }
        
        stats['alerts'] = alerts_by_severity
        
        return stats
    
    def reset(self):
        """Reset detection engine."""
        self.packet_capture.reset_stats()
        self.alerts.clear()
        self.alert_history.clear()
        logger.info("Detection engine reset")


