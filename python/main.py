"""Main entry point for SmartShield detection."""
import os
import sys
import argparse
import logging
import time
import json

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.detection import DetectionEngine
from src.utils import setup_logging

logger = setup_logging()

def main(args):
    """Main detection function."""
    logger.info("=" * 60)
    logger.info("SmartShield Detection System Starting...")
    logger.info("=" * 60)
    
    # Initialize detection engine
    engine = DetectionEngine(
        models_dir=args.models_dir,
        anomaly_threshold=args.threshold
    )
    
    # Initialize models
    engine.initialize_models()
    
    # Start detection
    try:
        engine.start_detection(
            interface=args.interface,
            packet_filter=args.filter
        )
        
        logger.info("Detection started. Press Ctrl+C to stop.")
        
        # Main loop - monitor and process packets
        last_stats_time = time.time()
        while True:
            time.sleep(5)  # Check every 5 seconds
            
            # Print statistics periodically
            if time.time() - last_stats_time > 30:  # Every 30 seconds
                stats = engine.get_statistics()
                logger.info("\n" + "=" * 60)
                logger.info("Statistics:")
                for key, value in stats.items():
                    logger.info(f"  {key}: {value}")
                
                # Show recent alerts
                recent_alerts = engine.get_recent_alerts(5)
                if recent_alerts:
                    logger.info("\nRecent Alerts:")
                    for alert in recent_alerts:
                        logger.warning(
                            f"  [{alert['severity']}] {alert['attack_type']} "
                            f"from {alert['src_ip']} to {alert['dst_ip']} "
                            f"(confidence: {alert['confidence']:.2f})"
                        )
                logger.info("=" * 60 + "\n")
                
                last_stats_time = time.time()
            
            # Process recent packets if in detection mode
            # In production, this would be done via callback
            # For now, we rely on the stats display
            
    except KeyboardInterrupt:
        logger.info("\nShutting down detection system...")
    finally:
        engine.stop_detection()
        logger.info("Detection system stopped.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SmartShield Detection System')
    parser.add_argument('--interface', type=str, default=None,
                        help='Network interface to monitor (e.g., eth0, wlan0)')
    parser.add_argument('--models-dir', type=str, default='models',
                        help='Directory containing trained models')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Anomaly detection threshold (0-1)')
    parser.add_argument('--filter', type=str, default=None,
                        help='BPF filter for packet capture (e.g., "tcp and port 80")')
    
    args = parser.parse_args()
    
    main(args)


