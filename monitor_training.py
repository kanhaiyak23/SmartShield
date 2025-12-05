#!/usr/bin/env python3
"""
Real-time monitoring of Random Forest training progress
"""

import os
import time
import psutil
import subprocess
from datetime import datetime

def get_process_info(pid):
    """Get process information"""
    try:
        process = psutil.Process(pid)
        return {
            'cpu_percent': process.cpu_percent(interval=0.1),
            'memory_mb': process.memory_info().rss / 1024 / 1024,
            'status': process.status(),
            'is_running': process.is_running()
        }
    except:
        return None

def check_training_progress():
    """Check training progress by monitoring files and process"""
    # Find training process
    training_pid = None
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = ' '.join(proc.info['cmdline'] or [])
            if 'train_enhanced_rf.py' in cmdline:
                training_pid = proc.info['pid']
                break
        except:
            continue
    
    if not training_pid:
        return None, "Training process not found"
    
    # Check model files
    model_files = {
        'random_forest_model.joblib': False,
        'rf_feature_scaler.joblib': False,
        'service_encoder.joblib': False,
        'state_encoder.joblib': False
    }
    
    for filename in model_files.keys():
        if os.path.exists(filename):
            model_files[filename] = True
    
    # Estimate progress
    files_created = sum(model_files.values())
    total_files = len(model_files)
    
    # If all files exist, training is complete
    if files_created == total_files:
        return training_pid, "COMPLETE", 100
    
    # Estimate based on process activity
    proc_info = get_process_info(training_pid)
    if not proc_info:
        return None, "Process not accessible"
    
    # Rough progress estimation (this is approximate)
    # Training has phases: loading (0-30%), feature extraction (30-60%), training (60-100%)
    # We can't know exact phase, so estimate based on time and activity
    progress = min(95, files_created * 25)  # Each file = ~25% progress
    
    return training_pid, proc_info, progress, model_files

def monitor_training():
    """Monitor training in real-time"""
    print("=" * 70)
    print("Random Forest Training Monitor")
    print("=" * 70)
    print()
    
    start_time = time.time()
    last_status = None
    
    while True:
        os.system('clear' if os.name != 'nt' else 'cls')
        print("=" * 70)
        print("Random Forest Training Monitor - Real-time Progress")
        print("=" * 70)
        print(f"Time: {datetime.now().strftime('%H:%M:%S')}")
        print()
        
        result = check_training_progress()
        
        if result[0] is None:
            print("❌ Training process not found!")
            print("   Training may have completed or crashed.")
            break
        
        if result[1] == "COMPLETE":
            print("✅ TRAINING COMPLETE!")
            print()
            print("Model files created:")
            for filename, exists in result[3].items():
                status = "✅" if exists else "❌"
                print(f"  {status} {filename}")
            print()
            elapsed = time.time() - start_time
            print(f"Total training time: {elapsed/60:.2f} minutes")
            break
        
        pid, proc_info, progress, model_files = result
        
        # Progress bar
        bar_width = 50
        filled = int(bar_width * progress / 100)
        bar = '█' * filled + '░' * (bar_width - filled)
        
        print(f"Progress: [{bar}] {progress:.1f}%")
        print()
        print(f"Process ID: {pid}")
        print(f"CPU Usage: {proc_info['cpu_percent']:.1f}%")
        print(f"Memory Usage: {proc_info['memory_mb']:.1f} MB")
        print(f"Status: {proc_info['status']}")
        print()
        
        print("Model Files Status:")
        for filename, exists in model_files.items():
            status = "✅ Created" if exists else "⏳ Pending"
            if exists:
                size = os.path.getsize(filename) / 1024 / 1024
                print(f"  {status:12s} {filename:30s} ({size:.2f} MB)")
            else:
                print(f"  {status:12s} {filename:30s}")
        
        print()
        elapsed = time.time() - start_time
        print(f"Elapsed Time: {elapsed/60:.2f} minutes")
        
        # Phase estimation
        if progress < 30:
            phase = "Loading Dataset"
        elif progress < 60:
            phase = "Extracting Features"
        elif progress < 90:
            phase = "Training Model"
        else:
            phase = "Finalizing"
        
        print(f"Current Phase: {phase}")
        print()
        print("Press Ctrl+C to stop monitoring (training will continue)")
        
        time.sleep(2)  # Update every 2 seconds

if __name__ == '__main__':
    try:
        monitor_training()
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped. Training continues in background.")
        print("Check for model files: random_forest_model.joblib")

