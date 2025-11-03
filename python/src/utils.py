"""Utility functions for SmartShield."""
import os
import json
import logging
from datetime import datetime
from pathlib import Path

def setup_logging(log_dir='logs', log_level=logging.INFO):
    """Setup logging configuration.
    
    Args:
        log_dir: Directory to store log files
        log_level: Logging level (logging.INFO, logging.DEBUG, etc.)
    """
    Path(log_dir).mkdir(exist_ok=True)
    
    log_file = os.path.join(log_dir, f'smartshield_{datetime.now().strftime("%Y%m%d")}.log')
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    simple_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File handler with detailed format (includes file:line)
    # Use append mode and handle permission errors gracefully
    try:
        # Try to fix permissions if file exists and is owned by root
        if os.path.exists(log_file):
            try:
                os.chmod(log_file, 0o664)  # Make sure it's writable
            except (OSError, PermissionError):
                pass  # If we can't fix permissions, try anyway
        
        file_handler = logging.FileHandler(log_file, mode='a')  # Append mode
        file_handler.setLevel(log_level)
        file_handler.setFormatter(detailed_formatter)
    except (OSError, PermissionError) as e:
        # If we can't write to the file (e.g., owned by root), skip file logging
        import sys
        print(f"Warning: Could not create log file {log_file}: {e}", file=sys.stderr)
        print("Continuing with console logging only...", file=sys.stderr)
        file_handler = None
    
    # Console handler with simple format
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(simple_formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Only add file handler if it was successfully created
    if file_handler is not None:
        root_logger.addHandler(file_handler)
        log_file_info = f"File: {log_file}"
    else:
        log_file_info = "File: (console only, permission denied)"
    
    root_logger.addHandler(console_handler)
    
    logger = logging.getLogger('SmartShield')
    logger.info(f"Logging initialized - Level: {logging.getLevelName(log_level)}, {log_file_info}")
    
    return logger

def get_project_root():
    """Get project root directory."""
    return Path(__file__).parent.parent.parent

def ensure_dir(path):
    """Ensure directory exists."""
    Path(path).mkdir(parents=True, exist_ok=True)

def save_json(data, filepath):
    """Save data to JSON file."""
    ensure_dir(os.path.dirname(filepath))
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

def load_json(filepath):
    """Load data from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)

