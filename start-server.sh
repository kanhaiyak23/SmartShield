#!/bin/bash
# Start Flask backend server for SmartShield

echo "Starting SmartShield Flask Backend..."
echo "=========================================="

# Activate virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
else
    source venv/bin/activate
fi

# Check if running as root/admin (required for Scapy packet capture)
if [ "$EUID" -ne 0 ]; then 
    echo ""
    echo "⚠️  WARNING: Not running as root/admin"
    echo "Packet capture requires elevated privileges."
    echo "On macOS, you may need to run: sudo ./start-server.sh"
    echo ""
    echo "Attempting to start anyway (may fail on packet capture)..."
    echo ""
fi

# Start the server
python3 server.py

