#!/bin/bash

# SmartShield Stop Script

echo "=========================================="
echo "Stopping SmartShield services..."
echo "=========================================="

# Kill processes by PID if files exist
if [ -f logs/ml_api.pid ]; then
    kill $(cat logs/ml_api.pid) 2>/dev/null
    rm logs/ml_api.pid
    echo "Stopped ML API"
fi

if [ -f logs/backend.pid ]; then
    kill $(cat logs/backend.pid) 2>/dev/null
    rm logs/backend.pid
    echo "Stopped Backend"
fi

if [ -f logs/frontend.pid ]; then
    kill $(cat logs/frontend.pid) 2>/dev/null
    rm logs/frontend.pid
    echo "Stopped Frontend"
fi

if [ -f logs/detection.pid ]; then
    kill $(cat logs/detection.pid) 2>/dev/null
    rm logs/detection.pid
    echo "Stopped Detection"
fi

# Also kill by process name (backup method)
pkill -f "api_server.py" 2>/dev/null
pkill -f "server.js" 2>/dev/null
pkill -f "next-server" 2>/dev/null
pkill -f "live_detection.py" 2>/dev/null

# Kill processes on specific ports
lsof -ti:5000 | xargs kill -9 2>/dev/null
lsof -ti:3001 | xargs kill -9 2>/dev/null
lsof -ti:3000 | xargs kill -9 2>/dev/null

echo ""
echo "All SmartShield services stopped"


