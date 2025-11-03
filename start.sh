#!/bin/bash

# SmartShield Startup Script
# Runs all components of the SmartShield system

echo "=========================================="
echo "SmartShield AI-Powered IDS"
echo "Starting all services..."
echo "=========================================="

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to check if a process is running
check_process() {
    if lsof -Pi :$1 -sTCP:LISTEN -t >/dev/null 2>&1 ; then
        echo -e "${RED}Port $1 is already in use${NC}"
        return 1
    else
        return 0
    fi
}

# Check ports
echo "Checking ports..."
check_process 5000 || exit 1
check_process 3001 || exit 1
check_process 3000 || exit 1

# Start Flask ML API
echo -e "${GREEN}Starting Flask ML API on port 5000...${NC}"
cd python
source venv/bin/activate
python api_server.py > ../logs/ml_api.log 2>&1 &
ML_API_PID=$!
cd ..

sleep 3

# Start Express Backend
echo -e "${GREEN}Starting Express Backend on port 3001...${NC}"
cd backend
npm start > ../logs/backend.log 2>&1 &
BACKEND_PID=$!
cd ..

sleep 3

# Start Frontend
echo -e "${GREEN}Starting Next.js Frontend on port 3000...${NC}"
cd frontend
npm run dev > ../logs/frontend.log 2>&1 &
FRONTEND_PID=$!
cd ..

sleep 5

# Start Live Detection (requires root for packet capture)
echo -e "${YELLOW}Starting Live Packet Detection...${NC}"
echo -e "${YELLOW}Note: This requires root privileges for packet capture${NC}"
cd python
source venv/bin/activate

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo -e "${RED}Not running as root. Starting detection without packet capture...${NC}"
    echo -e "${YELLOW}For real packet capture, run: sudo ./start.sh${NC}"
    python live_detection.py > ../logs/detection.log 2>&1 &
    DETECTION_PID=$!
else
    python live_detection.py --interface any > ../logs/detection.log 2>&1 &
    DETECTION_PID=$!
fi
cd ..

# Create logs directory if it doesn't exist
mkdir -p logs

# Save PIDs
echo $ML_API_PID > logs/ml_api.pid
echo $BACKEND_PID > logs/backend.pid
echo $FRONTEND_PID > logs/frontend.pid
echo $DETECTION_PID > logs/detection.pid

echo ""
echo "=========================================="
echo -e "${GREEN}SmartShield started successfully!${NC}"
echo "=========================================="
echo ""
echo "Services running:"
echo "  - ML API:      http://localhost:5000"
echo "  - Backend:     http://localhost:3001"
echo "  - Frontend:    http://localhost:3000"
echo ""
echo "PIDs saved in logs/ directory"
echo ""
echo "To stop all services: ./stop.sh"
echo "To view logs: tail -f logs/*.log"
echo ""
echo "Opening dashboard in browser..."
sleep 2
open http://localhost:3000 || xdg-open http://localhost:3000 || start http://localhost:3000


