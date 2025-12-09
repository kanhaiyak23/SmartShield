#!/bin/bash
# SmartShield Academic Demo Script
# Easy launcher for academic presentation

echo "╔════════════════════════════════════════════════════════════╗"
echo "║     SmartShield - Academic Project Demo Launcher          ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if running as root (needed for packet capture)
if [ "$EUID" -ne 0 ]; then 
    echo -e "${YELLOW}⚠️  Not running as root${NC}"
    echo "   Some features may require sudo privileges"
    echo ""
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo -e "${RED}❌ Virtual environment not found!${NC}"
    echo "   Please run: python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Check if model files exist
if [ ! -f "random_forest_model.joblib" ]; then
    echo -e "${YELLOW}⚠️  Model files not found!${NC}"
    echo "   Model will run in simulation mode"
    echo "   To train model: python3 train_enhanced_rf.py"
    echo ""
else
    echo -e "${GREEN}✅ Model files found${NC}"
fi

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  Demo Options:"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "1. 🚀 Start Backend Server (Terminal 1)"
echo "2. 🎨 Start Frontend Dashboard (Terminal 2)"
echo "3. 🎯 Run Attack Simulator (Terminal 3)"
echo "4. 📊 Full Demo Sequence (Auto-start everything)"
echo "5. ✅ Check System Status"
echo "6. ❌ Exit"
echo ""
read -p "Select option (1-6): " choice

case $choice in
    1)
        echo ""
        echo -e "${GREEN}🚀 Starting Backend Server...${NC}"
        echo "   Backend will run on: http://127.0.0.1:5000"
        echo "   Press Ctrl+C to stop"
        echo ""
        python3 server.py
        ;;
    2)
        echo ""
        echo -e "${GREEN}🎨 Starting Frontend Dashboard...${NC}"
        echo "   Dashboard will run on: http://localhost:3000"
        echo "   Press Ctrl+C to stop"
        echo ""
        npm run dev
        ;;
    3)
        echo ""
        echo -e "${YELLOW}🎯 Starting Attack Simulator...${NC}"
        echo "   This will generate attacks for detection"
        echo ""
        python3 attack_simulator.py
        ;;
    4)
        echo ""
        echo -e "${GREEN}📊 Full Demo Sequence${NC}"
        echo ""
        echo "This will:"
        echo "  1. Check if backend is running"
        echo "  2. Launch attack simulator with demo sequence"
        echo ""
        read -p "Press Enter to continue or Ctrl+C to cancel..."
        
        # Check if backend is running
        if curl -s http://127.0.0.1:5000/health > /dev/null 2>&1; then
            echo -e "${GREEN}✅ Backend is running${NC}"
        else
            echo -e "${RED}❌ Backend is not running!${NC}"
            echo "   Please start backend first (option 1)"
            echo "   Or run: sudo python3 server.py"
            exit 1
        fi
        
        echo ""
        echo -e "${YELLOW}🎬 Starting Attack Simulation...${NC}"
        echo "   Watch your dashboard at http://localhost:3000"
        echo ""
        python3 attack_simulator.py --attack 5
        ;;
    5)
        echo ""
        echo -e "${GREEN}✅ Checking System Status...${NC}"
        echo ""
        
        # Check backend
        if curl -s http://127.0.0.1:5000/health > /dev/null 2>&1; then
            echo -e "${GREEN}✅ Backend: Running on http://127.0.0.1:5000${NC}"
            STATUS=$(curl -s http://127.0.0.1:5000/health)
            echo "   Status: $STATUS"
        else
            echo -e "${RED}❌ Backend: Not running${NC}"
            echo "   Start with: sudo python3 server.py"
        fi
        
        # Check frontend
        if curl -s http://localhost:3000 > /dev/null 2>&1; then
            echo -e "${GREEN}✅ Frontend: Running on http://localhost:3000${NC}"
        else
            echo -e "${YELLOW}⚠️  Frontend: Not running${NC}"
            echo "   Start with: npm run dev"
        fi
        
        # Check model
        if [ -f "random_forest_model.joblib" ]; then
            MODEL_SIZE=$(ls -lh random_forest_model.joblib | awk '{print $5}')
            echo -e "${GREEN}✅ Model: Found (${MODEL_SIZE})${NC}"
        else
            echo -e "${YELLOW}⚠️  Model: Not found (using simulation mode)${NC}"
        fi
        
        # Check dependencies
        if command -v python3 &> /dev/null; then
            PYTHON_VERSION=$(python3 --version)
            echo -e "${GREEN}✅ Python: ${PYTHON_VERSION}${NC}"
        else
            echo -e "${RED}❌ Python: Not found${NC}"
        fi
        
        if command -v node &> /dev/null; then
            NODE_VERSION=$(node --version)
            echo -e "${GREEN}✅ Node.js: ${NODE_VERSION}${NC}"
        else
            echo -e "${RED}❌ Node.js: Not found${NC}"
        fi
        
        echo ""
        ;;
    6)
        echo "Exiting..."
        exit 0
        ;;
    *)
        echo -e "${RED}Invalid option${NC}"
        exit 1
        ;;
esac

