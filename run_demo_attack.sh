#!/bin/bash
# SmartShield - One-Click Attack Demo
# usage: sudo ./run_demo_attack.sh

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

if [ "$EUID" -ne 0 ]; then 
    echo -e "${RED}Please run as root (sudo)${NC}"
    exit 1
fi

echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║          ⚠️  SMARTSHIELD ATTACK SIMULATION  ⚠️              ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${YELLOW}>> Target: 127.0.0.1 (Localhost)${NC}"
echo -e "${YELLOW}>> Sequence: Port Scan -> DDoS -> SQL Injection -> ICMP${NC}"
echo ""
read -p "Press [Enter] to launch attacks..."

echo ""
./venv/bin/python3 attack_simulator.py --attack 5

echo ""
echo -e "${GREEN}✅ Demo Sequence Complete.${NC}"
echo "Check dashboard for results."
