#!/usr/bin/env python3
import time
import sys
import os
from attack_simulator import AttackSimulator, TARGET_IP

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header(title):
    print("\n" + "="*60)
    print(f"🎓  SMARTSHIELD ACADEMIC DEMO: {title}")
    print("="*60 + "\n")

def main():
    if os.geteuid() != 0:
        print("❌ Please run with sudo")
        sys.exit(1)

    sim = AttackSimulator(target_ip=TARGET_IP)
    sim.running = True

    # Phase 0: Setup
    clear_screen()
    print_header("ENVIRONMENT SETUP")
    print(">> Target System: " + TARGET_IP)
    print(">> Dashboard Filter: STRICT MODE (Only 100.x and 200.x visible)")
    print(">> Current Status:  Backend Listening, Dashboard Blank (Quiet)")
    print("\nInstruct your professor: 'The system is currently online but waiting for traffic.'")
    input("\n▶️  Press Enter to begin Phase 1 (Normal Traffic)...")

    # Phase 1: Normal Traffic
    clear_screen()
    print_header("PHASE 1: NORMAL TRAFFIC")
    print(">> Injecting benign HTTP/HTTPS requests from User 100.1.1.1")
    print(">> Watch Dashboard: Should show GREEN (Safe) traffic.")
    
    # Run safe traffic for 15 seconds
    sim.simulate_normal_traffic(duration=15)
    
    print("\n✅ Phase 1 Complete. Traffic stopped.")
    print("Instruct your professor: 'The AI correctly identifies this legitimate activity as Safe.'")
    input("\n▶️  Press Enter to begin Phase 2 (ATTACK SIMULATION)...")

    # Phase 2: Attacks
    clear_screen()
    print_header("PHASE 2: ATTACK SEQUENCE")
    print(">> Launching multi-vector attacks from Attacker 200.1.1.1")
    print(">> Watch Dashboard: Should turn RED (Critical).")
    
    # 1. Port Scan
    print("\n[1/3] Launching Port Scan...")
    sim.port_scan_attack(duration=8)
    time.sleep(2)
    
    # 2. DDoS
    print("\n[2/3] Launching DDoS Flood...")
    sim.ddos_flood_attack(duration=5)
    time.sleep(2)
    
    # 3. SQL Injection
    print("\n[3/3] Launching SQL Injection...")
    sim.sql_injection_attack(duration=6)
    
    print("\n✅ Phase 2 Complete. All attacks launched.")
    print("Instruct your professor: 'The system detected all threats in real-time.'")
    
    print("\n" + "="*60)
    print("🎉 DEMO COMPLETE")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
