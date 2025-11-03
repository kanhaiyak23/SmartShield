"""Real-time packet capture using Scapy."""
import socket
import struct
from scapy.all import sniff, IP, TCP, UDP, ICMP, ARP, Ether, Raw, get_if_list
from scapy.layers.inet6 import IPv6
from datetime import datetime
import logging
import threading
import time
from collections import deque

logger = logging.getLogger('SmartShield')

class PacketCapture:
    """Capture and analyze network packets in real-time."""
    
    def __init__(self, interface=None, packet_buffer_size=1000, log_packets=True, log_interval=100):
        self.interface = interface
        self.packet_buffer_size = packet_buffer_size
        self.log_packets = log_packets  # Enable/disable detailed packet logging
        self.log_interval = log_interval  # Log every N packets
        self.packets = deque(maxlen=packet_buffer_size)
        self.capture_thread = None
        self.is_capturing = False
        self.capture_start_time = None
        self.last_stats_log_time = None
        self.stats = {
            'total_packets': 0,
            'tcp_packets': 0,
            'udp_packets': 0,
            'icmp_packets': 0,
            'arp_packets': 0,
            'total_bytes': 0
        }
        self.packet_rates = {
            'packets_per_second': 0,
            'bytes_per_second': 0
        }
        self.last_stats = {
            'total_packets': 0,
            'total_bytes': 0
        }
        
    def _extract_packet_features(self, packet):
        """Extract features from a single packet for ML model."""
        features = {}
        
        # Layer information
        features['packet_length'] = len(packet)
        
        # IP Layer
        if IP in packet:
            ip_layer = packet[IP]
            features['src_ip'] = ip_layer.src
            features['dst_ip'] = ip_layer.dst
            features['ip_version'] = ip_layer.version
            features['ip_header_length'] = ip_layer.ihl * 4
            features['ip_tos'] = ip_layer.tos
            features['ip_ttl'] = ip_layer.ttl
            features['ip_flags'] = ip_layer.flags
            features['ip_proto'] = ip_layer.proto
        else:
            features['src_ip'] = 'unknown'
            features['dst_ip'] = 'unknown'
            features['ip_version'] = 4
            features['ip_header_length'] = 20
            features['ip_tos'] = 0
            features['ip_ttl'] = 64
            features['ip_flags'] = 0
            features['ip_proto'] = 0
        
        # TCP Layer
        if TCP in packet:
            tcp_layer = packet[TCP]
            features['protocol'] = 'tcp'
            features['src_port'] = tcp_layer.sport
            features['dst_port'] = tcp_layer.dport
            features['tcp_seq'] = tcp_layer.seq
            features['tcp_ack'] = tcp_layer.ack
            features['tcp_flags'] = tcp_layer.flags
            features['tcp_window'] = tcp_layer.window
        elif UDP in packet:
            udp_layer = packet[UDP]
            features['protocol'] = 'udp'
            features['src_port'] = udp_layer.sport
            features['dst_port'] = udp_layer.dport
        elif ICMP in packet:
            icmp_layer = packet[ICMP]
            features['protocol'] = 'icmp'
            features['icmp_type'] = icmp_layer.type
            features['icmp_code'] = icmp_layer.code
        elif ARP in packet:
            arp_layer = packet[ARP]
            features['protocol'] = 'arp'
            features['arp_op'] = arp_layer.op
        else:
            features['protocol'] = 'other'
        
        # Ethernet Layer
        if Ether in packet:
            ether_layer = packet[Ether]
            features['src_mac'] = ether_layer.src
            features['dst_mac'] = ether_layer.dst
        
        # Timestamp
        features['timestamp'] = datetime.now().isoformat()
        
        # Additional derived features
        features['has_payload'] = 1 if Raw in packet else 0
        features['payload_length'] = len(packet[Raw].load) if Raw in packet else 0
        
        return features
    
    def _packet_handler(self, packet):
        """Handle captured packets."""
        if not self.is_capturing:
            return
        
        try:
            # Update statistics
            self.stats['total_packets'] += 1
            packet_len = len(packet)
            self.stats['total_bytes'] += packet_len
            
            # Protocol classification and logging
            protocol = None
            src_ip = None
            dst_ip = None
            src_port = None
            dst_port = None
            
            if TCP in packet:
                self.stats['tcp_packets'] += 1
                protocol = 'TCP'
                src_ip = packet[IP].src if IP in packet else 'N/A'
                dst_ip = packet[IP].dst if IP in packet else 'N/A'
                src_port = packet[TCP].sport
                dst_port = packet[TCP].dport
            elif UDP in packet:
                self.stats['udp_packets'] += 1
                protocol = 'UDP'
                src_ip = packet[IP].src if IP in packet else 'N/A'
                dst_ip = packet[IP].dst if IP in packet else 'N/A'
                src_port = packet[UDP].sport
                dst_port = packet[UDP].dport
            elif ICMP in packet:
                self.stats['icmp_packets'] += 1
                protocol = 'ICMP'
                src_ip = packet[IP].src if IP in packet else 'N/A'
                dst_ip = packet[IP].dst if IP in packet else 'N/A'
            elif ARP in packet:
                self.stats['arp_packets'] += 1
                protocol = 'ARP'
                if ARP in packet:
                    src_ip = packet[ARP].psrc if hasattr(packet[ARP], 'psrc') else 'N/A'
                    dst_ip = packet[ARP].pdst if hasattr(packet[ARP], 'pdst') else 'N/A'
            else:
                protocol = 'OTHER'
                if IP in packet:
                    src_ip = packet[IP].src
                    dst_ip = packet[IP].dst
            
            # Extract features
            features = self._extract_packet_features(packet)
            self.packets.append(features)
            
            # Log packet details periodically
            if self.log_packets and self.stats['total_packets'] % self.log_interval == 0:
                logger.info(
                    f"Packet #{self.stats['total_packets']}: {protocol} | "
                    f"{src_ip or 'N/A'}:{src_port or 'N/A'} -> {dst_ip or 'N/A'}:{dst_port or 'N/A'} | "
                    f"Size: {packet_len} bytes"
                )
            
            # Log every packet at DEBUG level
            logger.debug(
                f"Captured {protocol} packet: {src_ip}:{src_port or ''} -> "
                f"{dst_ip}:{dst_port or ''} ({packet_len} bytes)"
            )
            
            # Calculate packet rate (every 1000 packets)
            if self.stats['total_packets'] % 1000 == 0:
                if self.capture_start_time:
                    elapsed = time.time() - self.capture_start_time
                    if elapsed > 0:
                        self.packet_rates['packets_per_second'] = self.stats['total_packets'] / elapsed
                        self.packet_rates['bytes_per_second'] = self.stats['total_bytes'] / elapsed
            
        except Exception as e:
            logger.error(f"Error processing packet #{self.stats.get('total_packets', 0)}: {e}", exc_info=True)
    
    def start_capture(self, packet_filter=None):
        """Start capturing packets."""
        if self.is_capturing:
            logger.warning("Capture already running")
            return
        
        self.is_capturing = True
        self.capture_start_time = time.time()
        self.last_stats_log_time = time.time()
        
        # Log interface information
        available_interfaces = get_if_list()
        logger.info("=" * 60)
        logger.info("Starting WiFi Network Packet Capture")
        logger.info("=" * 60)
        logger.info(f"Interface: {self.interface or 'ALL INTERFACES (any)'}")
        logger.info(f"Available interfaces: {', '.join(available_interfaces) if available_interfaces else 'None detected'}")
        logger.info(f"BPF Filter: {packet_filter or 'None (capturing all)'}")
        logger.info(f"Packet Buffer Size: {self.packet_buffer_size}")
        logger.info(f"Packet Logging: {'Enabled' if self.log_packets else 'Disabled'} (every {self.log_interval} packets)")
        logger.info(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 60)
        
        # Start periodic statistics logging in separate thread
        def stats_logger():
            while self.is_capturing:
                time.sleep(10)  # Log stats every 10 seconds
                if self.is_capturing:
                    self._log_statistics()
        
        stats_thread = threading.Thread(target=stats_logger, daemon=True)
        stats_thread.start()
        
        # Start capture in separate thread
        def capture_loop():
            try:
                logger.info("Packet capture thread started. Beginning to sniff packets...")
                sniff(
                    iface=self.interface,
                    prn=self._packet_handler,
                    filter=packet_filter,
                    store=False
                )
            except PermissionError as e:
                logger.error(f"Permission denied: {e}")
                logger.error("Packet capture requires root/administrator privileges!")
                logger.error("Please run with: sudo python live_detection.py --interface <interface>")
                self.is_capturing = False
            except OSError as e:
                logger.error(f"Interface error: {e}")
                logger.error(f"Interface '{self.interface}' may not exist or be accessible")
                logger.error(f"Available interfaces: {', '.join(get_if_list())}")
                self.is_capturing = False
            except Exception as e:
                logger.error(f"Capture error: {e}", exc_info=True)
                self.is_capturing = False
        
        self.capture_thread = threading.Thread(target=capture_loop, daemon=True)
        self.capture_thread.start()
        logger.info("Capture initialized. Waiting for packets...")
    
    def _log_statistics(self):
        """Log periodic capture statistics."""
        if not self.is_capturing:
            return
        
        current_time = time.time()
        elapsed = current_time - self.last_stats_log_time
        
        if elapsed > 0:
            # Calculate rate since last stats log (interval rate)
            packets_in_interval = self.stats['total_packets'] - self.last_stats['total_packets']
            bytes_in_interval = self.stats['total_bytes'] - self.last_stats['total_bytes']
            pps = packets_in_interval / elapsed if elapsed > 0 else 0
            bps = bytes_in_interval / elapsed if elapsed > 0 else 0
            mbps = bps / (1024 * 1024)  # Convert to MB/s
            
            logger.info("=" * 60)
            logger.info(f"Capture Statistics (last {elapsed:.1f}s)")
            logger.info(f"  Total Packets: {self.stats['total_packets']:,}")
            logger.info(f"  Total Bytes: {self.stats['total_bytes']:,} ({self.stats['total_bytes'] / (1024*1024):.2f} MB)")
            logger.info(f"  Packet Rate: {pps:.2f} packets/sec")
            logger.info(f"  Data Rate: {mbps:.2f} MB/s ({bps:.0f} bytes/sec)")
            logger.info(f"  Protocol Distribution:")
            logger.info(f"    - TCP:  {self.stats['tcp_packets']:,} ({self.stats['tcp_packets']/max(1,self.stats['total_packets'])*100:.1f}%)")
            logger.info(f"    - UDP:  {self.stats['udp_packets']:,} ({self.stats['udp_packets']/max(1,self.stats['total_packets'])*100:.1f}%)")
            logger.info(f"    - ICMP: {self.stats['icmp_packets']:,} ({self.stats['icmp_packets']/max(1,self.stats['total_packets'])*100:.1f}%)")
            logger.info(f"    - ARP:  {self.stats['arp_packets']:,} ({self.stats['arp_packets']/max(1,self.stats['total_packets'])*100:.1f}%)")
            
            if self.capture_start_time:
                total_elapsed = current_time - self.capture_start_time
                avg_pps = self.stats['total_packets'] / total_elapsed if total_elapsed > 0 else 0
                avg_bps = self.stats['total_bytes'] / total_elapsed if total_elapsed > 0 else 0
                logger.info(f"  Average Rate: {avg_pps:.2f} packets/sec, {avg_bps/(1024*1024):.2f} MB/s")
                logger.info(f"  Capture Duration: {total_elapsed:.1f}s ({total_elapsed/60:.1f} minutes)")
            
            logger.info("=" * 60)
        
        # Update last stats for next interval calculation
        self.last_stats['total_packets'] = self.stats['total_packets']
        self.last_stats['total_bytes'] = self.stats['total_bytes']
        self.last_stats_log_time = current_time
    
    def stop_capture(self):
        """Stop capturing packets."""
        if not self.is_capturing:
            logger.warning("Capture is not running")
            return
        
        self.is_capturing = False
        stop_time = time.time()
        
        logger.info("=" * 60)
        logger.info("Stopping Packet Capture")
        logger.info("=" * 60)
        logger.info(f"Stop Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        if self.capture_start_time:
            duration = stop_time - self.capture_start_time
            logger.info(f"Total Duration: {duration:.1f}s ({duration/60:.1f} minutes)")
            logger.info(f"Total Packets Captured: {self.stats['total_packets']:,}")
            logger.info(f"Total Bytes Captured: {self.stats['total_bytes']:,} ({self.stats['total_bytes']/(1024*1024):.2f} MB)")
            
            if duration > 0:
                avg_pps = self.stats['total_packets'] / duration
                avg_bps = self.stats['total_bytes'] / duration
                logger.info(f"Average Rate: {avg_pps:.2f} packets/sec, {avg_bps/(1024*1024):.2f} MB/s")
        
        logger.info("Final Statistics:")
        logger.info(f"  TCP Packets:  {self.stats['tcp_packets']:,}")
        logger.info(f"  UDP Packets:  {self.stats['udp_packets']:,}")
        logger.info(f"  ICMP Packets: {self.stats['icmp_packets']:,}")
        logger.info(f"  ARP Packets:  {self.stats['arp_packets']:,}")
        logger.info("=" * 60)
        logger.info("Packet capture stopped successfully")
    
    def get_recent_packets(self, n=10):
        """Get recent n packets."""
        return list(self.packets)[-n:]
    
    def get_stats(self):
        """Get capture statistics."""
        return self.stats.copy()
    
    def reset_stats(self):
        """Reset capture statistics."""
        logger.info("Resetting capture statistics")
        self.stats = {
            'total_packets': 0,
            'tcp_packets': 0,
            'udp_packets': 0,
            'icmp_packets': 0,
            'arp_packets': 0,
            'total_bytes': 0
        }
        self.packet_rates = {
            'packets_per_second': 0,
            'bytes_per_second': 0
        }
        self.last_stats = {
            'total_packets': 0,
            'total_bytes': 0
        }
        self.capture_start_time = time.time()
        self.last_stats_log_time = time.time()

