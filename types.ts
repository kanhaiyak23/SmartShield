export enum Protocol {
  TCP = 'TCP',
  UDP = 'UDP',
  ICMP = 'ICMP',
  HTTP = 'HTTP',
  HTTPS = 'HTTPS',
  SSH = 'SSH'
}

export enum RiskLevel {
  SAFE = 'SAFE',
  WARNING = 'WARNING',
  CRITICAL = 'CRITICAL'
}

export type View = 'hero' | 'dashboard' | 'threat_map' | 'intel' | 'system' | 'admin' | 'docs' | 'status';

export interface Packet {
  id: string;
  timestamp: number;
  sourceIp: string;
  destIp: string;
  sourcePort: number;
  destPort: number;
  protocol: Protocol;
  length: number; // bytes
  
  // Scapy specific fields
  ttl: number;
  windowSize?: number;
  seq?: number;
  ack?: number;
  flags: string[]; // [SYN, ACK, etc]
  
  // Data
  rawHex: string;
  asciiPayload: string;
  summary: string; // Scapy-like summary string: "IP / TCP 192.168.1.5:443 > 10.0.0.2:53422 PA"
  
  // Feature Extraction Layer (Step 2)
  featureVector: number[]; // [length, ttl, proto_enum, src_port, dst_port]
  
  // Random Forest Inference Layer (Step 4)
  anomalyScore: number; // e.g., -0.45 (attack detected) to 0.5 (normal traffic)
  riskLevel: RiskLevel; // Derived from Random Forest attack probability
}

export interface TrafficStats {
  totalPackets: number;
  bytesTransferred: number;
  threatsDetected: number;
  activeConnections: number;
}