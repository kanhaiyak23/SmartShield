import { Packet, Protocol, RiskLevel } from '../types';

// Determine if we should try to connect to local backend
// Only attempt connection if:
// 1. We're on localhost/127.0.0.1 (same origin)
// 2. Or explicitly configured via environment variable
const isLocalhost = typeof window !== 'undefined' && 
  (window.location.hostname === 'localhost' || 
   window.location.hostname === '127.0.0.1' ||
   window.location.hostname === '');

// Get backend URL from environment or use localhost only if we're on localhost
const getBackendURL = (): string | null => {
  // Check for environment variable (for production deployments)
  if (typeof window !== 'undefined' && (window as any).__BACKEND_URL__) {
    return (window as any).__BACKEND_URL__;
  }
  
  // Only use localhost if we're actually on localhost
  if (isLocalhost) {
    return 'http://127.0.0.1:5000/packets';
  }
  
  // For hosted/production, return null to skip fetch attempt
  return null;
};

const API_URL = getBackendURL();

// --- HELPERS ---
const randomInt = (min: number, max: number) => Math.floor(Math.random() * (max - min + 1)) + min;
const randomItem = <T>(arr: T[]): T => arr[Math.floor(Math.random() * arr.length)];
const generateIp = () => `${randomInt(10, 192)}.${randomInt(0, 255)}.${randomInt(0, 255)}.${randomInt(1, 254)}`;
const generatePort = () => randomItem([80, 443, 22, 53, 8080, 3000, randomInt(1024, 65535)]);

const MALICIOUS_PAYLOADS = [
  "UNION SELECT * FROM users--", "/etc/passwd", "<script>alert('pwned')</script>",
  "GET /admin/config.xml HTTP/1.1", "User-Agent: () { :;}; /bin/bash -i", "eval(base64_decode('...'))"
];

const stringToHex = (str: string) => {
  let hex = '';
  for(let i=0;i<str.length;i++) hex += str.charCodeAt(i).toString(16).padStart(2, '0');
  return hex;
};

const generateRandomHex = (len: number) => {
  let hex = '';
  for(let i=0;i<len;i++) hex += randomInt(0, 255).toString(16).padStart(2, '0');
  return hex;
};

// --- STEP 2: FEATURE EXTRACTION LAYER ---
// Maps raw packet data to numerical vector [length, ttl, proto_code, src_port, dst_port]
const extractFeatures = (
    length: number, 
    ttl: number, 
    protocol: Protocol, 
    srcPort: number, 
    dstPort: number
): number[] => {
    // 1. Protocol Encoding (Categorical -> Numerical)
    let protoCode = 0;
    switch (protocol) {
        case Protocol.TCP: protoCode = 1; break;
        case Protocol.UDP: protoCode = 2; break;
        case Protocol.ICMP: protoCode = 3; break;
        case Protocol.HTTP: protoCode = 4; break;
        case Protocol.HTTPS: protoCode = 5; break;
        case Protocol.SSH: protoCode = 6; break;
        default: protoCode = 0;
    }

    // 2. Normalization (Simplified for this simulation, usually done with StandardScaler)
    // We return the raw numbers here, but in a real ML pipeline, these would be scaled.
    return [length, ttl, protoCode, srcPort, dstPort];
};

// --- STEP 4: RANDOM FOREST INFERENCE LAYER (SIMULATION) ---
// Simulates Random Forest attack probability prediction
// Returns an anomaly score mapped from attack probability
// Range: -0.5 (Attack detected) to 0.5 (Normal traffic)
const runRandomForestInference = (features: number[], isSimulatedAttack: boolean): { score: number, risk: RiskLevel } => {
    // features = [length, ttl, protoCode, srcPort, dstPort]
    
    // In a real model, this would be: score = model.decision_function([features])
    
    // SIMULATION LOGIC:
    // We introduce "noise" based on the features to make it look realistic
    // but ultimately bias it based on whether we injected an attack pattern.
    
    let baseScore = isSimulatedAttack ? -0.3 : 0.25;
    
    // Feature 1: Length (Attacks often have unusual lengths, either very small or huge)
    const len = features[0];
    if (len > 1400 && len < 1500) baseScore += 0.1; // Normal MTU size
    if (len > 8000) baseScore -= 0.1; // Jumbo packet anomaly
    
    // Feature 4 & 5: Ports (High ports are slightly more suspicious for services)
    const dstPort = features[4];
    if ([80, 443, 53].includes(dstPort)) baseScore += 0.1; // Common ports are "safer"
    
    // Add some stochastic variance to mimic real model "uncertainty"
    const variance = (Math.random() * 0.1) - 0.05;
    let finalScore = baseScore + variance;
    
    // Clamp
    if (finalScore < -0.5) finalScore = -0.5;
    if (finalScore > 0.5) finalScore = 0.5;

    // Thresholds (as per user workflow Step 4)
    // score < -0.2 -> suspicious
    // score < -0.4 -> highly suspicious
    
    let risk = RiskLevel.SAFE;
    if (finalScore < -0.4) risk = RiskLevel.CRITICAL;
    else if (finalScore < -0.2) risk = RiskLevel.WARNING;
    
    return { score: Number(finalScore.toFixed(3)), risk };
};

let packetIdCounter = 0;

export const generateMockPacket = (): Packet => {
  packetIdCounter++;
  
  // 1. Packet Capture Simulation
  const isAttack = Math.random() < 0.15; // 15% chance of anomaly
  const protocol = randomItem([Protocol.TCP, Protocol.UDP, Protocol.HTTP, Protocol.SSH, Protocol.ICMP]);
  const srcIp = generateIp();
  const dstIp = generateIp();
  const srcPort = generatePort();
  const dstPort = generatePort();
  
  let asciiPayload = "";
  if (isAttack) asciiPayload = randomItem(MALICIOUS_PAYLOADS);
  else {
    if (protocol === Protocol.HTTP) asciiPayload = `GET /api/v1/status HTTP/1.1`;
    else if (protocol === Protocol.SSH) asciiPayload = "SSH-2.0-OpenSSH_8.2p1";
    else asciiPayload = "................";
  }
  
  if (asciiPayload.length < 16) asciiPayload = asciiPayload.padEnd(16, '.');
  const rawHex = stringToHex(asciiPayload) + generateRandomHex(randomInt(10, 50));
  const length = rawHex.length / 2;
  const ttl = randomInt(64, 128);

  const flags = [];
  if (protocol === Protocol.TCP || protocol === Protocol.HTTP) {
    if (Math.random() > 0.8) flags.push('SYN');
    if (Math.random() > 0.5) flags.push('ACK');
    if (Math.random() > 0.95) flags.push('RST');
  }

  // 2. Feature Extraction
  const featureVector = extractFeatures(length, ttl, protocol, srcPort, dstPort);

  // 3. Inference (Random Forest simulation)
  const { score, risk } = runRandomForestInference(featureVector, isAttack);

  // 4. Packet Construction
  return {
    id: `pkt-${packetIdCounter.toString().padStart(6, '0')}`,
    timestamp: Date.now(),
    sourceIp: srcIp,
    destIp: dstIp,
    sourcePort: srcPort,
    destPort: dstPort,
    protocol: protocol,
    length: length,
    ttl: ttl,
    windowSize: randomInt(1000, 65535),
    seq: randomInt(1000000, 4000000000),
    ack: randomInt(1000000, 4000000000),
    flags: flags,
    rawHex: rawHex.toUpperCase(),
    asciiPayload: asciiPayload,
    summary: `[RF_PROB: ${(score + 0.5).toFixed(3)}] ${protocol} ${srcIp} > ${dstIp}`,
    
    // New Fields
    featureVector: featureVector,
    anomalyScore: score,
    riskLevel: risk
  };
};

export const pollPacketStream = async (): Promise<{ packets: Packet[], connected: boolean }> => {
  // If no API URL (hosted/production), immediately return mock data
  // This prevents browser permission prompts for localhost access
  if (!API_URL) {
    return { packets: [generateMockPacket()], connected: false };
  }

  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 500);
    
    const response = await fetch(API_URL, { 
        signal: controller.signal,
        headers: { 'Content-Type': 'application/json' }
    });
    clearTimeout(timeoutId);

    if (!response.ok) {
        throw new Error("Server response not OK");
    }

    const data = await response.json();
    
    if (Array.isArray(data)) {
        return { packets: data as Packet[], connected: true };
    }
    return { packets: [], connected: true };

  } catch (error) {
    // Silently fall back to mock data on any error
    return { packets: [generateMockPacket()], connected: false };
  }
};