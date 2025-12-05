import { Packet, Protocol, RiskLevel } from '../types';

// Backend API URL - always attempt to connect to local backend
// Users will be prompted for local network permission if accessing from hosted link
const getBackendURL = (): string => {
  // Check for environment variable (for custom backend URLs)
  if (typeof window !== 'undefined' && (window as any).__BACKEND_URL__) {
    return (window as any).__BACKEND_URL__;
  }
  
  // Default to localhost backend
  return 'http://127.0.0.1:5000/packets';
};

const API_URL = getBackendURL();

export const pollPacketStream = async (): Promise<{ packets: Packet[], connected: boolean }> => {
  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 1000);
    
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
    // Return empty packets when backend is not connected
    // Dashboard will show "Network Not Connected" message
    return { packets: [], connected: false };
  }
};