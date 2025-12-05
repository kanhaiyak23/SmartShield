import React from 'react';
import { Globe, MapPin, AlertCircle } from 'lucide-react';

const ThreatMap: React.FC = () => {
  // Mock data for threat points
  const threatPoints = [
    { top: '30%', left: '20%', id: 'US-WEST', type: 'DDoS', level: 'CRITICAL' },
    { top: '45%', left: '48%', id: 'EU-CENTRAL', type: 'SQL Injection', level: 'WARNING' },
    { top: '60%', left: '75%', id: 'ASIA-PAC', type: 'Botnet', level: 'WARNING' },
    { top: '25%', left: '80%', id: 'RU-NORTH', type: 'Brute Force', level: 'CRITICAL' },
    { top: '75%', left: '30%', id: 'SA-EAST', type: 'Port Scan', level: 'SAFE' },
  ];

  return (
    <div className="h-full bg-slate-950 flex flex-col relative overflow-hidden">
        {/* Header */}
        <div className="p-6 border-b border-white/10 bg-slate-900/50 backdrop-blur z-10">
            <h1 className="text-2xl font-bold text-white flex items-center gap-3">
                <Globe className="text-cyan-400 w-8 h-8" /> 
                GLOBAL THREAT <span className="text-cyan-500">VECTOR MAP</span>
            </h1>
            <p className="text-slate-400 text-sm font-mono mt-2">Real-time geolocation of intercepted packet anomalies.</p>
        </div>

        {/* Map Container */}
        <div className="flex-1 relative flex items-center justify-center bg-black/60 overflow-hidden">
            {/* Abstract World Map Grid */}
            <div className="absolute inset-0 opacity-20 bg-[url('https://upload.wikimedia.org/wikipedia/commons/e/ec/World_map_blank_without_borders.svg')] bg-contain bg-no-repeat bg-center mix-blend-overlay filter invert"></div>
            <div className="absolute inset-0 bg-[linear-gradient(to_right,#083344_1px,transparent_1px),linear-gradient(to_bottom,#083344_1px,transparent_1px)] bg-[size:40px_40px] opacity-20 pointer-events-none"></div>

            {/* Threat Points */}
            {threatPoints.map((point, index) => (
                <div 
                    key={index} 
                    className="absolute group"
                    style={{ top: point.top, left: point.left }}
                >
                    {/* Pulse Effect */}
                    <div className={`absolute -inset-4 rounded-full opacity-50 animate-ping ${point.level === 'CRITICAL' ? 'bg-red-500' : 'bg-yellow-500'}`}></div>
                    
                    {/* Icon */}
                    <div className={`relative w-4 h-4 rounded-full border-2 border-white shadow-[0_0_15px_rgba(255,255,255,0.5)] cursor-pointer transition-transform hover:scale-125 ${point.level === 'CRITICAL' ? 'bg-red-600' : 'bg-yellow-500'}`}></div>

                    {/* Tooltip */}
                    <div className="absolute left-6 top-1/2 -translate-y-1/2 opacity-0 group-hover:opacity-100 transition-opacity bg-black/90 border border-white/20 p-3 rounded w-48 z-20 pointer-events-none">
                        <h4 className="font-bold text-white text-xs mb-1 flex items-center gap-2">
                            <MapPin className="w-3 h-3" /> {point.id}
                        </h4>
                        <div className="text-[10px] text-slate-300 space-y-1 font-mono">
                            <p>TYPE: <span className="text-cyan-400">{point.type}</span></p>
                            <p>RISK: <span className={point.level === 'CRITICAL' ? 'text-red-500' : 'text-yellow-500'}>{point.level}</span></p>
                            <p>LATENCY: {Math.floor(Math.random() * 200)}ms</p>
                        </div>
                    </div>
                </div>
            ))}

            {/* Scanning Radar Line */}
            <div className="absolute top-0 bottom-0 w-[2px] bg-gradient-to-b from-transparent via-cyan-500 to-transparent opacity-50 animate-scan-horizontal shadow-[0_0_20px_rgba(34,211,238,0.5)]"></div>
        </div>
        
        <style>{`
            @keyframes scan-horizontal {
                0% { left: 0%; }
                100% { left: 100%; }
            }
            .animate-scan-horizontal {
                animation: scan-horizontal 5s linear infinite;
            }
        `}</style>
    </div>
  );
};

export default ThreatMap;