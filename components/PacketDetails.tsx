import React from 'react';
import { Packet, RiskLevel } from '../types';
import { CheckCircle, ShieldAlert, Binary, Activity, Layers, Cpu } from 'lucide-react';

interface PacketDetailsProps {
  packet: Packet | null;
}

const PacketDetails: React.FC<PacketDetailsProps> = ({ packet }) => {

  if (!packet) {
    return (
      <div className="h-full flex flex-col items-center justify-center text-slate-600 space-y-4 border border-white/5 bg-black/40 backdrop-blur-md rounded-lg p-6">
        <Cpu className="w-16 h-16 opacity-20 animate-pulse" />
        <p className="font-mono text-sm">SELECT_PACKET_TO_INSPECT</p>
      </div>
    );
  }

  const isCritical = packet.riskLevel === RiskLevel.CRITICAL;
  const isWarning = packet.riskLevel === RiskLevel.WARNING;

  return (
    <div className="h-full border border-white/10 bg-slate-900/50 backdrop-blur-md rounded-lg overflow-hidden flex flex-col">
      <div className={`p-4 border-b ${isCritical ? 'border-red-500/30 bg-red-950/20' : 'border-white/10 bg-slate-900/50'} flex justify-between items-center`}>
        <h2 className="font-bold text-lg flex items-center gap-2">
          {isCritical ? <ShieldAlert className="text-red-500" /> : <CheckCircle className="text-green-500" />}
          PACKET_INSPECTOR
        </h2>
        <span className="font-mono text-xs text-slate-500">{packet.id}</span>
      </div>

      <div className="p-6 space-y-6 overflow-y-auto custom-scrollbar">
        
        {/* Step 4: Random Forest Score Display */}
        <div className="space-y-2 bg-black/40 p-4 rounded border border-white/10">
          <div className="flex justify-between items-center">
             <span className="text-xs font-bold text-slate-400 uppercase flex items-center gap-2">
                <Activity className="w-3 h-3 text-cyan-400" /> Random Forest Attack Probability
             </span>
             <span className={`font-mono text-lg font-bold ${isCritical ? 'text-red-500' : isWarning ? 'text-yellow-500' : 'text-green-500'}`}>
                {packet.anomalyScore.toFixed(3)}
             </span>
          </div>
          <div className="h-2 w-full bg-slate-800 rounded-full overflow-hidden relative">
             {/* Scale from -0.5 (left/anomaly) to 0.5 (right/normal) */}
             <div className="absolute left-1/2 top-0 bottom-0 w-0.5 bg-white/30 z-10"></div>
             <div 
               className={`absolute top-0 bottom-0 w-2 h-2 rounded-full transition-all duration-500 ${isCritical ? 'bg-red-500 shadow-[0_0_10px_red]' : 'bg-green-500 shadow-[0_0_10px_green]'}`}
               style={{ 
                   left: `${((packet.anomalyScore + 0.5) / 1) * 100}%`,
                   transform: 'translate(-50%, 0)'
               }}
             ></div>
          </div>
          <div className="flex justify-between text-[10px] text-slate-500 font-mono">
              <span>-0.5 (ATTACK)</span>
              <span>0.5 (NORMAL)</span>
          </div>
        </div>

        {/* Step 2: Feature Vector Display */}
        <div className="space-y-1">
             <p className="text-xs text-slate-500 uppercase flex items-center gap-2">
                 <Layers className="w-3 h-3"/> Extracted Features [Step 2]
             </p>
             <div className="font-mono text-xs bg-black/50 p-2 border border-white/5 text-cyan-500/80 break-all">
                [{packet.featureVector.join(', ')}]
             </div>
             <p className="text-[10px] text-slate-600 font-mono">
                 Format: [Length, TTL, ProtoCode, SrcPort, DstPort, Duration, PacketRate, ByteRate, ...] (20 features)
             </p>
        </div>

        {/* Risk Assessment */}
        <div className={`p-4 rounded border transition-colors duration-500 ${
            isCritical ? 'border-red-500/20 bg-red-500/5' : isWarning ? 'border-yellow-500/20 bg-yellow-500/5' : 'border-green-500/20 bg-green-500/5'
        }`}>
            <h3 className={`text-xs font-bold uppercase tracking-wider mb-2 flex items-center gap-2 ${
                isCritical ? 'text-red-400' : isWarning ? 'text-yellow-400' : 'text-green-400'
            }`}>
                <Activity className="w-3 h-3" /> 
                RANDOM FOREST ASSESSMENT
            </h3>
            <p className="text-sm font-mono text-slate-300 leading-relaxed">
                {isCritical 
                    ? "⚠️ CRITICAL ANOMALY DETECTED - This packet exhibits highly anomalous characteristics."
                    : isWarning
                    ? "⚠️ SUSPICIOUS ACTIVITY - This packet shows unusual patterns."
                    : "✓ NORMAL TRAFFIC - Packet characteristics are within expected parameters."
                }
            </p>
        </div>

        {/* Header Details */}
        <div className="grid grid-cols-2 gap-4">
            <div className="bg-black/40 p-3 rounded border border-white/5">
                <p className="text-xs text-slate-500 uppercase">Protocol</p>
                <p className="text-white font-mono">{packet.protocol}</p>
            </div>
            <div className="bg-black/40 p-3 rounded border border-white/5">
                <p className="text-xs text-slate-500 uppercase">Length</p>
                <p className="text-white font-mono">{packet.length} bytes</p>
            </div>
            <div className="bg-black/40 p-3 rounded border border-white/5 col-span-2">
                <p className="text-xs text-slate-500 uppercase mb-1">IPv4 Vector</p>
                <div className="flex justify-between items-center font-mono text-sm">
                    <span className="text-indigo-400">{packet.sourceIp}:{packet.sourcePort}</span>
                    <span className="text-slate-600">→</span>
                    <span className="text-pink-400">{packet.destIp}:{packet.destPort}</span>
                </div>
            </div>
        </div>

        {/* Payload Dump */}
        <div className="space-y-2">
            <p className="text-xs text-slate-500 uppercase flex items-center gap-2"><Binary className="w-3 h-3"/> Raw Payload (Hex)</p>
            <div className="bg-black p-3 rounded border border-white/10 font-mono text-xs text-slate-400 break-all leading-relaxed h-32 overflow-y-auto custom-scrollbar">
                {packet.rawHex}
            </div>
        </div>

      </div>
    </div>
  );
};

export default PacketDetails;