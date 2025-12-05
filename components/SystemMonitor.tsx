import React from 'react';
import { Settings, Cpu, HardDrive, Wifi, Activity } from 'lucide-react';

const SystemMonitor: React.FC = () => {
  const metrics = [
      { name: 'CPU Load (Core 0-11)', value: 42, icon: Cpu },
      { name: 'Memory Usage (32GB)', value: 68, icon: HardDrive },
      { name: 'Network Buffer', value: 23, icon: Wifi },
      { name: 'AI Inference Latency', value: 12, icon: Activity },
  ];

  return (
    <div className="h-full bg-slate-950 p-6 flex flex-col items-center justify-center">
        <div className="max-w-2xl w-full bg-black/40 border border-white/10 rounded-2xl p-8 backdrop-blur-xl">
             <div className="flex items-center gap-4 mb-8 border-b border-white/10 pb-6">
                 <div className="p-3 bg-cyan-500/10 rounded-lg border border-cyan-500/30">
                    <Settings className="w-8 h-8 text-cyan-400" />
                 </div>
                 <div>
                     <h1 className="text-2xl font-bold text-white">SYSTEM STATUS</h1>
                     <p className="text-slate-400 font-mono text-xs">UBUNTU-LTS-KERNEL-5.15 // NODE-01</p>
                 </div>
             </div>

             <div className="space-y-6">
                 {metrics.map((m, idx) => (
                     <div key={idx} className="space-y-2">
                         <div className="flex justify-between text-sm text-slate-300">
                             <div className="flex items-center gap-2">
                                 <m.icon className="w-4 h-4 text-slate-500" />
                                 {m.name}
                             </div>
                             <span className="font-mono text-cyan-400">{m.value}%</span>
                         </div>
                         <div className="h-2 bg-slate-800 rounded-full overflow-hidden">
                             <div 
                                className="h-full bg-cyan-500 relative overflow-hidden" 
                                style={{ width: `${m.value}%` }}
                             >
                                 <div className="absolute inset-0 bg-white/20 animate-[shimmer_2s_infinite]"></div>
                             </div>
                         </div>
                     </div>
                 ))}
             </div>

             <div className="mt-8 grid grid-cols-2 gap-4">
                 <div className="bg-slate-900/50 p-4 rounded border border-white/5 text-center">
                     <p className="text-xs text-slate-500 font-mono">UPTIME</p>
                     <p className="text-xl font-bold text-white">14d 03h 22m</p>
                 </div>
                 <div className="bg-slate-900/50 p-4 rounded border border-white/5 text-center">
                     <p className="text-xs text-slate-500 font-mono">SCAPY VERSION</p>
                     <p className="text-xl font-bold text-white">2.5.0</p>
                 </div>
             </div>
        </div>
        <style>{`
            @keyframes shimmer {
                0% { transform: translateX(-100%); }
                100% { transform: translateX(100%); }
            }
        `}</style>
    </div>
  );
};

export default SystemMonitor;