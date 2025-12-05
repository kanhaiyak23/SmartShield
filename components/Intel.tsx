import React from 'react';
import { Lock, FileText, Database, ShieldAlert, ExternalLink } from 'lucide-react';

const Intel: React.FC = () => {
  const intelItems = [
    { id: 'CVE-2024-3094', title: 'XZ Utils Backdoor Attempt', severity: 'CRITICAL', source: 'NIST NVD', date: '2024-03-29' },
    { id: 'INT-8821', title: 'New Botnet "Mirai-X" Detected in subnet 192.168.x.x', severity: 'HIGH', source: 'SmartShield Sensor', date: 'Today' },
    { id: 'INT-8820', title: 'Suspicious SSH Handshake Pattern from IP 45.33.22.11', severity: 'MEDIUM', source: 'AI Heuristic', date: 'Today' },
    { id: 'CVE-2024-1102', title: 'Buffer Overflow in Legacy TCP Stack', severity: 'CRITICAL', source: 'Vendor Advisory', date: '2024-04-15' },
    { id: 'INT-8819', title: 'Outbound Traffic Anomaly on Port 5432', severity: 'LOW', source: 'Traffic Analysis', date: 'Yesterday' },
  ];

  return (
    <div className="h-full bg-slate-950 p-6 overflow-y-auto custom-scrollbar">
       <div className="max-w-4xl mx-auto space-y-8">
            <div className="border-b border-white/10 pb-6">
                 <h1 className="text-2xl font-bold text-white flex items-center gap-3">
                    <Lock className="text-cyan-400 w-8 h-8" /> 
                    INTELLIGENCE <span className="text-cyan-500">FEED</span>
                </h1>
                <p className="text-slate-400 text-sm font-mono mt-2">Aggregated threat data from local sensors and global databases.</p>
            </div>

            <div className="grid gap-4">
                {intelItems.map((item) => (
                    <div key={item.id} className="bg-slate-900/50 border border-white/5 rounded-lg p-5 hover:border-cyan-500/30 transition-all group">
                        <div className="flex justify-between items-start mb-2">
                            <div className="flex items-center gap-3">
                                {item.severity === 'CRITICAL' ? <ShieldAlert className="text-red-500 w-5 h-5" /> : <Database className="text-cyan-500 w-5 h-5" />}
                                <h3 className="text-lg font-bold text-slate-200 group-hover:text-cyan-400 transition-colors">{item.title}</h3>
                            </div>
                            <span className={`px-2 py-1 rounded text-xs font-mono font-bold ${
                                item.severity === 'CRITICAL' ? 'bg-red-500/20 text-red-400' : 
                                item.severity === 'HIGH' ? 'bg-orange-500/20 text-orange-400' :
                                'bg-cyan-500/20 text-cyan-400'
                            }`}>
                                {item.severity}
                            </span>
                        </div>
                        <div className="flex justify-between items-end">
                            <div className="text-xs text-slate-500 font-mono space-y-1">
                                <p>ID: {item.id}</p>
                                <p>SOURCE: {item.source}</p>
                            </div>
                            <button className="text-xs flex items-center gap-1 text-slate-400 hover:text-white transition-colors">
                                <FileText className="w-3 h-3" /> FULL REPORT <ExternalLink className="w-3 h-3" />
                            </button>
                        </div>
                    </div>
                ))}
            </div>
       </div>
    </div>
  );
};

export default Intel;