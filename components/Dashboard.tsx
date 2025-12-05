import React, { useState, useEffect, useRef } from 'react';
import { pollPacketStream } from '../services/scapyService';
import { Packet, RiskLevel } from '../types';
import PacketDetails from './PacketDetails';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, AreaChart, Area, ReferenceLine } from 'recharts';
import { Shield, Zap, Wifi, AlertOctagon, Terminal, Pause, Play, Network, Server, ServerOff } from 'lucide-react';

const Dashboard: React.FC = () => {
  const [packets, setPackets] = useState<Packet[]>([]);
  const [isScanning, setIsScanning] = useState(true);
  const [isConnected, setIsConnected] = useState(false); // Track backend connection
  const [selectedPacket, setSelectedPacket] = useState<Packet | null>(null);
  const [chartData, setChartData] = useState<{time: string, anomalyScore: number, traffic: number}[]>([]);
  
  // Stats
  const [stats, setStats] = useState({ total: 0, threats: 0, bytes: 0 });

  const scrollRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom of list
  useEffect(() => {
    if (isScanning && scrollRef.current) {
        scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [packets, isScanning]);

  // Auto-select the most recent packet if none is selected
  useEffect(() => {
    if (selectedPacket === null && packets.length > 0) {
      setSelectedPacket(packets[packets.length - 1]);
    }
  }, [packets, selectedPacket]);

  useEffect(() => {
    if (!isScanning) return;

    const interval = setInterval(async () => {
      
      const { packets: newPackets, connected } = await pollPacketStream();
      setIsConnected(connected);

      if (newPackets.length === 0) return;

      setPackets(prev => {
        const updated = [...prev, ...newPackets];
        if (updated.length > 500) return updated.slice(updated.length - 500); // Keep last 500
        return updated;
      });

      // Aggregate stats from the batch
      let batchThreats = 0;
      let batchBytes = 0;
      let worstAnomalyScore = 0.5; // Start at normal (0.5)

      newPackets.forEach(p => {
        if (p.riskLevel === RiskLevel.CRITICAL) batchThreats++;
        batchBytes += p.length;
        // Find minimum score (most anomalous)
        if (p.anomalyScore < worstAnomalyScore) worstAnomalyScore = p.anomalyScore;
      });

      setStats(prev => ({
        total: prev.total + newPackets.length,
        threats: prev.threats + batchThreats,
        bytes: prev.bytes + batchBytes
      }));

      // Only update chart once per interval with the aggregate/max of the batch
      setChartData(prev => {
        const now = new Date();
        const timeStr = `${now.getHours()}:${now.getMinutes()}:${now.getSeconds()}`;
        const newEntry = {
            time: timeStr,
            anomalyScore: worstAnomalyScore, 
            traffic: batchBytes
        };
        const updated = [...prev, newEntry];
        if (updated.length > 30) updated.shift();
        return updated;
      });

    }, 1000); // Poll every 1s to match Scapy batching somewhat

    return () => clearInterval(interval);
  }, [isScanning]);

  return (
    <div className="h-full bg-slate-950/50 text-slate-200 p-2 md:p-4 font-sans flex flex-col relative overflow-hidden">
      {/* Background Grid for Dashboard */}
      <div className="absolute inset-0 bg-[linear-gradient(to_right,#80808008_1px,transparent_1px),linear-gradient(to_bottom,#80808008_1px,transparent_1px)] bg-[size:16px_16px] pointer-events-none"></div>

      {/* Top Bar */}
      <header className="flex-none flex justify-between items-center mb-4 p-4 border border-white/10 bg-slate-900/50 backdrop-blur-md rounded-xl z-10">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 bg-cyan-500/20 rounded-lg flex items-center justify-center border border-cyan-500/30">
             <Shield className="text-cyan-400 w-6 h-6" />
          </div>
          <div>
            <h1 className="text-xl font-bold tracking-tight text-white">LIVE <span className="text-cyan-400">TRAFFIC</span> CONTROL</h1>
            <div className="flex items-center gap-2">
                <span className={`w-2 h-2 rounded-full ${isScanning ? 'bg-green-500 animate-pulse' : 'bg-red-500'}`}></span>
                <span className="text-xs font-mono text-slate-400">{isScanning ? 'MONITOR_ACTIVE' : 'MONITOR_PAUSED'}</span>
                <span className="text-slate-600">|</span>
                {isConnected ? (
                   <span className="text-xs font-mono text-cyan-400 flex items-center gap-1"><Server className="w-3 h-3"/> LOCAL_SERVER_CONNECTED</span>
                ) : (
                   <span className="text-xs font-mono text-yellow-500 flex items-center gap-1"><ServerOff className="w-3 h-3"/> SIMULATION_MODE</span>
                )}
            </div>
          </div>
        </div>

        <div className="flex items-center gap-4">
             {/* Simple Stats */}
             <div className="hidden md:flex gap-6 mr-6">
                <div className="text-right">
                    <p className="text-xs text-slate-500 font-mono">PACKETS_CAPTURED</p>
                    <p className="text-xl font-bold text-white">{stats.total.toLocaleString()}</p>
                </div>
                <div className="text-right">
                    <p className="text-xs text-slate-500 font-mono">ANOMALIES</p>
                    <p className="text-xl font-bold text-red-500">{stats.threats.toLocaleString()}</p>
                </div>
             </div>

             <button 
                onClick={() => setIsScanning(!isScanning)}
                className={`p-3 rounded-lg border transition-all ${isScanning ? 'border-red-500/30 bg-red-500/10 hover:bg-red-500/20 text-red-400' : 'border-green-500/30 bg-green-500/10 hover:bg-green-500/20 text-green-400'}`}
             >
                {isScanning ? <Pause className="w-5 h-5" /> : <Play className="w-5 h-5" />}
             </button>
        </div>
      </header>

      {/* Main Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-4 flex-1 min-h-0 z-10">
        
        {/* Left Col: Charts & Live Feed (Width 8) */}
        <div className="lg:col-span-8 flex flex-col gap-4 min-h-0">
            
            {/* Charts Row */}
            <div className="flex-none h-48 md:h-64 grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="bg-slate-900/50 border border-white/10 rounded-xl p-4 backdrop-blur-sm">
                    <div className="flex items-center justify-between mb-2">
                        <h3 className="text-xs font-bold text-slate-400 uppercase flex items-center gap-2"><Wifi className="w-3 h-3" /> Bandwidth (Bytes/sec)</h3>
                    </div>
                    <ResponsiveContainer width="100%" height="85%">
                        <AreaChart data={chartData}>
                            <defs>
                                <linearGradient id="colorTraffic" x1="0" y1="0" x2="0" y2="1">
                                    <stop offset="5%" stopColor="#06b6d4" stopOpacity={0.3}/>
                                    <stop offset="95%" stopColor="#06b6d4" stopOpacity={0}/>
                                </linearGradient>
                            </defs>
                            <XAxis dataKey="time" hide />
                            <YAxis hide />
                            <Tooltip 
                                contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #334155' }}
                                itemStyle={{ color: '#22d3ee' }}
                            />
                            <Area type="monotone" dataKey="traffic" stroke="#22d3ee" fillOpacity={1} fill="url(#colorTraffic)" isAnimationActive={false} />
                        </AreaChart>
                    </ResponsiveContainer>
                </div>

                <div className="bg-slate-900/50 border border-white/10 rounded-xl p-4 backdrop-blur-sm">
                    <div className="flex items-center justify-between mb-2">
                        <h3 className="text-xs font-bold text-slate-400 uppercase flex items-center gap-2"><Zap className="w-3 h-3" /> Random Forest Score</h3>
                    </div>
                    <ResponsiveContainer width="100%" height="85%">
                        <LineChart data={chartData}>
                            <XAxis dataKey="time" hide />
                            {/* Domain set from -0.5 to 0.5 to match Random Forest anomaly score range */}
                            <YAxis hide domain={[-0.5, 0.5]} />
                            <ReferenceLine y={-0.2} stroke="#f59e0b" strokeDasharray="3 3" label={{ value: "Suspicious", fill: '#f59e0b', fontSize: 10 }} />
                            <ReferenceLine y={-0.4} stroke="#ef4444" strokeDasharray="3 3" label={{ value: "Critical", fill: '#ef4444', fontSize: 10 }} />
                            <Tooltip 
                                contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #334155' }}
                                itemStyle={{ color: '#ef4444' }}
                            />
                            <Line type="step" dataKey="anomalyScore" stroke="#ef4444" strokeWidth={2} dot={false} isAnimationActive={false} />
                        </LineChart>
                    </ResponsiveContainer>
                </div>
            </div>

            {/* Live Feed List */}
            <div className="flex-1 bg-black/40 border border-white/10 rounded-xl overflow-hidden flex flex-col relative">
                <div className="bg-slate-900/80 p-3 border-b border-white/10 flex justify-between items-center backdrop-blur z-10">
                    <h3 className="text-sm font-bold text-white flex items-center gap-2">
                        <Terminal className="w-4 h-4 text-cyan-400" /> 
                        {isConnected ? 'LIVE_PACKET_STREAM (127.0.0.1:5000)' : 'SIMULATED_PACKET_STREAM (MOCK)'}
                    </h3>
                    <div className="flex gap-2">
                         <span className="w-2 h-2 rounded-full bg-red-500"></span>
                         <span className="w-2 h-2 rounded-full bg-yellow-500"></span>
                         <span className="w-2 h-2 rounded-full bg-green-500"></span>
                    </div>
                </div>
                
                {/* Table Header */}
                <div className="grid grid-cols-12 gap-2 px-4 py-2 bg-slate-900/40 text-xs font-mono text-slate-500 border-b border-white/5 uppercase tracking-wider">
                    <div className="col-span-1">No.</div>
                    <div className="col-span-2">Time</div>
                    <div className="col-span-3">Source</div>
                    <div className="col-span-3">Destination</div>
                    <div className="col-span-1">Proto</div>
                    <div className="col-span-2 text-right">Class</div>
                </div>

                {/* Table Body */}
                <div ref={scrollRef} className="flex-1 overflow-y-auto custom-scrollbar p-2 space-y-1 scroll-smooth">
                    {packets.map((pkt) => (
                        <div 
                            key={pkt.id}
                            onClick={() => setSelectedPacket(pkt)}
                            className={`grid grid-cols-12 gap-2 px-3 py-1.5 rounded text-xs font-mono cursor-pointer border border-transparent transition-colors items-center
                                ${selectedPacket?.id === pkt.id ? 'bg-white/10 border-white/20' : 'hover:bg-white/5'}
                                ${pkt.riskLevel === RiskLevel.CRITICAL ? 'text-red-400 bg-red-950/20' : pkt.riskLevel === RiskLevel.WARNING ? 'text-yellow-400' : 'text-slate-300'}
                            `}
                        >
                            <div className="col-span-1 opacity-50">{pkt.id.split('-')[1]}</div>
                            <div className="col-span-2 opacity-70">{new Date(pkt.timestamp).toLocaleTimeString().split(' ')[0]}.{Math.floor(pkt.timestamp % 1000)}</div>
                            <div className="col-span-3 truncate" title={pkt.sourceIp}>{pkt.sourceIp}</div>
                            <div className="col-span-3 truncate" title={pkt.destIp}>{pkt.destIp}</div>
                            <div className="col-span-1 font-bold">{pkt.protocol}</div>
                            <div className="col-span-2 text-right flex justify-end items-center gap-2 overflow-hidden whitespace-nowrap">
                                {pkt.riskLevel === RiskLevel.CRITICAL && <AlertOctagon className="w-3 h-3 flex-shrink-0" />}
                                {pkt.riskLevel === RiskLevel.CRITICAL ? 'ANOMALY' : pkt.riskLevel === RiskLevel.WARNING ? 'SUSPICIOUS' : 'NORMAL'}
                                <span className="opacity-50 text-[10px]">({pkt.anomalyScore.toFixed(2)})</span>
                            </div>
                        </div>
                    ))}
                    {packets.length === 0 && (
                        <div className="text-center py-20 text-slate-600 font-mono animate-pulse">
                            <Network className="w-12 h-12 mx-auto mb-4 opacity-20" />
                            {isConnected ? 'LISTENING_FOR_TRAFFIC...' : 'INITIALIZING SCAPY SOCKET...'}
                        </div>
                    )}
                </div>
                
                {/* Decorative scan line */}
                <div className="absolute top-0 left-0 w-full h-1 bg-cyan-500/20 shadow-[0_0_10px_rgba(34,211,238,0.5)] z-0 pointer-events-none animate-scan opacity-30"></div>
            </div>
        </div>

        {/* Right Col: Details (Width 4) */}
        <div className="lg:col-span-4 min-h-0">
            <PacketDetails packet={selectedPacket} />
        </div>

      </div>
    </div>
  );
};

export default Dashboard;