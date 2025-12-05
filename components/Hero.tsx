import React from 'react';
import { Activity, Shield, Cpu, ArrowRight } from 'lucide-react';

interface HeroProps {
  onEnter: () => void;
}

const Hero: React.FC<HeroProps> = ({ onEnter }) => {
  return (
    <div className="relative w-full h-screen overflow-hidden bg-black flex flex-col items-center justify-center">
      {/* Background Effects */}
      <div className="absolute inset-0 z-0 pointer-events-none">
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_center,_var(--tw-gradient-stops))] from-blue-900/20 via-slate-950 to-black"></div>
        <div className="absolute inset-0 bg-[url('https://grainy-gradients.vercel.app/noise.svg')] opacity-20 brightness-100 contrast-150 mix-blend-overlay"></div>
        {/* Animated Grid */}
        <div className="absolute inset-0 bg-[linear-gradient(to_right,#80808012_1px,transparent_1px),linear-gradient(to_bottom,#80808012_1px,transparent_1px)] bg-[size:24px_24px] [mask-image:radial-gradient(ellipse_60%_50%_at_50%_0%,#000_70%,transparent_100%)]"></div>
      </div>

      <div className="z-10 text-center space-y-6 px-4 max-w-5xl">
        <div className="inline-flex items-center space-x-2 border border-cyan-500/30 bg-cyan-500/10 px-3 py-1 rounded-full text-cyan-400 text-sm font-mono tracking-wider animate-pulse">
          <span className="w-2 h-2 bg-cyan-400 rounded-full"></span>
          <span>SYSTEM ONLINE // RANDOM FOREST ACTIVE</span>
        </div>

        <h1 className="text-6xl md:text-8xl font-black tracking-tighter text-transparent bg-clip-text bg-gradient-to-b from-white to-slate-500 drop-shadow-[0_0_30px_rgba(255,255,255,0.2)]">
          <span className="text-cyan-500">Smart</span>Shield
        </h1>

        <p className="text-slate-400 text-lg md:text-xl max-w-2xl mx-auto font-light">
          Real-time packet interception with Random Forest machine learning. 
          Detect network attacks with 99.62% accuracy before they become breaches.
        </p>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 py-4">
          {[
            { icon: Activity, title: "Live Packet Scan", desc: "Microsecond latency capture via Scapy API." },
            { icon: Cpu, title: "Random Forest", desc: "Pre-trained UNSW-NB15 model (200 trees, 20 features)." },
            { icon: Shield, title: "Real-Time Defense", desc: "Instant anomaly scoring and risk assessment." }
          ].map((feature, idx) => (
            <div key={idx} className="group border border-white/5 bg-white/5 hover:bg-white/10 p-6 rounded-xl transition-all duration-300 backdrop-blur-sm">
              <feature.icon className="w-8 h-8 text-cyan-400 mb-4 group-hover:scale-110 transition-transform" />
              <h3 className="text-white font-bold mb-2">{feature.title}</h3>
              <p className="text-slate-400 text-sm">{feature.desc}</p>
            </div>
          ))}
        </div>

        <button 
          onClick={onEnter}
          className="group relative inline-flex items-center justify-center px-8 py-4 text-lg font-bold text-black bg-cyan-400 overflow-hidden rounded-full transition-all hover:bg-cyan-300 hover:scale-105 hover:shadow-[0_0_40px_rgba(34,211,238,0.5)] focus:outline-none focus:ring-2 focus:ring-cyan-400 focus:ring-offset-2 focus:ring-offset-slate-900"
        >
          <span className="absolute w-0 h-0 transition-all duration-500 ease-out bg-white rounded-full group-hover:w-56 group-hover:h-56 opacity-10"></span>
          <span className="relative flex items-center gap-2">
            INITIALIZE DASHBOARD <ArrowRight className="w-5 h-5" />
          </span>
        </button>
      </div>

      {/* Decorative floating code snippets */}
      <div className="absolute left-10 bottom-20 opacity-20 font-mono text-xs text-green-500 hidden md:block pointer-events-none">
        {`>> INIT_SEQUENCE_START`}<br/>
        {`>> LOADING_MODEL_WEIGHTS... [OK]`}<br/>
        {`>> ESTABLISHING_SOCKET... [OK]`}
      </div>
      <div className="absolute right-10 top-20 opacity-20 font-mono text-xs text-cyan-500 hidden md:block text-right pointer-events-none">
        {`Mem: 0x48F2A1`}<br/>
        {`Threads: 12 Active`}<br/>
        {`Latency: 4ms`}
      </div>
    </div>
  );
};

export default Hero;