import React from 'react';
import { Shield, BarChart3, Lock, Settings, User, Globe } from 'lucide-react';
import { View } from '../types';

interface NavbarProps {
  currentView: View;
  onChangeView: (view: View) => void;
}

const Navbar: React.FC<NavbarProps> = ({ currentView, onChangeView }) => {
  const getButtonClass = (view: View) => 
    `flex items-center gap-2 text-sm font-medium transition-all hover:scale-105 ${currentView === view ? 'text-cyan-400 shadow-[0_0_10px_rgba(34,211,238,0.2)]' : 'text-slate-400 hover:text-white'}`;

  return (
    <nav className="flex-none h-16 border-b border-white/10 bg-black/80 backdrop-blur-md flex items-center px-4 md:px-8 justify-between z-50">
      <div 
        className="flex items-center gap-2 cursor-pointer group"
        onClick={() => onChangeView('hero')}
      >
        <div className="relative">
             <div className="absolute inset-0 bg-cyan-500 blur-lg opacity-20 group-hover:opacity-40 transition-opacity"></div>
             <Shield className="w-6 h-6 text-cyan-400 relative z-10 group-hover:scale-110 transition-transform" />
        </div>
        <span className="font-bold text-xl tracking-tight text-white group-hover:text-cyan-100 transition-colors">
          <span className="text-cyan-500">Smart</span>Shield
        </span>
      </div>

      <div className="hidden md:flex items-center gap-8">
        <button 
          onClick={() => onChangeView('dashboard')}
          className={getButtonClass('dashboard')}
        >
          <BarChart3 className="w-4 h-4" /> MONITOR
        </button>
        <button 
          onClick={() => onChangeView('threat_map')}
          className={getButtonClass('threat_map')}
        >
          <Globe className="w-4 h-4" /> THREAT_MAP
        </button>
        <button 
          onClick={() => onChangeView('intel')}
          className={getButtonClass('intel')}
        >
          <Lock className="w-4 h-4" /> INTEL
        </button>
        <button 
          onClick={() => onChangeView('system')}
          className={getButtonClass('system')}
        >
          <Settings className="w-4 h-4" /> SYSTEM
        </button>
      </div>

      <div className="flex items-center gap-6">
        <div className="hidden lg:flex flex-col items-end border-r border-white/10 pr-6">
            <span className="text-[10px] font-mono text-green-500 flex items-center gap-1">
                <span className="w-1.5 h-1.5 rounded-full bg-green-500 animate-pulse"></span> ENCRYPTED
            </span>
            <span className="text-[10px] font-mono text-slate-500">NODE: US-EAST-4</span>
        </div>
        <button 
            onClick={() => onChangeView('admin')}
            className={`group flex items-center gap-2 hover:bg-white/5 p-2 rounded-lg transition-colors ${currentView === 'admin' ? 'bg-white/10' : ''}`}
        >
            <div className={`w-8 h-8 rounded bg-gradient-to-tr from-cyan-900 to-slate-800 border flex items-center justify-center transition-colors ${currentView === 'admin' ? 'border-cyan-400' : 'border-white/10 group-hover:border-cyan-500/50'}`}>
                <User className="w-4 h-4 text-cyan-400" />
            </div>
            <span className="hidden sm:block text-xs font-mono text-slate-400 group-hover:text-white">ADMIN</span>
        </button>
      </div>
    </nav>
  );
};

export default Navbar;