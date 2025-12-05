import React from 'react';
import { User, Lock, Key } from 'lucide-react';

const AdminPanel: React.FC = () => {
  return (
    <div className="h-full bg-slate-950 flex flex-col items-center justify-center p-4 relative overflow-hidden">
        {/* Background Animation */}
        <div className="absolute inset-0 flex items-center justify-center pointer-events-none opacity-20">
             <div className="w-[500px] h-[500px] bg-cyan-500/30 rounded-full blur-[100px] animate-pulse"></div>
        </div>

        <div className="relative z-10 w-full max-w-md bg-black/60 border border-white/10 p-8 rounded-2xl backdrop-blur-xl shadow-2xl">
            <div className="text-center mb-8">
                <div className="w-16 h-16 bg-gradient-to-br from-slate-800 to-black rounded-full border border-white/20 flex items-center justify-center mx-auto mb-4 shadow-[0_0_20px_rgba(34,211,238,0.2)]">
                    <User className="w-8 h-8 text-cyan-400" />
                </div>
                <h1 className="text-2xl font-bold text-white">ADMINISTRATOR ACCESS</h1>
                <p className="text-xs text-slate-500 font-mono mt-2">SECURE GATEWAY // LEVEL 4 CLEARANCE REQUIRED</p>
            </div>

            <div className="space-y-4">
                <div className="space-y-1">
                    <label className="text-xs font-mono text-slate-400">USERNAME</label>
                    <div className="relative">
                        <User className="absolute left-3 top-3 w-4 h-4 text-slate-600" />
                        <input type="text" className="w-full bg-slate-900/50 border border-white/10 rounded p-2.5 pl-10 text-white focus:border-cyan-500 focus:outline-none focus:ring-1 focus:ring-cyan-500 transition-all font-mono text-sm" placeholder="root" />
                    </div>
                </div>
                <div className="space-y-1">
                    <label className="text-xs font-mono text-slate-400">ACCESS KEY</label>
                    <div className="relative">
                        <Key className="absolute left-3 top-3 w-4 h-4 text-slate-600" />
                        <input type="password" className="w-full bg-slate-900/50 border border-white/10 rounded p-2.5 pl-10 text-white focus:border-cyan-500 focus:outline-none focus:ring-1 focus:ring-cyan-500 transition-all font-mono text-sm" placeholder="••••••••••••••••" />
                    </div>
                </div>

                <button className="w-full bg-cyan-500 hover:bg-cyan-400 text-black font-bold py-3 rounded mt-4 transition-all flex items-center justify-center gap-2">
                    <Lock className="w-4 h-4" /> AUTHENTICATE
                </button>
            </div>

            <div className="mt-6 text-center">
                 <p className="text-[10px] text-red-500/80 font-mono blink">
                     WARNING: UNAUTHORIZED ACCESS ATTEMPTS ARE LOGGED AND REPORTED.
                 </p>
            </div>
        </div>
    </div>
  );
};

export default AdminPanel;