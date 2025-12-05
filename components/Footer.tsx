import React from 'react';
import { Github, Twitter, Terminal, Command } from 'lucide-react';
import { View } from '../types';

interface FooterProps {
    onChangeView: (view: View) => void;
}

const Footer: React.FC<FooterProps> = ({ onChangeView }) => {
  return (
    <footer className="flex-none border-t border-white/10 bg-black/90 py-4 z-50">
      <div className="max-w-[1920px] mx-auto px-6 flex flex-col md:flex-row justify-between items-center gap-4">
        
        <div className="flex items-center gap-4">
           <div className="flex items-center gap-2 px-3 py-1 rounded-full bg-slate-900 border border-white/5">
              <Terminal className="w-3 h-3 text-cyan-500" />
              <span className="text-[10px] font-mono text-slate-400">VERSION 2.5.0-ALPHA</span>
           </div>
           <p className="hidden md:block text-slate-600 text-xs">
             &copy; {new Date().getFullYear()} SmartShield Defense.
           </p>
        </div>

        <div className="flex items-center gap-6 font-mono text-xs text-slate-500">
           <button onClick={() => onChangeView('docs')} className="flex items-center gap-1 hover:text-cyan-400 cursor-pointer transition-colors">
             <Command className="w-3 h-3" /> COMMANDS
           </button>
           <button onClick={() => onChangeView('system')} className="hover:text-cyan-400 cursor-pointer transition-colors">STATUS</button>
           <button onClick={() => onChangeView('docs')} className="hover:text-cyan-400 cursor-pointer transition-colors">DOCS</button>
           <span className="hover:text-cyan-400 cursor-pointer transition-colors">PRIVACY</span>
        </div>

        <div className="flex items-center gap-3">
            <a href="#" className="p-2 rounded bg-white/5 hover:bg-cyan-500/20 text-slate-400 hover:text-cyan-400 transition-all">
                <Github className="w-3 h-3" />
            </a>
            <a href="#" className="p-2 rounded bg-white/5 hover:bg-cyan-500/20 text-slate-400 hover:text-cyan-400 transition-all">
                <Twitter className="w-3 h-3" />
            </a>
        </div>
      </div>
    </footer>
  );
};

export default Footer;