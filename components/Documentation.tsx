import React from 'react';
import { Book, Code, Terminal, ChevronRight, AlertTriangle } from 'lucide-react';

const Documentation: React.FC = () => {
  return (
    <div className="h-full bg-slate-950 overflow-y-auto custom-scrollbar p-6 md:p-12">
        <div className="max-w-4xl mx-auto">
             <div className="mb-12 border-b border-white/10 pb-6">
                <h1 className="text-4xl font-bold text-white mb-2">Documentation</h1>
                <p className="text-slate-400 text-lg">SmartShield Architecture & Usage Guide</p>
             </div>

             <div className="space-y-12">
                 <section className="bg-slate-900/40 p-6 rounded-xl border border-cyan-500/20">
                     <h2 className="text-2xl font-bold text-cyan-400 mb-4 flex items-center gap-2">
                         <Terminal className="w-6 h-6" /> 1. Running Real-Time Mode
                     </h2>
                     <p className="text-slate-300 mb-4">
                         To switch from Simulation Mode to Real-Time Packet Capture with Random Forest inference, run the included python server locally.
                     </p>
                     
                     <div className="bg-black/80 border border-white/10 p-6 rounded-lg font-mono text-sm text-slate-300 shadow-xl">
                         <div className="flex items-center gap-2 text-yellow-500 mb-4 border-b border-white/10 pb-2">
                             <AlertTriangle className="w-4 h-4" /> 
                             <span>REQUIREMENTS: Python 3.8+, Root/Admin Privileges (for Scapy)</span>
                         </div>
                         
                         <p className="mb-2 text-slate-500"># 1. Install Python Libraries</p>
                         <p className="text-green-400 mb-6">$ pip install flask flask-cors scapy scikit-learn numpy pandas</p>
                         
                         <p className="mb-2 text-slate-500"># 2. Run the Backend Server (Must use sudo/admin for raw sockets)</p>
                         <p className="text-green-400 mb-6">$ sudo python3 server.py</p>
                         
                         <p className="mb-2 text-slate-500"># 3. Verify Connection</p>
                         <p className="text-slate-300">The dashboard top-bar will change from <span className="text-yellow-500">SIMULATION_MODE</span> to <span className="text-cyan-400">LOCAL_SERVER_CONNECTED</span> automatically.</p>
                     </div>
                 </section>

                 <section>
                     <h2 className="text-2xl font-bold text-cyan-400 mb-4 flex items-center gap-2">
                         <Code className="w-6 h-6" /> 2. API Endpoints
                     </h2>
                     <div className="space-y-4">
                         <div className="border-l-2 border-cyan-500 pl-4">
                             <h3 className="text-white font-bold text-lg">GET /packets</h3>
                             <p className="text-slate-400 mt-1">Retrieves the latest buffer of captured packets.</p>
                             <div className="mt-2 bg-slate-900 p-2 rounded text-xs font-mono text-yellow-500">
                                 Returns: JSON Array of Packet Objects
                             </div>
                         </div>
                     </div>
                 </section>

                 <section>
                     <h2 className="text-2xl font-bold text-cyan-400 mb-4 flex items-center gap-2">
                         <Book className="w-6 h-6" /> 3. Architecture
                     </h2>
                     <p className="text-slate-300 leading-relaxed">
                         SmartShield uses a hybrid architecture. The Python backend utilizes <span className="text-white font-bold">Scapy</span> for low-level packet manipulation and sniffing. 
                         Captured data is processed through <span className="text-white font-bold">SciKit-Learn (Random Forest)</span> for real-time attack detection with 99.62% accuracy. The model uses 20 enhanced features including flow statistics, protocol info, and network behavior patterns. Data is streamed via a Flask REST API to the React frontend.
                         <br/><br/>
                         The frontend visualizes this data in real-time, showing attack probabilities and risk levels (SAFE/WARNING/CRITICAL) for each captured packet.
                     </p>
                 </section>
             </div>
        </div>
    </div>
  );
};

export default Documentation;