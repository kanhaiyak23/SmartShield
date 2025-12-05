import React, { useState } from 'react';
import Hero from './components/Hero';
import Dashboard from './components/Dashboard';
import ThreatMap from './components/ThreatMap';
import Intel from './components/Intel';
import SystemMonitor from './components/SystemMonitor';
import AdminPanel from './components/AdminPanel';
import Documentation from './components/Documentation';
import Navbar from './components/Navbar';
import Footer from './components/Footer';
import { View } from './types';

const App: React.FC = () => {
  const [view, setView] = useState<View>('hero');

  const renderView = () => {
      switch (view) {
          case 'hero': return <Hero onEnter={() => setView('dashboard')} />;
          case 'dashboard': return <Dashboard />;
          case 'threat_map': return <ThreatMap />;
          case 'intel': return <Intel />;
          case 'system': return <SystemMonitor />;
          case 'admin': return <AdminPanel />;
          case 'docs': return <Documentation />;
          case 'status': return <SystemMonitor />; // Reuse system for status
          default: return <Hero onEnter={() => setView('dashboard')} />;
      }
  };

  return (
    <div className="bg-black h-screen text-white flex flex-col overflow-hidden selection:bg-cyan-500/30">
      <Navbar currentView={view} onChangeView={setView} />
      
      <main className="flex-1 relative overflow-hidden flex flex-col">
        {renderView()}
      </main>

      <Footer onChangeView={setView} />
    </div>
  );
};

export default App;