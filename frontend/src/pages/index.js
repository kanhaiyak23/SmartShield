import { useState, useEffect } from 'react';

import Layout from '../components/Layout';
import StatisticsCards from '../components/StatisticsCards';
import ProtocolChart from '../components/ProtocolChart';
import AlertsTable from '../components/AlertsTable';
import TopIPs from '../components/TopIPs';
import { useSocket } from '../hooks/useSocket';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:3001';

export default function Home() {
  const [statistics, setStatistics] = useState(null);
  const [alerts, setAlerts] = useState([]);
  const [isConnected, socket] = useSocket();

  useEffect(() => {
    // Fetch initial data
    fetch(`${API_URL}/api/statistics`)
      .then(res => res.json())
      .then(data => setStatistics(data))
      .catch(err => console.error('Failed to fetch statistics:', err));

    fetch(`${API_URL}/api/alerts?limit=100`)
      .then(res => res.json())
      .then(data => setAlerts(data))
      .catch(err => console.error('Failed to fetch alerts:', err));
  }, []);

  useEffect(() => {
    if (!socket) return;

    socket.on('statistics', (data) => {
      setStatistics(data);
    });

    socket.on('new_alert', (alert) => {
      setAlerts(prev => [alert, ...prev].slice(0, 1000));
    });

    return () => {
      socket.off('statistics');
      socket.off('new_alert');
    };
  }, [socket]);

  if (!statistics) {
    return (
      <Layout>
        <div className="flex justify-center items-center min-h-screen">
          <div className="text-center">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto"></div>
            <p className="mt-4 text-gray-600">Loading SmartShield...</p>
          </div>
        </div>
      </Layout>
    );
  }

  return (
    <Layout>
      <div className="space-y-6">
        {/* Header */}
        <div className="flex justify-between items-center">
          <div>
            <h1 className="text-3xl font-bold text-gray-900">SmartShield Dashboard</h1>
            <p className="text-gray-600 mt-1">Real-time Campus Intrusion Detection</p>
          </div>
          <div className="flex items-center space-x-2">
            <div className={`w-3 h-3 rounded-full ${isConnected ? 'bg-green-500' : 'bg-red-500'} animate-pulse`}></div>
            <span className="text-sm text-gray-600">
              {isConnected ? 'Connected' : 'Disconnected'}
            </span>
          </div>
        </div>

        {/* Statistics Cards */}
        <StatisticsCards statistics={statistics} />

        {/* Charts */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <ProtocolChart statistics={statistics} />
          <TopIPs alerts={alerts} />
        </div>

        {/* Alerts Table */}
        <AlertsTable alerts={alerts} />
      </div>
    </Layout>
  );
}

