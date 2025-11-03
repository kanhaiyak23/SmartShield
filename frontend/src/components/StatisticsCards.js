import { useMemo } from 'react';

export default function StatisticsCards({ statistics }) {
  const cards = useMemo(() => {
    if (!statistics) return [];

    return [
      {
        title: 'Total Packets',
        value: statistics.totalPackets?.toLocaleString() || '0',
        icon: '📦',
        color: 'blue',
        change: '+12.5%'
      },
      {
        title: 'Total Alerts',
        value: statistics.alerts?.Total || 0,
        icon: '🚨',
        color: 'red',
        change: '+5.2%'
      },
      {
        title: 'TCP Packets',
        value: statistics.tcpPackets?.toLocaleString() || '0',
        icon: '🔗',
        color: 'green',
        change: '+8.1%'
      },
      {
        title: 'Bytes Transferred',
        value: formatBytes(statistics.totalBytes || 0),
        icon: '💾',
        color: 'purple',
        change: '+15.3%'
      }
    ];
  }, [statistics]);

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
      {cards.map((card, index) => (
        <div key={index} className="bg-white rounded-lg shadow p-6 border-l-4 border-blue-500">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">{card.title}</p>
              <p className="text-2xl font-bold text-gray-900 mt-2">{card.value}</p>
              <p className="text-xs text-green-600 mt-1">{card.change}</p>
            </div>
            <div className="text-4xl">{card.icon}</div>
          </div>
        </div>
      ))}
    </div>
  );
}

function formatBytes(bytes) {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
}

