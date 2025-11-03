import { useMemo } from 'react';

export default function TopIPs({ alerts }) {
  const topIPs = useMemo(() => {
    const ipCounts = {};
    const ipDetails = {};

    alerts.forEach(alert => {
      const srcIP = alert.src_ip;
      if (srcIP && srcIP !== 'unknown') {
        ipCounts[srcIP] = (ipCounts[srcIP] || 0) + 1;
        if (!ipDetails[srcIP]) {
          ipDetails[srcIP] = {
            attack_types: new Set(),
            severity: alert.severity
          };
        }
        if (alert.attack_type) {
          ipDetails[srcIP].attack_types.add(alert.attack_type);
        }
      }
    });

    return Object.entries(ipCounts)
      .map(([ip, count]) => ({
        ip,
        count,
        attack_types: Array.from(ipDetails[ip].attack_types),
        severity: ipDetails[ip].severity
      }))
      .sort((a, b) => b.count - a.count)
      .slice(0, 10);
  }, [alerts]);

  return (
    <div className="bg-white rounded-lg shadow p-6">
      <h2 className="text-xl font-bold text-gray-900 mb-4">Top Source IPs by Alerts</h2>
      {topIPs.length === 0 ? (
        <p className="text-gray-500 text-center py-8">No alerts to display</p>
      ) : (
        <div className="space-y-4">
          {topIPs.map((item, index) => (
            <div key={item.ip} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg hover:bg-gray-100 transition">
              <div className="flex items-center space-x-4">
                <div className="w-8 h-8 flex items-center justify-center bg-blue-100 rounded-full text-blue-600 font-bold">
                  {index + 1}
                </div>
                <div>
                  <p className="font-medium text-gray-900">{item.ip}</p>
                  <p className="text-sm text-gray-600">
                    {item.attack_types.slice(0, 2).join(', ')}
                    {item.attack_types.length > 2 && '...'}
                  </p>
                </div>
              </div>
              <div className="flex items-center space-x-4">
                <span className="text-sm text-gray-600">
                  {item.attack_types.length} types
                </span>
                <span className="px-3 py-1 bg-blue-600 text-white rounded-full font-semibold">
                  {item.count}
                </span>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

