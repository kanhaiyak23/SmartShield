const express = require('express');
const cors = require('cors');
const morgan = require('morgan');
const helmet = require('helmet');
const { createServer } = require('http');
const { Server } = require('socket.io');

const app = express();
const httpServer = createServer(app);
const io = new Server(httpServer, {
  cors: {
    origin: process.env.FRONTEND_URL || 'http://localhost:3000',
    methods: ['GET', 'POST']
  }
});

// Middleware
app.use(cors());
app.use(morgan('combined'));
app.use(helmet());
app.use(express.json());

// In-memory storage for alerts and statistics
const alerts = [];
const statistics = {
  totalPackets: 0,
  tcpPackets: 0,
  udpPackets: 0,
  icmpPackets: 0,
  arpPackets: 0,
  totalBytes: 0,
  alerts: {
    Critical: 0,
    High: 0,
    Medium: 0,
    Low: 0,
    Total: 0
  }
};

// Routes
app.get('/api/health', (req, res) => {
  res.json({ status: 'ok', timestamp: new Date().toISOString() });
});

// Get statistics
app.get('/api/statistics', (req, res) => {
  res.json(statistics);
});

// Get alerts
app.get('/api/alerts', (req, res) => {
  const { limit = 100, severity } = req.query;
  let filteredAlerts = [...alerts];
  
  if (severity) {
    filteredAlerts = filteredAlerts.filter(alert => alert.severity === severity);
  }
  
  res.json(filteredAlerts.slice(-limit));
});

// Get alert by ID
app.get('/api/alerts/:id', (req, res) => {
  const alert = alerts.find(a => a.id === req.params.id);
  if (!alert) {
    return res.status(404).json({ error: 'Alert not found' });
  }
  res.json(alert);
});

// Update statistics (called by packet capture)
app.post('/api/statistics', (req, res) => {
  Object.assign(statistics, req.body);
  
  // Emit to all connected clients
  io.emit('statistics', statistics);
  
  res.json({ success: true });
});

// Add new alert (called by detection engine)
app.post('/api/alerts', (req, res) => {
  const alert = {
    id: Date.now().toString(),
    timestamp: new Date().toISOString(),
    ...req.body
  };
  
  alerts.push(alert);
  
  // Update alert count by severity
  if (statistics.alerts[alert.severity] !== undefined) {
    statistics.alerts[alert.severity]++;
    statistics.alerts.Total++;
  }
  
  // Emit alert to all connected clients
  io.emit('new_alert', alert);
  
  // Keep only last 1000 alerts
  if (alerts.length > 1000) {
    alerts.shift();
  }
  
  res.json({ success: true, alert });
});

// Reset statistics
app.post('/api/reset', (req, res) => {
  Object.keys(statistics).forEach(key => {
    if (key === 'alerts') {
      Object.keys(statistics.alerts).forEach(severity => {
        statistics.alerts[severity] = 0;
      });
    } else if (typeof statistics[key] === 'number') {
      statistics[key] = 0;
    }
  });
  alerts.length = 0;
  
  io.emit('reset');
  
  res.json({ success: true });
});

// WebSocket connection handling
io.on('connection', (socket) => {
  console.log(`Client connected: ${socket.id}`);
  
  // Send current state to new client
  socket.emit('statistics', statistics);
  socket.emit('alerts', alerts.slice(-50));
  
  socket.on('disconnect', () => {
    console.log(`Client disconnected: ${socket.id}`);
  });
});

const PORT = process.env.PORT || 3001;

httpServer.listen(PORT, () => {
  console.log(`SmartShield Backend Server running on port ${PORT}`);
  console.log(`WebSocket server ready for connections`);
});

module.exports = app;


