# SmartShield Deployment Guide

## Pre-Deployment Checklist

Before deploying SmartShield to production, ensure you have:

- [ ] All dependencies installed
- [ ] UNSW-NB15 dataset downloaded and prepared
- [ ] Models trained and saved
- [ ] Network interface identified for monitoring
- [ ] Proper permissions (root/sudo access)
- [ ] Firewall rules configured
- [ ] Monitoring and alerting set up
- [ ] Backups configured
- [ ] Documentation reviewed

## Deployment Options

### Option 1: Single Server Deployment (Recommended for Testing)

**Best for**: Small campuses, development, initial deployment

```
┌─────────────────────────────────┐
│  Single Server                  │
│  ┌───────────────────────────┐  │
│  │  ML API (Port 5000)      │  │
│  │  Backend (Port 3001)     │  │
│  │  Frontend (Port 3000)    │  │
│  │  Detection Service       │  │
│  └───────────────────────────┘  │
└─────────────────────────────────┘
```

**Requirements:**
- Ubuntu 20.04+ or similar Linux distribution
- 8GB RAM minimum, 16GB recommended
- 100GB storage
- Root/sudo access
- Network interface with promiscuous mode

**Steps:**
1. Clone repository on server
2. Follow SETUP.md installation steps
3. Train models
4. Configure network interface
5. Start services with `./start.sh`
6. Access dashboard at http://server-ip:3000

### Option 2: Distributed Deployment

**Best for**: Large campuses, production, high availability

```
┌─────────────────────────────────────────────────────────┐
│                     Load Balancer                        │
└────────┬──────────────┬──────────────┬──────────────────┘
         │              │              │
    ┌────▼────┐   ┌────▼────┐   ┌────▼────┐
    │ ML API  │   │ Backend │   │ Frontend│
    │ Server  │   │ Server  │   │ Server  │
    └─────────┘   └─────────┘   └─────────┘
         │              │              │
         └──────────────┴──────────────┘
                          │
                    ┌─────▼─────┐
                    │ Detection │
                    │  Service  │
                    └───────────┘
```

**Requirements per server:**
- ML API Server: 8GB RAM, 4 CPU cores
- Backend Server: 4GB RAM, 2 CPU cores
- Frontend Server: 2GB RAM, 1 CPU core
- Detection Service: 8GB RAM, 4 CPU cores

### Option 3: Docker Container Deployment (Future)

**Best for**: Scalability, consistency, DevOps workflows

```yaml
services:
  ml-api:
    image: smartshield/ml-api:latest
    ports: ["5000:5000"]
  
  backend:
    image: smartshield/backend:latest
    ports: ["3001:3001"]
  
  frontend:
    image: smartshield/frontend:latest
    ports: ["3000:3000"]
  
  detection:
    image: smartshield/detection:latest
    network_mode: host
```

## Production Configuration

### Environment Variables

**Backend (`backend/.env`):**
```bash
PORT=3001
NODE_ENV=production
FRONTEND_URL=https://your-domain.com
ML_API_URL=http://localhost:5000
REDIS_URL=redis://localhost:6379  # Future
DATABASE_URL=postgresql://...      # Future
```

**ML API:**
Create `python/.env`:
```bash
FLASK_ENV=production
MODEL_PATH=/opt/smartshield/models
LOG_LEVEL=INFO
CORS_ORIGINS=https://your-domain.com
```

### System Service Setup

**Create systemd service for detection:**

```bash
sudo nano /etc/systemd/system/smartshield-detection.service
```

```ini
[Unit]
Description=SmartShield Detection Service
After=network.target

[Service]
Type=simple
User=smartshield
Group=smartshield
WorkingDirectory=/opt/smartshield/python
Environment="PATH=/opt/smartshield/python/venv/bin"
ExecStart=/opt/smartshield/python/venv/bin/python live_detection.py --interface eth0
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

**Enable and start:**
```bash
sudo systemctl daemon-reload
sudo systemctl enable smartshield-detection
sudo systemctl start smartshield-detection
sudo systemctl status smartshield-detection
```

### Security Hardening

**1. Firewall Configuration**
```bash
# Allow only necessary ports
sudo ufw allow 3000/tcp  # Frontend
sudo ufw allow 3001/tcp  # Backend
sudo ufw allow 5000/tcp  # ML API (internal only)
sudo ufw enable
```

**2. SSL/TLS Certificate**
```bash
# Install certbot
sudo apt install certbot

# Obtain certificate
sudo certbot --nginx -d your-domain.com

# Or use Let's Encrypt with reverse proxy
```

**3. Authentication (Future)**
- Add JWT authentication
- Implement role-based access control
- Use OAuth2 for SSO

**4. Network Security**
- Run ML API on localhost only
- Use VPN for backend access
- Implement rate limiting
- Enable DDoS protection

### Monitoring & Alerting

**1. Application Logs**
- Centralized logging with syslog or ELK stack
- Log rotation configuration
- Alert on errors

**2. System Monitoring**
- CPU, memory, disk usage
- Network interface utilization
- Model inference latency

**3. Health Checks**
```bash
# ML API health
curl http://localhost:5000/health

# Backend health
curl http://localhost:3001/api/health

# Automated monitoring script
*/5 * * * * /opt/smartshield/scripts/health_check.sh
```

**4. Alerting**
- Email alerts on critical issues
- Slack integration for team
- SMS for emergency alerts
- PagerDuty for 24/7 support

### Database Integration (Future)

**PostgreSQL Setup:**
```sql
CREATE DATABASE smartshield;
CREATE USER smartshield_user WITH PASSWORD 'secure_password';
GRANT ALL PRIVILEGES ON DATABASE smartshield TO smartshield_user;
```

**Schema for Alerts:**
```sql
CREATE TABLE alerts (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT NOW(),
    src_ip INET,
    dst_ip INET,
    protocol VARCHAR(20),
    attack_type VARCHAR(100),
    severity VARCHAR(20),
    confidence FLOAT,
    processed BOOLEAN DEFAULT FALSE
);

CREATE INDEX idx_timestamp ON alerts(timestamp);
CREATE INDEX idx_severity ON alerts(severity);
```

### Backup & Recovery

**Critical Data to Backup:**
1. Trained models (`models/*.pkl`, `models/*.h5`)
2. Configuration files
3. Database (if implemented)
4. Custom rules and signatures

**Backup Script:**
```bash
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backup/smartshield"

mkdir -p $BACKUP_DIR/$DATE

# Backup models
tar -czf $BACKUP_DIR/$DATE/models.tar.gz /opt/smartshield/python/models

# Backup configs
cp /opt/smartshield/*.env $BACKUP_DIR/$DATE/

# Backup database (if exists)
pg_dump smartshield > $BACKUP_DIR/$DATE/database.sql

# Keep only last 7 days
find $BACKUP_DIR -type d -mtime +7 -exec rm -rf {} \;
```

**Schedule with cron:**
```bash
0 2 * * * /opt/smartshield/scripts/backup.sh
```

### Performance Optimization

**1. Model Optimization**
- Quantize TensorFlow models
- Use ONNX runtime
- Implement model caching
- Batch predictions

**2. Database Optimization**
- Implement read replicas
- Use connection pooling
- Index frequently queried fields
- Archive old data

**3. Network Optimization**
- Use packet sampling for high-traffic interfaces
- Implement BPF filters to reduce load
- Distribute detection across multiple interfaces

**4. Resource Limits**
```bash
# Set process limits
ulimit -n 65536  # Open files
ulimit -u 32768  # User processes
```

## Deployment Steps

### Initial Deployment

```bash
# 1. Prepare server
sudo apt update && sudo apt upgrade -y
sudo apt install python3 python3-pip nodejs npm git -y

# 2. Clone repository
cd /opt
sudo git clone https://github.com/yourusername/smartshield.git
sudo chown -R $USER:$USER smartshield

# 3. Install Python dependencies
cd smartshield/python
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 4. Download and prepare dataset
# Follow DATASET_SETUP.md

# 5. Train models
python train_models.py --train-all

# 6. Install backend dependencies
cd ../backend
npm install

# 7. Install frontend dependencies
cd ../frontend
npm install

# 8. Build frontend for production
npm run build

# 9. Configure environment
cp backend/env.example backend/.env
nano backend/.env  # Edit as needed

# 10. Start services
cd ..
sudo ./start.sh

# 11. Verify deployment
curl http://localhost:3001/api/health
```

### Updates and Maintenance

**1. Code Updates**
```bash
cd /opt/smartshield
sudo git pull origin main
sudo ./stop.sh
sudo ./start.sh
```

**2. Model Retraining**
```bash
cd python
source venv/bin/activate
python train_models.py --train-all
sudo systemctl restart smartshield-detection
```

**3. Dependency Updates**
```bash
# Python
cd python
source venv/bin/activate
pip install --upgrade -r requirements.txt

# Node.js
cd ../backend && npm update
cd ../frontend && npm update
```

## Troubleshooting

### Common Issues

**1. Packet Capture Fails**
```bash
# Check permissions
sudo setcap 'cap_net_raw,cap_net_admin+eip' /opt/smartshield/python/venv/bin/python3

# Check interface
ip link show

# Test capture
sudo tcpdump -i eth0 -c 10
```

**2. High Memory Usage**
```bash
# Reduce packet buffer
nano python/src/packet_capture.py
# Change packet_buffer_size

# Implement packet sampling
# Add sampling rate to live_detection.py
```

**3. Model Loading Slow**
```bash
# Use model caching
# Implement lazy loading
# Optimize model files
```

**4. Dashboard Not Updating**
```bash
# Check WebSocket connection
# Verify backend is running
# Check browser console for errors
# Verify CORS settings
```

## Scaling

### Horizontal Scaling

- Deploy multiple ML API instances
- Use load balancer for backend
- CDN for frontend static assets
- Multiple detection services per interface

### Vertical Scaling

- Increase RAM for larger models
- Faster CPUs for inference
- SSD for faster I/O
- Higher bandwidth network cards

## Support

For deployment issues:
1. Check logs in `python/logs/`
2. Review system logs: `journalctl -u smartshield`
3. Check health endpoints
4. Review documentation
5. Contact support team

## Appendix

### Useful Commands

```bash
# View logs
tail -f python/logs/*.log
journalctl -u smartshield-detection -f

# Check services
systemctl status smartshield-detection

# Restart services
sudo ./stop.sh && sudo ./start.sh

# Test API
curl http://localhost:5000/health
curl http://localhost:3001/api/health

# Monitor resources
htop
nethogs  # Network usage
iotop    # Disk I/O
```

### Performance Benchmarks

Expected performance on standard server:
- **Packet processing**: 10,000+ packets/sec
- **Detection latency**: < 100ms
- **Model accuracy**: 95%+
- **Memory usage**: ~4GB idle, ~8GB under load
- **CPU usage**: ~20% idle, ~60% under load

### Capacity Planning

| Campus Size | Students | Devices | Recommended Setup |
|------------|----------|---------|-------------------|
| Small | < 1,000 | < 2,000 | Single server |
| Medium | 1,000-5,000 | 2,000-10,000 | Single server + load balancer |
| Large | 5,000+ | 10,000+ | Distributed deployment |


