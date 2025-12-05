# SmartShield Deployment Guide

## Overview

SmartShield consists of two components:
1. **Frontend** (React/Vite) - Web dashboard
2. **Backend** (Flask/Python) - Packet capture and ML analysis

## Deployment Options

### Option 1: Local Development (Recommended for Testing)

Both frontend and backend run on your local machine.

**Frontend:**
```bash
npm install
npm run dev
# Access at http://localhost:3000
```

**Backend:**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
sudo python3 server.py
# Runs on http://127.0.0.1:5000
```

### Option 2: Frontend Hosted, Backend Local

Deploy frontend to Vercel/Netlify, but backend must run locally on each user's machine.

**Frontend Deployment (Vercel):**
```bash
# Set environment variable in Vercel dashboard:
# VITE_BACKEND_URL=http://127.0.0.1:5000

vercel deploy
```

**Backend (User's Machine):**
- Each user must run the backend locally
- Users will be prompted for local network permission
- Backend URL defaults to `http://127.0.0.1:5000`

### Option 3: Full Cloud Deployment (Advanced)

Deploy both frontend and backend to cloud services.

#### Backend Deployment Options:

**A. Railway.app:**
```bash
# Create railway.json
{
  "build": {
    "builder": "NIXPACKS"
  },
  "deploy": {
    "startCommand": "python3 server.py",
    "restartPolicyType": "ON_FAILURE"
  }
}

# Note: Packet capture requires privileged mode
# Railway may not support raw socket access
```

**B. Render.com:**
```bash
# Use Dockerfile for deployment
# Note: Cloud services typically don't allow raw packet capture
# Consider using a different approach for cloud deployment
```

**C. Custom VPS (DigitalOcean, AWS EC2, etc.):**
```bash
# Full control, can enable packet capture
# Requires root access and proper network configuration
```

#### Frontend Configuration:

Set the backend URL via environment variable:

```bash
# .env.production
VITE_BACKEND_URL=https://your-backend-url.com
```

Or configure at runtime:
```javascript
// In index.html or config
window.__BACKEND_URL__ = 'https://your-backend-url.com';
```

## Important Notes

### Packet Capture Limitations

⚠️ **Cloud services typically cannot capture network packets** because:
- They don't have access to raw network interfaces
- Security restrictions prevent privileged network access
- Virtualized environments isolate network traffic

### Recommended Architecture

For production use:
1. **Frontend**: Deploy to Vercel/Netlify (static hosting)
2. **Backend**: Run locally on each user's machine OR deploy to a VPS with network access
3. **Alternative**: Use a VPN/tunneling solution to route traffic to a cloud backend

### Environment Variables

**Frontend (.env):**
```
VITE_BACKEND_URL=http://127.0.0.1:5000  # Local development
# OR
VITE_BACKEND_URL=https://api.smartshield.com  # Production backend
```

**Backend (server.py):**
- No environment variables needed for basic setup
- Model files should be in the same directory or configured path

## Troubleshooting

### "Backend Server Not Connected" Error

1. **Check if backend is running:**
   ```bash
   curl http://127.0.0.1:5000/packets
   ```

2. **Check firewall settings:**
   - Ensure port 5000 is not blocked
   - On macOS, check System Preferences > Security & Privacy

3. **Check browser permissions:**
   - Grant local network access when prompted
   - Check browser console for CORS errors

4. **Verify backend URL:**
   - Check browser console for the actual URL being requested
   - Ensure it matches your backend configuration

### CORS Issues

If you see CORS errors, ensure `flask-cors` is installed and configured in `server.py`:
```python
from flask_cors import CORS
CORS(app)  # Enable CORS for all routes
```

## Security Considerations

1. **Never expose backend with packet capture to public internet** without proper security
2. **Use HTTPS** for production deployments
3. **Implement authentication** for cloud-deployed backends
4. **Restrict access** to backend API endpoints
5. **Monitor and log** all network access

## Support

For deployment issues, check:
- [README.md](README.md) - Basic setup instructions
- [GitHub Issues](https://github.com/kanhaiyak23/SmartShield/issues) - Known issues and solutions

