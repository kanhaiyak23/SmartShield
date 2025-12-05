# Deploy SmartShield Backend to Render

## Quick Deploy Steps

### 1. Create Render Account
- Go to https://render.com
- Sign up with GitHub

### 2. Create New Web Service
1. Click "New +" → "Web Service"
2. Connect your GitHub repository: `kanhaiyak23/SmartShield`
3. Configure the service:
   - **Name:** `smartshield-backend`
   - **Region:** Choose closest to you
   - **Branch:** `main`
   - **Root Directory:** (leave empty)
   - **Environment:** `Python 3`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `python3 server.py`

### 3. Environment Variables (Optional)
Add these in Render dashboard → Environment:
- `PORT` - Automatically set by Render (don't override)
- `PYTHON_VERSION` - `3.11.0` (optional)

### 4. Deploy
- Click "Create Web Service"
- Render will build and deploy automatically
- Wait for deployment to complete (5-10 minutes)

### 5. Get Your Backend URL
- After deployment, you'll get a URL like: `https://smartshield-backend.onrender.com`
- Copy this URL

### 6. Update Frontend
Update your Vercel deployment with the backend URL:

**In Vercel Dashboard:**
1. Go to your project settings
2. Navigate to "Environment Variables"
3. Add: `VITE_BACKEND_URL` = `https://your-backend-url.onrender.com`
4. Redeploy

**Or update `services/scapyService.ts` directly:**
```typescript
// Temporarily hardcode for testing
const API_URL = 'https://your-backend-url.onrender.com/packets';
```

## Important Notes

⚠️ **Packet Capture Limitation:**
- Render (and most cloud services) **cannot capture network packets**
- The backend will run but won't receive real packet data
- This is a limitation of cloud infrastructure (no raw socket access)

### Workarounds:
1. **Use for API/Demo:** Backend can still serve as an API endpoint
2. **Local Backend:** Users can run backend locally for real packet capture
3. **Hybrid:** Use cloud backend for API, local for packet capture

## Testing the Deployment

1. Check health endpoint:
   ```bash
   curl https://your-backend-url.onrender.com/health
   ```

2. Check packets endpoint:
   ```bash
   curl https://your-backend-url.onrender.com/packets
   ```

3. Check status:
   ```bash
   curl https://your-backend-url.onrender.com/status
   ```

## Troubleshooting

### Build Fails
- Check Render logs for error messages
- Ensure `requirements.txt` is correct
- Verify Python version compatibility

### Service Crashes
- Check logs in Render dashboard
- Verify model files are not too large (Render has size limits)
- Check if port binding is correct

### CORS Errors
- Ensure `flask-cors` is installed
- Check CORS configuration in `server.py`

## Cost Considerations

- **Free Tier:** Render free tier has limitations:
  - Services spin down after 15 minutes of inactivity
  - Limited resources
  - May take time to wake up

- **Paid Plans:** For production use, consider paid plans for:
  - Always-on service
  - Better performance
  - More resources

## Alternative: Railway.app

Railway is another option with similar setup:

1. Go to https://railway.app
2. Connect GitHub repo
3. Deploy Python service
4. Set start command: `python3 server.py`
5. Get deployment URL

