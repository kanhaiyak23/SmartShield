# ⚡ Quick Start - Academic Demo

## 🚀 Fastest Way to Run Demo (3 Commands)

### Terminal 1: Backend
```bash
cd /Users/navnitnaman/SmartShield-1
source venv/bin/activate
sudo python3 server.py
```

### Terminal 2: Frontend
```bash
cd /Users/navnitnaman/SmartShield-1
npm run dev
```
Then open: http://localhost:3000 → Click "Dashboard"

### Terminal 3: Attack Simulator
```bash
cd /Users/navnitnaman/SmartShield-1
source venv/bin/activate
sudo python3 attack_simulator.py
# Type: 5 (for Full Demo Sequence)
```

**That's it! Watch the dashboard for real-time attack detection! 🎉**

---

## 🎯 Or Use the Interactive Demo Script

```bash
./demo.sh
```

Choose option 4 for full demo sequence.

---

## ✅ What to Look For

1. **Backend:** Should show "✅ Loaded pretrained Random Forest model"
2. **Frontend:** Should show "LOCAL_SERVER_CONNECTED" (green)
3. **Dashboard:** Packets appearing in real-time table
4. **Attacks:** RED (CRITICAL) or YELLOW (WARNING) flagged packets

---

## 📝 Need Help?

See `ACADEMIC_DEMO_GUIDE.md` for complete instructions.

