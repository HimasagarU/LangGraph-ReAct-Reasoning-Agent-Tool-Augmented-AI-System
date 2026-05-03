# Render Deployment Guide

## Quick Deploy Steps

### 1. **Connect GitHub to Render**
   - Go to [Render Dashboard](https://dashboard.render.com)
   - Click "New" → "Web Service"
   - Connect your GitHub account
   - Select this repository: `HimasagarU/LangGraph-ReAct-Reasoning-Agent-Tool-Augmented-AI-System`

### 2. **Configure Deployment**
   - **Name:** `langgraph-react-agent` (or any name you prefer)
   - **Environment:** Docker
   - **Plan:** Free tier
   - **Branch:** `main`

### 3. **Add Environment Variables**
   In the Render dashboard, add these secrets under "Environment":
   ```
   GROQ_API_KEY=<your-groq-api-key>
   TAVILY_API_KEY=<your-tavily-api-key>
   MODEL_NAME=llama-3.3-70b-versatile
   MAX_ITERATIONS=5
   MODEL_TEMPERATURE=0.2
   ```

### 4. **Deploy**
   - Click "Create Web Service"
   - Render will automatically build and deploy using the `Dockerfile` and `render.yaml`

---

## Keep Server Alive on Free Tier

Render's free tier spins down services after **15 minutes** of inactivity. To keep your server alive:

### **Option A: GitHub Actions (Recommended)**

1. **Get your Render URL:**
   - After deployment, copy your URL (e.g., `https://langgraph-react-agent.onrender.com`)

2. **Add GitHub Secret:**
   - Go to your GitHub repo → Settings → Secrets and variables → Actions
   - Click "New repository secret"
   - Name: `RENDER_URL`
   - Value: `https://your-render-url.onrender.com`

3. **Activate Workflow:**
   - The `.github/workflows/keep-alive.yml` will automatically:
     - Run every 13 minutes
     - Send a ping to `/ping` endpoint
     - Keep your server alive 24/7

### **Option B: External Cron Service (EasyCron)**

If you prefer manual setup:
1. Go to [EasyCron.com](https://www.easycron.com)
2. Create an account (free)
3. Add a new cron job:
   - **URL:** `https://your-render-url.onrender.com/ping`
   - **Schedule:** Every 10 minutes
   - **HTTP Method:** GET

---

## Testing Your Deployment

### Test the health endpoint:
```bash
curl https://your-render-url.onrender.com/ping
# Response: {"status":"alive"}
```

### Test the full health check:
```bash
curl https://your-render-url.onrender.com/health
```

### Test the agent:
```bash
curl -X POST https://your-render-url.onrender.com/agent/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is 2+2?",
    "max_iterations": 3
  }'
```

---

## Monitoring

- **View Logs:** Render Dashboard → Your Service → Logs
- **Check Status:** Render Dashboard → Your Service → Overview
- **Monitor GitHub Actions:** GitHub Repo → Actions tab

---

## Cost

- **Free tier:** $0/month (perfect for testing)
- **Limitations:** 
  - Spins down after 15 minutes of inactivity
  - Limited compute resources
  - No SSL on custom domains

---

## Next Steps

1. Deploy to Render using steps above
2. Add `RENDER_URL` secret to GitHub
3. GitHub Actions will automatically ping your server every 13 minutes
4. Monitor the `/ping` endpoint to ensure it's working

For issues, check:
- Render logs: `Render Dashboard → Logs`
- GitHub Actions: `GitHub Repo → Actions → Keep Render Server Alive`
