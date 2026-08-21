# AI Meeting Intelligence Platform

**Stack:** Deepgram → Lyzr (Chief of Staff Agent) → Qdrant → **Google ADK Executor**

Live demo: https://meeting-ai-orcin.vercel.app/

---

## Architecture

```
🎙️ Browser audio
    │
    ▼ WebSocket (PCM audio)
🔊 Deepgram Nova-2 (live transcription)
    │
    ▼ Final transcript
🤖 Lyzr Agent 2 — Chief of Staff
    │ Extracts: action items, owners, deadlines, decisions, summary
    │
    ▼ JSON extraction
🧠 Qdrant (long-term organizational memory via Gemini embeddings)
    │
    ▼ /api/propose
✅ Agent 3 — Google ADK Executor (HITL)
    │ Drafts Calendar events + Google Doc
    │ Shows review panel in UI
    │ Human edits & clicks Approve / Reject
    ▼ /api/execute
📅 Google Calendar + 📄 Google Docs
```

---

## Quick Start (local)

### Backend
```bash
cd backend
pip install -r requirements.txt
# Copy and fill in your env vars
copy .env.example .env
uvicorn main:app --reload --port 8000
```

### Frontend
```bash
cd frontend
npm install
# Copy and fill in your env vars
copy .env.example .env.local
npm run dev
```

---

## Deployment

### Backend → Railway
1. Push repo to GitHub
2. Go to [railway.app](https://railway.app) → New Project → Deploy from GitHub
3. Select the `backend` folder as root
4. Add env vars from `.env.example` in Railway dashboard
5. Copy the Railway public URL

### Frontend → Vercel
1. Go to [vercel.com](https://vercel.com) → New Project → Import from GitHub
2. Set **Root Directory** to `frontend`
3. Add env vars (set `NEXT_PUBLIC_BACKEND_WS_URL` and `NEXT_PUBLIC_BACKEND_API_URL` to your Railway URL)
4. Deploy → copy Vercel URL → paste into Railway `FRONTEND_URL` env var

---

## Google ADK Executor (Agent 3)

The executor runs in **DEMO/MOCK mode by default** — no credentials needed. The full HITL flow is visible to judges: propose → review → approve/reject → result links.

To enable real Google API execution:
1. Create a Google Cloud project
2. Enable Calendar API + Docs API
3. Create a Service Account → download JSON key
4. Set `GOOGLE_SERVICE_ACCOUNT_JSON` and `GOOGLE_DELEGATED_EMAIL` in Railway

---

## Environment Variables

### Backend (Railway)
| Variable | Description |
|---|---|
| `DEEPGRAM_API_KEY` | Deepgram speech-to-text |
| `LYZR_API_KEY` | Lyzr platform key |
| `LYZR_AGENT_ID` | Chief of Staff agent ID |
| `GEMINI_API_KEY` | Google Gemini for embeddings |
| `QDRANT_URL` | Qdrant cluster URL |
| `QDRANT_API_KEY` | Qdrant API key |
| `FRONTEND_URL` | Vercel URL (for CORS) |
| `GOOGLE_SERVICE_ACCOUNT_JSON` | Path to SA JSON (optional) |
| `GOOGLE_DELEGATED_EMAIL` | Email to impersonate (optional) |

### Frontend (Vercel)
| Variable | Description |
|---|---|
| `NEXT_PUBLIC_BACKEND_WS_URL` | Railway WebSocket URL |
| `NEXT_PUBLIC_BACKEND_API_URL` | Railway HTTP API URL |
