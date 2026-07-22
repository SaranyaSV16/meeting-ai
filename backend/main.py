"""
WebSocket gateway with Deepgram live transcription + Lyzr Chief of
Staff agent + Google ADK Executor (Agent 3).

Flow: browser sends PCM audio -> Deepgram transcribes -> on STOP,
Lyzr extracts action items / decisions / summary -> stored in Qdrant
-> /api/propose builds Calendar+Docs draft -> UI shows HITL panel
-> user clicks Approve -> /api/execute calls Google APIs (or mock).

Run:
    set DEEPGRAM_API_KEY=your_key_here
    set LYZR_API_KEY=your_key_here
    set LYZR_AGENT_ID=your_agent_id_here
    set GEMINI_API_KEY=your_key_here
    set QDRANT_URL=your_qdrant_url
    set QDRANT_API_KEY=your_qdrant_key
    # Optional — if missing, Google actions run in MOCK mode
    set GOOGLE_SERVICE_ACCOUNT_JSON=C:\\path\\to\\sa.json
    set GOOGLE_DELEGATED_EMAIL=you@yourorg.com
    py -m uvicorn main:app --reload --port 8000
"""

import os
import uuid
import json
import time
import asyncio
import logging
import httpx
from datetime import datetime, timedelta, timezone
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from deepgram import AsyncDeepgramClient
from deepgram.core.events import EventType
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gateway")

app = FastAPI()

# Allow both localhost dev and any deployed Vercel frontend
ALLOWED_ORIGINS = [
    "http://localhost:3000",
    os.environ.get("FRONTEND_URL", ""),
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o for o in ALLOWED_ORIGINS if o],
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Env vars ────────────────────────────────────────────────────────────────
DEEPGRAM_API_KEY = os.environ.get("DEEPGRAM_API_KEY")
LYZR_API_KEY = os.environ.get("LYZR_API_KEY")
LYZR_AGENT_ID = os.environ.get("LYZR_AGENT_ID")
LYZR_CHAT_URL = "https://agent-prod.studio.lyzr.ai/v3/inference/chat/"

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_EMBED_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:embedContent"
EMBEDDING_DIM = 768

QDRANT_URL = os.environ.get("QDRANT_URL")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")
QDRANT_COLLECTION = "meeting_memory"

# Google ADK — optional; if absent we run in MOCK mode
GOOGLE_SA_JSON = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON")  # path to file
GOOGLE_DELEGATED_EMAIL = os.environ.get("GOOGLE_DELEGATED_EMAIL", "")
GOOGLE_MOCK_MODE = not bool(GOOGLE_SA_JSON and GOOGLE_DELEGATED_EMAIL)

# ── Qdrant client ────────────────────────────────────────────────────────────
qdrant_client = None
if QDRANT_URL and QDRANT_API_KEY:
    qdrant_client = AsyncQdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# ── In-memory pending bundles (Agent 3 HITL state) ──────────────────────────
# Maps bundle_id -> { "events": [...], "doc": {...} }
pending_bundles: dict[str, dict] = {}


# ── Try to import Google libs (not hard failure) ─────────────────────────────
try:
    from google.oauth2 import service_account
    from googleapiclient.discovery import build
    GOOGLE_LIBS_AVAILABLE = True
except ImportError:
    GOOGLE_LIBS_AVAILABLE = False
    logger.warning("google-api-python-client not installed — Google actions in mock mode")


# ── Startup ──────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def ensure_qdrant_collection():
    if not qdrant_client:
        logger.warning("Qdrant not configured — storage disabled")
        return
    try:
        collections = await qdrant_client.get_collections()
        existing_names = [c.name for c in collections.collections]
        if QDRANT_COLLECTION not in existing_names:
            await qdrant_client.create_collection(
                collection_name=QDRANT_COLLECTION,
                vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
            )
            logger.info(f"created Qdrant collection '{QDRANT_COLLECTION}'")
        else:
            logger.info(f"Qdrant collection '{QDRANT_COLLECTION}' already exists")
    except Exception as e:
        logger.error(f"could not set up Qdrant collection: {e}")


# ── Health ───────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "deepgram_key_set": bool(DEEPGRAM_API_KEY),
        "lyzr_key_set": bool(LYZR_API_KEY),
        "lyzr_agent_id_set": bool(LYZR_AGENT_ID),
        "gemini_key_set": bool(GEMINI_API_KEY),
        "qdrant_configured": bool(qdrant_client),
        "google_configured": not GOOGLE_MOCK_MODE,
        "google_mock_mode": GOOGLE_MOCK_MODE,
    }


# ── Embedding helper ─────────────────────────────────────────────────────────
async def embed_text(text: str) -> list[float] | None:
    if not GEMINI_API_KEY:
        return None
    try:
        async with httpx.AsyncClient(timeout=20.0) as http_client:
            resp = await http_client.post(
                GEMINI_EMBED_URL,
                headers={"x-goog-api-key": GEMINI_API_KEY, "Content-Type": "application/json"},
                json={
                    "model": "models/gemini-embedding-001",
                    "content": {"parts": [{"text": text}]},
                    "outputDimensionality": EMBEDDING_DIM,
                },
            )
            resp.raise_for_status()
            return resp.json()["embedding"]["values"]
    except Exception as e:
        logger.error(f"embedding call failed: {e}")
        return None


# ── Qdrant storage ───────────────────────────────────────────────────────────
async def store_meeting_memory(combined_text: str, extraction: dict, session_id: str) -> bool:
    if not qdrant_client:
        return False
    vector = await embed_text(combined_text)
    if not vector:
        return False
    try:
        await qdrant_client.upsert(
            collection_name=QDRANT_COLLECTION,
            points=[
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload={
                        "session_id": session_id,
                        "transcript": combined_text,
                        "action_items": extraction.get("action_items", []),
                        "decisions": extraction.get("decisions", []),
                        "summary": extraction.get("summary", ""),
                        "timestamp": time.time(),
                    },
                )
            ],
        )
        logger.info("stored meeting extraction in Qdrant")
        return True
    except Exception as e:
        logger.error(f"Qdrant upsert failed: {e}")
        return False


# ── Lyzr Chief of Staff ──────────────────────────────────────────────────────
async def call_chief_of_staff_agent(transcript_text: str, session_id: str) -> dict | None:
    if not (LYZR_API_KEY and LYZR_AGENT_ID):
        return None
    payload = {
        "user_id": "hackathon_user",
        "agent_id": LYZR_AGENT_ID,
        "session_id": session_id,
        "message": transcript_text,
    }
    headers = {
        "accept": "application/json",
        "Content-Type": "application/json",
        "x-api-key": LYZR_API_KEY,
    }
    try:
        async with httpx.AsyncClient(timeout=40.0) as http_client:
            resp = await http_client.post(LYZR_CHAT_URL, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
        raw_reply = data.get("response") or data.get("agent_response") or ""
        cleaned = raw_reply.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:]
            cleaned = cleaned.strip()
        return json.loads(cleaned)
    except Exception as e:
        logger.error(f"Chief of Staff agent call failed: {e}")
        return None


# ═════════════════════════════════════════════════════════════════════════════
# AGENT 3 — Google ADK Executor (HITL Propose / Execute / Reject)
# ═════════════════════════════════════════════════════════════════════════════

def _parse_deadline_to_datetime(deadline_str: str | None) -> tuple[str, str]:
    """
    Best-effort parse of a natural-language deadline into RFC-3339 start/end
    strings (1-hour event). Falls back to tomorrow 10am if parsing fails.
    """
    from datetime import date
    import re

    now = datetime.now()
    target = None

    if deadline_str:
        dl = deadline_str.lower().strip()
        if "tomorrow" in dl:
            target = now + timedelta(days=1)
        elif "today" in dl:
            target = now
        elif "monday" in dl:
            target = _next_weekday(now, 0)
        elif "tuesday" in dl:
            target = _next_weekday(now, 1)
        elif "wednesday" in dl:
            target = _next_weekday(now, 2)
        elif "thursday" in dl:
            target = _next_weekday(now, 3)
        elif "friday" in dl:
            target = _next_weekday(now, 4)
        elif "saturday" in dl:
            target = _next_weekday(now, 5)
        elif "sunday" in dl:
            target = _next_weekday(now, 6)
        elif "next week" in dl:
            target = now + timedelta(weeks=1)
        elif "eod" in dl or "end of day" in dl:
            target = now
        else:
            # Try to find a date pattern like "July 25" or "25th"
            m = re.search(r"(\d{1,2})(st|nd|rd|th)?", dl)
            if m:
                day = int(m.group(1))
                target = now.replace(day=min(day, 28))

    if not target:
        target = now + timedelta(days=1)

    start = target.replace(hour=10, minute=0, second=0, microsecond=0)
    end = start + timedelta(hours=1)
    fmt = "%Y-%m-%dT%H:%M:%S"
    return start.strftime(fmt), end.strftime(fmt)


def _next_weekday(dt: datetime, weekday: int) -> datetime:
    days_ahead = weekday - dt.weekday()
    if days_ahead <= 0:
        days_ahead += 7
    return dt + timedelta(days=days_ahead)


def _build_bundle(extraction: dict) -> dict:
    """
    From a Lyzr extraction dict, build the proposed Google Workspace actions:
    - One Calendar event per action_item (if it has a deadline or owner)
    - One Google Doc with the full meeting summary + decisions
    """
    action_items = extraction.get("action_items", [])
    decisions = extraction.get("decisions", [])
    summary = extraction.get("summary", "Meeting summary not available.")

    events = []
    for item in action_items:
        task = item.get("task") or item.get("description") or str(item)
        owner = item.get("owner", "")
        deadline = item.get("deadline") or item.get("due_date") or ""
        start_str, end_str = _parse_deadline_to_datetime(deadline)
        events.append({
            "id": str(uuid.uuid4()),
            "title": f"[Action] {task[:80]}",
            "owner": owner,
            "deadline_raw": deadline,
            "start": start_str,
            "end": end_str,
            "description": f"Owner: {owner}\nDeadline: {deadline}\n\nTask: {task}",
        })

    # Build doc body
    doc_lines = [f"# Meeting Summary\n\n{summary}\n"]
    if decisions:
        doc_lines.append("\n## Decisions\n")
        for d in decisions:
            doc_lines.append(f"- {d}")
    if action_items:
        doc_lines.append("\n## Action Items\n")
        for item in action_items:
            task = item.get("task") or str(item)
            owner = item.get("owner", "")
            deadline = item.get("deadline", "")
            parts = [f"- {task}"]
            if owner:
                parts.append(f"  Owner: {owner}")
            if deadline:
                parts.append(f"  Due: {deadline}")
            doc_lines.append("\n".join(parts))

    doc_body = "\n".join(doc_lines)
    doc_title = f"Meeting Notes — {datetime.now().strftime('%B %d, %Y')}"

    return {
        "events": events,
        "doc": {"title": doc_title, "body": doc_body},
    }


def _get_google_services():
    """Returns (calendar_service, docs_service) or (None, None) if not configured."""
    if GOOGLE_MOCK_MODE or not GOOGLE_LIBS_AVAILABLE:
        return None, None
    try:
        scopes = [
            "https://www.googleapis.com/auth/calendar.events",
            "https://www.googleapis.com/auth/documents",
        ]
        credentials = service_account.Credentials.from_service_account_file(
            GOOGLE_SA_JSON, scopes=scopes
        )
        if GOOGLE_DELEGATED_EMAIL:
            credentials = credentials.with_subject(GOOGLE_DELEGATED_EMAIL)
        cal = build("calendar", "v3", credentials=credentials)
        docs = build("docs", "v1", credentials=credentials)
        return cal, docs
    except Exception as e:
        logger.error(f"Google auth failed: {e}")
        return None, None


def _execute_google_actions(bundle: dict) -> dict:
    """
    Calls Google Calendar + Docs APIs synchronously.
    Returns dict with calendar_links and doc_url.
    Falls back to mock links on any failure.
    """
    cal, docs = _get_google_services()
    mock = cal is None

    calendar_links = []
    doc_url = ""

    if mock:
        # Demo mode — return plausible-looking mock links
        for event in bundle.get("events", []):
            calendar_links.append({
                "title": event["title"],
                "url": f"https://calendar.google.com/calendar/r/eventedit?text={event['title'].replace(' ', '+')}&dates={event['start'].replace('-','').replace(':','').replace('T','')}/{event['end'].replace('-','').replace(':','').replace('T','')}",
                "mock": True,
            })
        doc_url = "https://docs.google.com/document/d/MOCK_DEMO_DOC/edit"
        return {"calendar_links": calendar_links, "doc_url": doc_url, "mock": True}

    # Real execution
    for event in bundle.get("events", []):
        try:
            body = {
                "summary": event["title"],
                "description": event["description"],
                "start": {"dateTime": event["start"], "timeZone": "UTC"},
                "end": {"dateTime": event["end"], "timeZone": "UTC"},
            }
            result = cal.events().insert(calendarId="primary", body=body).execute()
            calendar_links.append({
                "title": event["title"],
                "url": result.get("htmlLink", ""),
                "mock": False,
            })
        except Exception as e:
            logger.error(f"Calendar event creation failed: {e}")
            calendar_links.append({"title": event["title"], "url": "", "error": str(e), "mock": False})

    # Create the Google Doc
    doc_info = bundle.get("doc", {})
    try:
        doc = docs.documents().create(body={"title": doc_info.get("title", "Meeting Notes")}).execute()
        doc_id = doc.get("documentId")
        body_text = doc_info.get("body", "")
        # Insert text via batchUpdate
        docs.documents().batchUpdate(
            documentId=doc_id,
            body={
                "requests": [
                    {
                        "insertText": {
                            "location": {"index": 1},
                            "text": body_text,
                        }
                    }
                ]
            },
        ).execute()
        doc_url = f"https://docs.google.com/document/d/{doc_id}/edit"
    except Exception as e:
        logger.error(f"Google Docs creation failed: {e}")
        doc_url = ""

    return {"calendar_links": calendar_links, "doc_url": doc_url, "mock": False}


# ── Agent 3 HTTP endpoints ───────────────────────────────────────────────────

from fastapi import HTTPException
from pydantic import BaseModel


class ProposeRequest(BaseModel):
    extraction: dict


class ExecuteRequest(BaseModel):
    bundle_id: str
    events: list[dict] | None = None  # optional edited events from UI
    doc: dict | None = None           # optional edited doc from UI


class RejectRequest(BaseModel):
    bundle_id: str


@app.post("/api/propose")
async def propose(req: ProposeRequest):
    """
    Agent 3 — Step 1 (PROPOSE):
    Receives the Lyzr extraction, builds a draft of Calendar events + Docs,
    stores it under a UUID, and returns the preview to the frontend.
    The human then reviews and clicks Approve or Reject.
    """
    bundle = _build_bundle(req.extraction)
    bundle_id = str(uuid.uuid4())
    pending_bundles[bundle_id] = bundle
    logger.info(f"proposed bundle {bundle_id}: {len(bundle['events'])} events")
    return {
        "bundle_id": bundle_id,
        "mock_mode": GOOGLE_MOCK_MODE,
        **bundle,
    }


@app.post("/api/execute")
async def execute(req: ExecuteRequest):
    """
    Agent 3 — Step 2 (EXECUTE, after human approval):
    Applies any edits from the UI, calls Google APIs (or mock),
    cleans up the pending bundle, returns resource links.
    """
    bundle = pending_bundles.pop(req.bundle_id, None)
    if bundle is None:
        raise HTTPException(status_code=404, detail="bundle not found or already executed")

    # Merge UI edits if provided
    if req.events is not None:
        bundle["events"] = req.events
    if req.doc is not None:
        bundle["doc"] = req.doc

    # Run in a thread pool to avoid blocking the event loop
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, _execute_google_actions, bundle)
    logger.info(f"executed bundle {req.bundle_id}: mock={result.get('mock')}")
    return result


@app.post("/api/reject")
async def reject(req: RejectRequest):
    """
    Agent 3 — Reject path:
    Discards the pending bundle without executing anything.
    """
    discarded = pending_bundles.pop(req.bundle_id, None)
    logger.info(f"rejected bundle {req.bundle_id} (existed={discarded is not None})")
    return {"status": "rejected"}


# ═════════════════════════════════════════════════════════════════════════════
# WebSocket — Deepgram live transcription
# ═════════════════════════════════════════════════════════════════════════════

@app.websocket("/v1/streaming/ingress")
async def streaming_ingress(websocket: WebSocket):
    await websocket.accept()
    logger.info("client connected")

    sample_rate = websocket.query_params.get("sample_rate", "16000")
    logger.info(f"using sample_rate={sample_rate}")

    if not DEEPGRAM_API_KEY:
        await websocket.send_json({
            "type": "error",
            "message": "Server missing DEEPGRAM_API_KEY. Set it and restart.",
        })
        await websocket.close()
        return

    lyzr_session_id = str(uuid.uuid4())
    transcript_queue: asyncio.Queue = asyncio.Queue()
    full_transcript: list[str] = []

    def on_message(message):
        msg_type = getattr(message, "type", "")
        if msg_type == "Results":
            try:
                alternative = message.channel.alternatives[0]
                text = alternative.transcript
                is_final = bool(getattr(message, "is_final", False))
                if text:
                    transcript_queue.put_nowait({
                        "type": "transcript",
                        "text": text,
                        "is_final": is_final,
                    })
            except Exception as e:
                logger.error(f"failed to parse Deepgram message: {e}")

    def on_error(error):
        logger.error(f"Deepgram error: {error}")
        transcript_queue.put_nowait({"type": "error", "message": str(error)})

    client = AsyncDeepgramClient(api_key=DEEPGRAM_API_KEY)

    async def relay_transcripts():
        while True:
            msg = await transcript_queue.get()
            await websocket.send_json(msg)
            if msg["type"] == "transcript" and msg["is_final"]:
                full_transcript.append(msg["text"])

    relay_task = None
    listen_task = None
    chunk_count = 0
    extraction_already_sent = False

    try:
        async with client.listen.v1.connect(
            model="nova-2",
            encoding="linear16",
            sample_rate=sample_rate,
            channels="1",
            interim_results="true",
            punctuate="true",
            smart_format="true",
            language="en-US",
        ) as connection:
            connection.on(EventType.OPEN, lambda _: logger.info("Deepgram connection opened"))
            connection.on(EventType.MESSAGE, on_message)
            connection.on(EventType.CLOSE, lambda _: logger.info("Deepgram connection closed"))
            connection.on(EventType.ERROR, on_error)

            listen_task = asyncio.create_task(connection.start_listening())
            relay_task = asyncio.create_task(relay_transcripts())

            while True:
                message = await websocket.receive()
                if message["type"] == "websocket.disconnect":
                    break
                if message.get("bytes") is not None:
                    chunk_count += 1
                    await connection.send_media(message["bytes"])
                elif message.get("text") is not None:
                    try:
                        control = json.loads(message["text"])
                    except Exception:
                        control = {}
                    if control.get("type") == "stop":
                        logger.info("received stop signal from client")
                        break

            await asyncio.sleep(1.0)

            if full_transcript:
                combined_text = " ".join(full_transcript)
                logger.info(f"running end-of-meeting extraction on {len(combined_text)} chars")
                extraction = await call_chief_of_staff_agent(combined_text, lyzr_session_id)
                if extraction:
                    await websocket.send_json({"type": "extraction", "data": extraction})
                    extraction_already_sent = True

                    stored = await store_meeting_memory(combined_text, extraction, lyzr_session_id)
                    try:
                        await websocket.send_json({"type": "stored", "success": stored})
                    except Exception:
                        pass

    except WebSocketDisconnect:
        logger.info(f"client disconnected after {chunk_count} chunks")
    except Exception as e:
        logger.error(f"error in streaming session: {e}")
    finally:
        if relay_task:
            relay_task.cancel()
        if listen_task:
            listen_task.cancel()
        if full_transcript and not extraction_already_sent:
            logger.warning("session ended without successful extraction")