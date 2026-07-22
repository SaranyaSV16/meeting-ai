"use client";

import { useRef, useState, useEffect, useCallback } from "react";

const WS_URL =
  process.env.NEXT_PUBLIC_BACKEND_WS_URL || "ws://localhost:8000/v1/streaming/ingress";
const API_URL =
  process.env.NEXT_PUBLIC_BACKEND_API_URL || "http://localhost:8000";

const VU_BAR_COUNT = 20;

export default function Home() {
  const [status, setStatus] = useState("idle"); // idle | connecting | live | finishing
  const [transcriptLines, setTranscriptLines] = useState([]);
  const [extractions, setExtractions] = useState([]);
  const [sysLog, setSysLog] = useState([]);
  const [showSysLog, setShowSysLog] = useState(false);
  const [pipeline, setPipeline] = useState({
    mic: "idle",
    deepgram: "idle",
    lyzr: "idle",
    qdrant: "idle",
    google: "idle",
  });

  // ── Agent 3 HITL state ────────────────────────────────────────────────────
  const [pendingBundle, setPendingBundle] = useState(null);   // { bundle_id, events, doc, mock_mode }
  const [editedEvents, setEditedEvents] = useState([]);        // editable copy of events
  const [editedDoc, setEditedDoc] = useState(null);            // editable copy of doc
  const [googleResult, setGoogleResult] = useState(null);      // { calendar_links, doc_url, mock }
  const [googleLoading, setGoogleLoading] = useState(false);

  const wsRef = useRef(null);
  const audioContextRef = useRef(null);
  const streamRef = useRef(null);
  const processorRef = useRef(null);
  const analyserRef = useRef(null);
  const awaitingFinalRef = useRef(false);
  const vuBarRefs = useRef([]);
  const rafRef = useRef(null);

  const appendSysLog = (line) => {
    const time = new Date().toLocaleTimeString();
    setSysLog((prev) => [...prev.slice(-49), `${time}  ${line}`]);
  };

  const setStage = (stage, state) => {
    setPipeline((prev) => ({ ...prev, [stage]: state }));
  };

  // ── VU meter ──────────────────────────────────────────────────────────────
  const runVuMeterLoop = () => {
    const analyser = analyserRef.current;
    if (!analyser) return;
    const data = new Uint8Array(analyser.fftSize);
    analyser.getByteTimeDomainData(data);
    let peak = 0;
    for (let i = 0; i < data.length; i++) {
      const v = Math.abs(data[i] - 128) / 128;
      if (v > peak) peak = v;
    }
    const level = Math.min(1, peak * 3.2);
    const litBars = Math.round(level * VU_BAR_COUNT);
    vuBarRefs.current.forEach((bar, i) => {
      if (!bar) return;
      const heightPct = 25 + (i / VU_BAR_COUNT) * 75;
      bar.style.height = `${heightPct}%`;
      bar.style.background = !i < litBars
        ? "var(--accent-idle)"
        : i < VU_BAR_COUNT * 0.7
        ? "var(--accent-ok)"
        : i < VU_BAR_COUNT * 0.9
        ? "var(--accent-info)"
        : "var(--accent-rec)";
      if (i < litBars) {
        bar.style.background =
          i < VU_BAR_COUNT * 0.7
            ? "var(--accent-ok)"
            : i < VU_BAR_COUNT * 0.9
            ? "var(--accent-info)"
            : "var(--accent-rec)";
      } else {
        bar.style.background = "var(--accent-idle)";
      }
    });
    rafRef.current = requestAnimationFrame(runVuMeterLoop);
  };

  const floatTo16BitPCM = (float32Array) => {
    const int16Array = new Int16Array(float32Array.length);
    for (let i = 0; i < float32Array.length; i++) {
      const s = Math.max(-1, Math.min(1, float32Array[i]));
      int16Array[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
    }
    return int16Array;
  };

  // ── Agent 3: Propose ─────────────────────────────────────────────────────
  const proposeActions = useCallback(async (extraction) => {
    setStage("google", "active");
    appendSysLog("Agent 3: proposing Google Workspace actions…");
    try {
      const resp = await fetch(`${API_URL}/api/propose`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ extraction }),
      });
      if (!resp.ok) throw new Error(`propose failed: ${resp.status}`);
      const data = await resp.json();
      setPendingBundle(data);
      setEditedEvents(data.events || []);
      setEditedDoc(data.doc || null);
      appendSysLog(
        `Agent 3: ${data.events?.length || 0} events + 1 doc drafted${data.mock_mode ? " (mock mode)" : ""}`
      );
    } catch (err) {
      appendSysLog(`Agent 3 propose error: ${err.message}`);
      setStage("google", "error");
    }
  }, []);

  // ── Agent 3: Approve ─────────────────────────────────────────────────────
  const approveActions = async () => {
    if (!pendingBundle) return;
    setGoogleLoading(true);
    appendSysLog("Agent 3: executing approved actions…");
    try {
      const resp = await fetch(`${API_URL}/api/execute`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          bundle_id: pendingBundle.bundle_id,
          events: editedEvents,
          doc: editedDoc,
        }),
      });
      if (!resp.ok) throw new Error(`execute failed: ${resp.status}`);
      const result = await resp.json();
      setGoogleResult(result);
      setPendingBundle(null);
      setStage("google", "done");
      appendSysLog(
        result.mock
          ? "Agent 3: actions executed (mock — links are Google Calendar quick-add URLs)"
          : "Agent 3: Calendar events + Doc created ✓"
      );
    } catch (err) {
      appendSysLog(`Agent 3 execute error: ${err.message}`);
      setStage("google", "error");
    } finally {
      setGoogleLoading(false);
    }
  };

  // ── Agent 3: Reject ──────────────────────────────────────────────────────
  const rejectActions = async () => {
    if (!pendingBundle) return;
    setGoogleLoading(true);
    try {
      await fetch(`${API_URL}/api/reject`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ bundle_id: pendingBundle.bundle_id }),
      });
    } catch (_) {}
    setPendingBundle(null);
    setStage("google", "idle");
    appendSysLog("Agent 3: proposed actions rejected");
    setGoogleLoading(false);
  };

  // ── Start meeting ────────────────────────────────────────────────────────
  const startMeeting = async () => {
    setStatus("connecting");
    setTranscriptLines([]);
    setExtractions([]);
    setPendingBundle(null);
    setGoogleResult(null);
    setPipeline({ mic: "active", deepgram: "idle", lyzr: "idle", qdrant: "idle", google: "idle" });

    let stream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      appendSysLog("Mic permission granted");
    } catch (err) {
      appendSysLog(`Mic denied — ${err.name}: ${err.message}`);
      setStage("mic", "error");
      setStatus("idle");
      return;
    }
    streamRef.current = stream;

    const audioContext = new AudioContext();
    audioContextRef.current = audioContext;
    const nativeSampleRate = audioContext.sampleRate;

    const ws = new WebSocket(`${WS_URL}?sample_rate=${nativeSampleRate}`);
    ws.binaryType = "arraybuffer";
    wsRef.current = ws;

    ws.onmessage = async (event) => {
      const msg = JSON.parse(event.data);

      if (msg.type === "transcript" && msg.text) {
        setStage("deepgram", "active");
        setTranscriptLines((prev) => [
          ...prev.slice(-39),
          { text: msg.text, isFinal: msg.is_final },
        ]);
      } else if (msg.type === "extraction") {
        setStage("lyzr", "done");
        setStage("qdrant", "active");
        const ex = { ...msg.data, at: Date.now() };
        setExtractions((prev) => [...prev, ex]);
        appendSysLog("Extraction received from Lyzr Chief of Staff");
        // Immediately propose Google actions
        await proposeActions(msg.data);
      } else if (msg.type === "stored") {
        setStage("qdrant", msg.success ? "done" : "error");
        appendSysLog(msg.success ? "Stored in Qdrant memory" : "Qdrant storage failed");
        if (awaitingFinalRef.current) {
          awaitingFinalRef.current = false;
          wsRef.current?.close();
        }
      } else if (msg.type === "error") {
        appendSysLog(`ERROR: ${msg.message}`);
        setStage("deepgram", "error");
      }
    };

    ws.onerror = () => {
      appendSysLog("WebSocket error — is backend running?");
    };

    ws.onclose = () => {
      setStatus("idle");
      setPipeline((prev) => ({ ...prev, mic: "idle" }));
      appendSysLog("Session closed");
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
    };

    await new Promise((resolve, reject) => {
      ws.onopen = () => {
        setStatus("live");
        appendSysLog("Connected — streaming live");
        resolve();
      };
      ws.addEventListener("error", reject, { once: true });
    });

    const source = audioContext.createMediaStreamSource(stream);
    const analyser = audioContext.createAnalyser();
    analyser.fftSize = 256;
    analyserRef.current = analyser;
    source.connect(analyser);
    rafRef.current = requestAnimationFrame(runVuMeterLoop);

    const processor = audioContext.createScriptProcessor(4096, 1, 1);
    processorRef.current = processor;
    processor.onaudioprocess = (event) => {
      if (ws.readyState !== WebSocket.OPEN) return;
      const pcm16 = floatTo16BitPCM(event.inputBuffer.getChannelData(0));
      ws.send(pcm16.buffer);
    };
    source.connect(processor);
    processor.connect(audioContext.destination);

    appendSysLog(`Streaming at native ${nativeSampleRate}Hz`);
  };

  const stopMeeting = () => {
    processorRef.current?.disconnect();
    analyserRef.current = null;
    audioContextRef.current?.close();
    streamRef.current?.getTracks().forEach((t) => t.stop());
    if (rafRef.current) cancelAnimationFrame(rafRef.current);
    vuBarRefs.current.forEach((bar) => {
      if (bar) bar.style.background = "var(--accent-idle)";
    });
    setStatus("finishing");
    setStage("mic", "idle");
    setStage("lyzr", "active");
    appendSysLog("Stop — running final extraction…");
    awaitingFinalRef.current = true;
    wsRef.current?.send(JSON.stringify({ type: "stop" }));
    setTimeout(() => {
      if (awaitingFinalRef.current) {
        appendSysLog("Timed out waiting for extraction — closing");
        awaitingFinalRef.current = false;
        wsRef.current?.close();
      }
    }, 45000);
  };

  useEffect(() => {
    return () => { if (rafRef.current) cancelAnimationFrame(rafRef.current); };
  }, []);

  const stageLabel = { idle: "idle", active: "processing", done: "done", error: "error" };

  const statusText = {
    idle: "Ready",
    connecting: "Connecting…",
    live: "Recording live",
    finishing: "Finishing — extracting…",
  }[status];

  // ── Render ────────────────────────────────────────────────────────────────
  return (
    <main className="console">
      {/* Header */}
      <div className="header">
        <div className="brand">
          <span className="brand-mark">CHIEF OF STAFF</span>
          <span className="brand-sub">Meeting Intelligence · ADK Edition</span>
        </div>
        <div className="header-controls">
          <div className="status-text">{statusText}</div>
          <div className="vu-meter">
            <span className="vu-label">MIC</span>
            {Array.from({ length: VU_BAR_COUNT }).map((_, i) => (
              <div key={i} className="vu-bar" ref={(el) => (vuBarRefs.current[i] = el)} style={{ height: "25%" }} />
            ))}
          </div>
          <button
            className={`rec-button ${status === "live" ? "is-live" : "is-idle"}`}
            onClick={status === "live" ? stopMeeting : startMeeting}
            disabled={status === "connecting" || status === "finishing"}
          >
            <span className="rec-dot" />
            {status === "live" ? "STOP" : "START"}
          </button>
        </div>
      </div>

      {/* Pipeline status strip */}
      <div className="pipeline">
        {["mic", "deepgram", "lyzr", "qdrant", "google"].map((stage, i, arr) => (
          <div key={stage} style={{ display: "flex", alignItems: "center" }}>
            <div className={`pipeline-stage is-${pipeline[stage]}`}>
              <span className="pipeline-dot" />
              {stage === "google" ? "GOOGLE ADK" : stage.toUpperCase()} · {stageLabel[pipeline[stage]]}
            </div>
            {i < arr.length - 1 && <div className="pipeline-connector" />}
          </div>
        ))}
      </div>

      {/* Main grid */}
      <div className="grid">
        {/* Transcript panel */}
        <div className="panel">
          <div className="panel-header">
            <span className="panel-title">Live Transcript</span>
            <span className="panel-count">{transcriptLines.length} segments</span>
          </div>
          <div className="panel-body scanlines">
            {transcriptLines.length === 0 ? (
              <div className="transcript-empty">Click START and speak — transcript appears here in real time.</div>
            ) : (
              transcriptLines.map((line, i) => (
                <div key={i} className={`transcript-line ${line.isFinal ? "is-final" : "is-interim"}`}>
                  {line.text}
                </div>
              ))
            )}
          </div>
        </div>

        {/* Extractions panel */}
        <div className="panel">
          <div className="panel-header">
            <span className="panel-title">Extracted Intelligence</span>
            <span className="panel-count">{extractions.length} entries</span>
          </div>
          <div className="panel-body">
            {extractions.length === 0 ? (
              <div className="extraction-empty">Action items, decisions &amp; summaries appear here after meeting ends.</div>
            ) : (
              extractions.map((ex, i) => (
                <div key={i} className="ex-card">
                  <div className="ex-timestamp">{new Date(ex.at).toLocaleTimeString()}</div>
                  {ex.summary && <div className="ex-summary">{ex.summary}</div>}
                  {ex.action_items?.length > 0 && (
                    <>
                      <div className="ex-section-label">Action Items</div>
                      {ex.action_items.map((item, j) => (
                        <div key={j} className="ex-action-item">
                          <span style={{ flex: 1 }}>{item.task}</span>
                          {item.owner && <span className="ex-chip owner">{item.owner}</span>}
                          {item.deadline && <span className="ex-chip deadline">{item.deadline}</span>}
                        </div>
                      ))}
                    </>
                  )}
                  {ex.decisions?.length > 0 && (
                    <>
                      <div className="ex-section-label" style={{ marginTop: 12 }}>Decisions</div>
                      {ex.decisions.map((d, j) => (
                        <div key={j} className="ex-decision">{d}</div>
                      ))}
                    </>
                  )}
                </div>
              ))
            )}
          </div>
        </div>
      </div>

      {/* ═══ AGENT 3: Google ADK HITL Approval Panel ═══ */}
      {pendingBundle && (
        <div className="google-panel">
          <div className="google-panel-header">
            <div className="google-panel-title-row">
              <span className="google-icon">🤖</span>
              <span className="google-panel-title">Agent 3 — Google ADK Executor</span>
              {pendingBundle.mock_mode && (
                <span className="mock-badge">DEMO MODE</span>
              )}
            </div>
            <p className="google-panel-sub">
              Review proposed actions below. Edit any field, then <strong>Approve</strong> to execute or <strong>Reject</strong> to discard.
            </p>
          </div>

          {/* Calendar Events */}
          {editedEvents.length > 0 && (
            <div className="google-section">
              <div className="google-section-label">📅 Calendar Events ({editedEvents.length})</div>
              <div className="google-events-grid">
                {editedEvents.map((ev, i) => (
                  <div key={ev.id || i} className="google-event-card">
                    <label className="google-field-label">Event Title</label>
                    <input
                      className="google-input"
                      value={ev.title}
                      onChange={(e) => {
                        const updated = [...editedEvents];
                        updated[i] = { ...updated[i], title: e.target.value };
                        setEditedEvents(updated);
                      }}
                    />
                    <div className="google-event-meta">
                      <div>
                        <label className="google-field-label">Start</label>
                        <input
                          className="google-input google-input-sm"
                          type="datetime-local"
                          value={ev.start}
                          onChange={(e) => {
                            const updated = [...editedEvents];
                            updated[i] = { ...updated[i], start: e.target.value };
                            setEditedEvents(updated);
                          }}
                        />
                      </div>
                      <div>
                        <label className="google-field-label">Owner</label>
                        <input
                          className="google-input google-input-sm"
                          value={ev.owner || ""}
                          onChange={(e) => {
                            const updated = [...editedEvents];
                            updated[i] = { ...updated[i], owner: e.target.value };
                            setEditedEvents(updated);
                          }}
                        />
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Google Doc */}
          {editedDoc && (
            <div className="google-section">
              <div className="google-section-label">📄 Google Doc</div>
              <div className="google-doc-preview">
                <label className="google-field-label">Document Title</label>
                <input
                  className="google-input"
                  value={editedDoc.title}
                  onChange={(e) => setEditedDoc({ ...editedDoc, title: e.target.value })}
                />
                <label className="google-field-label" style={{ marginTop: 10 }}>Content Preview</label>
                <textarea
                  className="google-textarea"
                  value={editedDoc.body}
                  rows={6}
                  onChange={(e) => setEditedDoc({ ...editedDoc, body: e.target.value })}
                />
              </div>
            </div>
          )}

          {/* HITL action buttons */}
          <div className="google-actions">
            <button
              className="approve-btn"
              onClick={approveActions}
              disabled={googleLoading}
            >
              {googleLoading ? "Executing…" : "✅ Approve & Execute"}
            </button>
            <button
              className="reject-btn"
              onClick={rejectActions}
              disabled={googleLoading}
            >
              ❌ Reject
            </button>
          </div>
        </div>
      )}

      {/* Google result links */}
      {googleResult && (
        <div className="google-result">
          <div className="google-result-title">
            ✓ Google ADK Executor Complete
            {googleResult.mock && <span className="mock-badge" style={{ marginLeft: 10 }}>DEMO LINKS</span>}
          </div>
          <div className="google-result-links">
            {googleResult.calendar_links?.map((link, i) => (
              link.url && (
                <a key={i} href={link.url} target="_blank" rel="noopener noreferrer" className="google-link calendar-link">
                  📅 {link.title || `Event ${i + 1}`}
                </a>
              )
            ))}
            {googleResult.doc_url && (
              <a href={googleResult.doc_url} target="_blank" rel="noopener noreferrer" className="google-link doc-link">
                📄 Open Meeting Notes Doc
              </a>
            )}
          </div>
        </div>
      )}

      {/* System log */}
      <button className="syslog-toggle" onClick={() => setShowSysLog((v) => !v)}>
        {showSysLog ? "▾" : "▸"} System log ({sysLog.length})
      </button>
      {showSysLog && (
        <div className="syslog">
          {sysLog.map((line, i) => <div key={i}>{line}</div>)}
        </div>
      )}
    </main>
  );
}