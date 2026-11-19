import time
import uuid
import logging
import threading
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel, Field

from answer import setup, MODEL
from agent import run_agent
from pipeline import build_pipeline_from_pdf, doc_id_for

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("rag-api")

STATE = {}              # built-in Tesla index: chunks, bm25, collection (loaded ONCE at startup)
DOCS = {}               # doc_id -> uploaded document (chunks, bm25, collection)
SESSIONS = {}           # session_id -> [{"user": ..., "assistant": ...}]
MAX_TURNS = 6           # remembered exchanges per session
MAX_SESSIONS = 200      # oldest session is dropped beyond this
MAX_DOCS = 10           # oldest uploaded document is dropped beyond this
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
LOCK = threading.Lock()
METRICS = {"requests": 0, "errors": 0, "clarifications": 0, "gave_up": 0, "uploads": 0,
           "total_latency_ms": 0.0, "llm_calls": 0, "tokens": 0, "empty_replies": 0}


@asynccontextmanager
async def lifespan(app):
    chunks, bm25, collection = setup()
    STATE.update(chunks=chunks, bm25=bm25, collection=collection)
    log.info("pipeline ready: %d chunks", len(chunks))
    yield
    STATE.clear()


app = FastAPI(title="Agentic RAG over your documents", version="2.1", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    session_id: Optional[str] = Field(default=None, max_length=64)
    doc_id: Optional[str] = Field(default=None, max_length=64)   # None = the built-in Tesla 10-K


@app.get("/health")
def health():
    return {"status": "ok" if STATE else "starting", "model": MODEL,
            "chunks": len(STATE.get("chunks", [])), "uploaded_documents": len(DOCS)}

@app.get("/metrics")
def metrics():
    with LOCK:
        m = dict(METRICS)
    n = max(m["requests"], 1)
    m["avg_latency_ms"] = round(m.pop("total_latency_ms") / n, 1)
    m["avg_tokens_per_request"] = round(m["tokens"] / n, 1)
    m["active_sessions"] = len(SESSIONS)
    return m


@app.post("/upload")
def upload(file: UploadFile = File(...)):
    """Upload any PDF; returns a doc_id to send with /ask."""
    if not STATE:
        raise HTTPException(status_code=503, detail="Pipeline is still starting")
    name = file.filename or "uploaded.pdf"
    if not name.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    data = file.file.read(MAX_UPLOAD_BYTES + 1)
    if not data:
        raise HTTPException(status_code=400, detail="The file is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="The file is larger than 20 MB.")

    doc = DOCS.get(doc_id_for(data))           # same file uploaded before: reuse it
    if doc is None:
        t0 = time.perf_counter()
        try:
            doc = build_pipeline_from_pdf(data, name)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        DOCS[doc["doc_id"]] = doc
        while len(DOCS) > MAX_DOCS:
            DOCS.pop(next(iter(DOCS)))
        with LOCK:
            METRICS["uploads"] += 1
        log.info("indexed %r: %d pages, %d chunks in %.1fs", name, doc["pages"],
                 len(doc["chunks"]), time.perf_counter() - t0)

    return {"doc_id": doc["doc_id"], "name": doc["name"], "pages": doc["pages"],
            "chunks": len(doc["chunks"])}


@app.post("/ask")
def ask(req: AskRequest):
    if not STATE:
        raise HTTPException(status_code=503, detail="Pipeline is still starting")

    if req.doc_id:
        doc = DOCS.get(req.doc_id)
        if doc is None:
            raise HTTPException(status_code=404, detail="Document not found. Please upload it again.")
        chunks, bm25, collection = doc["chunks"], doc["bm25"], doc["collection"]
    else:
        chunks, bm25, collection = STATE["chunks"], STATE["bm25"], STATE["collection"]

    session_id = req.session_id or str(uuid.uuid4())
    history = SESSIONS.get(session_id, [])
    question = req.question.strip()
    t0 = time.perf_counter()

    try:
        result = run_agent(question, chunks, bm25, collection, history=history[-MAX_TURNS:])
    except RuntimeError as e:
        with LOCK:
            METRICS["requests"] += 1
            METRICS["errors"] += 1
        log.error("agent failed: %s", e)
        raise HTTPException(status_code=502, detail=f"The language model call failed: {str(e)[:300]}")
    latency_ms = round((time.perf_counter() - t0) * 1000, 1)
    stats = result.get("stats", {})

    if result["type"] == "clarify":
        reply, pages = result["question"], []
    else:
        reply, pages = result["answer"], result.get("pages", [])

    # remember this exchange (the agent's clarifying question counts too)
    saved = reply + (f" (pages {', '.join(map(str, pages))})" if pages else "")
    history.append({"user": question, "assistant": saved})
    SESSIONS[session_id] = history[-MAX_TURNS:]
    while len(SESSIONS) > MAX_SESSIONS:
        SESSIONS.pop(next(iter(SESSIONS)))

    with LOCK:
        METRICS["requests"] += 1
        METRICS["total_latency_ms"] += latency_ms
        METRICS["llm_calls"] += stats.get("llm_calls", 0)
        METRICS["tokens"] += stats.get("tokens", 0)
        METRICS["empty_replies"] += stats.get("empty_replies", 0)
        if result["type"] == "clarify":
            METRICS["clarifications"] += 1
        if result["type"] == "gave_up":
            METRICS["gave_up"] += 1

    log.info("session=%s doc=%s type=%s steps=%d latency_ms=%s tokens=%s",
             session_id[:8], req.doc_id or "tesla", result["type"], len(result["steps"]),
             latency_ms, stats.get("tokens"))

    return {
        "session_id": session_id,
        "type": result["type"],          # answer | clarify | gave_up
        "answer": reply,
        "pages": pages,
        "steps": [s["action"] for s in result["steps"]],
        "latency_ms": latency_ms,
        "llm_calls": stats.get("llm_calls", 0),
        "tokens": stats.get("tokens", 0),
    }