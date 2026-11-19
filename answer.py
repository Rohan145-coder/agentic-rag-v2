import os
import json
import time
import logging
from dotenv import load_dotenv
from groq import Groq, BadRequestError
from ingest import load_pages, join_pages, chunk_with_page
from index import build_index
from search import build_bm25
from retrieve import hybrid_search
import re

load_dotenv()
log = logging.getLogger("rag")

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise RuntimeError("Missing GROQ_API_KEY. Put it in your .env file.")
groq_client = Groq(api_key=api_key,max_retries=0)

MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")
# Optional experiment: set REASONING_EFFORT=low in .env to try faster replies
EXTRA = {"reasoning_effort": os.getenv("REASONING_EFFORT")} if os.getenv("REASONING_EFFORT") else {}

SYSTEM_PROMPT = """You are a financial document assistant. Answer using ONLY the sources provided.

Rules:
1. If the sources do not contain the answer, set "answer" to "I cannot find this information in the provided documents" and "sources" to [].
2. Never guess or use outside knowledge.
3. Financial tables often have similar row labels and several year columns. In "reasoning", list the matching rows and say which column belongs to which year before answering.
4. In "sources", list the numbers of the sources you actually used.

Reply with ONLY valid JSON, nothing else, in exactly this shape:
{"reasoning": "...", "answer": "...", "sources": [1, 2]}"""


def failed_generation(error):
    """When Groq rejects a reply with a 400 error, the rejected text is inside the error."""
    body = getattr(error, "body", None)
    if isinstance(body, dict):
        inner = body.get("error", body)
        if isinstance(inner, dict):
            return inner.get("failed_generation")
    return None


def retry_delay(message):
    """Reads Groq's 'Please try again in 7.5s' hint. Returns seconds to wait (default 10, max 65)."""
    m = re.search(r"try again in\s+(?:(\d+)m)?\s*(\d+(?:\.\d+)?)(ms|s)", message)
    if not m:
        return 10.0
    seconds = int(m.group(1) or 0) * 60 + float(m.group(2)) / (1000 if m.group(3) == "ms" else 1)
    return min(seconds + 1, 65.0)


def call_llm(messages, retries=4, stats=None):
    """Every LLM call goes through here: retries with visible waits, token counting,
    and recovery of tool-style replies. `stats` is a dict the caller owns."""
    for attempt in range(retries):
        try:
            t0 = time.perf_counter()
            response = groq_client.chat.completions.create(
                model=MODEL, temperature=0, messages=messages, **EXTRA
            )
            log.info("llm call took %.1fs", time.perf_counter() - t0)
            if stats is not None:
                stats["llm_calls"] = stats.get("llm_calls", 0) + 1
                usage = getattr(response, "usage", None)
                if usage is not None:
                    stats["tokens"] = stats.get("tokens", 0) + (getattr(usage, "total_tokens", 0) or 0)
            choice = response.choices[0]
            if not choice.message.content:
                log.warning("empty reply, finish_reason=%s", choice.finish_reason)
                if stats is not None:
                    stats["empty_replies"] = stats.get("empty_replies", 0) + 1
            return choice.message.content
        except BadRequestError as e:
            recovered = failed_generation(e)
            if recovered:
                log.info("recovered a rejected tool-style reply")
                return recovered
            raise RuntimeError(f"Bad request (not retrying): {e}") from e
        except Exception as e:
            msg = str(e)
            status = getattr(e, "status_code", None)
            # Too-large requests and daily limits cannot be fixed by retrying
            if status == 413 or "tokens per day" in msg:
                raise RuntimeError(f"Token limit reached (not retrying): {msg[:200]}") from e
            if attempt == retries - 1:
                raise RuntimeError(f"LLM call failed after {retries} attempts: {e}") from e
            wait = retry_delay(msg) if status == 429 else 3 * (2 ** attempt)
            log.warning("LLM call failed (status %s). Waiting %.1fs before retry %d", status, wait, attempt + 1)
            time.sleep(wait)

def build_context(top_chunks):
    parts = []
    for i, c in enumerate(top_chunks, start=1):
        parts.append(f"[Source {i}] (page {c['page']}):\n{c['text']}")
    return "\n\n".join(parts)


def generate_answer(question, top_chunks):
    """Simple (non-agentic) answer path."""
    user_prompt = f"Sources:\n{build_context(top_chunks)}\n\nQuestion: {question}"
    raw = call_llm([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]) or ""
    raw = raw.replace("```json", "").replace("```", "").strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {"answer": raw, "pages": [], "invalid_sources": [], "warning": "model did not return valid JSON"}
    sources = data.get("sources", [])
    valid = [s for s in sources if isinstance(s, int) and 1 <= s <= len(top_chunks)]
    invalid = [s for s in sources if s not in valid]
    pages = sorted({top_chunks[s - 1]["page"] for s in valid})
    return {"answer": data.get("answer", ""), "pages": pages, "invalid_sources": invalid}


def setup():
    pages = load_pages("data/tesla.pdf")
    full_text, page_starts = join_pages(pages)
    chunks = chunk_with_page(full_text, page_starts)
    collection = build_index(chunks)
    bm25 = build_bm25(chunks)
    return chunks, bm25, collection