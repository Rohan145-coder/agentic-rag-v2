import json
import re
import logging
from answer import call_llm, setup
from retrieve import hybrid_search, expand_with_neighbors

log = logging.getLogger("rag")

MAX_STEPS = 6
MAX_SEARCHES = 3   # after this many searches the agent must give its final answer
VALID_ACTIONS = {"search_document", "calculator", "ask_user", "final_answer"}

AGENT_PROMPT = """You are a research agent working over a document the user provided (often a financial report such as a 10-K).

1. search_document: search the report. input = a short search query.
2. calculator: do arithmetic. input = a plain math expression with only numbers and + - * / ( ), for example (27 - 20) / 20 * 100
3. ask_user: ask the user ONE clarifying question when the question is genuinely ambiguous (several similar figures could be meant). input = your question.
4. final_answer: give the final answer. input = the answer text. Only use facts from search results.

Rules:
- You do NOT have function calling. Never make a function call or tool call. Reply with plain text that contains one JSON object.
- You MUST use search_document at least once before final_answer. Never answer or refuse from memory.
- Never do arithmetic in your head, not even simple subtraction. Always use the calculator action for any calculation.
- Never use outside knowledge. If searching does not find the answer, use final_answer with "I cannot find this information in the provided documents" and sources [].
- Search results are labeled [Source N]. In final_answer, list the numbers of the sources you used in "sources".
- In final_answer, put the source numbers ONLY in the "sources" field, never inside the answer text.
- If a search did not help, search again with different words.
- If the search results show several similarly named figures that could each answer the question (for example different "automotive revenue" lines), use ask_user to ask which one the user means, instead of picking one.
- If a "Conversation so far" section is given, use it to understand short follow-ups like "what about 2022?" or answers to your own earlier question. Your search queries must be complete on their own (include the company, the metric and the year).
- Financial tables have similar row labels and several year columns. Check the right row and the right year column before using a number.
- For questions comparing two or more years, your search query MUST include the metric and ALL years being compared.
- Prefer a sentence that explicitly states the comparison between those years.
- Do not give a final answer to a comparison question until you have found evidence for every year being compared.
- If a number comes from a table and the sources do not show which column belongs to which year, do NOT assume. Search again for a sentence in the text that states the figure (for example "effective tax rate increased from"), or say you cannot determine it with confidence.
- Prefer figures stated in a sentence (for example "increased from 20% to 27%") over bare numbers in a table. For a question about a specific year, if your number comes only from a table, run one more search for a sentence that states it before you answer.
- If two sources give different values for the same figure and the same year, search again to resolve it before you answer or ask the user.

Reply with ONLY one JSON object, nothing else, exactly in this shape:
{"thought": "short reasoning", "action": "search_document", "input": "...", "sources": [1, 2]}
The "action" must be one of: search_document, calculator, ask_user, final_answer. "sources" is only needed for final_answer."""


def calculator(expression):
    # Only allow digits and math symbols, so nothing dangerous can run
    if not re.fullmatch(r"[0-9+\-*/(). ]+", expression):
        return "Error: use only numbers and + - * / ( )"
    try:
        return str(round(eval(expression, {"__builtins__": {}}, {}), 4))
    except Exception as e:
        return f"Error: {e}"


def parse_decision(raw):
    raw = raw.replace("```json", "").replace("```", "").strip()
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end == -1:
        return None
    try:
        data = json.loads(raw[start:end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None

    # Function-call shape: {"name": "...", "arguments": {...}}
    args = data.get("arguments")
    if "action" not in data and args is not None:
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = None
        if isinstance(args, dict):
            merged = dict(args)
            merged.setdefault("action", data.get("name"))
            if "input" not in merged:   # the model sometimes names the field "query" etc.
                for k in ("query", "expression", "question", "text", "answer"):
                    if k in merged:
                        merged["input"] = merged[k]
                        break
            return merged
    return data


def format_history(history):
    lines = []
    for turn in history:
        lines.append(f"User: {turn['user']}")
        lines.append(f"Assistant: {turn['assistant']}")
    return "\n".join(lines)
def trim_old_observations(messages, keep=2):
    """Keep only the last `keep` observations in full; shorten older ones.
    Without this the conversation grows with every search and hits the token-per-minute limit."""
    idx = [i for i, m in enumerate(messages)
           if m["role"] == "user" and m["content"].startswith("Observation:")]
    for i in idx[:-keep]:
        messages[i] = {"role": "user", "content": "Observation: (older results omitted to save tokens)"}

def run_agent(question, chunks, bm25, collection, history=None, max_steps=MAX_STEPS):
    first_message = question
    if history:
        first_message = ("Conversation so far:\n" + format_history(history)
                         + "\n\nNew user message: " + question)

    messages = [
        {"role": "system", "content": AGENT_PROMPT},
        {"role": "user", "content": first_message},
    ]
    sources = []    # every chunk the agent has seen; position + 1 = its Source number
    steps = []      # log of what the agent did
    stats = {"llm_calls": 0, "tokens": 0, "empty_replies": 0}   # monitoring numbers
    nudged = False  # the "fix your sources" reminder is sent only once
    searches_done = 0
    queries_seen = set()

    for step in range(1, max_steps + 1):
        # Ask for a valid decision; retry up to 3 times INSIDE the same step
        decision = None
        raw = ""
        retry_messages = messages
        for attempt in range(3):
            raw = call_llm(retry_messages, stats=stats) or ""
            decision = parse_decision(raw)
            if decision is not None and decision.get("action") in VALID_ACTIONS:
                break
            if raw.strip():
                log.warning("bad reply: %s", raw[:200])
            decision = None
            retry_messages = messages + [{
                "role": "user",
                "content": 'Your last reply was empty, not valid JSON, or had no valid "action". '
                           "Reply with plain text containing ONLY one JSON object in the required shape."}]

        messages.append({"role": "assistant", "content": raw})

        if decision is None:
            steps.append({"step": step, "action": "invalid_json", "input": repr(raw[:80])})
            continue

        action = decision.get("action")
        inp = str(decision.get("input", ""))
        steps.append({"step": step, "action": action, "input": inp})

        if action == "final_answer":
            # Code rule: no answer (not even a refusal) before at least one search
            if not sources:
                messages.append({"role": "user", "content":
                    "You have not searched the document yet. Use search_document first, "
                    "then give your final_answer."})
                continue

            cited = decision.get("sources", [])
            valid = [s for s in cited if isinstance(s, int) and 1 <= s <= len(sources)]
            invalid = [s for s in cited if s not in valid]
            is_refusal = "cannot find" in inp.lower()
            if not valid and not is_refusal and not nudged:
                nudged = True
                messages.append({"role": "user", "content":
                    'Your final_answer had no valid "sources". Put the Source numbers '
                    'in the JSON field "sources", not inside the answer text. '
                    'Reply again with final_answer.'})
                continue

            pages = sorted({p for s in valid
                            for p in range(sources[s - 1]["page"],
                                           sources[s - 1].get("end_page", sources[s - 1]["page"]) + 1)})
            return {"type": "answer", "answer": inp, "pages": pages,
                    "invalid_sources": invalid, "steps": steps, "stats": stats}
        if action == "ask_user":
            # Code rule: never ask the user before searching
            if not sources:
                messages.append({"role": "user", "content":
                    "You have not searched the document yet. Use search_document first. "
                    "There is exactly one document, so never ask which company or which document."})
                continue
            return {"type": "clarify", "question": inp, "steps": steps, "stats": stats}
        if action == "search_document" and searches_done >= MAX_SEARCHES:
            messages.append({"role": "user", "content":
                "Search limit reached. Do not search again. Use final_answer now: give the answer "
                "if you found it, otherwise say \"I cannot find this information in the provided documents\"."})
            continue
        if action == "search_document" and not inp.strip():
            messages.append({"role": "user", "content":
                'Your search query was empty. Put a short search query in the "input" field.'})
            continue
        if action == "search_document" and " ".join(inp.lower().split()) in queries_seen:     
            messages.append({"role": "user", "content":                                         
                "You already ran that exact search. Do not repeat it. "                         
                "Search with different words, or use final_answer now."})                       
            continue   
        if action == "search_document":
            searches_done += 1
            queries_seen.add(" ".join(inp.lower().split()))
            hits = hybrid_search(inp, chunks, bm25, collection, top_k=5)
            # Add the previous chunk for the best hits: a table header usually sits just before its rows
            results = expand_with_neighbors(hits, chunks, top_n=2)
            log.info("search %r -> chunks %s", inp, [c["id"] for c in results])
            lines = []
            for c in results:
                known_ids = [s["id"] for s in sources]
                if c["id"] in known_ids:
                    n = known_ids.index(c["id"]) + 1
                else:
                    sources.append(c)
                    n = len(sources)
                lines.append(f"[Source {n}] (page {c['page']}):\n{c['text']}")
            observation = "Search results:\n\n" + "\n\n".join(lines)
        elif action == "calculator":
            observation = "Calculator result: " + calculator(inp)
        else:
            observation = f"Unknown action '{action}'. Use one of the four actions."

        messages.append({"role": "user", "content": "Observation:\n" + observation})
        trim_old_observations(messages)

    return {"type": "gave_up", "answer": "I could not finish within the step limit.",
            "pages": [], "steps": steps, "stats": stats}