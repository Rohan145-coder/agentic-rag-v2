import os
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Document Agent", page_icon="📊")
st.title("Ask your document")

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.session_id = None
    st.session_state.doc = None          # None = the built-in Tesla 10-K


def reset_chat():
    st.session_state.messages = []
    st.session_state.session_id = None


with st.sidebar:
    st.subheader("Document")
    uploaded = st.file_uploader("Upload a PDF", type=["pdf"])
    if uploaded is not None:
        key = f"{uploaded.name}:{uploaded.size}"
        if st.session_state.get("upload_key") != key:
            st.session_state.upload_key = key        # remember it, so reruns do not upload again
            with st.spinner("Reading and indexing the PDF..."):
                try:
                    r = requests.post(
                        f"{API_URL}/upload",
                        files={"file": (uploaded.name, uploaded.getvalue(), "application/pdf")},
                        timeout=600,
                    )
                    if r.status_code == 200:
                        st.session_state.doc = r.json()
                        reset_chat()
                    else:
                        st.error(r.json().get("detail", r.text))
                except Exception as e:
                    st.error(f"Upload failed: {e}")

    doc = st.session_state.doc
    if doc:
        st.success(f"Using: {doc['name']} ({doc['pages']} pages)")
        if st.button("Use the Tesla 10-K instead"):
            st.session_state.doc = None
            reset_chat()
            st.rerun()
    else:
        st.info("Using the built-in Tesla 10-K")

    if st.button("New conversation"):
        reset_chat()
        st.rerun()

    st.subheader("Live metrics")
    try:
        m = requests.get(f"{API_URL}/metrics", timeout=5).json()
        st.metric("Requests", m["requests"])
        st.metric("Avg latency (ms)", m["avg_latency_ms"])
        st.metric("Avg tokens / request", m["avg_tokens_per_request"])
    except Exception:
        st.caption("API not reachable")

doc = st.session_state.doc
st.caption(f"Agentic RAG over: {doc['name'] if doc else 'Tesla 10-K (built in)'}. "
           "Answers cite PDF pages, and follow-up questions work.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("footer"):
            st.caption(msg["footer"])

question = st.chat_input("Ask about the document...")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        footer = ""
        with st.spinner("Searching the document..."):
            try:
                resp = requests.post(
                    f"{API_URL}/ask",
                    json={"question": question,
                          "session_id": st.session_state.session_id,
                          "doc_id": doc["doc_id"] if doc else None},
                    timeout=300,
                )
                if resp.status_code != 200:
                    raise RuntimeError(resp.json().get("detail", resp.text))
                data = resp.json()
                st.session_state.session_id = data["session_id"]
                text = data["answer"]
                if data["pages"]:
                    text += f"\n\n*Source: PDF page(s) {', '.join(map(str, data['pages']))}*"
                if data["type"] == "clarify":
                    text = "I need one detail: " + text
                footer = (f"steps: {' → '.join(data['steps'])} · "
                          f"{data['latency_ms'] / 1000:.1f}s · {data['tokens']} tokens")
            except Exception as e:
                text = f"Error: {e}"
        st.markdown(text)
        if footer:
            st.caption(footer)
    st.session_state.messages.append({"role": "assistant", "content": text, "footer": footer})