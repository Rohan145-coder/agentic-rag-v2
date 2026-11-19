import io
import hashlib
import chromadb
from ingest import load_pages, join_pages, chunk_with_page
from index import embed_model
from search import build_bm25

MAX_CHUNKS = 3000            # protects the server from huge documents
_client = chromadb.EphemeralClient()   # in-memory vector DB for uploaded documents


def doc_id_for(file_bytes):
    """Same file -> same id, so uploading it twice does not rebuild anything."""
    return hashlib.sha256(file_bytes).hexdigest()[:16]


def build_pipeline_from_pdf(file_bytes, name="uploaded.pdf"):
    """Builds chunks + BM25 + a vector collection for ANY uploaded PDF.
    Raises ValueError with a clear message for bad input."""
    try:
        pages = load_pages(io.BytesIO(file_bytes))
    except Exception as e:
        raise ValueError(f"Could not read '{name}'. It may be corrupted or password-protected.") from e
    if not pages:
        raise ValueError(f"No readable text in '{name}'. It may be a scanned (image-only) PDF.")

    full_text, page_starts = join_pages(pages)
    chunks = chunk_with_page(full_text, page_starts)
    if len(chunks) > MAX_CHUNKS:
        raise ValueError(f"'{name}' is too large ({len(chunks)} chunks, the limit is {MAX_CHUNKS}).")

    doc_id = doc_id_for(file_bytes)
    collection_name = "doc_" + doc_id
    try:
        _client.delete_collection(collection_name)
    except Exception:
        pass
    collection = _client.create_collection(collection_name)

    texts = [c["text"] for c in chunks]
    collection.add(
        ids=[str(c["id"]) for c in chunks],
        documents=texts,
        embeddings=embed_model.encode(texts).tolist(),
        metadatas=[{"page": c["page"]} for c in chunks],
    )
    return {"doc_id": doc_id, "name": name, "pages": len(pages), "chunks": chunks,
            "bm25": build_bm25(chunks), "collection": collection}