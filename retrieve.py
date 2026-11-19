from sentence_transformers import CrossEncoder
from ingest import load_pages, join_pages, chunk_with_page
from index import embed_model, build_index
from search import build_bm25, keyword_search

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def vector_search(question, collection, chunks, k=10):
    q = embed_model.encode(question).tolist()
    res = collection.query(query_embeddings=[q], n_results=k)
    ids = [int(i) for i in res["ids"][0]]
    return [chunks[i] for i in ids]


def merge(keyword_results, vector_results):
    seen = set()
    merged = []
    for c in keyword_results + vector_results:
        if c["id"] not in seen:
            seen.add(c["id"])
            merged.append(c)
    return merged


def rerank(question, candidates, top_k=8):
    scores = reranker.predict([[question, c["text"]] for c in candidates])
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [c for c, _ in ranked[:top_k]]


def hybrid_search(question, chunks, bm25, collection, top_k=8, candidates=10):
    kw = keyword_search(question, chunks, bm25, k=candidates)
    vec = vector_search(question, collection, chunks, k=candidates)
    return rerank(question, merge(kw, vec), top_k)


def expand_with_neighbors(results, chunks, top_n=2):
    """Context expansion: for the best `top_n` hits, also include the PREVIOUS chunk,
    because a table's header usually sits just before its rows.
    Chunk id == position in `chunks`. Duplicates are removed."""
    out, seen = [], set()

    def add(c):
        if c["id"] not in seen:
            seen.add(c["id"])
            out.append(c)

    for rank, c in enumerate(results):
        if rank < top_n and c["id"] > 0:
            add(chunks[c["id"] - 1])
        add(c)
    return out


if __name__ == "__main__":
    pages = load_pages("data/tesla.pdf")
    full_text, page_starts = join_pages(pages)
    chunks = chunk_with_page(full_text, page_starts)
    collection = build_index(chunks)
    bm25 = build_bm25(chunks)
    for c in hybrid_search("What was Tesla's effective tax rate?", chunks, bm25, collection)[:5]:
        print("id", c["id"], "| page", c["page"])