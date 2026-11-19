import hashlib  # NEW: used to make a "fingerprint" of the chunks
import chromadb
from sentence_transformers import SentenceTransformer
from ingest import load_pages, join_pages,chunk_with_page

embed_model = SentenceTransformer("all-MiniLM-L6-v2")
def chunks_fingerprint(chunks):
    h = hashlib.sha256()
    for c in chunks:
        h.update(f'{c["id"]}|{c["page"]}|'.encode("utf-8"))
        h.update(c["text"].encode("utf-8"))
    return h.hexdigest()
def build_index(chunks, db_path="chroma_db"):
    client = chromadb.PersistentClient(path=db_path)

    # CHANGED: no more delete_collection. get_or_create_collection reuses
    # the collection if it exists, and creates it on the very first run.
    collection = client.get_or_create_collection("tesla")

    # NEW: fingerprint of the chunks we are about to store
    new_fingerprint = chunks_fingerprint(chunks)

    # NEW: look at ONE stored chunk to see which fingerprint it was saved with
    stored_fingerprint = None
    if collection.count() > 0:
        one = collection.get(limit=1, include=["metadatas"])
        stored_fingerprint = one["metadatas"][0].get("fingerprint")

    # NEW: same fingerprint and same number of chunks -> nothing changed.
    # Skip the slow embedding step and reuse what is already stored.
    if stored_fingerprint == new_fingerprint and collection.count() == len(chunks):
        print("Index is up to date. Skipping embeddings.")
        return collection

    # NEW: something changed (or first run). Remove ALL old chunks first,
    # so no stale chunks from the old PDF are left behind.
    old_ids = collection.get(include=[])["ids"]
    if old_ids:
        print(f"PDF changed. Removing {len(old_ids)} old chunks and rebuilding...")
        collection.delete(ids=old_ids)

    texts = [c["text"] for c in chunks]
    embeddings = embed_model.encode(texts).tolist()
    collection.add(
        ids=[str(c["id"]) for c in chunks],
        documents=texts,
        embeddings=embeddings,
        # CHANGED: each chunk now also remembers the fingerprint it was saved with
        metadatas=[{"page": c["page"], "fingerprint": new_fingerprint} for c in chunks],
    )
    return collection


if __name__ == "__main__":
    pages = load_pages("data/tesla.pdf")
    full_text, page_starts = join_pages(pages)
    chunks = chunk_with_page(full_text, page_starts)
    collection = build_index(chunks)
    print("Stored:", collection.count())

    q = embed_model.encode("What was Tesla's effective tax rate?").tolist()
    res = collection.query(query_embeddings=[q], n_results=3)
    for doc, meta in zip(res["documents"][0], res["metadatas"][0]):
        print("page", meta["page"], "|", doc[:100].replace("\n", " "))