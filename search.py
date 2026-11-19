import re
from rank_bm25 import BM25Okapi
from ingest import load_pages, join_pages, chunk_with_page
def tokenize(text):
    return re.findall(r"[a-z0-9]+(?:[.,][0-9]+)*", text.lower())
def build_bm25(chunks): 
    return BM25Okapi([tokenize(c["text"]) for c in chunks])
def keyword_search(question,chunks,bm25,k=10):
    scores= bm25.get_scores(tokenize(question))
    top= sorted(range(len(chunks)), key=lambda i: scores[i] , reverse=True)[:k]
    return [chunks[i] for i in top]
if __name__ == "__main__":
    pages = load_pages("data/tesla.pdf")
    full_text, page_starts = join_pages(pages)
    chunks = chunk_with_page(full_text, page_starts)
    bm25 = build_bm25(chunks)
    results= keyword_search("What was Tesla's effective tax rate?" , chunks,bm25,k=3)
    for c in results:
        print("page", c["page"], "|", c["text"][:100].replace("\n", " "))
        

