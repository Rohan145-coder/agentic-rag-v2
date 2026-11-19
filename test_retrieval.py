
from answer import setup
from retrieve import hybrid_search

chunks, bm25, collection = setup()

queries = [
    "Tesla effective tax rate 2024",
    "Tesla effective tax rate 2024 2025",
    "Tesla effective tax rate comparison 2024 2025",
    "Tesla effective tax rate increased from 20% to 27%",
    "Tesla provision for income taxes effective tax rate 2024 2025"
]

for query in queries:

    print("\n" + "=" * 80)
    print("QUERY:", query)
    print("=" * 80)

    results = hybrid_search(
        query,
        chunks,
        bm25,
        collection,
        top_k=5
    )

    for rank, c in enumerate(results, start=1):

        print(
            f"\nRank: {rank} | "
            f"Chunk ID: {c['id']} | "
            f"PDF Page: {c['page']}"
        )

        print(c["text"][:500])
        print("-" * 60)
