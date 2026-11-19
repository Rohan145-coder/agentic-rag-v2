from answer import setup

chunks, bm25, collection = setup()
for c in chunks:
    if c["page"] == 101 and "27.0" in c["text"]:
        print("chunk id", c["id"], "| page", c["page"])
        print(c["text"])
        print("-" * 60)