from answer import setup
from agent import run_agent

MAX_TURNS = 6   # remember only the last 6 exchanges, to save tokens


def main():
    chunks, bm25, collection = setup()
    history = []
    print("\nAsk about the Tesla 10-K. Type 'quit' to stop.\n")

    while True:
        question = input("You: ").strip()
        if not question:
            continue
        if question.lower() in ("quit", "exit"):
            break

        try:
            result = run_agent(question, chunks, bm25, collection,
                               history=history[-MAX_TURNS:])
        except RuntimeError as e:
            print("Error:", e)
            continue

        if result["type"] == "clarify":
            reply = result["question"]
            print("Agent asks:", reply)
        else:
            reply = result["answer"]
            if result["pages"]:
                reply += f" (pages {', '.join(str(p) for p in result['pages'])})"
            print("Agent:", reply)

        # the notepad: save this exchange so the next question can use it
        history.append({"user": question, "assistant": reply})


if __name__ == "__main__":
    main()