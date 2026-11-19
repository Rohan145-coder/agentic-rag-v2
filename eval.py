import re
import sys
import json
import time
import os

from answer import setup
from agent import run_agent


# kind "answer": the document has it.
# expected = correct number.
# must_find = phrase that must appear in the chunks actually retrieved by the agent.
#
# kind "refuse": agent must say it cannot find it.
#
# split "dev" = cases you may tune against.
# split "held_out" = never tune against.

CASES = [

    {
        "q": "What was Tesla's effective tax rate in 2025?",
        "kind": "answer",
        "expected": 27,
        "must_find": "effective tax rate increased",
        "split": "dev"
    },

    {
        "q": "What was Tesla's effective tax rate in 2024?",
        "kind": "answer",
        "expected": 20,
        "must_find": "effective tax rate increased",
        "split": "dev"
    },

    {
        "q": "By how many percentage points did Tesla's effective tax rate change from 2024 to 2025?",
        "kind": "answer",
        "expected": 7,
        "must_find": "effective tax rate increased",
        "split": "dev"
    },

    {
        "q": "What was Tesla's provision for income taxes in 2025, in millions?",
        "kind": "answer",
        "expected": 1423,
        "must_find": "provision for (benefit from) income taxes",
        "split": "dev"
    },

    {
    "q": "What was Tesla's provision for income taxes in 2024, in millions?",
    "kind": "answer",
    "expected": 1837,
    "must_find": "total provision for (benefit from) income taxes",
    "split": "held_out"
    },

    {
        "q": "What was Tesla's total cash paid for income taxes, net of refunds, in 2025, in millions?",
        "kind": "answer",
        "expected": 1232,
        "must_find": "total cash paid for income taxes",
        "split": "held_out"
    },

    {
        "q": "What is Tesla's CEO's favorite color?",
        "kind": "refuse",
        "split": "dev"
    },

    {
        "q": "What was Tesla's stock price on January 1, 2030?",
        "kind": "refuse",
        "split": "dev"
    },

    {
        "q": "What is the phone number of Tesla's CFO?",
        "kind": "refuse",
        "split": "held_out"
    }

]


def normalize(text):
    return re.sub(r"\s+", " ", text.lower())


def number_matches(answer, expected):

    numbers = [
        float(x)
        for x in re.findall(
            r"\d+\.?\d*",
            answer.replace(",", "")
        )
    ]

    tolerance = max(
        abs(expected) * 0.01,
        0.05
    )

    return any(
        abs(n - expected) <= tolerance
        for n in numbers
    )


def main():

    split = "all"

    if "--split" in sys.argv:
        split = sys.argv[
            sys.argv.index("--split") + 1
        ]

    cases = [
        c for c in CASES
        if split == "all" or c["split"] == split
    ]

    threshold = float(
        os.getenv("EVAL_MIN", "0.8")
    )

    chunks, bm25, collection = setup()

    rows = []

    for case in cases:

        q = case["q"]

        row = {
            "question": q,
            "split": case["split"],
            "kind": case["kind"]
        }

        # Run the agent once.
        # The returned chunks are the exact evidence it saw.

        t0 = time.perf_counter()

        try:
            result = run_agent(
                q,
                chunks,
                bm25,
                collection
            )

        except RuntimeError as e:

            row.update(
                passed=False,
                error=str(e)
            )

            rows.append(row)

            print(
                "ERROR |",
                q[:60],
                "|",
                e
            )

            continue

        stats = result.get("stats", {})

        retrieved = result.get(
            "retrieved_chunks",
            []
        )

        # Check the actual chunks supplied to the agent,
        # not a separate top-8 search.

        if "must_find" in case:

            phrase = case["must_find"]

            row["retrieval_ok"] = any(
                phrase in normalize(c.get("text", ""))
                for c in retrieved
            )

            row["retrieved_chunk_ids"] = [
                c.get("id")
                for c in retrieved
            ]

            row["retrieved_pages"] = sorted({

                page

                for c in retrieved

                for page in range(
                    c.get("page", 1),
                    c.get(
                        "end_page",
                        c.get("page", 1)
                    ) + 1
                )

            })

        # ANSWER EVALUATION

        if result["type"] != "answer":

            ok = False

        elif case["kind"] == "refuse":

            ok = (
                "cannot find"
                in result["answer"].lower()
            )

        else:

            ok = number_matches(
                result["answer"],
                case["expected"]
            )

        row.update(

            passed=ok,

            latency_s=round(
                time.perf_counter() - t0,
                1
            ),

            steps=len(result["steps"]),

            tokens=stats.get("tokens", 0),

            empty_replies=stats.get(
                "empty_replies",
                0
            ),

            pages=result.get("pages", []),

            got=result.get(
                "answer",
                result.get("question", "")
            )

        )

        rows.append(row)

        flag = "PASS" if ok else "FAIL"

        retr = ""

        if "retrieval_ok" in row:

            retr = (
                " | retrieval "
                + (
                    "OK"
                    if row["retrieval_ok"]
                    else "MISS"
                )
            )

        print(
            f"{flag} [{case['split']}] "
            f"{q[:55]}{retr} | "
            f"steps {row['steps']} | "
            f"{row['latency_s']}s | "
            f"pages {row['pages']}"
        )

        if not ok:

            print(
                "     got:",
                row["got"]
            )

            print(
                "     actions:",
                [
                    s["action"]
                    for s in result["steps"]
                ]
            )

        # Be gentle with free-tier rate limits.
        time.sleep(10)

    # SUMMARY

    done = [
        r for r in rows
        if "passed" in r
    ]

    passed = sum(
        r["passed"]
        for r in done
    )

    retr_rows = [
        r for r in rows
        if "retrieval_ok" in r
    ]

    retr_ok = sum(
        r["retrieval_ok"]
        for r in retr_rows
    )

    accuracy = (
        passed / len(done)
        if done
        else 0.0
    )

    print(
        f"\nRetrieval: {retr_ok}/{len(retr_rows)}"
    )

    print(
        f"Answers:   {passed}/{len(done)}"
    )

    for s in ("dev", "held_out"):

        part = [
            r for r in done
            if r["split"] == s
        ]

        if part:

            print(
                f"  {s}: "
                f"{sum(r['passed'] for r in part)}/{len(part)}"
            )

    timed = [
        r for r in rows
        if "latency_s" in r
    ]

    if timed:

        print(
            f"Avg latency: "
            f"{sum(r['latency_s'] for r in timed) / len(timed):.1f}s | "
            f"tokens: {sum(r['tokens'] for r in timed)} | "
            f"empty replies: {sum(r['empty_replies'] for r in timed)}"
        )

    # SAVE EVALUATION REPORT

    with open(
        "eval_report.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            {
                "retrieval": f"{retr_ok}/{len(retr_rows)}",
                "answers": f"{passed}/{len(done)}",
                "cases": rows
            },
            f,
            indent=2
        )

    sys.exit(
        0 if accuracy >= threshold else 1
    )


if __name__ == "__main__":
    main()