"""Step 6a: score retrieval against the question set.

For every question in data/questions.jsonl we know the one correct page.
We run a search and record the rank at which that page comes back.

  Recall@1  share of questions whose correct page is the top result
  Recall@5  share of questions whose correct page is in the top 5
  MRR       mean of 1/rank (0 when the page is not found): rewards ranking it higher

Results are broken down by question type (figure / table / text), because
the interesting claim is about figure questions, not the average.

Usage:
  python eval_retrieval.py
"""

import json
from collections import defaultdict
from pathlib import Path

from text_index import search

QUESTIONS = Path("data/questions.jsonl")
K = 5


def rank_of(hits: list[dict], doc_id: str, page: int) -> int | None:
    for rank, hit in enumerate(hits, start=1):
        if hit["doc_id"] == doc_id and hit["page"] == page:
            return rank
    return None


def evaluate(name: str, search_fn, questions: list[dict]) -> None:
    ranks = defaultdict(list)
    misses = []
    for q in questions:
        rank = rank_of(search_fn(q["question"]), q["doc_id"], q["page"])
        ranks[q["type"]].append(rank)
        ranks["ALL"].append(rank)
        if rank is None:
            misses.append(q)

    print(f"\n=== {name} ===")
    print(f"{'type':<8}{'n':>4}{'R@1':>8}{'R@5':>8}{'MRR':>8}")
    for qtype in sorted(ranks, key=lambda t: (t == "ALL", t)):
        r = ranks[qtype]
        n = len(r)
        r1 = sum(x == 1 for x in r) / n
        r5 = sum(x is not None for x in r) / n
        mrr = sum(1 / x for x in r if x) / n
        print(f"{qtype:<8}{n:>4}{r1:>8.2f}{r5:>8.2f}{mrr:>8.2f}")
    for q in misses:
        print(f"  missed: [{q['type']}] {q['doc_id']} p{q['page']}: {q['question']}")


def main() -> None:
    with QUESTIONS.open(encoding="utf-8") as f:
        questions = [json.loads(line) for line in f if line.strip()]
    print(f"{len(questions)} questions, top-{K} pages per search")

    evaluate("Dense, text only", lambda q: search(q, K, use_captions=False), questions)
    evaluate("Dense (embeddings)", lambda q: search(q, K, method="dense"), questions)
    evaluate("BM25 (keywords)", lambda q: search(q, K, method="bm25"), questions)
    evaluate("Hybrid (dense + BM25)", lambda q: search(q, K, method="hybrid"), questions)


if __name__ == "__main__":
    main()
