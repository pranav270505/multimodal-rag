"""Step 6: the full pipeline - question in, answer and source page out.

  1. Retrieve the top pages with hybrid search (embeddings + keywords, text_index.py).
  2. Read the pages one at a time, best first. For each, show the model the
     page image AND its extracted text (the best mode from step 5).
  3. Stop at the first page the model can answer from, and return that
     answer together with the page it came from.

Why one page at a time instead of all pages in one prompt? A page image is
large; several of them plus their text would overflow a small local model's
context window. Reading in rank order also tells us exactly which page an
answer came from, so every answer can cite its source.

Two prompt variants, to test whether the model can be made to skip wrong pages:
  basic     answer, or say NOT FOUND
  evidence  must first quote the words or name the figure/table it relies on,
            then answer; a page it cannot quote from is NOT FOUND

Commands:
  python rag.py ask "your question"
  python rag.py eval basic       run all test questions (resumable) -> data/rag_answers_basic.jsonl
  python rag.py eval evidence                                       -> data/rag_answers_evidence.jsonl
"""

import json
import sys
import time
from pathlib import Path

from llm import ask
from text_index import search

MANIFEST = Path("data/pages.jsonl")
QUESTIONS = Path("data/questions.jsonl")
K = 5  # pages to retrieve; retrieval found the right page in the top 5 for 95% of test questions
METHOD = "hybrid"  # ranked the right page first for 65% of test questions, vs 50% for embeddings alone

PAGE_BLOCK = (
    "The page is given as an image and as extracted text. "
    "Use the image for plots and diagrams, and the text for exact numbers.\n\n"
    "Extracted text:\n{text}\n\nQuestion: {question}"
)

PROMPTS = {
    "basic": (
        "Answer the question using only the page provided. Reply in one short sentence. "
        "If the page does not contain the answer, reply exactly: NOT FOUND\n\n" + PAGE_BLOCK
    ),
    "evidence": (
        "Decide whether this page answers the question. Many pages are on a similar topic "
        "but do not contain the specific fact asked for; for those, reply exactly: NOT FOUND\n"
        "Do not guess and do not answer a different question.\n"
        "If the page does contain it, reply in exactly this format:\n"
        "EVIDENCE: <the exact words from the page, or the figure/table label, that contain the answer>\n"
        "ANSWER: <one short sentence>\n\n" + PAGE_BLOCK
    ),
}

# Used when the page image is left out (much faster on a CPU, weaker on figures).
TEXT_ONLY_PROMPT = (
    "Answer the question using only the page provided. Reply in one short sentence. "
    "If the page does not contain the answer, reply exactly: NOT FOUND\n\n"
    "Page content:\n{text}\n\nQuestion: {question}"
)


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


_pages = None


def get_page(doc_id: str, page: int) -> dict:
    global _pages
    if _pages is None:
        _pages = {(p["doc_id"], p["page"]): p for p in load_jsonl(MANIFEST)}
    return _pages[(doc_id, page)]


def answer_question(question: str, k: int = K, variant: str = "basic", use_image: bool = True) -> dict:
    """Run the whole pipeline for one question."""
    hits = search(question, k, method=METHOD)
    retrieved = [[h["doc_id"], h["page"]] for h in hits]
    for pages_read, hit in enumerate(hits, start=1):
        page = get_page(hit["doc_id"], hit["page"])
        if use_image:
            reply = ask(PROMPTS[variant].format(text=page["text"], question=question), [page["image_path"]])
        else:
            reply = ask(TEXT_ONLY_PROMPT.format(text=page["text"], question=question))
        if "NOT FOUND" not in reply.upper():
            evidence = ""
            if "ANSWER:" in reply:
                evidence, reply = reply.split("ANSWER:", 1)
                evidence = evidence.replace("EVIDENCE:", "").strip()
            return {"answer": reply.strip(), "evidence": evidence, "source": [hit["doc_id"], hit["page"]],
                    "pages_read": pages_read, "retrieved": retrieved}
    return {"answer": "NOT FOUND", "evidence": "", "source": None, "pages_read": len(hits), "retrieved": retrieved}


def run_eval(variant: str) -> None:
    RESULTS = Path(f"data/rag_answers_{variant}.jsonl")
    questions = load_jsonl(QUESTIONS)
    done = {r["question"] for r in load_jsonl(RESULTS)}
    todo = [q for q in questions if q["question"] not in done]
    print(f"[{variant}] {len(questions)} questions; {len(done)} done; {len(todo)} to do")

    with RESULTS.open("a", encoding="utf-8") as f:
        for i, q in enumerate(todo, start=1):
            start = time.time()
            result = answer_question(q["question"], variant=variant)
            gold = [q["doc_id"], q["page"]]
            record = {
                "question": q["question"],
                "type": q["type"],
                "expected": q["answer"],
                "gold_page": gold,
                **result,
                "gold_retrieved": gold in result["retrieved"],
                "source_is_gold": result["source"] == gold,
                "seconds": round(time.time() - start, 1),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            mark = "right page" if record["source_is_gold"] else "WRONG PAGE" if result["source"] else "no answer"
            print(f"[{i}/{len(todo)}] {mark:<11}{record['seconds']:>6}s  read {result['pages_read']}  {result['answer'][:60]}")

    rows = load_jsonl(RESULTS)
    n = len(rows)
    print(f"\nCorrect page retrieved (top {K}):   {sum(r['gold_retrieved'] for r in rows)}/{n}")
    print(f"Answered from the correct page:   {sum(r['source_is_gold'] for r in rows)}/{n}")
    print(f"Answered from a wrong page:       {sum(bool(r['source']) and not r['source_is_gold'] for r in rows)}/{n}")
    print(f"No answer (NOT FOUND everywhere): {sum(r['source'] is None for r in rows)}/{n}")
    print("Answer correctness still needs grading against 'expected'.")


def main() -> None:
    if len(sys.argv) == 3 and sys.argv[1] == "ask":
        result = answer_question(sys.argv[2])
        print(f"\nAnswer: {result['answer']}")
        if result["source"]:
            doc_id, page = result["source"]
            print(f"Source: {doc_id}, page {page}  (read {result['pages_read']} of {len(result['retrieved'])} pages)")
    elif len(sys.argv) == 3 and sys.argv[1] == "eval" and sys.argv[2] in PROMPTS:
        run_eval(sys.argv[2])
    else:
        sys.exit('Usage: python rag.py ask "question"   |   python rag.py eval [basic|evidence]')


if __name__ == "__main__":
    main()
