"""Step 5: answer the test questions four ways and compare.

To measure answering on its own, every question is answered from its
CORRECT page (taken from questions.jsonl), so retrieval mistakes cannot
blur the result. The only thing that changes is what the model is shown:

  text          the page's extracted text
  text+caption  extracted text plus the figure/table caption from caption.py
  image         the page image, nothing else
  image+text    the page image AND its extracted text (shapes from one, exact digits from the other)

Commands:
  python answer.py run       answer every question in every mode (resumable)
  python answer.py grade     show each answer next to the expected one; you mark y/n
  python answer.py report    accuracy by mode and question type

Output: data/answers.jsonl
"""

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

from llm import ask

MANIFEST = Path("data/pages.jsonl")
CAPTIONS = Path("data/captions.jsonl")
QUESTIONS = Path("data/questions.jsonl")
ANSWERS = Path("data/answers.jsonl")
MODES = ["text", "text+caption", "image", "image+text"]

INSTRUCTION = (
    "Answer the question using only the page provided. "
    "Reply in one short sentence. "
    "If the page does not contain the answer, reply exactly: NOT FOUND"
)


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def answer_one(question: str, page: dict, caption: str, mode: str) -> str:
    if mode == "image":
        return ask(f"{INSTRUCTION}\n\nQuestion: {question}", [page["image_path"]])
    if mode == "image+text":
        prompt = (
            f"{INSTRUCTION}\n\nThe page is given as an image and as extracted text. "
            "Use the image for plots and diagrams, and the text for exact numbers.\n\n"
            f"Extracted text:\n{page['text']}\n\nQuestion: {question}"
        )
        return ask(prompt, [page["image_path"]])
    context = page["text"]
    if mode == "text+caption" and caption:
        context += f"\n\nDescription of the figures and tables on this page:\n{caption}"
    return ask(f"{INSTRUCTION}\n\nPage content:\n{context}\n\nQuestion: {question}")


def run() -> None:
    pages = {(p["doc_id"], p["page"]): p for p in load_jsonl(MANIFEST)}
    captions = {(c["doc_id"], c["page"]): c["caption"] for c in load_jsonl(CAPTIONS)}
    questions = load_jsonl(QUESTIONS)
    done = {(a["question"], a["mode"]) for a in load_jsonl(ANSWERS)}

    todo = [(q, m) for q in questions for m in MODES if (q["question"], m) not in done]
    print(f"{len(questions)} questions x {len(MODES)} modes; {len(done)} done; {len(todo)} to do")

    with ANSWERS.open("a", encoding="utf-8") as f:
        for i, (q, mode) in enumerate(todo, start=1):
            key = (q["doc_id"], q["page"])
            if key not in pages:
                sys.exit(f"Page not in manifest: {key} - check doc_id/page in questions.jsonl")
            start = time.time()
            answer = answer_one(q["question"], pages[key], captions.get(key, ""), mode)
            record = {
                "question": q["question"],
                "type": q["type"],
                "doc_id": q["doc_id"],
                "page": q["page"],
                "mode": mode,
                "expected": q["answer"],
                "answer": answer,
                "seconds": round(time.time() - start, 1),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{i}/{len(todo)}] {mode:<13}{record['seconds']:>7}s  {answer[:70]}")


def grade() -> None:
    rows = load_jsonl(ANSWERS)
    todo = [r for r in rows if "correct" not in r]
    print(f"{len(todo)} answers to grade. y = correct, n = wrong, q = stop and save.\n")
    for i, row in enumerate(todo, start=1):
        print(f"({i}/{len(todo)}) [{row['type']} | {row['mode']}] {row['question']}")
        print(f"  expected: {row['expected']}")
        print(f"  answer:   {row['answer']}")
        choice = ""
        while choice not in ("y", "n", "q"):
            choice = input("  correct? [y/n/q] ").strip().lower()
        if choice == "q":
            break
        row["correct"] = choice == "y"
        print()
    save_jsonl(ANSWERS, rows)
    print("Saved.")


def report() -> None:
    rows = [r for r in load_jsonl(ANSWERS) if "correct" in r]
    if not rows:
        sys.exit("Nothing graded yet - run: python answer.py grade")
    table = defaultdict(list)  # (mode, type) -> list of True/False
    for r in rows:
        table[(r["mode"], r["type"])].append(r["correct"])
        table[(r["mode"], "ALL")].append(r["correct"])
    types = sorted({t for _, t in table}, key=lambda t: (t == "ALL", t))
    print(f"{'mode':<14}" + "".join(f"{t:>12}" for t in types))
    for mode in MODES:
        cells = []
        for t in types:
            marks = table.get((mode, t), [])
            cells.append(f"{sum(marks)}/{len(marks)}" if marks else "-")
        print(f"{mode:<14}" + "".join(f"{c:>12}" for c in cells))


if __name__ == "__main__":
    commands = {"run": run, "grade": grade, "report": report}
    if len(sys.argv) != 2 or sys.argv[1] not in commands:
        sys.exit("Usage: python answer.py [run|grade|report]")
    commands[sys.argv[1]]()
