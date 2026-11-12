"""Step 3a: caption the figures and tables on "visual" pages.

Text extraction (step 1) cannot see what a plot shows. Here a vision model
writes that down as text, so ordinary text retrieval can find it later.

Captioning is slow on CPU, so this script:
  - only captions pages that look visual (embedded images or many vector drawings)
  - appends each result to data/captions.jsonl as soon as it is done
  - skips pages already captioned, so you can stop (Ctrl+C) and resume any time

Usage:
  python caption.py --dry-run      # show which pages would be captioned, call no model
  python caption.py --limit 3      # caption just 3 pages to check quality and speed
  python caption.py                # caption everything selected (resumable)
"""

import argparse
import json
import time
from collections import Counter
from pathlib import Path

from llm import ask

MANIFEST = Path("data/pages.jsonl")
CAPTIONS = Path("data/captions.jsonl")

# A plot drawn as vector graphics is made of many small drawing items;
# a plain text page has only a few (rules, underlines).
MIN_DRAWINGS = 30

PROMPT = (
    "List every figure and table on this page. For each one give its label "
    "(for example 'Figure 3'), what it shows, the axes or columns, the named "
    "series or groups, and the main trend or key values. "
    "Do not summarize the body text. "
    "If the page has no figure or table, reply with exactly: NONE"
)


def is_visual(page: dict, min_drawings: int) -> bool:
    return page["n_images"] > 0 or page["n_drawings"] >= min_drawings


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="show the selection, call no model")
    parser.add_argument("--limit", type=int, help="caption at most this many pages this run")
    parser.add_argument("--min-drawings", type=int, default=MIN_DRAWINGS)
    args = parser.parse_args()

    pages = load_jsonl(MANIFEST)
    if not pages:
        raise SystemExit(f"{MANIFEST} not found or empty - run ingest.py first.")

    selected = [p for p in pages if is_visual(p, args.min_drawings)]
    done = {(c["doc_id"], c["page"]) for c in load_jsonl(CAPTIONS)}
    todo = [p for p in selected if (p["doc_id"], p["page"]) not in done]

    per_doc_all = Counter(p["doc_id"] for p in pages)
    per_doc_sel = Counter(p["doc_id"] for p in selected)
    for doc_id, n in per_doc_all.items():
        print(f"{doc_id}: {per_doc_sel[doc_id]} of {n} pages look visual")
    print(f"\nSelected {len(selected)} of {len(pages)} pages; {len(done)} already captioned; {len(todo)} to do")

    if args.dry_run:
        for p in todo:
            print(f"  {p['doc_id']} p{p['page']}: images={p['n_images']} drawings={p['n_drawings']}")
        return

    if args.limit:
        todo = todo[: args.limit]

    with CAPTIONS.open("a", encoding="utf-8") as f:
        for i, page in enumerate(todo, start=1):
            start = time.time()
            caption = ask(PROMPT, [page["image_path"]])
            seconds = round(time.time() - start, 1)
            if caption.strip().upper() == "NONE":
                caption = ""
            record = {
                "doc_id": page["doc_id"],
                "page": page["page"],
                "caption": caption,
                "seconds": seconds,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()  # written to disk now, so a Ctrl+C never loses finished work
            print(f"[{i}/{len(todo)}] {page['doc_id']} p{page['page']}: {seconds}s, {len(caption)} chars")


if __name__ == "__main__":
    main()
