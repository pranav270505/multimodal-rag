"""Step 3b: Pipeline A - text retrieval over page text + figure captions.

Build:
  1. Cut every page's extracted text into overlapping chunks.
  2. Cut every figure/table caption (from caption.py) into chunks too.
  3. Embed all chunks and save the vectors.

Search:
  Embed the question, score every chunk by cosine similarity, and rank
  PAGES by their best-scoring chunk. Pages are the unit we return because
  the evaluation asks "did we find the right page?" for both pipelines.

Each chunk remembers whether it came from "text" or "caption", so we can
switch captions off and measure exactly how much they help.

Output:
  data/index/chunks.jsonl    one record per chunk (doc_id, page, source, text)
  data/index/vectors.npy     one row per chunk, unit length

Usage:
  python text_index.py build
  python text_index.py search "what does the stellar mass histogram show?"
  python text_index.py search "..." --no-captions
  python text_index.py search "..." --method hybrid     (dense | bm25 | hybrid)
"""

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np

from llm import embed

MANIFEST = Path("data/pages.jsonl")
CAPTIONS = Path("data/captions.jsonl")
INDEX_DIR = Path("data/index")
CHUNKS_FILE = INDEX_DIR / "chunks.jsonl"
VECTORS_FILE = INDEX_DIR / "vectors.npy"

CHUNK_CHARS = 1000  # small enough that the embedding model never truncates
OVERLAP = 200       # so a sentence cut at a boundary still appears whole somewhere
BATCH = 32

# nomic-embed-text was trained with these prefixes; leaving them out hurts retrieval.
DOC_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def split(text: str) -> list[str]:
    """Cut text into windows of CHUNK_CHARS that overlap by OVERLAP."""
    text = " ".join(text.split())  # collapse newlines and repeated spaces
    step = CHUNK_CHARS - OVERLAP
    return [text[i : i + CHUNK_CHARS] for i in range(0, len(text), step) if text[i : i + CHUNK_CHARS].strip()]


def normalize(vectors: np.ndarray) -> np.ndarray:
    """Scale each row to length 1, so a dot product equals cosine similarity."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.clip(norms, 1e-12, None)


def build() -> None:
    pages = load_jsonl(MANIFEST)
    if not pages:
        raise SystemExit(f"{MANIFEST} not found or empty - run ingest.py first.")
    captions = load_jsonl(CAPTIONS)

    chunks = []
    for page in pages:
        for piece in split(page["text"]):
            chunks.append({"doc_id": page["doc_id"], "page": page["page"], "source": "text", "text": piece})
    for cap in captions:
        for piece in split(cap["caption"]):
            chunks.append({"doc_id": cap["doc_id"], "page": cap["page"], "source": "caption", "text": piece})

    n_caption = sum(c["source"] == "caption" for c in chunks)
    print(f"{len(chunks)} chunks ({len(chunks) - n_caption} text, {n_caption} caption). Embedding...")

    vectors = []
    for start in range(0, len(chunks), BATCH):
        batch = chunks[start : start + BATCH]
        vectors.extend(embed([DOC_PREFIX + c["text"] for c in batch]))
        print(f"  {min(start + BATCH, len(chunks))}/{len(chunks)}", end="\r")

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    np.save(VECTORS_FILE, normalize(np.array(vectors, dtype=np.float32)))
    with CHUNKS_FILE.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    print(f"\nSaved index to {INDEX_DIR}/")


_cache = None


def _load_index():
    global _cache
    if _cache is None:
        if not VECTORS_FILE.exists():
            raise SystemExit("No index found - run: python text_index.py build")
        _cache = (load_jsonl(CHUNKS_FILE), np.load(VECTORS_FILE))
    return _cache


_bm25 = None


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _bm25_scores(query: str) -> np.ndarray:
    """Keyword scores for every chunk (BM25): rewards rare words that the
    question and the chunk share, such as names, acronyms and numbers."""
    global _bm25
    chunks, _ = _load_index()
    if _bm25 is None:
        docs = [_tokens(c["text"]) for c in chunks]
        counts = [Counter(d) for d in docs]
        doc_freq = Counter(word for d in docs for word in set(d))
        lengths = np.array([len(d) for d in docs], dtype=np.float32)
        _bm25 = (counts, doc_freq, lengths, lengths.mean())
    counts, doc_freq, lengths, avg_len = _bm25
    n = len(counts)
    k1, b = 1.5, 0.75
    scores = np.zeros(n, dtype=np.float32)
    for word in set(_tokens(query)):
        if word not in doc_freq:
            continue
        idf = math.log(1 + (n - doc_freq[word] + 0.5) / (doc_freq[word] + 0.5))
        tf = np.array([c.get(word, 0) for c in counts], dtype=np.float32)
        scores += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * lengths / avg_len))
    return scores


def _rank_pages(chunks: list[dict], scores, use_captions: bool) -> list[dict]:
    """Collapse chunk scores to pages (best chunk wins) and sort best first."""
    best = {}
    for chunk, score in zip(chunks, scores):
        if not use_captions and chunk["source"] == "caption":
            continue
        key = (chunk["doc_id"], chunk["page"])
        if key not in best or score > best[key]["score"]:
            best[key] = {**chunk, "score": float(score)}
    return sorted(best.values(), key=lambda r: r["score"], reverse=True)


def search(query: str, k: int = 5, use_captions: bool = True, method: str = "dense") -> list[dict]:
    """Return the top-k pages for a question, best first.

    method:
      dense   embedding similarity - matches meaning, even with different words
      bm25    keyword overlap      - matches exact terms, names and numbers
      hybrid  both, merged with reciprocal rank fusion (RRF)
    """
    chunks, vectors = _load_index()
    rankings = []
    if method in ("dense", "hybrid"):
        q = normalize(np.array(embed([QUERY_PREFIX + query]), dtype=np.float32))[0]
        rankings.append(_rank_pages(chunks, vectors @ q, use_captions))
    if method in ("bm25", "hybrid"):
        rankings.append(_rank_pages(chunks, _bm25_scores(query), use_captions))
    if not rankings:
        raise ValueError(f"unknown method: {method}")
    if len(rankings) == 1:
        return rankings[0][:k]

    # RRF: a page earns 1/(60 + rank) from each ranking. Only ranks are used,
    # so the two score scales never have to be compared directly.
    fused = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking, start=1):
            key = (hit["doc_id"], hit["page"])
            entry = fused.setdefault(key, {**hit, "score": 0.0})
            entry["score"] += 1 / (60 + rank)
    return sorted(fused.values(), key=lambda r: r["score"], reverse=True)[:k]


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("build")
    s = sub.add_parser("search")
    s.add_argument("query")
    s.add_argument("-k", type=int, default=5)
    s.add_argument("--no-captions", action="store_true", help="ignore caption chunks")
    s.add_argument("--method", choices=["dense", "bm25", "hybrid"], default="dense")
    args = parser.parse_args()

    if args.command == "build":
        build()
        return

    for rank, hit in enumerate(search(args.query, args.k, not args.no_captions, args.method), start=1):
        print(f"{rank}. {hit['doc_id']} p{hit['page']}  score={hit['score']:.3f}  via {hit['source']}")
        print(f"   {hit['text'][:160]}...")


if __name__ == "__main__":
    main()
