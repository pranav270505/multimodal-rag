"""Step 1: turn PDFs (papers, slide decks) into per-page records.

For every page we save:
  - a PNG render of the whole page   -> used later by page-image retrieval
  - the extracted text               -> used later by caption/text retrieval
  - a few stats (chars, images)      -> tells us which pages are "visual"

Output:
  data/pages/<doc_id>/<page>.png
  data/pages.jsonl   (one JSON record per page: the manifest every later step reads)

Usage:
  python ingest.py                 # reads data/raw/*.pdf
  python ingest.py path/to/pdfs    # reads another folder
"""

import json
import sys
from pathlib import Path

import pymupdf

RAW_DIR = Path("data/raw")
PAGES_DIR = Path("data/pages")
MANIFEST = Path("data/pages.jsonl")
DPI = 150  # enough for a vision model to read axis labels; raise if small text is lost


def ingest_pdf(pdf_path: Path) -> list[dict]:
    doc_id = pdf_path.stem
    out_dir = PAGES_DIR / doc_id
    out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    with pymupdf.open(pdf_path) as doc:
        for page_index, page in enumerate(doc):
            page_num = page_index + 1
            image_path = out_dir / f"{page_num:04d}.png"
            page.get_pixmap(dpi=DPI).save(image_path)

            text = page.get_text("text").strip()
            records.append(
                {
                    "doc_id": doc_id,
                    "page": page_num,
                    "image_path": image_path.as_posix(),
                    "text": text,
                    "n_chars": len(text),
                    "n_images": len(page.get_images(full=True)),
                    "n_drawings": len(page.get_drawings()),
                }
            )
    return records


def main() -> None:
    raw_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else RAW_DIR
    pdfs = sorted(raw_dir.glob("*.pdf"))
    if not pdfs:
        sys.exit(f"No PDFs found in {raw_dir}/ - put some papers or slide decks there.")

    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with MANIFEST.open("w", encoding="utf-8") as f:
        for pdf in pdfs:
            records = ingest_pdf(pdf)
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            total += len(records)
            print(f"{pdf.name}: {len(records)} pages")

    print(f"\nWrote {total} page records to {MANIFEST}")


if __name__ == "__main__":
    main()
