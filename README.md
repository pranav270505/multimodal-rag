# Multimodal RAG over papers and slides

A question-answering system for PDFs whose answers often sit in a figure, a
diagram or a table, not in the body text. It retrieves the right page with
hybrid search, then shows a vision model both the page image and its extracted
text, and returns the answer together with the page it came from.

The focus of the project is measurement. Each design choice was tested on a
fixed set of 20 questions, and the results, including the ones that went
against expectations, are reported below.

## Results at a glance

| Question | Result |
|---|---|
| Does showing the model the page image help? | Yes. Image + text answered 18/20 from the correct page, against 12/20 for text alone. |
| Do generated figure captions help retrieval? | Barely. Recall@5 was 0.95 with or without them. |
| Does hybrid search beat embeddings alone? | Yes. The correct page ranked first for 65% of questions, against 50%. |
| Does good Recall@5 mean good answers? | No. With Recall@5 at 0.95 the first full pipeline still scored only 10/20. |
| Does asking the model to quote evidence fix that? | No. It helped text and table questions but hurt figure questions. |

All numbers come from `qwen2.5vl:7b` (vision) and `nomic-embed-text`
(embeddings), served by Ollama. With 20 questions, one question is worth 5
points, so read the differences as directions, not precise measurements.

## How it works

```
PDFs ──ingest.py──► page images + extracted text
                         │
            caption.py   │   (vision model describes figures and tables)
                         ▼
                  text_index.py ──► chunks + embeddings
                         │
question ──► hybrid search (embeddings + BM25, merged with RRF) ──► top 5 pages
                         │
                    rag.py: read pages best first; each page is given to the
                    vision model as image + text; stop at the first page that
                    yields an answer
                         ▼
                 answer + source page  (app.py shows both)
```

Design decisions:

- **The page is the unit.** Text is chunked for embedding (1000 characters,
  200 overlap), but search returns pages, ranked by their best chunk. That
  gives one ground truth, "the answer is on page N", for every experiment.
- **No vector database.** About 950 chunks fit in one NumPy matrix, and a
  matrix multiplication searches them instantly.
- **One model client, two backends.** `llm.py` talks to local Ollama or to
  OpenRouter through the same OpenAI-style API, chosen by an environment
  variable. Embeddings always run locally, so retrieval is identical whichever
  model writes the answers.
- **Pages are read one at a time.** Several page images plus their text would
  overflow a small model's context window, and reading in rank order means
  every answer can cite its source page.

## The test set

`data/questions.jsonl` holds 20 questions over 8 documents (4 arXiv papers and
4 cloud-computing lecture decks, 234 pages). Each has one correct page and a
checked answer.

| Type | Count | The answer is found in |
|---|---|---|
| figure | 10 | a plot or diagram |
| table | 4 | a table cell |
| text | 6 | body or slide text |

Questions are worded differently from the pages so that retrieval cannot win
by matching phrases. Answers were graded by hand; a wrong number or a
"not found" counts as wrong.

## Experiments

### 1. Retrieval: where does the correct page rank?

| Method | Recall@1 | Recall@5 | MRR |
|---|---|---|---|
| Embeddings, page text only | 0.50 | 0.95 | 0.68 |
| Embeddings, text + captions | 0.50 | 0.95 | 0.70 |
| BM25 keywords | 0.60 | 0.90 | 0.71 |
| **Hybrid (embeddings + BM25)** | **0.65** | **0.95** | **0.77** |

Captions added almost nothing. The PDFs are born-digital, so text extraction
already captures printed figure captions, axis labels and diagram labels. I
expect captions to matter more for scanned documents, which I did not test.

Hybrid search gained most on table and text questions, where exact terms such
as model or dataset names matter and embeddings blur them.

### 2. Answering: what should the model be shown?

Each question was answered from its correct page, so that retrieval errors
could not affect the comparison.

| Model input | Figure | Table | Text | All |
|---|---|---|---|---|
| Extracted text | 4/10 | 2/4 | 6/6 | 12/20 |
| Text + generated caption | 5/10 | 3/4 | 6/6 | 14/20 |
| Page image | 8/10 | 1/4 | 6/6 | 15/20 |
| **Page image + text** | **9/10** | **3/4** | **6/6** | **18/20** |

The image doubles accuracy on figure questions ("which curve is lowest",
"which bar is tallest"), which text cannot answer because it has the labels
but not the shape of the plot. The image alone misreads dense tables, where
extracted text keeps the digits exact. Giving both is best.

One image + text figure answer said "roughly 0.3" for a bar at about 0.25 and
was graded correct. Graded strictly, that row is 17/20.

### 3. The full pipeline

| Search | Prompt | Figure | Table | Text | All | From correct page | From wrong page | No answer |
|---|---|---|---|---|---|---|---|---|
| Embeddings | basic | 7/10 | 0/4 | 3/6 | 10/20 | 12 | 8 | 0 |
| **Hybrid** | **basic** | **8/10** | **1/4** | **4/6** | **13/20** | 14 | 6 | 0 |
| Hybrid | evidence | 5/10 | 2/4 | 5/6 | 12/20 | 14 | 4 | 2 |

Accuracy fell from 18/20 with the correct page to 10/20 end to end, even
though the correct page was in the top 5 for 19 of 20 questions. The cause is
a combination: the correct page was ranked first only half the time, and the
model almost never said "not found". When a wrong page was read first, it
moved on to the next page 2 times in 10 and otherwise answered from the wrong
page.

Hybrid search recovered 3 questions by ranking the correct page first more
often.

The "evidence" prompt requires the model to quote the words it relies on
before answering. It correctly skipped wrong pages on two table and text
questions, but it refused the correct page on two slide diagrams and left the
correct chart on a third. A quote requirement penalises answers that exist
only in a picture. The basic prompt remains the default.

### Other measurements

- Captioning a page took about 3.5 minutes on a laptop CPU and about 16
  seconds on a Colab T4 GPU.
- The caption model made factual errors: it read 32 arcseconds as 32 degrees
  and reported the order of curves in a plot wrongly.
- One table question fails in every configuration: the model reads the value
  from the wrong row.

## Limitations

- 20 questions and 8 documents is a small test. The table column has 4
  questions.
- The questions and the grading were done by hand by one person.
- All results use one 7B vision model. A larger model may change the picture,
  especially on tables and on saying "not found".
- The rule that picks pages for captioning (an embedded image or 30+ vector
  drawings) misses most slide diagrams, which use fewer shapes. It was left as
  is because captions did not affect retrieval.
- Page-image retrieval (ColPali-style) is not implemented, so the retrieval
  comparison covers text-based methods only.

## Run it yourself

Requirements: Python 3.10+, and [Ollama](https://ollama.com) for local models.

```
pip install -r requirements.txt
ollama pull qwen2.5vl:7b
ollama pull nomic-embed-text
```

Put your PDFs in `data/raw/`, then:

```
python ingest.py                 # page images + text
python caption.py --dry-run      # see which pages would be captioned
python caption.py                # caption them (resumable)
python text_index.py build       # build the search index
python rag.py ask "your question"
streamlit run app.py             # web UI
```

To reproduce the evaluations, write your own `data/questions.jsonl` for your
documents in the same format, then:

```
python eval_retrieval.py         # experiment 1
python answer.py run             # experiment 2, then: grade, report
python rag.py eval basic         # experiment 3
python rag.py eval evidence
```

The source PDFs are not included in this repository. The question set, the
generated captions and the graded answers from my runs are in `data/`, so
`python answer.py report` reproduces the table for experiment 2.

### Using a hosted model

```
LLM_BACKEND=openrouter
OPENROUTER_API_KEY=your-key
OPENROUTER_MODEL=a-vision-capable-model-id
```

Set these as environment variables. Never commit the key.

### Without a GPU

The vision steps are slow on a CPU. `notebooks/run_on_colab.ipynb` runs the
same scripts on a free Colab T4: zip the scripts and data, upload, run,
download the result file.

## Repository layout

| File | Purpose |
|---|---|
| `ingest.py` | PDFs to page images and extracted text |
| `llm.py` | Model client for Ollama or OpenRouter |
| `caption.py` | Vision model describes figures and tables |
| `text_index.py` | Chunking, embeddings, and dense, BM25 and hybrid search |
| `eval_retrieval.py` | Recall@1, Recall@5 and MRR by question type |
| `answer.py` | Answering from the correct page in four input modes, with grading |
| `rag.py` | The full pipeline and its evaluation |
| `app.py` | Streamlit UI |
| `notebooks/run_on_colab.ipynb` | Runs any of the scripts on a Colab GPU |
| `data/questions.jsonl` | The 20 test questions |
| `data/captions.jsonl` | Generated captions for 65 pages |
| `data/answers.jsonl` | Graded answers for experiment 2 |
| `data/rag_answers_*.jsonl` | Graded answers for the three pipeline runs |

## What I would do next

- Get the model to abstain properly: a separate yes/no relevance check per
  page, or answering from the top 3 pages and keeping the best-supported one.
- Add page-image retrieval and compare it with hybrid text search on the same
  questions, including on scanned documents.
- Grow the question set, especially table questions, and have a second person
  grade a sample.
