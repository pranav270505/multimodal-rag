"""Step 7: a small web UI for the pipeline.

Type a question, get the answer, and see the page it came from.
All the logic lives in rag.py; this file only draws the screen.

Run from the project folder:
  pip install streamlit
  streamlit run app.py
"""

import json
import time
from pathlib import Path

import streamlit as st

import llm
import rag

st.set_page_config(page_title="Multimodal RAG", page_icon="📄", layout="wide")

QUESTIONS = Path("data/questions.jsonl")


def sample_questions() -> list[str]:
    if not QUESTIONS.exists():
        return []
    with QUESTIONS.open(encoding="utf-8") as f:
        return [json.loads(line)["question"] for line in f if line.strip()]


# ---- Sidebar: settings -------------------------------------------------------
with st.sidebar:
    st.header("Settings")
    mode = st.radio(
        "What the model reads",
        ["Page image + text", "Text only"],
        help="Image + text was the most accurate in testing (18/20 against 12/20 for text only "
        "when given the correct page). Text only is much faster on a CPU.",
    )
    k = st.slider("Pages to retrieve", 1, 5, rag.K, help="Pages are read best first until one answers.")
    st.divider()
    model = llm.LOCAL_VISION_MODEL if llm.BACKEND == "local" else "set by OPENROUTER_MODEL"
    st.caption(f"Backend: **{llm.BACKEND}**  \nModel: `{model}`  \nSearch: {rag.METHOD}")
    if llm.BACKEND == "local":
        st.caption("Running on this computer. Without a GPU, each page image takes a few minutes.")

# ---- Main: question in -------------------------------------------------------
st.title("Ask your documents")
st.caption("Answers come from research papers and lecture slides, including their figures and tables.")

samples = sample_questions()
picked = st.selectbox("Try a test question", [""] + samples, format_func=lambda q: q or "Choose one...")
question = st.text_input("Or type your own", value=picked, placeholder="What does the PUE chart show for modern cloud facilities?")

if st.button("Ask", type="primary", disabled=not question.strip()):
    start = time.time()
    with st.spinner("Searching and reading pages..."):
        try:
            result = rag.answer_question(question.strip(), k=k, use_image=(mode == "Page image + text"))
        except Exception as error:  # most often: Ollama is not running
            st.error(f"Could not get an answer: {error}")
            st.info("If you use the local backend, check that Ollama is running.")
            st.stop()
    st.session_state["result"] = result
    st.session_state["seconds"] = time.time() - start

# ---- Main: answer out --------------------------------------------------------
result = st.session_state.get("result")
if result:
    left, right = st.columns([2, 3], gap="large")

    with left:
        st.subheader("Answer")
        if result["source"]:
            st.success(result["answer"])
            doc_id, page = result["source"]
            st.markdown(f"**Source:** {doc_id}, page {page}")
        else:
            st.warning("No answer was found in the retrieved pages.")
        st.caption(f"Read {result['pages_read']} of {len(result['retrieved'])} pages in {st.session_state['seconds']:.0f} s")

        st.markdown("**Pages retrieved, best first**")
        for rank, (doc_id, page) in enumerate(result["retrieved"], start=1):
            used = result["source"] == [doc_id, page]
            st.markdown(f"{rank}. {doc_id}, page {page}" + ("  ← answer came from here" if used else ""))

    with right:
        if result["source"]:
            doc_id, page = result["source"]
            st.subheader("Source page")
            st.image(rag.get_page(doc_id, page)["image_path"])
