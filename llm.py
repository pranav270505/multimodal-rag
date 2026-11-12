"""Step 2: one model client, two backends.

Ollama (local) and OpenRouter (hosted) both speak the OpenAI-style API,
so the only things that change between them are the base URL, the key
and the model name. Everything else in the project calls ask() and embed()
and never needs to know which backend is active.

Switch backend with an environment variable:
  LLM_BACKEND=local        (default) -> Ollama on this machine
  LLM_BACKEND=openrouter             -> needs OPENROUTER_API_KEY and OPENROUTER_MODEL

Embeddings always run locally, so retrieval stays identical no matter
which model writes the answers. That keeps later comparisons fair.

Quick test:
  python llm.py data/pages/<doc_id>/0007.png
"""

import base64
import os
import sys
from pathlib import Path

from openai import OpenAI

OLLAMA_URL = "http://localhost:11434/v1"
OPENROUTER_URL = "https://openrouter.ai/api/v1"

BACKEND = os.getenv("LLM_BACKEND", "local")
LOCAL_VISION_MODEL = os.getenv("LOCAL_VISION_MODEL", "qwen2.5vl:7b")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")

_local = OpenAI(base_url=OLLAMA_URL, api_key="ollama")  # Ollama ignores the key


def _chat_client() -> tuple[OpenAI, str]:
    """Return (client, model name) for the active backend."""
    if BACKEND == "local":
        return _local, LOCAL_VISION_MODEL
    if BACKEND == "openrouter":
        key = os.getenv("OPENROUTER_API_KEY")
        model = os.getenv("OPENROUTER_MODEL")
        if not key or not model:
            sys.exit("Set OPENROUTER_API_KEY and OPENROUTER_MODEL to use the openrouter backend.")
        return OpenAI(base_url=OPENROUTER_URL, api_key=key), model
    sys.exit(f"Unknown LLM_BACKEND: {BACKEND!r} (use 'local' or 'openrouter')")


def _image_part(path: str | Path) -> dict:
    """Encode a PNG as a data URL, the format the chat API expects for images."""
    data = base64.b64encode(Path(path).read_bytes()).decode()
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{data}"}}


def ask(prompt: str, images: list[str | Path] | None = None) -> str:
    """Send a prompt, plus optional page images, to the vision model."""
    client, model = _chat_client()
    content = [{"type": "text", "text": prompt}]
    content += [_image_part(p) for p in images or []]
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
        temperature=0,  # same input -> same output, so eval runs are repeatable
    )
    return response.choices[0].message.content.strip()


def embed(texts: list[str]) -> list[list[float]]:
    """Turn texts into vectors with the local embedding model."""
    response = _local.embeddings.create(model=EMBED_MODEL, input=texts)
    return [item.embedding for item in response.data]


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python llm.py path/to/page.png")
    print(f"Backend: {BACKEND}\n")
    print(ask("Describe every figure and table on this page in 2-3 sentences each.", [sys.argv[1]]))
    vector = embed(["hello world"])[0]
    print(f"\nEmbedding works: {len(vector)} dimensions")
