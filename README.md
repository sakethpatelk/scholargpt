# ScholarGPT

![CI](https://github.com/sakethpatelk/scholargpt/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

A master's-level **Retrieval-Augmented Generation** web app. Upload research
PDFs, ask questions in natural language, and receive grounded answers with
page-level citations — powered by ChromaDB, sentence-transformers, Ollama, and Claude.

---

## Architecture

```
User
 │
 ▼
Streamlit UI (app.py)
 │
 ▼
RAGPipeline (src/rag_pipeline.py)        <- single public facade
 ├── pdf_loader.py   PyMuPDF text extraction
 ├── chunker.py      RecursiveCharacterTextSplitter (800 chars, 150 overlap)
 ├── vector_store.py ChromaDB + sentence-transformers bi-encoder (all-MiniLM-L6-v2)
 ├── retriever.py    Bi-encoder recall -> cross-encoder reranking
 └── llm.py          Ollama (local) | Claude API | retrieval-only fallback
```

**Two-stage retrieval** (interview talking point):

| Stage | Model type | Speed | Accuracy |
|---|---|---|---|
| Bi-encoder recall | Dual-encoder | Fast (HNSW) | Good |
| Cross-encoder reranking | Single-encoder | Slow (O(K) passes) | Better |

Bi-encoder retrieves `TOP_K=5` candidates; cross-encoder re-scores and keeps
the top `RERANK_TOP_K=3`. The combination is the industry-standard approach
for production RAG.

---

## Features

- PDF upload and text extraction (multi-page, multi-document)
- Semantic chunking with filename + page metadata preserved
- Local embedding generation — no external API needed for embeddings
- Persistent ChromaDB vector store (survives restarts)
- Cross-encoder reranking for higher retrieval precision
- Streaming answers via Ollama (llama3.2:3b / llama3.1) — fully local and free
- Claude API support for cloud-powered answers
- `[Source N]` citations with filename and page number on every answer
- Multi-turn chat history (last 4 turns passed as context)
- Three-tier fallback: Claude → Ollama → retrieval-only (never crashes)
- Document management: list and delete individual PDFs
- Evaluation script (`src/evaluation.py`) for batch testing

---

## Quick start

### Prerequisites
- Python 3.11+
- **Ollama** (recommended, free, local) — or an Anthropic API key for Claude

### Install

```bash
git clone https://github.com/sakethpatelk/scholargpt.git
cd scholargpt
python -m venv .venv

# macOS/Linux:
source .venv/bin/activate
# Windows:
.venv\Scripts\activate

pip install -r requirements.txt
```

### Configure

```bash
cp .env.example .env
# Edit .env to choose your LLM backend (see section below)
```

### Run

```bash
streamlit run app.py
```

Open **http://localhost:8501**, upload a PDF, and start asking questions.

---

## LLM Backend

ScholarGPT selects a backend automatically (`LLM_BACKEND=auto` in `.env`):

| Priority | Backend | Requirement |
|---|---|---|
| 1 | **Anthropic Claude** | `ANTHROPIC_API_KEY` set in `.env` |
| 2 | **Ollama (local)** | `ollama serve` running on port 11434 |
| 3 | **Retrieval-only** | No LLM — raw passages shown with citations |

### Using Ollama (free, runs locally)

1. Download and install Ollama from **https://ollama.com**

2. Pull a model (choose one):
   ```bash
   ollama pull llama3.2:3b   # ~2 GB, fast on CPU — recommended
   ollama pull llama3.1      # ~4.7 GB, higher quality
   ollama pull mistral       # ~4.1 GB, alternative
   ```

3. Start the Ollama server:
   ```bash
   ollama serve
   ```

4. Set the model in `.env`:
   ```
   OLLAMA_MODEL=llama3.2:3b
   ```

5. Launch the app — the sidebar will show **"LLM: Ollama (local)"**.
   Answers stream token-by-token as the model generates them.

### Using Claude (Anthropic API)

Set in `.env`:
```
ANTHROPIC_API_KEY=sk-ant-api03-...
LLM_BACKEND=auto   # or "anthropic" to force it
```

The sidebar will show **"LLM: Claude (Anthropic API)"**.

---

## Docker

```bash
docker build -t scholargpt .

# With Ollama on the host:
docker run -p 8501:8501 \
  -e OLLAMA_BASE_URL=http://host.docker.internal:11434 \
  -e OLLAMA_MODEL=llama3.2:3b \
  -v "$(pwd)/data:/app/data" \
  scholargpt

# With Claude API:
docker run -p 8501:8501 \
  -e ANTHROPIC_API_KEY=sk-ant-... \
  -v "$(pwd)/data:/app/data" \
  scholargpt
```

The `-v` flag persists ChromaDB and uploaded files between container restarts.

---

## Evaluation

After uploading at least one PDF:

```bash
python -m src.evaluation
```

Runs 5 generic research questions through the full pipeline and prints
latency, source counts, and answer previews.

---

## Configuration reference

| Variable | Default | Description |
|---|---|---|
| `LLM_BACKEND` | `auto` | `auto` / `anthropic` / `ollama` / `none` |
| `ANTHROPIC_API_KEY` | _(empty)_ | Claude API key; skipped if unset |
| `CLAUDE_MODEL` | `claude-opus-4-8` | Claude model ID |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `llama3.2:3b` | Ollama model name |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformers model |
| `CHUNK_SIZE` | `800` | Characters per chunk |
| `CHUNK_OVERLAP` | `150` | Overlap between adjacent chunks |
| `TOP_K` | `5` | Candidates retrieved from vector store |
| `RERANK_TOP_K` | `3` | Chunks kept after cross-encoder reranking |
| `USE_RERANKER` | `true` | Enable/disable reranking |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder model |
| `CHROMA_COLLECTION` | `research_papers` | ChromaDB collection name |

---

## Known limitations

- English-only (no multilingual embedding model configured by default)
- Plain text extraction only — tables and figures are not handled
- ChromaDB is single-machine; not suitable for distributed deployments as-is
- First run downloads embedding/reranker models from HuggingFace (~80–100 MB)
- Ollama responses on CPU are slow for large models (use llama3.2:3b for speed)

## Future improvements

- Hybrid search (BM25 sparse + semantic dense)
- RAGAS-based automated evaluation (faithfulness, answer relevancy)
- Streaming Claude responses via `client.messages.stream()`
- Per-user document isolation with session keys
- Table and figure extraction (PyMuPDF block-level API)
- Distributed vector store (Qdrant, Weaviate, Pinecone)
