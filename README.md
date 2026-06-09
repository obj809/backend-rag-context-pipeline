# backend-rag-context-pipeline

The HTTP API for the [RAG Context Pipeline](../). A FastAPI service that answers
questions about the indexed document over JSON, reusing the same retrieval +
answer chain as the project's REPL. Answers carry inline `[page N]` citations.

This is one of several repos the pipeline is being split into
(`backend-`, `engine-`, `indexing-`, `vector-db-rag-context-pipeline`). This repo
owns the **HTTP layer only**.

## Transitional dependencies (split in progress)

This service is **not yet standalone**. Until the sibling repos land it borrows
from the umbrella project that contains it:

| Needs | Provided today by | Will move to |
|---|---|---|
| Query engine (`chain`, `load_index`, `retriever`) | umbrella `querying/` (added to `sys.path`) | `engine-rag-context-pipeline` |
| Postgres + pgvector | umbrella `docker-compose.yml` | `vector-db-rag-context-pipeline` |
| The built `chunks` index | umbrella `python indexing/build_index.py` | `indexing-rag-context-pipeline` |

When those repos exist, repoint `ENGINE_DIR` and the env loading in `api/main.py`
and drop the `sys.path` bridge.

## Run

Prerequisites: the umbrella Postgres running (`docker compose up -d`) and the
index built (`python indexing/build_index.py`) — both from the umbrella project
root for now.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env        # then fill in OPENAI_API_KEY (DATABASE_URL is preset)

uvicorn api.main:app --reload
```

The service listens on `http://localhost:8000` (interactive docs at `/docs`).
`api/main.py` reads this repo's `.env` first, falling back to the umbrella `.env`,
so an existing umbrella `.env` keeps it running with no extra setup.

## Endpoints

| Endpoint | Description |
|---|---|
| `GET /health` | Liveness check plus a real DB round-trip; returns `{ "status": "ok" }`. |
| `POST /ask` | Body `{ "question": string, "k"?: int }` (`k` defaults to 6, range 1–20). Returns the `question` and an `answer` with inline `[page N]` citations. |

Validation errors (empty `question`, `k` out of range) return **422** (no LLM
call); upstream DB/OpenAI failures return **502** with the error in `detail`.
Runnable `curl` examples live in [`curl-commands.md`](curl-commands.md).

## Required environment variables

- `OPENAI_API_KEY` — answer generation (the embedding model runs locally, no key).
- `DATABASE_URL` — e.g. `postgresql://rag:rag@localhost:5432/rag`, matching the
  umbrella `docker-compose.yml`.
