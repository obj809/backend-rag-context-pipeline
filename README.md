# Backend RAG Context Pipeline

The HTTP API for the [RAG Context Pipeline](../). A FastAPI service that answers
questions about the indexed document over JSON, reusing the same retrieval +
answer chain as the project's REPL. Answers carry inline `[page N]` citations.

This is one of several repos the pipeline is being split into
(`backend-`, `engine-`, `indexing-`, `vector-db-rag-context-pipeline`). This repo
owns the **HTTP layer only**.

## Dependencies (sibling repos)

This service is **not yet standalone** — it composes its retriever + chain from
the engine repo, and needs the database and a built index:

| Needs | Provided by |
|---|---|
| Query engine (`chain`, `load_index`, `retriever`) | `engine-rag-context-pipeline` (its root on `sys.path` via `ENGINE_DIR`) |
| Postgres + pgvector | `vector-db-rag-context-pipeline` (`docker compose up -d`) |
| The built `chunks` index | `indexing-rag-context-pipeline` (`python build_index.py`) |

The engine is reached by a `sys.path` bridge (`ENGINE_DIR` in `api/main.py`)
rather than a package import — replace that with a real package dependency once
the engine is published.

## Run

Prerequisites: Postgres running (`cd vector-db-rag-context-pipeline && docker compose up -d`)
and the index built (`cd indexing-rag-context-pipeline && python build_index.py`).

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

## Run in Docker

Same prerequisites as above: the database container up and the index built
(the index build stays a **host-side** step via `indexing-rag-context-pipeline`).
Then:

```bash
docker compose up --build
```

Notes on how the image works:

- The build context is the **umbrella root** (`context: ..`) because the engine's
  leaf modules (`retriever.py`, `chain.py`, `load_index.py`) are copied into the
  image at `/app/engine-rag-context-pipeline/`, mirroring the umbrella layout so
  the `ENGINE_DIR` `sys.path` bridge works unchanged.
- The container joins the vector-db repo's Compose network
  (`vector-db-rag-context-pipeline_default`, declared `external`) and reaches
  Postgres as `db:5432` — `DATABASE_URL` is set in `docker-compose.yml`;
  `OPENAI_API_KEY` is interpolated from this repo's `.env`.
- The embedding model (`BAAI/bge-small-en-v1.5`) is baked into the image at build
  time, so startup needs no HuggingFace download. If you change `EMBEDDING_MODEL`
  in the indexer, rebuild with `--build-arg EMBEDDING_MODEL=...`.
- `.env` files are deliberately excluded from the image
  (`Dockerfile.dockerignore`): a baked-in `.env` would override the injected
  `DATABASE_URL` and embed the API key.

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
- `DATABASE_URL` — e.g. `postgresql://rag:rag@localhost:5432/rag`, matching
  `vector-db-rag-context-pipeline/docker-compose.yml`.
