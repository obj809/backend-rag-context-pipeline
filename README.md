# Backend RAG Context Pipeline

[![tests](https://github.com/obj809/backend-rag-context-pipeline/actions/workflows/tests.yml/badge.svg)](https://github.com/obj809/backend-rag-context-pipeline/actions/workflows/tests.yml)

The HTTP API for the [RAG Context Pipeline](../). A FastAPI service that answers
questions about the indexed corpus (the three-volume EPBC Act 1999) — as blocking
JSON (`/ask`) or as a streamed chat endpoint (`/chat`) — reusing the same retrieval
+ answer chain as the project's REPL. Answers carry inline `[Volume N, p.M]` citations.

This is one of the per-concern repos the pipeline is split into
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
| `POST /ask` | Body `{ "question": string, "k"?: int }` (`k` defaults to 6, range 1–20). Returns the `question` and an `answer` with inline `[Volume N, p.M]` citations. |
| `POST /chat` | Body `{ "messages": [{ "role": "user" \| "assistant", "content": string }, ...] }` — full conversation, last message must be from the user. **Streams** the answer as raw `text/plain; charset=utf-8` fragments (no SSE/JSON framing); stream close = end of reply. v1 answers only the last user message (prior turns are accepted but ignored); `k` is fixed at 6. Built for the chat frontend's proxy route. |

Validation errors return **422** (malformed body/roles) or **400** (`messages`
empty or not ending with a user message) — no LLM call; upstream DB/OpenAI
failures return **502** with the error in `detail`. `/chat` only fails with a
status code *before* the first streamed byte; a mid-stream failure surfaces as
an aborted body. Runnable `curl` examples live in
[`curl-commands.md`](curl-commands.md).

## Tests

Contract tests for the three endpoints — the behaviors the chat frontend's
proxy relies on (status codes, streaming content type, last-user-message
semantics). They run **offline**: the app is wired with fakes
(`tests/conftest.py`) instead of the real pool/embedder/LLM, so no database,
OpenAI key, or model download is needed.

```bash
pip install -r requirements-dev.txt   # pytest + httpx, on top of requirements.txt
python -m pytest
```

The engine repo must still be checked out as a sibling (the tests import
`api.main`, which bridges to it over `sys.path`) — but only its source modules
are used; the two repos' test suites are otherwise independent.

## Required environment variables

- `OPENAI_API_KEY` — answer generation (the embedding model runs locally, no key).
- `DATABASE_URL` — e.g. `postgresql://rag:rag@localhost:5432/rag`, matching
  `vector-db-rag-context-pipeline/docker-compose.yml`.
- `RAG_API_KEY` — **optional** shared secret. When set (production), `/ask` and
  `/chat` require the `X-API-Key: <value>` header (401 otherwise, before any
  retrieval/LLM work) and the auto docs (`/docs`, `/openapi.json`) are disabled.
  When unset (local dev), no check — everything behaves as before. `GET /health`
  is always open.
