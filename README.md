# Backend RAG Context Pipeline

[![tests](https://github.com/obj809/backend-rag-context-pipeline/actions/workflows/tests.yml/badge.svg)](https://github.com/obj809/backend-rag-context-pipeline/actions/workflows/tests.yml)

The HTTP API for the [RAG Context Pipeline](../). A FastAPI service that answers
questions about the indexed corpus (the three-volume EPBC Act 1999) — as blocking
JSON (`/ask`) or as a streamed chat endpoint (`/chat`) — reusing the same retrieval
+ answer chain as the project's REPL. Answers carry inline `[Volume N, p.M]` citations.

This is one of the per-concern repos the pipeline is split into
(`backend-`, `engine-`, `indexing-`, `vector-db-rag-context-pipeline`). This repo
owns the **HTTP layer only**.

## Dependencies

This service composes its retriever + chain from the **engine** package, and
needs the database and a built index:

| Needs | Provided by |
|---|---|
| Query engine (`chain`, `load_index`, `retriever`) | the `rag-engine` package, pinned in `requirements.txt` (`git+https://…/engine-rag-context-pipeline@<tag>`) |
| Postgres + pgvector | `vector-db-rag-context-pipeline` (`docker compose up -d`) |
| The built `chunks` index | `indexing-rag-context-pipeline` (`python build_index.py`) |

The engine is an ordinary installed dependency — no `sys.path` bridge and no
side-by-side checkout. For active engine development, swap the pinned line in
`requirements.txt` for an editable local install (`-e ../engine-rag-context-pipeline`).

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

- The build context is **this repo's own root** (`context: .`). The engine is
  installed from its Git repo as the `rag-engine` package (the image adds `git`
  for the `git+https` install), so no sibling source is copied in.
- The container joins two **external** Compose networks: the vector-db repo's
  (`vector-db-rag-context-pipeline_default`) to reach Postgres as `db:5432`, and
  the LiteLLM proxy's (`litellm-docker-container_default`) to reach the LLM
  gateway as `litellm:4000`. Both projects must be up first. `DATABASE_URL` and
  `OPENAI_BASE_URL` (`http://litellm:4000/v1`) are set in `docker-compose.yml`;
  `OPENAI_API_KEY` (the LiteLLM virtual key) is interpolated from this repo's `.env`.
- The embedding model (`BAAI/bge-small-en-v1.5`) is baked into the image at build
  time, so startup needs no HuggingFace download. If you change `EMBEDDING_MODEL`
  in the indexer, rebuild with `--build-arg EMBEDDING_MODEL=...`.
- `.env` files are deliberately excluded from the image
  (`Dockerfile.dockerignore`): a baked-in `.env` would embed the OpenAI key. (It
  can't override the injected `DATABASE_URL` — `api/main.py` loads `.env` without
  `override=True`, so the container's env wins.)

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

The tests import `api.main`, which imports the engine as the installed
`rag-engine` package — so the engine just needs to be installed (it is, via
`requirements.txt`); no sibling checkout is required. The two repos' test suites
are otherwise independent.

## Required environment variables

- `OPENAI_API_KEY` — answer generation (the embedding model runs locally, no key).
  With `OPENAI_BASE_URL` set this is the **LiteLLM virtual key**, not the raw
  OpenAI key.
- `OPENAI_BASE_URL` — **optional**. Unset → the LLM call goes direct to OpenAI
  (default). Set → it's routed through an OpenAI-compatible proxy: `http://localhost:4000/v1`
  for host runs, `http://litellm:4000/v1` in Docker (already set in `docker-compose.yml`).
- `DATABASE_URL` — e.g. `postgresql://rag:rag@localhost:5432/rag`, matching
  `vector-db-rag-context-pipeline/docker-compose.yml`.
- `RAG_API_KEY` — **optional** shared secret. When set (production), `/ask` and
  `/chat` require the `X-API-Key: <value>` header (401 otherwise, before any
  retrieval/LLM work) and the auto docs (`/docs`, `/openapi.json`) are disabled.
  When unset (local dev), no check — everything behaves as before. `GET /health`
  is always open.

### Routing through the LiteLLM proxy

`OPENAI_BASE_URL` lets you gate the OpenAI call behind a local/VPS
[LiteLLM](https://docs.litellm.ai/) proxy (`litellm-docker-container/`), so the raw
OpenAI key lives only in the proxy and the backend authenticates with a scoped,
budget-capped **virtual key**. Paired with `RAG_API_KEY` (inbound auth) and the
proxy's per-key budget (outbound spend cap), the public endpoint can't burn
uncapped OpenAI credit.

1. Start the proxy: `cd ../litellm-docker-container && docker compose up -d`
   (its `.env` holds the real `OPENAI_API_KEY` + `LITELLM_MASTER_KEY`).
2. Mint a virtual key scoped to the answer model, once:
   ```bash
   curl -s http://localhost:4000/key/generate \
     -H "Authorization: Bearer $LITELLM_MASTER_KEY" -H "Content-Type: application/json" \
     -d '{"models": ["gpt-5.4-mini"], "max_budget": 5, "key_alias": "rag-backend"}'
   ```
   (or use the admin UI at `http://localhost:4000/ui`). Put the returned `sk-…`
   in this repo's `.env` as `OPENAI_API_KEY`.
3. Host runs: also set `OPENAI_BASE_URL=http://localhost:4000/v1` in `.env`. The
   Docker container already gets `http://litellm:4000/v1` from `docker-compose.yml`.
