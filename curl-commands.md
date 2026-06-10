# curl commands

`curl` examples for the FastAPI service (`api/main.py`). Start it first, from this
repo root:

```bash
uvicorn api.main:app --reload
```

The service listens on `http://localhost:8000` by default. Interactive,
auto-generated docs live at `http://localhost:8000/docs`.

> Requires the index to be built (`cd indexing-rag-context-pipeline && python
> build_index.py`), Postgres running (`cd vector-db-rag-context-pipeline &&
> docker compose up -d`), and `OPENAI_API_KEY` + `DATABASE_URL` in `.env`.

See the [Endpoints section in the README](README.md#endpoints) for the endpoint reference (request schema, error codes).

## `GET /health`

Liveness check plus a real DB round-trip.

```bash
curl localhost:8000/health
```

```json
{ "status": "ok" }
```

## `POST /ask`

Basic request:

```bash
curl -s localhost:8000/ask \
  -H 'content-type: application/json' \
  -d '{"question": "What is Australia'\''s 2035 emissions target?"}'
```

```json
{
  "question": "What is Australia's 2035 emissions target?",
  "answer": "Australia's 2035 target is to cut emissions 62–70% below 2005 levels [page 4] ..."
}
```

(The apostrophe in "Australia's" is escaped as `'\''` to survive the single-quoted
shell string. Answers carry inline `[page N]` citations.)

With a custom `k`, pretty-printed via `jq`:

```bash
curl -s localhost:8000/ask \
  -H 'content-type: application/json' \
  -d '{"question": "How big is the Net Zero Fund?", "k": 4}' | jq
```

From a JSON file instead of an inline body:

```bash
echo '{"question": "What share of methane comes from agriculture?"}' > q.json
curl -s localhost:8000/ask -H 'content-type: application/json' -d @q.json | jq
```

## Error responses

Empty/missing `question`, or `k` out of range → **422** (request validation, no LLM call):

```bash
curl -i -s localhost:8000/ask -H 'content-type: application/json' -d '{"question": ""}'
# HTTP/1.1 422 Unprocessable Entity
```

Upstream failure (DB or the OpenAI call) → **502** with the error in `detail`.
