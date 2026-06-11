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

## `POST /chat`

Streaming chat endpoint (consumed by the chat frontend's proxy route). Takes the
full conversation; answers the **last user message** and streams the reply as raw
`text/plain` fragments. `-N` disables curl's buffering so chunks show as they
arrive:

```bash
curl -N -X POST localhost:8000/chat \
  -H 'content-type: application/json' \
  -d '{"messages": [{"role": "user", "content": "What is the net-zero target year?"}]}'
```

Prior turns are accepted (and currently ignored — the chain is single-question):

```bash
curl -N -X POST localhost:8000/chat \
  -H 'content-type: application/json' \
  -d '{"messages": [
        {"role": "user", "content": "What is the Net Zero Fund?"},
        {"role": "assistant", "content": "A fund that..."},
        {"role": "user", "content": "How big is it?"}
      ]}'
```

The stream is plain UTF-8 text — no SSE `data:` prefixes, no JSON framing; the
connection closing is the end-of-reply signal. Answers carry inline `[page N]`
citations.

## Error responses

Empty/missing `question`, or `k` out of range → **422** (request validation, no LLM call):

```bash
curl -i -s localhost:8000/ask -H 'content-type: application/json' -d '{"question": ""}'
# HTTP/1.1 422 Unprocessable Entity
```

For `/chat`: a malformed body or unknown `role` → **422**; an empty `messages`
array, or a conversation not ending with a `user` message → **400**:

```bash
curl -i -s -X POST localhost:8000/chat -H 'content-type: application/json' -d '{"messages": []}'
# HTTP/1.1 400 Bad Request
```

Upstream failure (DB or the OpenAI call) → **502** with the error in `detail`.
`/chat` returns error statuses only **before** the first streamed byte; once
streaming has begun, a failure aborts the body instead.
