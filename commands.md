# commands.md

Command cheatsheet for this repo (the HTTP API). Run everything from this repo
root. Endpoint reference lives in the [README](README.md#endpoints); runnable
`curl` examples in [`curl-commands.md`](curl-commands.md).

Prerequisites (sibling repos): Postgres up and the index built —

```bash
cd ../vector-db-rag-context-pipeline  && docker compose up -d
cd ../indexing-rag-context-pipeline   && python build_index.py
```

```bash
# environment (one-time)
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env       # then fill in the real OPENAI_API_KEY
```

```bash
# run natively (docs at http://localhost:8000/docs)
uvicorn api.main:app --reload
uvicorn api.main:app --host 0.0.0.0 --port 8000     # bind explicitly
```

```bash
# run in Docker (joins the vector-db compose network; index build stays host-side)
docker compose up --build                            # build + run in foreground
docker compose up -d --build                         # ... detached
docker compose logs -f api                           # follow logs
docker compose down                                  # stop and remove the container
docker compose build --build-arg EMBEDDING_MODEL=... # rebake a different embedding model
```

```bash
# smoke test
curl localhost:8000/health
curl -s localhost:8000/ask \
  -H 'content-type: application/json' \
  -d '{"question": "What is the net zero target year?"}'
```
