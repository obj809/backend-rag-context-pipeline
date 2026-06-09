"""FastAPI service exposing the RAG pipeline over HTTP.

Wraps the query-side components (`PgVectorRetriever` + the LCEL chain) behind a
JSON API. The embedding model and a Postgres connection pool are created once at
startup; each request borrows a pooled connection, builds the (cheap) retriever +
chain, and invokes it. Answers carry inline `[page N]` citations.

Run:  uvicorn api.main:app --reload   (from this repo root; build the index first)
Env:  OPENAI_API_KEY, DATABASE_URL    (see .env.example)

Transitional layout (multi-repo split in progress):
  - The query engine (chain/load_index/retriever) currently lives in the umbrella
    project's `querying/` dir; it will move to `engine-rag-context-pipeline`. Until
    then this service reaches it by putting that dir on sys.path (UMBRELLA/querying).
  - The vector store + index are owned by `vector-db-` / `indexing-rag-context-pipeline`
    (today: the umbrella project's docker-compose + `python indexing/build_index.py`).
  When those repos land, repoint ENGINE_DIR / env loading below and drop the bridge.
"""

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from langchain_openai import ChatOpenAI
from pgvector.psycopg import register_vector
from psycopg_pool import ConnectionPool
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

BACKEND_ROOT = Path(__file__).resolve().parent.parent   # backend-rag-context-pipeline/
UMBRELLA = BACKEND_ROOT.parent                           # rag-context-pipeline/ (transitional)

# Transitional: the query engine still lives in the umbrella project's querying/ dir.
# Put it on sys.path so `chain`/`load_index`/`retriever` import as before. Replace
# with a dependency on engine-rag-context-pipeline once that repo exists.
ENGINE_DIR = UMBRELLA / "querying"
sys.path.insert(0, str(ENGINE_DIR))
from chain import build_chain           # noqa: E402
from load_index import load_index        # noqa: E402
from retriever import PgVectorRetriever  # noqa: E402

DEFAULT_TOP_K = 6
OPENAI_MODEL = "gpt-5.4-nano"


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Question to answer from the document.")
    k: int = Field(DEFAULT_TOP_K, ge=1, le=20, description="How many chunks to retrieve as context.")


class AskResponse(BaseModel):
    question: str
    answer: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the (expensive) model and open the pool once, before serving requests."""
    # Repo-local .env wins; fall back to the umbrella .env during the multi-repo split.
    load_dotenv(UMBRELLA / ".env")
    load_dotenv(BACKEND_ROOT / ".env", override=True)

    # register_vector runs on every pooled connection so numpy arrays adapt to the
    # pgvector `vector` type.
    pool = ConnectionPool(os.environ["DATABASE_URL"], configure=register_vector, open=False)
    pool.open()
    with pool.connection() as conn:
        embedding_model = load_index(conn)   # the model name recorded at index time

    app.state.pool = pool
    app.state.model = SentenceTransformer(embedding_model)
    app.state.llm = ChatOpenAI(model=OPENAI_MODEL)   # reads OPENAI_API_KEY
    try:
        yield
    finally:
        pool.close()


app = FastAPI(title="RAG Context Pipeline — Backend", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    """Liveness + a cheap DB round-trip."""
    with app.state.pool.connection() as conn:
        conn.execute("SELECT 1")
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    """Answer a question against the indexed document."""
    try:
        with app.state.pool.connection() as conn:
            retriever = PgVectorRetriever(conn=conn, embedder=app.state.model, k=req.k)
            chain = build_chain(retriever, app.state.llm)
            answer = chain.invoke(req.question)
    except Exception as exc:  # surface DB / LLM / upstream failures as 502
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return AskResponse(question=req.question, answer=answer)
