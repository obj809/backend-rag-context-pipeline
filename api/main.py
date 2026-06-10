"""FastAPI service exposing the RAG pipeline over HTTP.

Wraps the query-side components (`PgVectorRetriever` + the LCEL chain) behind a
JSON API. The embedding model and a Postgres connection pool are created once at
startup; each request borrows a pooled connection, builds the (cheap) retriever +
chain, and invokes it. Answers carry inline `[page N]` citations.

Run:  uvicorn api.main:app --reload   (from this repo root; build the index first)
Env:  OPENAI_API_KEY, DATABASE_URL    (see .env.example)

Transitional layout (multi-repo split):
  - The query engine (chain/load_index/retriever) lives in the sibling
    `engine-rag-context-pipeline` repo; this service reaches its leaf modules by
    putting that repo's root on sys.path (UMBRELLA/engine-rag-context-pipeline).
    Replace with a real package dependency once the engine is published.
  - The vector store + index are owned by `vector-db-rag-context-pipeline`
    (docker compose) and `indexing-rag-context-pipeline` (`python build_index.py`).
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
UMBRELLA = BACKEND_ROOT.parent                           # rag-context-pipeline/

# The query engine lives in the sibling engine repo (leaves flattened at its root).
# Put it on sys.path so `chain`/`load_index`/`retriever` import directly. Replace
# with a real package dependency once the engine is published.
ENGINE_DIR = UMBRELLA / "engine-rag-context-pipeline"
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
