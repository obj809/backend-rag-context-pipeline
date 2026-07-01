"""FastAPI service exposing the RAG pipeline over HTTP.

Wraps the query-side components (`PgVectorRetriever` + the LCEL chain) behind
HTTP: `/ask` (blocking JSON) and `/chat` (chat-message contract, streamed
text/plain — consumed by the llm-user-interface frontend's proxy route). The
embedding model and a Postgres connection pool are created once at startup; each
request borrows a pooled connection and builds the (cheap) retriever + chain.
Answers carry inline `[Volume N, p.M]` citations.

Run:  uvicorn api.main:app --reload   (from this repo root; build the index first)
Env:  OPENAI_API_KEY, DATABASE_URL    (see .env.example)
      OPENAI_BASE_URL (optional) routes the LLM call through a LiteLLM/OpenAI-
      compatible proxy; when set, OPENAI_API_KEY is that proxy's key (unset = direct OpenAI).

Layout (multi-repo split):
  - The query engine (chain/load_index/retriever) is the installed `rag-engine`
    package (from the `engine-rag-context-pipeline` repo, pinned in
    requirements.txt); its leaf modules import as ordinary top-level modules.
  - The vector store + index are owned by `vector-db-rag-context-pipeline`
    (docker compose) and `indexing-rag-context-pipeline` (`python build_index.py`).
"""

import os
import secrets
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from chain import build_answer_chain, build_chain, build_condense_chain, format_docs, format_history
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import StreamingResponse
from langchain_openai import ChatOpenAI
from load_index import load_index
from pgvector.psycopg import register_vector
from psycopg_pool import ConnectionPool
from pydantic import BaseModel, Field
from retriever import PgVectorRetriever
from sentence_transformers import SentenceTransformer

BACKEND_ROOT = Path(__file__).resolve().parent.parent
UMBRELLA = BACKEND_ROOT.parent

DEFAULT_TOP_K = 6
OPENAI_MODEL = "gpt-5.4-mini"


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Question to answer from the document.")
    k: int = Field(DEFAULT_TOP_K, ge=1, le=20, description="How many chunks to retrieve as context.")


class AskResponse(BaseModel):
    question: str
    answer: str


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    messages: list[Message] = Field(..., description="Full conversation; last message must be from the user.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the (expensive) model and open the pool once, before serving requests."""
    # Precedence: real environment > repo-local .env > umbrella .env — injected
    # env (e.g. the container's DATABASE_URL) is never overridden by a .env file.
    load_dotenv(BACKEND_ROOT / ".env")
    load_dotenv(UMBRELLA / ".env")

    # register_vector runs on every pooled connection so numpy arrays adapt to the
    # pgvector `vector` type.
    pool = ConnectionPool(os.environ["DATABASE_URL"], configure=register_vector, open=False)
    pool.open()
    with pool.connection() as conn:
        embedding_model = load_index(conn)

    app.state.pool = pool
    app.state.model = SentenceTransformer(embedding_model)
    # base_url unset → direct OpenAI; set → route via a LiteLLM/OpenAI-compatible proxy.
    app.state.llm = ChatOpenAI(model=OPENAI_MODEL, base_url=os.getenv("OPENAI_BASE_URL") or None)
    app.state.api_key = os.environ.get("RAG_API_KEY")  # None → auth disabled (local dev)
    try:
        yield
    finally:
        pool.close()


# Auto docs are public once the service is proxied; ship them only when running
# keyless (local dev). Read at import time — in production the key arrives as
# real env (compose-injected), not just a .env file, so this stays consistent
# with the lifespan check.
_KEYLESS = os.environ.get("RAG_API_KEY") is None
app = FastAPI(
    title="RAG Context Pipeline — Backend",
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs" if _KEYLESS else None,
    redoc_url="/redoc" if _KEYLESS else None,
    openapi_url="/openapi.json" if _KEYLESS else None,
)


def require_api_key(
    request: Request, x_api_key: str | None = Header(default=None)
) -> None:
    """Shared-secret gate for the credit-spending endpoints.

    Active only when RAG_API_KEY is set (production); keyless local dev is
    unchanged. Rejects with 401 before any embedding/retrieval/LLM work.
    """
    expected = getattr(request.app.state, "api_key", None)
    if expected is not None and not secrets.compare_digest(x_api_key or "", expected):
        raise HTTPException(status_code=401, detail="invalid or missing X-API-Key")


@app.get("/health")
def health() -> dict:
    """Liveness + a cheap DB round-trip."""
    with app.state.pool.connection() as conn:
        conn.execute("SELECT 1")
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse, dependencies=[Depends(require_api_key)])
def ask(req: AskRequest) -> AskResponse:
    """Answer a question against the indexed document."""
    try:
        with app.state.pool.connection() as conn:
            retriever = PgVectorRetriever(conn=conn, embedder=app.state.model, k=req.k)
            chain = build_chain(retriever, app.state.llm)
            answer = chain.invoke(req.question)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return AskResponse(question=req.question, answer=answer)


@app.post("/chat", dependencies=[Depends(require_api_key)])
def chat(req: ChatRequest) -> StreamingResponse:
    """Answer the last user message, streamed as raw text/plain fragments.

    Contract (see the umbrella's integration-plan.md): full conversation
    in, plain UTF-8 token stream out (no SSE/JSON framing), non-200 for any failure
    before the first token. Multi-turn: when prior turns exist, a cheap LLM pass
    condenses the recent history + latest message into a self-contained question
    before retrieval, so follow-ups with pronouns/ellipsis resolve their referent.
    """
    if not req.messages or req.messages[-1].role != "user":
        raise HTTPException(status_code=400, detail="messages must be non-empty and end with a user message")
    question = req.messages[-1].content
    history = [(m.role, m.content) for m in req.messages[:-1]]

    try:
        # With prior turns, fold history + the latest message into a standalone
        # question so retrieval isn't embedding an under-specified follow-up. The
        # first turn (no history) skips the extra LLM call and retrieves on the raw
        # question — unchanged single-turn cost and latency.
        search_query = (
            build_condense_chain(app.state.llm).invoke(
                {"chat_history": format_history(history), "question": question}
            )
            if history
            else question
        )
        # Retrieve eagerly so the pooled connection is held only for the SQL
        # lookup, not for the multi-second LLM stream.
        with app.state.pool.connection() as conn:
            retriever = PgVectorRetriever(conn=conn, embedder=app.state.model, k=DEFAULT_TOP_K)
            docs = retriever.invoke(search_query)
        stream = build_answer_chain(app.state.llm).stream(
            {"context": format_docs(docs), "question": search_query}
        )
        # Pull the first token before returning: once StreamingResponse starts,
        # the 200 status is already sent, so OpenAI failures must surface here.
        first = next(stream, "")
    except Exception as exc:  # surface DB / LLM / upstream failures as 502
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    def generate():
        yield first
        yield from stream

    return StreamingResponse(generate(), media_type="text/plain; charset=utf-8")
