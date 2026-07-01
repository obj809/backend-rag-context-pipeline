"""Shared test fixtures: the FastAPI app with fakes on `app.state`.

The real lifespan loads a SentenceTransformer, opens a Postgres pool, and
constructs ChatOpenAI — none of which belong in contract tests. TestClient is
deliberately NOT used as a context manager, so the lifespan never runs;
`app.state` is assembled here with stand-ins instead. The suite therefore runs
offline: no database, no OpenAI key, no model download.
"""

import sys
from contextlib import contextmanager
from itertools import cycle
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatResult

# Make `api` importable regardless of how pytest was invoked (bare `pytest`,
# any cwd) — pytest only puts this tests/ dir on sys.path, not the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api.main import app  # imports the engine leaves from the installed rag-engine package

# GenericFakeChatModel streams this back in word-sized chunks; concatenated
# chunks reproduce it exactly, so tests can assert on the full text.
FAKE_ANSWER = "Protected matters are listed in Part 3 [Volume 1, p.96]."
# (content, page, volume) — matches the engine retriever's SELECT content, page, volume.
FAKE_ROWS = [
    ("Matters of national environmental significance are in Part 3.", 96, "Volume 1"),
    ("Bilateral agreements are covered in Part 5.", 4, "Volume 2"),
]


class FakeEmbedder:
    """Records every encoded text so tests can assert which query was embedded."""

    def __init__(self):
        self.encoded: list[str] = []

    def encode(self, text, **kwargs):
        self.encoded.append(text)
        return [0.0, 0.0, 0.0]


class _FakeResult:
    def __init__(self, rows):
        self._rows = rows

    def fetchall(self):
        return self._rows


class FakeConn:
    def __init__(self, rows):
        self._rows = rows

    def execute(self, sql, params=None):
        return _FakeResult(self._rows)


class FailingConn:
    def execute(self, sql, params=None):
        raise RuntimeError("db down")


class FakePool:
    def __init__(self, conn):
        self._conn = conn

    @contextmanager
    def connection(self):
        yield self._conn


class FailingChatModel(BaseChatModel):
    """Raises on first use — exercises the pre-stream 502 path. No `_stream`
    implementation, so `.stream()` falls back to `_generate` and raises before
    the first token, exactly like an OpenAI auth/connection failure."""

    @property
    def _llm_type(self) -> str:
        return "failing"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        raise RuntimeError("llm down")


# The standalone question the condensing fake rewrites a follow-up into. Distinct
# from FAKE_ANSWER so a multi-turn test can tell the condense call from the answer
# call by which string reaches the retriever.
CONDENSED_QUESTION = "What are the penalties under the EPBC Act?"

# What the keyed client's gate expects; tests send (or omit) this value.
SHARED_KEY = "test-shared-secret"


def _wire(pool_conn, llm, embedder, api_key=None):
    app.state.pool = FakePool(pool_conn)
    app.state.model = embedder
    app.state.llm = llm
    app.state.api_key = api_key  # always reset — `app` is shared across tests
    return TestClient(app)


@pytest.fixture
def embedder():
    return FakeEmbedder()


@pytest.fixture
def client(embedder):
    """Happy-path app: working fake DB + a fake LLM that streams FAKE_ANSWER."""
    llm = GenericFakeChatModel(messages=cycle([AIMessage(content=FAKE_ANSWER)]))
    return _wire(FakeConn(FAKE_ROWS), llm, embedder)


@pytest.fixture
def client_db_down(embedder):
    llm = GenericFakeChatModel(messages=cycle([AIMessage(content=FAKE_ANSWER)]))
    return _wire(FailingConn(), llm, embedder)


@pytest.fixture
def client_llm_down(embedder):
    return _wire(FakeConn(FAKE_ROWS), FailingChatModel(), embedder)


@pytest.fixture
def client_condensing(embedder):
    """Multi-turn app: the fake LLM returns CONDENSED_QUESTION for the first
    (condense) call, then streams FAKE_ANSWER for the answer call — a plain
    iterator (not cycle) so the two calls consume distinct messages in order."""
    llm = GenericFakeChatModel(
        messages=iter([AIMessage(content=CONDENSED_QUESTION), AIMessage(content=FAKE_ANSWER)])
    )
    return _wire(FakeConn(FAKE_ROWS), llm, embedder)


@pytest.fixture
def client_keyed(embedder):
    """Happy-path app with the shared-secret gate armed (RAG_API_KEY set)."""
    llm = GenericFakeChatModel(messages=cycle([AIMessage(content=FAKE_ANSWER)]))
    return _wire(FakeConn(FAKE_ROWS), llm, embedder, api_key=SHARED_KEY)
