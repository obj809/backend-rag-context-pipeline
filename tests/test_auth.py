"""Contract tests for the shared-secret gate (X-API-Key) — deployment step 4a.

The gate arms only when RAG_API_KEY is set; keyless local dev is untouched.
"""

from conftest import FAKE_ANSWER, SHARED_KEY

CHAT_BODY = {"messages": [{"role": "user", "content": "what is the target year?"}]}
ASK_BODY = {"question": "what is the target year?"}


def test_chat_without_key_is_401_before_any_work(client_keyed, embedder):
    resp = client_keyed.post("/chat", json=CHAT_BODY)
    assert resp.status_code == 401
    assert embedder.encoded == []  # rejected before embedding/retrieval/LLM


def test_ask_without_key_is_401(client_keyed):
    assert client_keyed.post("/ask", json=ASK_BODY).status_code == 401


def test_wrong_key_is_401(client_keyed):
    resp = client_keyed.post("/chat", json=CHAT_BODY, headers={"X-API-Key": "wrong"})
    assert resp.status_code == 401


def test_correct_key_passes(client_keyed):
    resp = client_keyed.post("/chat", json=CHAT_BODY, headers={"X-API-Key": SHARED_KEY})
    assert resp.status_code == 200
    assert resp.text == FAKE_ANSWER

    resp = client_keyed.post("/ask", json=ASK_BODY, headers={"X-API-Key": SHARED_KEY})
    assert resp.status_code == 200
    assert resp.json()["answer"] == FAKE_ANSWER


def test_health_stays_open_with_gate_armed(client_keyed):
    assert client_keyed.get("/health").status_code == 200


def test_keyless_app_skips_the_check(client):
    """RAG_API_KEY unset (local dev) → no header needed, nothing changes."""
    assert client.post("/chat", json=CHAT_BODY).status_code == 200
