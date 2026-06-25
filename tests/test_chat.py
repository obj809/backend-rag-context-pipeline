"""Contract tests for POST /chat — the behaviors the frontend's proxy relies on
(see the umbrella's integration-plan.md §3)."""

from conftest import FAKE_ANSWER

from retriever import QUERY_PREFIX  # from the installed rag-engine package


def _user(content):
    return {"role": "user", "content": content}


def _assistant(content):
    return {"role": "assistant", "content": content}


def test_streams_plain_text_answer(client):
    resp = client.post("/chat", json={"messages": [_user("What is the target year?")]})
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "text/plain; charset=utf-8"
    assert resp.text == FAKE_ANSWER


def test_answers_the_last_user_message(client, embedder):
    resp = client.post(
        "/chat",
        json={
            "messages": [
                _user("What is RAG?"),
                _assistant("Retrieval-augmented generation."),
                _user("What about scope 3?"),
            ]
        },
    )
    assert resp.status_code == 200
    # The retriever embeds QUERY_PREFIX + query; prior turns must be ignored.
    assert embedder.encoded == [QUERY_PREFIX + "What about scope 3?"]


def test_empty_messages_is_400(client):
    resp = client.post("/chat", json={"messages": []})
    assert resp.status_code == 400


def test_conversation_not_ending_with_user_is_400(client):
    resp = client.post("/chat", json={"messages": [_user("hi"), _assistant("hello")]})
    assert resp.status_code == 400


def test_unknown_role_is_422(client):
    resp = client.post("/chat", json={"messages": [{"role": "system", "content": "x"}]})
    assert resp.status_code == 422


def test_missing_messages_key_is_422(client):
    resp = client.post("/chat", json={})
    assert resp.status_code == 422


def test_db_failure_is_502_before_any_bytes(client_db_down):
    resp = client_db_down.post("/chat", json={"messages": [_user("hi")]})
    assert resp.status_code == 502
    assert "db down" in resp.json()["detail"]


def test_llm_failure_is_502_before_any_bytes(client_llm_down):
    resp = client_llm_down.post("/chat", json={"messages": [_user("hi")]})
    assert resp.status_code == 502
    assert "llm down" in resp.json()["detail"]
