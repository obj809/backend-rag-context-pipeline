"""Contract tests for POST /ask (blocking JSON) and GET /health."""

from conftest import FAKE_ANSWER


def test_health_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_ask_returns_question_and_answer(client):
    resp = client.post("/ask", json={"question": "What is the target year?"})
    assert resp.status_code == 200
    assert resp.json() == {"question": "What is the target year?", "answer": FAKE_ANSWER}


def test_ask_empty_question_is_422(client):
    resp = client.post("/ask", json={"question": ""})
    assert resp.status_code == 422


def test_ask_k_out_of_range_is_422(client):
    assert client.post("/ask", json={"question": "q", "k": 0}).status_code == 422
    assert client.post("/ask", json={"question": "q", "k": 21}).status_code == 422


def test_ask_db_failure_is_502(client_db_down):
    resp = client_db_down.post("/ask", json={"question": "q"})
    assert resp.status_code == 502
