# ============================================================
# test_api_endpoints.py
#
# WHAT THIS FILE DOES:
# Unit tests for the FastAPI endpoints, written with pytest and
# FastAPI's TestClient.
#
# TestClient lets us call the API directly from code — no need to
# start a real server with uvicorn. Each test sends a request and
# checks (asserts) that the response is what we expect.
#
# Tests cover:
#   /health   -> service is alive
#   /metadata -> corpus info is returned
#   /query    -> valid question works, and bad input returns the
#                correct HTTP error codes (400 / 422)
#
# Run with:  pytest test_api_endpoints.py -v
# ============================================================

# TestClient -> lets us send requests to the app inside tests
from fastapi.testclient import TestClient

# Import the FastAPI app object from our api.py file
from api import app

# Wrap the app in a TestClient so we can call its endpoints
client = TestClient(app)


# ============================================================
# TESTS FOR /health
# ============================================================
def test_health_returns_ok() -> None:
    """/health should return 200 and report the service is running.

    Checks that the status is 'ok' and that the database actually
    contains chunks (so the system started up properly).
    """
    response = client.get("/health")

    # 200 means the request succeeded
    assert response.status_code == 200

    # Read the JSON body and check its contents
    data = response.json()
    assert data["status"] == "ok"
    assert data["chunks_in_db"] > 0


# ============================================================
# TESTS FOR /metadata
# ============================================================
def test_metadata_returns_corpus_info() -> None:
    """/metadata should return 200 and include the expected fields.

    Confirms the endpoint reports chunk count, chunk size, K, and
    the embedding model being used.
    """
    response = client.get("/metadata")
    assert response.status_code == 200

    data = response.json()
    # Each of these keys must be present in the response
    assert "total_chunks" in data
    assert "chunk_size" in data
    assert "top_k" in data
    assert data["embedding_model"] == "all-MiniLM-L6-v2"


# ============================================================
# TESTS FOR /query — ERROR CASES
# ============================================================
def test_query_empty_question_returns_400() -> None:
    """An empty (whitespace-only) question should return 400 Bad Request.

    This checks our own validation inside the /query endpoint.
    """
    response = client.post("/query", json={"question": "   "})

    # 400 = the client sent a bad request
    assert response.status_code == 400
    # The error message should mention that the question is empty
    assert "empty" in response.json()["detail"].lower()


def test_query_missing_field_returns_422() -> None:
    """A body with no 'question' field should return 422.

    422 is returned automatically by FastAPI/pydantic when the
    request body does not match the expected shape.
    """
    response = client.post("/query", json={})
    assert response.status_code == 422


def test_query_wrong_type_returns_422() -> None:
    """A question that is a number instead of a string should return 422.

    Again this is pydantic's automatic validation doing its job.
    """
    response = client.post("/query", json={"question": 12345})
    assert response.status_code == 422


# ============================================================
# TEST FOR /query — SUCCESS CASE
# ============================================================
def test_query_valid_question_returns_answer() -> None:
    """A valid question should return 200 with a complete response.

    Checks that the response contains an answer, source citations,
    and a latency measurement — i.e. the full RAG pipeline ran.
    """
    response = client.post("/query", json={"question": "what is physics"})
    assert response.status_code == 200

    data = response.json()
    assert "answer" in data
    assert "sources" in data
    assert "latency_seconds" in data
    # The system should have retrieved at least one source chunk
    assert len(data["sources"]) > 0