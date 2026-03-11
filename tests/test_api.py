"""
API endpoint tests. Heavy dependencies are patched in conftest.py.
build_query_vector, _qdrant_query, and presign_image are patched at the
function level so tests exercise routing and validation logic only.
"""
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _make_mock_result(title="Test Product", score=0.95):
    hit = MagicMock()
    hit.score = score
    hit.payload = {
        "item_id": "item_001",
        "title": title,
        "caption": "A great product",
        "image_key": "images/test.jpg",
    }
    result = MagicMock()
    result.points = [hit]
    return result


@pytest.fixture(scope="module")
def client():
    fake_vec = [1.0] + [0.0] * 511
    mock_result = _make_mock_result()

    with (
        patch("main.build_query_vector", return_value=fake_vec),
        patch("main._qdrant_query", return_value=mock_result),
        patch("main.presign_image", return_value="https://example.com/img.jpg"),
    ):
        import main
        yield TestClient(main.app)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

def test_health_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["ok"] is True


# ---------------------------------------------------------------------------
# GET /search — input validation
# ---------------------------------------------------------------------------

def test_get_search_returns_results(client):
    resp = client.get("/search", params={"q": "blue phone case"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["mode"] == "text"
    assert len(data["results"]) == 1
    assert data["results"][0]["title"] == "Test Product"


def test_get_search_query_too_short(client):
    resp = client.get("/search", params={"q": "a"})
    assert resp.status_code == 422


def test_get_search_query_too_long(client):
    resp = client.get("/search", params={"q": "x" * 501})
    assert resp.status_code == 422


def test_get_search_k_too_large(client):
    resp = client.get("/search", params={"q": "shoes", "k": 100})
    assert resp.status_code == 422


def test_get_search_k_zero(client):
    resp = client.get("/search", params={"q": "shoes", "k": 0})
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# POST /search — input validation
# ---------------------------------------------------------------------------

def test_post_search_text_only(client):
    resp = client.post("/search", params={"q": "wireless headphones"})
    assert resp.status_code == 200
    assert resp.json()["mode"] == "text"


def test_post_search_no_query_no_image(client):
    resp = client.post("/search", params={"q": ""})
    assert resp.status_code == 400


def test_post_search_whitespace_query_no_image(client):
    resp = client.post("/search", params={"q": "   "})
    assert resp.status_code == 400


def test_post_search_image_unsupported_type(client):
    resp = client.post(
        "/search",
        files={"image": ("file.gif", b"GIF87a\x00", "image/gif")},
    )
    assert resp.status_code == 415


def test_post_search_image_too_large(client):
    big = b"\x00" * (11 * 1024 * 1024)  # 11 MB > 10 MB limit
    resp = client.post(
        "/search",
        files={"image": ("photo.jpg", big, "image/jpeg")},
    )
    assert resp.status_code == 413


def test_post_search_query_too_long(client):
    resp = client.post("/search", params={"q": "x" * 501})
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# format_hits — pure function unit test
# ---------------------------------------------------------------------------

def test_format_hits_structure():
    with patch("main.presign_image", return_value=""):
        import main

        hit = MagicMock()
        hit.score = 0.8
        hit.payload = {
            "item_id": "abc",
            "title": "Widget",
            "caption": "nice widget",
            "image_key": "",
        }
        results = main.format_hits([hit])

    assert len(results) == 1
    r = results[0]
    assert r["title"] == "Widget"
    assert r["item_id"] == "abc"
    assert r["score"] == pytest.approx(0.8)
    assert r["caption"] == "nice widget"


def test_format_hits_empty():
    import main
    assert main.format_hits([]) == []


def test_format_hits_missing_payload():
    with patch("main.presign_image", return_value=""):
        import main

        hit = MagicMock()
        hit.score = 0.5
        hit.payload = None
        results = main.format_hits([hit])

    assert results[0]["title"] == ""
    assert results[0]["item_id"] == ""
