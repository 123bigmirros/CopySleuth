"""Tests for the /v1/detect endpoint."""

from tests.conftest import make_test_image


def test_detect_single_query(app_client):
    query = make_test_image()
    target = make_test_image(color="green")
    resp = app_client.post(
        "/v1/detect",
        files=[
            ("query_images", ("query.png", query, "image/png")),
            ("target_image", ("target.png", target, "image/png")),
        ],
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["media_type"] == "image"
    assert len(data["results"]) == 1
    result = data["results"][0]
    assert result["is_match"] is True
    assert len(result["candidate_results"]) >= 1


def test_detect_missing_query_returns_400(app_client):
    target = make_test_image()
    resp = app_client.post(
        "/v1/detect",
        files=[("target_image", ("target.png", target, "image/png"))],
    )
    assert resp.status_code == 400


def test_detect_fallback_to_query_image(app_client):
    """When query_images is empty, falls back to query_image."""
    query = make_test_image()
    target = make_test_image(color="green")
    resp = app_client.post(
        "/v1/detect",
        files=[
            ("query_image", ("q.png", query, "image/png")),
            ("target_image", ("target.png", target, "image/png")),
        ],
    )
    assert resp.status_code == 200
    assert resp.json()["media_type"] == "image"


def test_detect_invalid_image_returns_400(app_client):
    target = make_test_image()
    resp = app_client.post(
        "/v1/detect",
        files=[
            ("query_images", ("bad.png", b"not-an-image", "image/png")),
            ("target_image", ("target.png", target, "image/png")),
        ],
    )
    assert resp.status_code == 400
