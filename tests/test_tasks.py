"""Tests for the /v1/tasks endpoints."""

import time

from tests.conftest import make_test_image


def test_create_image_task(app_client):
    query = make_test_image()
    target = make_test_image(color="green")
    resp = app_client.post(
        "/v1/tasks",
        files=[
            ("query_images", ("query.png", query, "image/png")),
            ("target_file", ("target.png", target, "image/png")),
        ],
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "task_id" in data
    assert data["media_type"] == "image"


def test_get_task_status(app_client):
    query = make_test_image()
    target = make_test_image(color="green")
    create_resp = app_client.post(
        "/v1/tasks",
        files=[
            ("query_images", ("query.png", query, "image/png")),
            ("target_file", ("target.png", target, "image/png")),
        ],
    )
    task_id = create_resp.json()["task_id"]
    # Give worker a moment to finish
    time.sleep(1)
    resp = app_client.get(f"/v1/tasks/{task_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["task_id"] == task_id
    assert data["status"] in {"pending", "running", "done"}


def test_get_nonexistent_task_returns_404(app_client):
    resp = app_client.get("/v1/tasks/nonexistent-id")
    assert resp.status_code == 404


def test_cancel_nonexistent_task_returns_404(app_client):
    resp = app_client.post("/v1/tasks/nonexistent-id/cancel")
    assert resp.status_code == 404


def test_create_task_missing_query_returns_400(app_client):
    target = make_test_image()
    resp = app_client.post(
        "/v1/tasks",
        files=[("target_file", ("target.png", target, "image/png"))],
    )
    assert resp.status_code == 422  # FastAPI validation error


def test_task_events_nonexistent_returns_404(app_client):
    resp = app_client.get("/v1/tasks/nonexistent-id/events")
    assert resp.status_code == 404
