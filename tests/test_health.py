"""Tests for the /v1/health endpoint."""


def test_health_returns_ok(app_client):
    resp = app_client.get("/v1/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}
