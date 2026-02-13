"""Health-check endpoint."""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter()


@router.get("/v1/health")
def health() -> dict:
    """Return service health status."""
    return {"status": "ok"}
