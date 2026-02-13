"""Shared fixtures for algo_service tests."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable
from unittest.mock import MagicMock

import pytest
from PIL import Image

# Ensure backend package is importable
ROOT_DIR = Path(__file__).resolve().parents[1]
BACKEND_DIR = ROOT_DIR / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


# ---------------------------------------------------------------------------
# Lightweight domain stubs (avoid importing heavy model code)
# ---------------------------------------------------------------------------
@dataclass
class FakeMatchResult:
    embedding_similarity: float = 0.95
    embedding_pass: bool = True
    ransac_ok: bool = True
    inliers: int = 50
    total_matches: int = 100
    inlier_ratio: float = 0.5
    score: float = 0.85
    is_match: bool = True


@dataclass
class FakeCandidateResult:
    kind: str = "segment"
    bbox: tuple[int, int, int, int] | None = (10, 20, 100, 200)
    match: FakeMatchResult = field(default_factory=FakeMatchResult)
    score: float = 0.85
    image: Image.Image = field(default_factory=lambda: Image.new("RGB", (32, 32), "red"))


@dataclass
class FakeDetectionResult:
    is_match: bool = True
    best_bbox: tuple[int, int, int, int] | None = (10, 20, 100, 200)
    best_score: float | None = 0.85
    best_match: FakeMatchResult | None = field(default_factory=FakeMatchResult)
    candidates: int = 1
    candidate_results: list = field(default_factory=lambda: [FakeCandidateResult()])


class FakeAlgorithmService:
    """Mock algorithm service that returns canned results."""

    def __init__(self, results: list | None = None):
        self._results = results or [FakeDetectionResult()]

    def detect_image_batch(self, query_images, target_image, **kwargs):
        return self._results

    def detect_video_batch(self, query_images, video_path, **kwargs):
        return []


@pytest.fixture()
def fake_algorithm():
    return FakeAlgorithmService()


@pytest.fixture()
def app_client(fake_algorithm, tmp_path, monkeypatch):
    """Create a FastAPI TestClient with mocked algorithm service."""
    monkeypatch.setenv("ALGO_DATA_DIR", str(tmp_path / "algo_data"))

    # Patch config before importing app
    import algo_service.config as cfg
    monkeypatch.setattr(cfg, "DATA_DIR", tmp_path / "algo_data")

    from algo_service.app import app

    # Replace lifespan with a no-op so heavy model imports are skipped
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def _noop_lifespan(_app):
        yield

    monkeypatch.setattr(app.router, "lifespan_context", _noop_lifespan)

    # Manually set state (bypass lifespan which loads real models)
    from app.core.tasks import TaskManager
    app.state.algorithm = fake_algorithm
    app.state.tasks = TaskManager(worker_count=1)
    (tmp_path / "algo_data").mkdir(parents=True, exist_ok=True)

    from starlette.testclient import TestClient
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client


def make_test_image(width: int = 64, height: int = 64, color: str = "blue") -> bytes:
    """Create a small PNG image in memory and return its bytes."""
    import io
    img = Image.new("RGB", (width, height), color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
