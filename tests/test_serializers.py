"""Unit tests for algo_service.serializers."""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
BACKEND_DIR = ROOT_DIR / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from algo_service.serializers import (
    bbox_to_list,
    candidate_payload,
    match_to_payload,
    video_candidate_payload,
)
from tests.conftest import FakeCandidateResult, FakeMatchResult


class TestBboxToList:
    def test_none(self):
        assert bbox_to_list(None) is None

    def test_tuple(self):
        assert bbox_to_list((1, 2, 3, 4)) == [1, 2, 3, 4]


class TestMatchToPayload:
    def test_none(self):
        assert match_to_payload(None) is None

    def test_valid_match(self):
        m = FakeMatchResult(score=0.9, is_match=True)
        p = match_to_payload(m)
        assert p["score"] == 0.9
        assert p["is_match"] is True
        assert "embedding_similarity" in p
        assert "inlier_ratio" in p


class TestCandidatePayload:
    def test_without_image_path(self):
        c = FakeCandidateResult()
        p = candidate_payload(c)
        assert p["kind"] == "segment"
        assert p["score"] == 0.85
        assert "image_path" not in p

    def test_with_image_path(self):
        c = FakeCandidateResult()
        p = candidate_payload(c, image_path=Path("/tmp/test.png"))
        assert p["image_path"] == "/tmp/test.png"


class TestVideoCandidatePayload:
    def test_basic(self):
        from dataclasses import dataclass, field

        @dataclass
        class FakeVideoCandidate:
            obj_id: int = 1
            kind: str = "tracked"
            start_time: float = 0.0
            end_time: float = 5.0
            first_frame_idx: int = 0
            last_frame_idx: int = 150
            bbox: tuple = (10, 20, 100, 200)
            score: float = 0.9
            match: FakeMatchResult = field(default_factory=FakeMatchResult)

        c = FakeVideoCandidate()
        p = video_candidate_payload(c)
        assert p["obj_id"] == 1
        assert p["start_time"] == 0.0
        assert p["end_time"] == 5.0
        assert "image_path" not in p
