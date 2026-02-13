"""Payload serialization helpers for detection results."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from PIL import Image

if TYPE_CHECKING:
    from app.pipelines.detection import CandidateResult, DetectionResult
    from app.pipelines.video_detection import VideoMatchCandidate
    from app.services.roma_service import MatchResult


def bbox_to_list(bbox: tuple[int, int, int, int] | None) -> list[int] | None:
    """Convert a bounding-box tuple to a JSON-friendly list."""
    if bbox is None:
        return None
    return [int(v) for v in bbox]


def match_to_payload(match: MatchResult | None) -> dict | None:
    """Serialize a ``MatchResult`` to a plain dict."""
    if match is None:
        return None
    return {
        "embedding_similarity": float(match.embedding_similarity),
        "embedding_pass": bool(match.embedding_pass),
        "ransac_ok": bool(match.ransac_ok),
        "inliers": int(match.inliers),
        "total_matches": int(match.total_matches),
        "inlier_ratio": float(match.inlier_ratio),
        "score": float(match.score),
        "is_match": bool(match.is_match),
    }


def save_image(path: Path, image: Image.Image) -> None:
    """Persist a PIL image as PNG, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG")


def candidate_payload(
    candidate: CandidateResult,
    *,
    image_path: Path | None = None,
) -> dict:
    """Build a JSON-serializable dict for an image candidate."""
    payload: dict = {
        "kind": candidate.kind,
        "bbox": bbox_to_list(candidate.bbox),
        "score": float(candidate.score),
        "match": match_to_payload(candidate.match),
    }
    if image_path is not None:
        payload["image_path"] = str(image_path)
    return payload


def video_candidate_payload(
    candidate: VideoMatchCandidate,
    *,
    image_path: Path | None = None,
    full_image_path: Path | None = None,
) -> dict:
    """Build a JSON-serializable dict for a video candidate segment."""
    payload: dict = {
        "obj_id": int(candidate.obj_id),
        "kind": candidate.kind,
        "start_time": float(candidate.start_time),
        "end_time": float(candidate.end_time),
        "first_frame_index": int(candidate.first_frame_idx),
        "last_frame_index": int(candidate.last_frame_idx),
        "bbox": bbox_to_list(candidate.bbox),
        "score": float(candidate.score),
        "match": match_to_payload(candidate.match),
    }
    if image_path is not None:
        payload["image_path"] = str(image_path)
    if full_image_path is not None:
        payload["full_image_path"] = str(full_image_path)
    return payload


def build_image_result_payload(
    result: DetectionResult,
    *,
    task_dir: Path,
    query_index: int | None = None,
) -> dict:
    """Build the final payload for a single-query image detection result."""
    candidates = []
    saved_count = 0
    for cand in result.candidate_results:
        if not cand.match.is_match:
            continue
        saved_count += 1
        name = f"q{(query_index or 0) + 1:02d}_{saved_count:06d}.png"
        path = task_dir / "candidates" / name
        save_image(path, cand.image)
        candidates.append(candidate_payload(cand, image_path=path))
    return {
        "is_match": bool(candidates),
        "best_bbox": bbox_to_list(result.best_bbox),
        "best_score": float(result.best_score) if result.best_score is not None else None,
        "best_match": match_to_payload(result.best_match),
        "candidates": int(result.candidates),
        "candidate_results": candidates,
    }
