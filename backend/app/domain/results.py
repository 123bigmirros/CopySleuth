from __future__ import annotations

from pathlib import Path

from PIL import Image

from app.schemas import CandidateResult, DetectionResponse, MatchStats, VideoSegmentResult
from app.services.media_client import MediaClient


def image_to_data_url(image: Image.Image | Path | str, max_size: int = 256) -> str:
    # Preview generation disabled for performance.
    return ""


def image_to_data_url_with_bbox(
    image: Image.Image,
    bbox: tuple[int, int, int, int] | None,
    max_size: int = 640,
    color: tuple[int, int, int] = (255, 0, 0),
    width: int = 6,
) -> str:
    # Preview generation disabled for performance.
    return ""


def serialize_image_candidate(
    candidate,
    label: str,
    idx: int,
    full_preview: str | None = None,
    full_size: tuple[int, int] | None = None,
    full_image: Image.Image | None = None,
    media_client: MediaClient | None = None,
) -> dict:
    return CandidateResult(
        id=idx,
        label=label,
        kind=candidate.kind,
        bbox=candidate.bbox,
        score=candidate.score,
        match=MatchStats(
            embedding_similarity=candidate.match.embedding_similarity,
            embedding_pass=candidate.match.embedding_pass,
            ransac_ok=candidate.match.ransac_ok,
            inliers=candidate.match.inliers,
            total_matches=candidate.match.total_matches,
            inlier_ratio=candidate.match.inlier_ratio,
        score=candidate.match.score,
        is_match=candidate.match.is_match,
    ),
        preview="",
        full_preview=full_preview if full_preview else "",
        full_width=full_size[0] if full_size else None,
        full_height=full_size[1] if full_size else None,
    ).dict()


def serialize_image_result(
    result,
    full_preview: str | None = None,
    full_size: tuple[int, int] | None = None,
    full_image: Image.Image | None = None,
    query_preview: str | None = None,
    query_label: str | None = None,
    media_client: MediaClient | None = None,
) -> dict:
    match = None
    candidate_results = []
    segment_index = 0
    best_score = None
    best_bbox = None
    for idx, candidate in enumerate(result.candidate_results, start=1):
        if not candidate.match.is_match:
            continue
        if candidate.kind == "full":
            label = "原图"
        else:
            segment_index += 1
            label = f"切分 {segment_index}"
        payload = serialize_image_candidate(
            candidate,
            label,
            idx,
            full_preview=full_preview,
            full_size=full_size,
            full_image=full_image,
            media_client=media_client,
        )
        candidate_results.append(payload)
        if best_score is None or candidate.score > best_score:
            best_score = candidate.score
            best_bbox = candidate.bbox
            match = payload["match"]
    return DetectionResponse(
        media_type="image",
        is_match=bool(candidate_results),
        best_bbox=best_bbox,
        best_score=best_score,
        candidates=result.candidates,
        match=match,
        query_preview=query_preview,
        query_label=query_label,
        candidate_results=candidate_results,
    ).dict()


def serialize_video_candidate(
    candidate,
    idx: int,
    *,
    include_preview: bool = True,
    media_client: MediaClient | None = None,
) -> dict:
    payload = VideoSegmentResult(
        id=idx,
        obj_id=candidate.obj_id,
        kind=getattr(candidate, "kind", None),
        start_time=candidate.start_time,
        end_time=candidate.end_time,
        first_frame_index=candidate.first_frame_idx,
        last_frame_index=candidate.last_frame_idx,
        bbox=candidate.bbox,
        score=candidate.score,
        match=MatchStats(
            embedding_similarity=candidate.match.embedding_similarity,
            embedding_pass=candidate.match.embedding_pass,
            ransac_ok=candidate.match.ransac_ok,
            inliers=candidate.match.inliers,
            total_matches=candidate.match.total_matches,
        inlier_ratio=candidate.match.inlier_ratio,
        score=candidate.match.score,
        is_match=candidate.match.is_match,
        ),
        preview="",
        full_preview="",
        full_width=None,
        full_height=None,
    ).dict()
    return payload


def serialize_video_result(
    result,
    query_preview: str | None = None,
    query_label: str | None = None,
    *,
    preview_limit: int | None = None,
    media_client: MediaClient | None = None,
) -> dict:
    match = None
    segments = []
    best_score = None
    preview_indices: set[int] | None = None
    if preview_limit is not None and preview_limit > 0:
        if len(result.matches) > preview_limit:
            top_indices = sorted(
                range(len(result.matches)),
                key=lambda idx: result.matches[idx].score,
                reverse=True,
            )[:preview_limit]
            preview_indices = set(top_indices)
    for idx, candidate in enumerate(result.matches, start=1):
        if not candidate.match.is_match:
            continue
        include_preview = (
            True
            if preview_indices is None
            else (idx - 1) in preview_indices
        )
        payload = serialize_video_candidate(
            candidate,
            idx,
            include_preview=include_preview,
            media_client=media_client,
        )
        segments.append(payload)
        if best_score is None or candidate.score > best_score:
            best_score = candidate.score
            match = payload["match"]
    return DetectionResponse(
        media_type="video",
        is_match=bool(segments),
        best_bbox=None,
        best_score=best_score,
        candidates=result.candidates,
        match=match,
        query_preview=query_preview,
        query_label=query_label,
        segments=segments,
        fps=result.fps,
        frame_count=result.frame_count,
        duration=result.duration,
    ).dict()


def apply_threshold(result: dict | None, threshold: float | None) -> dict | None:
    if result is None or threshold is None:
        return result
    try:
        threshold = float(threshold)
    except (TypeError, ValueError):
        return result
    if threshold < 0:
        return result
    if threshold > 1:
        if threshold <= 100:
            threshold = threshold / 100
        else:
            return result

    if "query_results" in result:
        updated_queries = []
        any_match = False
        best_score = None
        best_match = None
        for item in result.get("query_results", []):
            payload = dict(item)
            payload["result"] = apply_threshold(payload.get("result"), threshold)
            updated_queries.append(payload)
            query_result = payload.get("result") or {}
            if query_result.get("is_match"):
                any_match = True
                score = query_result.get("best_score")
                if score is not None and (best_score is None or score > best_score):
                    best_score = score
                    best_match = query_result.get("match")
        return {
            **result,
            "query_results": updated_queries,
            "is_match": any_match,
            "best_score": best_score,
            "match": best_match,
        }

    media_type = result.get("media_type")
    if media_type == "image":
        candidates = [
            item
            for item in result.get("candidate_results", [])
            if item.get("score", 0) >= threshold
        ]
        best = max(candidates, key=lambda item: item.get("score", 0), default=None)
        return {
            **result,
            "candidate_results": candidates,
            "is_match": bool(candidates),
            "best_score": best.get("score") if best else None,
            "best_bbox": best.get("bbox") if best else None,
            "match": best.get("match") if best else None,
        }
    if media_type == "video":
        segments = [
            item for item in result.get("segments", []) if item.get("score", 0) >= threshold
        ]
        best = max(segments, key=lambda item: item.get("score", 0), default=None)
        return {
            **result,
            "segments": segments,
            "is_match": bool(segments),
            "best_score": best.get("score") if best else None,
            "match": best.get("match") if best else None,
        }
    return result


def hydrate_full_preview(
    result: dict | None,
    record: dict | None,
    media_client: MediaClient | None = None,
) -> dict | None:
    # Preview hydration disabled for performance.
    return result
