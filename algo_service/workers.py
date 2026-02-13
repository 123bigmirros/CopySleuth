"""Task worker functions for async image and video detection."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from PIL import Image

from algo_service.serializers import (
    bbox_to_list,
    candidate_payload,
    match_to_payload,
    save_image,
    video_candidate_payload,
)

if TYPE_CHECKING:
    from app.core.tasks import TaskState
    from app.services.algorithm import AlgorithmService

logger = logging.getLogger(__name__)


def image_worker(
    state: TaskState,
    *,
    algorithm: AlgorithmService,
    query_list: list[Image.Image],
    query_origins: list[dict] | None,
    target_image: Image.Image,
    task_dir: Path,
    threshold: float | None,
) -> None:
    """Run synchronous image detection inside a task worker thread."""
    try:
        state.set_progress(0.2, "segmenting", "分割目标图")
        state.check_cancelled()

        total_queries = len(query_list)
        partial_results: list[dict] = []
        completed_payloads: list[dict | None] = []
        next_match_ids: list[int] = []
        for _ in range(total_queries):
            partial_results.append({
                "media_type": "image",
                "is_match": False,
                "best_bbox": None,
                "best_score": None,
                "best_match": None,
                "candidates": 0,
                "candidate_results": [],
            })
            completed_payloads.append(None)
            next_match_ids.append(1)

        def on_candidate(query_index: int, candidate, candidate_idx: int, total: int | None) -> None:
            if total and partial_results[query_index]["candidates"] != total:
                partial_results[query_index]["candidates"] = total
            if not candidate.match.is_match:
                return
            img_id = next_match_ids[query_index]
            next_match_ids[query_index] += 1
            name = f"q{query_index + 1:02d}_{img_id:06d}.png"
            path = task_dir / "candidates" / name
            save_image(path, candidate.image)
            payload = candidate_payload(candidate, image_path=path)
            partial_results[query_index]["candidate_results"].append(payload)
            if (
                partial_results[query_index]["best_score"] is None
                or candidate.score > partial_results[query_index]["best_score"]
            ):
                partial_results[query_index]["best_score"] = float(candidate.score)
                partial_results[query_index]["best_bbox"] = bbox_to_list(candidate.bbox)
                partial_results[query_index]["best_match"] = match_to_payload(candidate.match)
            partial_results[query_index]["is_match"] = True
            if total_queries == 1:
                state.result = dict(partial_results[0])
                if query_origins is not None:
                    state.result["excel_origin"] = query_origins[0]
                state.publish("partial", {"result": state.result})
                return
            result_items = []
            any_match = False
            for item_idx in range(total_queries):
                payload_item = completed_payloads[item_idx]
                if payload_item is None and item_idx == query_index:
                    payload_item = dict(partial_results[item_idx])
                if payload_item is None:
                    continue
                if payload_item.get("is_match"):
                    any_match = True
                ri: dict = {"query_index": item_idx + 1, "result": payload_item}
                if query_origins is not None:
                    ri["excel_origin"] = query_origins[item_idx]
                result_items.append(ri)
            state.result = {
                "media_type": "multi",
                "is_match": any_match,
                "query_results": result_items,
            }
            state.publish("partial", {"result": state.result})

        def on_progress(query_idx: int, candidate_idx: int, total_queries_: int, total_candidates: int) -> None:
            span = 0.6 / max(total_queries_, 1)
            progress = 0.2 + span * (query_idx - 1) + span * (
                candidate_idx / max(total_candidates, 1)
            )
            state.set_progress(progress, "matching", "相似度匹配中")

        batch_results = algorithm.detect_image_batch(
            query_list,
            target_image,
            progress_cb=on_progress,
            candidate_cb=on_candidate,
            cancel_cb=state.is_cancelled,
            embedding_threshold=threshold,
        )
        state.check_cancelled()

        query_results = []
        overall_best_score = None
        overall_match = None
        for idx, result in enumerate(batch_results, start=1):
            payload = dict(partial_results[idx - 1])
            payload["best_bbox"] = bbox_to_list(result.best_bbox)
            payload["best_score"] = (
                float(result.best_score) if result.best_score is not None else None
            )
            payload["best_match"] = match_to_payload(result.best_match)
            payload["candidates"] = int(result.candidates)
            completed_payloads[idx - 1] = payload
            query_item: dict = {"query_index": idx, "result": payload}
            if query_origins is not None:
                query_item["excel_origin"] = query_origins[idx - 1]
            query_results.append(query_item)
            score = payload.get("best_score")
            if score is not None and (
                overall_best_score is None or score > overall_best_score
            ):
                overall_best_score = score
                overall_match = payload.get("best_match")

        if total_queries == 1:
            final_payload = query_results[0]["result"]
            if query_origins is not None:
                final_payload["excel_origin"] = query_origins[0]
        else:
            final_payload = {
                "media_type": "multi",
                "is_match": any(item["result"].get("is_match") for item in query_results),
                "best_score": overall_best_score,
                "match": overall_match,
                "query_results": query_results,
            }
        state.result = final_payload
        state.publish("result", {"result": final_payload})
    finally:
        for img in query_list:
            try:
                img.close()
            except Exception:
                pass
        query_list.clear()
        if target_image is not None:
            try:
                target_image.close()
            except Exception:
                pass


def video_worker(
    state: TaskState,
    *,
    algorithm: AlgorithmService,
    query_list: list[Image.Image],
    query_origins: list[dict] | None,
    target_path: Path,
    task_dir: Path,
    threshold: float | None,
    video_start: float | None,
    video_end: float | None,
) -> None:
    """Run asynchronous video detection inside a task worker thread."""
    from app.core.tasks import TaskCancelled

    try:
        state.set_progress(0.08, "loading", "保存视频")
        state.check_cancelled()

        def on_progress(frame_idx: int, total_frames: int, stage: str = "segmenting") -> None:
            if state.is_cancelled():
                raise TaskCancelled("Task canceled")
            if stage == "keyframe":
                progress = 0.05 * (frame_idx / max(total_frames, 1))
                state.set_progress(progress, "keyframe", "关键帧提取中")
                return
            progress = 0.05 + 0.65 * (frame_idx / max(total_frames, 1))
            state.set_progress(progress, "segmenting", "视频关键帧分割中")

        cache_dir = task_dir / "cache"
        results = algorithm.detect_video_batch(
            query_list,
            target_path,
            start_time=video_start,
            end_time=video_end,
            progress_cb=on_progress,
            candidate_cb=None,
            cancel_cb=state.is_cancelled,
            embedding_threshold=threshold,
            cache_dir=cache_dir,
        )
        state.check_cancelled()
        state.set_progress(0.75, "matching", "匹配候选目标")

        query_results = []
        for idx, result in enumerate(results, start=1):
            segments = []
            best_score = None
            best_match = None
            for candidate in result.matches:
                if not candidate.match.is_match:
                    continue
                if isinstance(candidate.image, (str, Path)):
                    image_path = Path(candidate.image)
                else:
                    image_path = (
                        task_dir / "segments" / f"seg_{idx:02d}_{len(segments) + 1:06d}.png"
                    )
                    save_image(image_path, candidate.image)
                if isinstance(candidate.full_image, (str, Path)):
                    full_image_path = Path(candidate.full_image)
                else:
                    full_image_path = (
                        task_dir / "frames" / f"frame_{idx:02d}_{len(segments) + 1:06d}.png"
                    )
                    save_image(full_image_path, candidate.full_image)
                segments.append(
                    video_candidate_payload(
                        candidate,
                        image_path=image_path,
                        full_image_path=full_image_path,
                    )
                )
                if best_score is None or candidate.score > best_score:
                    best_score = candidate.score
                    best_match = candidate.match
            payload: dict = {
                "media_type": "video",
                "is_match": bool(segments),
                "best_score": float(best_score) if best_score is not None else None,
                "best_match": match_to_payload(best_match),
                "candidates": int(result.candidates),
                "segments": segments,
                "fps": float(result.fps),
                "frame_count": int(result.frame_count),
                "duration": float(result.duration),
            }
            query_item: dict = {"query_index": idx, "result": payload}
            if query_origins is not None:
                query_item["excel_origin"] = query_origins[idx - 1]
            query_results.append(query_item)

        if len(query_results) == 1:
            final_payload = query_results[0]["result"]
            if query_origins is not None:
                final_payload["excel_origin"] = query_origins[0]
        else:
            overall_best_score = None
            overall_match = None
            any_match = False
            for item in query_results:
                result_payload = item["result"]
                if result_payload.get("is_match"):
                    any_match = True
                score = result_payload.get("best_score")
                if score is not None and (
                    overall_best_score is None or score > overall_best_score
                ):
                    overall_best_score = score
                    overall_match = result_payload.get("best_match")
            final_payload = {
                "media_type": "multi",
                "is_match": any_match,
                "best_score": overall_best_score,
                "match": overall_match,
                "query_results": query_results,
            }
        state.result = final_payload
        state.publish("result", {"result": final_payload})
    except TaskCancelled:
        raise
    finally:
        for img in query_list:
            try:
                img.close()
            except Exception:
                pass
        query_list.clear()
