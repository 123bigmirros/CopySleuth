from __future__ import annotations

from fastapi import APIRouter, File, UploadFile, HTTPException, Request, Form, Query, FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
import json
import os
import time
import tempfile
import uuid
from pathlib import Path

from app.core.tasks import sse_event_stream, TaskState, TaskCancelled
from app.api.utils import (
    load_excel_queries,
    load_excel_keywords,
    load_image_bytes,
    load_image_file,
)
from app.domain.results import (
    apply_threshold,
    hydrate_full_preview,
    serialize_image_candidate,
    serialize_image_result,
    serialize_video_result,
)
from app.schemas import (
    DetectionResponse,
    MatchStats,
    MultiDetectionResponse,
    TaskResponse,
    TaskStatusResponse,
)
from app.services.algorithm_api import parse_image_result, parse_video_result


router = APIRouter()


def _image_preview_with_size(
    image,
    max_size: int = 640,
    media_client=None,
) -> tuple[str, int, int]:
    # Preview generation is disabled for performance.
    if image is None:
        return "", 0, 0
    try:
        width, height = image.size
    except Exception:
        return "", 0, 0
    return "", int(width), int(height)


def _image_preview(image, max_size: int = 256, media_client=None) -> str:
    preview, _, _ = _image_preview_with_size(
        image,
        max_size=max_size,
        media_client=media_client,
    )
    return preview


def _scale_lines(lines: list[dict], scale_x: float, scale_y: float) -> list[dict]:
    if not lines:
        return []
    output = []
    for line in lines:
        if not isinstance(line, dict):
            continue
        updated = dict(line)
        bbox = line.get("bbox")
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            updated["bbox"] = [
                bbox[0] * scale_x,
                bbox[1] * scale_y,
                bbox[2] * scale_x,
                bbox[3] * scale_y,
            ]
        polygon = line.get("polygon")
        if isinstance(polygon, (list, tuple)):
            scaled_polygon = []
            ok = True
            for point in polygon:
                if not isinstance(point, (list, tuple)) or len(point) < 2:
                    ok = False
                    break
                scaled_polygon.append([point[0] * scale_x, point[1] * scale_y])
            if ok:
                updated["polygon"] = scaled_polygon
        output.append(updated)
    return output


def _is_excel_upload(filename: str | None, content_type: str | None) -> bool:
    name = (filename or "").lower()
    content = (content_type or "").lower()
    if name.endswith((".xlsx", ".xlsm")):
        return True
    return "spreadsheetml" in content or "ms-excel" in content


def _format_keywords_preview(keywords: list[str], max_items: int = 6, max_len: int = 80) -> str:
    if not keywords:
        return ""
    preview = ", ".join(keywords[:max_items])
    if len(keywords) > max_items:
        preview = f"{preview} ...(共{len(keywords)}项)"
    if len(preview) > max_len:
        preview = preview[:max_len].rstrip() + "..."
    return preview


def _run_ocr_match(
    app,
    target_path: Path,
    keywords: list[str],
    task_id: str | None = None,
    target_image=None,
) -> dict:
    try:
        from app.core.gpu_lock import gpu_lock

        with gpu_lock():
            payload = app.state.ocr.match_image_keywords(target_path, keywords)
    except Exception as exc:  # noqa: BLE001
        payload = {
            "enabled": True,
            "keywords": keywords,
            "keyword_count": len(keywords),
            "texts": [],
            "text": "",
            "matches": [],
            "match_count": 0,
            "is_match": False,
            "positions": None,
            "error": str(exc),
        }
    if target_image is not None and payload.get("positions"):
        positions = payload.get("positions")
        if isinstance(positions, dict):
            preview, preview_w, preview_h = _image_preview_with_size(
                target_image,
                max_size=640,
                media_client=app.state.media,
            )
            if preview:
                orig_w, orig_h = target_image.size
                scale_x = preview_w / orig_w if orig_w else 1.0
                scale_y = preview_h / orig_h if orig_h else 1.0
                lines = positions.get("lines") or []
                if lines and (scale_x != 1.0 or scale_y != 1.0):
                    positions["lines"] = _scale_lines(lines, scale_x, scale_y)
                positions["preview"] = preview
                positions["width"] = preview_w
                positions["height"] = preview_h
                payload["positions"] = positions
    if task_id:
        try:
            app.state.ocr_store.save(task_id, payload)
        except Exception:  # noqa: BLE001
            pass
    return payload


def _run_ocr_video_match(app, video_path: Path, keywords: list[str], task_id: str | None = None) -> dict:
    try:
        from app.core.gpu_lock import gpu_lock

        with gpu_lock():
            payload = app.state.ocr.match_video_keywords(video_path, keywords)
    except Exception as exc:  # noqa: BLE001
        payload = {
            "enabled": True,
            "keywords": keywords,
            "keyword_count": len(keywords),
            "texts": [],
            "text": "",
            "matches": [],
            "match_count": 0,
            "is_match": False,
            "video": None,
            "error": str(exc),
        }
    if task_id:
        try:
            app.state.ocr_store.save(task_id, payload)
        except Exception:  # noqa: BLE001
            pass
    return payload


def _attach_ocr_if_missing(result: dict | None, task_id: str, app: FastAPI) -> dict | None:
    if isinstance(result, dict) and "ocr" not in result:
        ocr_saved = app.state.ocr_store.load(task_id)
        if ocr_saved:
            result = dict(result)
            result["ocr"] = ocr_saved
    return result


def _render_algo_image_result(
    raw_result: dict,
    *,
    query_preview: str | None,
    query_label: str | None,
    full_preview: str | None,
    full_size: tuple[int, int] | None,
    full_image,
    media_client,
) -> dict:
    result = parse_image_result(raw_result)
    return serialize_image_result(
        result,
        full_preview=full_preview,
        full_size=full_size,
        full_image=full_image,
        query_preview=query_preview,
        query_label=query_label,
        media_client=media_client,
    )


def _render_algo_video_result(
    raw_result: dict,
    *,
    query_preview: str | None,
    query_label: str | None,
    media_client,
    preview_limit: int | None = None,
) -> dict:
    result = parse_video_result(raw_result)
    return serialize_video_result(
        result,
        query_preview=query_preview,
        query_label=query_label,
        preview_limit=preview_limit,
        media_client=media_client,
    )


def _render_algo_payload(
    raw_result: dict,
    *,
    query_previews: list[str],
    query_labels: list[str],
    query_ids: list[int],
    full_preview: str | None,
    full_size: tuple[int, int] | None,
    full_image,
    media_client,
    include_best: bool,
    preview_limit: int | None = None,
) -> dict:
    media_type = raw_result.get("media_type")
    if media_type == "multi":
        query_results = []
        any_match = False
        best_score = None
        best_match = None
        for item in raw_result.get("query_results", []):
            query_index = int(item.get("query_index", 0))
            if query_index <= 0 or query_index > len(query_previews):
                continue
            idx = query_index - 1
            result_payload = item.get("result") or {}
            result_media = result_payload.get("media_type") or "image"
            if result_media == "video":
                payload = _render_algo_video_result(
                    result_payload,
                    query_preview=query_previews[idx],
                    query_label=query_labels[idx],
                    media_client=media_client,
                    preview_limit=preview_limit,
                )
            else:
                payload = _render_algo_image_result(
                    result_payload,
                    query_preview=query_previews[idx],
                    query_label=query_labels[idx],
                    full_preview=full_preview,
                    full_size=full_size,
                    full_image=full_image,
                    media_client=media_client,
                )
            query_results.append(
                {
                    "query_id": query_ids[idx],
                    "query_label": query_labels[idx],
                    "query_preview": query_previews[idx],
                    "result": payload,
                }
            )
            if payload.get("is_match"):
                any_match = True
            score = payload.get("best_score")
            if score is not None and (best_score is None or score > best_score):
                best_score = score
                best_match = payload.get("match")
        output = {
            "media_type": "multi",
            "is_match": any_match,
            "query_results": query_results,
        }
        if include_best:
            output["best_score"] = raw_result.get("best_score", best_score)
            output["match"] = raw_result.get("match", best_match)
        return output
    if media_type == "video":
        return _render_algo_video_result(
            raw_result,
            query_preview=query_previews[0] if query_previews else None,
            query_label=query_labels[0] if query_labels else None,
            media_client=media_client,
            preview_limit=preview_limit,
        )
    return _render_algo_image_result(
        raw_result,
        query_preview=query_previews[0] if query_previews else None,
        query_label=query_labels[0] if query_labels else None,
        full_preview=full_preview,
        full_size=full_size,
        full_image=full_image,
        media_client=media_client,
    )


@router.get("/health")
def health() -> dict:
    return {"status": "ok"}


@router.post("/v1/detect", response_model=DetectionResponse | MultiDetectionResponse)
async def detect(
    request: Request,
    query_image: UploadFile = File(...),
    target_image: UploadFile = File(...),
    use_ocr: bool = Form(False),
):
    pipeline = request.app.state.algorithm_api or request.app.state.algorithm
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Algorithm service not available")
    media_client = request.app.state.media
    target = load_image_file(target_image)
    is_excel = _is_excel_upload(query_image.filename, query_image.content_type)
    ocr_payload = None

    if is_excel:
        query_bytes = await query_image.read()
        queries = load_excel_queries(query_bytes, query_image.filename)
        query_images = []
        query_labels = []
        query_previews = []
        query_ids = []
        for idx, (row_index, row_text, query_image_obj) in enumerate(queries, start=1):
            query_images.append(query_image_obj)
            query_labels.append(row_text or f"第{row_index}行")
            query_previews.append(_image_preview(query_image_obj, max_size=256, media_client=media_client))
            query_ids.append(idx)

        if use_ocr:
            keywords = load_excel_keywords(query_bytes, query_image.filename)
            with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
                target.save(tmp.name)
                ocr_payload = _run_ocr_match(
                    request.app, Path(tmp.name), keywords, target_image=target
                )

        batch_results = pipeline.detect_image_batch(query_images, target)
        full_preview = _image_preview(target, max_size=640, media_client=media_client)
        full_size = target.size
        query_results = []
        overall_best_score = None
        overall_match = None
        for idx, result in enumerate(batch_results, start=1):
            payload = serialize_image_result(
                result,
                full_preview=full_preview,
                full_size=full_size,
                full_image=target,
                query_preview=query_previews[idx - 1],
                query_label=query_labels[idx - 1],
                media_client=media_client,
            )
            query_results.append(
                {
                    "query_id": query_ids[idx - 1],
                    "query_label": query_labels[idx - 1],
                    "query_preview": query_previews[idx - 1],
                    "result": payload,
                }
            )
            if payload.get("best_score") is not None and (
                overall_best_score is None or payload["best_score"] > overall_best_score
            ):
                overall_best_score = payload["best_score"]
                overall_match = payload.get("match")

        if len(query_results) == 1:
            payload = query_results[0]["result"]
            if ocr_payload is not None:
                payload["ocr"] = ocr_payload
            return payload
        payload = {
            "media_type": "multi",
            "is_match": any(item["result"].get("is_match") for item in query_results),
            "best_score": overall_best_score,
            "match": overall_match,
            "query_results": query_results,
            "ocr": ocr_payload,
        }
        return payload

    query = load_image_file(query_image)
    result = pipeline.detect_image(query, target)
    match = None
    if result.best_match is not None:
        match = MatchStats(
            embedding_similarity=result.best_match.embedding_similarity,
            embedding_pass=result.best_match.embedding_pass,
            ransac_ok=result.best_match.ransac_ok,
            inliers=result.best_match.inliers,
            total_matches=result.best_match.total_matches,
            inlier_ratio=result.best_match.inlier_ratio,
            score=result.best_match.score,
            is_match=result.best_match.is_match,
        )

    candidate_results = []
    segment_index = 0
    full_preview = _image_preview(target, max_size=640, media_client=media_client)
    full_size = target.size
    for idx, candidate in enumerate(result.candidate_results, start=1):
        if not candidate.match.is_match:
            continue
        if candidate.kind == "full":
            label = "原图"
        else:
            segment_index += 1
            label = f"切分 {segment_index}"
        candidate_payload = serialize_image_candidate(
            candidate,
            label,
            idx,
            full_preview=full_preview,
            full_size=full_size,
            full_image=target,
            media_client=media_client,
        )
        candidate_results.append(candidate_payload)

    return DetectionResponse(
        media_type="image",
        is_match=result.is_match,
        best_bbox=result.best_bbox,
        best_score=result.best_score,
        candidates=result.candidates,
        match=match,
        candidate_results=candidate_results,
        ocr=ocr_payload,
    )


@router.post("/v1/tasks", response_model=TaskResponse)
async def create_task(
    request: Request,
    query_image: UploadFile = File(...),
    target_file: UploadFile = File(...),
    video_start: float | None = Form(None),
    video_end: float | None = Form(None),
    match_threshold: float | None = Form(None),
    embedding_threshold: float | None = Form(None),
    task_name: str | None = Form(None),
    use_ocr: bool | None = Form(False),
):
    content_type = (target_file.content_type or "").lower()
    query_content_type = (query_image.content_type or "").lower()
    app = request.app
    tasks = app.state.tasks
    try:
        threshold = 0.95 if match_threshold is None else float(match_threshold)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="Invalid match_threshold") from exc
    if threshold < 0 or threshold > 1:
        raise HTTPException(status_code=400, detail="Invalid match_threshold")
    try:
        embedding = None if embedding_threshold is None else float(embedding_threshold)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="Invalid embedding_threshold") from exc
    if embedding is not None and embedding > 1 and embedding <= 100:
        embedding = embedding / 100.0
    if embedding is not None and (embedding < 0 or embedding > 1):
        raise HTTPException(status_code=400, detail="Invalid embedding_threshold")
    name = (task_name or "").strip()
    if not name:
        suffix = time.strftime("%Y%m%d-%H%M%S")
        base_name = target_file.filename or "task"
        name = f"{base_name}-{suffix}"

    data_dir = Path(os.getenv("BACKEND_DATA_DIR", "./data/backend"))
    data_dir.mkdir(parents=True, exist_ok=True)
    is_video = content_type.startswith("video/")
    is_excel = _is_excel_upload(query_image.filename, query_content_type)
    task_id = str(uuid.uuid4())

    if is_excel:
        query_suffix = Path(query_image.filename or "").suffix or ".xlsx"
    else:
        query_suffix = Path(query_image.filename or "").suffix or ".png"
    query_path = data_dir / f"{task_id}_query{query_suffix}"
    target_name = target_file.filename or ("target.mp4" if is_video else "target.png")
    target_suffix = Path(target_name).suffix or (".mp4" if is_video else ".png")
    target_content_type = target_file.content_type or "application/octet-stream"
    target_path = data_dir / f"{task_id}_target{target_suffix}"

    query_bytes = await query_image.read()
    query_path.write_bytes(query_bytes)
    target_bytes = None
    if is_video:
        with open(target_path, "wb") as target_fp:
            while True:
                chunk = await target_file.read(8 * 1024 * 1024)
                if not chunk:
                    break
                target_fp.write(chunk)
    else:
        target_bytes = await target_file.read()
        target_path.write_bytes(target_bytes)

    def worker(state: TaskState) -> None:
        try:
            state.set_progress(0.05, "loading", "解析输入")
            state.check_cancelled()

            algo_tasks = getattr(app.state, "algorithm_tasks", None)
            default_embedding = getattr(
                getattr(app.state, "settings", None),
                "embedding_threshold",
                0.5,
            )
            if embedding is None and app.state.algorithm is not None:
                default_embedding = getattr(
                    getattr(app.state.algorithm, "_settings", None),
                    "embedding_threshold",
                    default_embedding,
                )
            app.state.db.create_task(
                state.task_id,
                name,
                "video" if is_video else "image",
                query_path,
                target_path,
                threshold,
                embedding if embedding is not None else default_embedding,
            )

            enable_ocr = bool(use_ocr) and is_excel
            ocr_payload = None
            target_image_for_ocr = None
            if enable_ocr and not is_video:
                target_image_for_ocr = load_image_bytes(target_bytes or b"", target_file.filename)
            if enable_ocr:
                keywords = load_excel_keywords(query_bytes, query_image.filename)
                keyword_preview = _format_keywords_preview(keywords)
                keyword_message = (
                    f"OCR关键词: {keyword_preview}" if keyword_preview else "OCR关键词: (空)"
                )
                state.set_ocr_progress(0.01, "keywords", keyword_message)
                state.set_progress(0.12, "ocr", "OCR 文字识别中")
                if is_video:
                    ocr_payload = _run_ocr_video_match(
                        app, target_path, keywords, task_id=state.task_id
                    )
                else:
                    ocr_payload = _run_ocr_match(
                        app,
                        target_path,
                        keywords,
                        task_id=state.task_id,
                        target_image=target_image_for_ocr,
                    )
                if ocr_payload.get("error"):
                    state.set_ocr_progress(1.0, "error", f"OCR 失败: {ocr_payload['error']}")
                else:
                    state.set_ocr_progress(1.0, "done", "OCR 完成")
                state.result = {"ocr": ocr_payload}
                state.publish("partial", {"result": state.result})

            if is_video:
                if algo_tasks is None:
                    state.set_progress(0.08, "loading", "保存视频")
                    state.check_cancelled()

                query_images = []
                query_labels = []
                query_previews = []
                query_ids = []
                if is_excel:
                    queries = load_excel_queries(query_bytes, query_image.filename)
                    for idx, (row_index, row_text, query_image_obj) in enumerate(
                        queries, start=1
                    ):
                        query_images.append(query_image_obj)
                        query_labels.append(row_text or f"第{row_index}行")
                        query_previews.append(
                            _image_preview(
                                query_image_obj,
                                max_size=256,
                                media_client=app.state.media,
                            )
                        )
                        query_ids.append(idx)
                else:
                    query = load_image_bytes(query_bytes, query_image.filename)
                    query_images.append(query)
                    query_labels.append(query_image.filename or "查询图")
                    query_previews.append(
                        _image_preview(query, max_size=256, media_client=app.state.media)
                    )
                    query_ids.append(1)

                if algo_tasks is not None:
                    state.check_cancelled()
                    target_payload = target_path.read_bytes()
                    algo_task_id = algo_tasks.create_task(
                        query_images=query_images,
                        target_bytes=target_payload,
                        target_name=target_name,
                        target_content_type=target_content_type,
                        embedding_threshold=embedding,
                        video_start=video_start,
                        video_end=video_end,
                        query_excel_bytes=query_bytes if is_excel else None,
                        query_excel_name=query_image.filename if is_excel else None,
                    )
                    setattr(state, "remote_task_id", algo_task_id)
                    for event in algo_tasks.iter_events(algo_task_id):
                        if state.is_cancelled():
                            algo_tasks.cancel_task(algo_task_id)
                            raise TaskCancelled("Task canceled")
                        event_type = event.get("event")
                        data = event.get("data") or {}
                        if event_type in {None, "keep-alive"}:
                            continue
                        if event_type == "progress":
                            state.set_progress(
                                float(data.get("progress", 0.0)),
                                data.get("stage", ""),
                                data.get("message", ""),
                            )
                            continue
                        if event_type in {"partial", "result"}:
                            raw_result = data.get("result")
                            if not isinstance(raw_result, dict):
                                continue
                            payload = _render_algo_payload(
                                raw_result,
                                query_previews=query_previews,
                                query_labels=query_labels,
                                query_ids=query_ids,
                                full_preview=None,
                                full_size=None,
                                full_image=None,
                                media_client=app.state.media,
                                include_best=event_type == "result",
                                preview_limit=120,
                            )
                            if ocr_payload is not None:
                                payload = dict(payload)
                                payload["ocr"] = ocr_payload
                            state.result = payload
                            state.publish(event_type, {"result": payload})
                            if event_type == "result":
                                app.state.db.update_result(state.task_id, payload)
                            continue
                        if event_type == "error":
                            raise RuntimeError(data.get("message") or "Algorithm task failed")
                        if event_type == "canceled":
                            raise TaskCancelled("Task canceled")
                        if event_type == "done":
                            break
                    return

                pipeline = app.state.algorithm

                total_queries = len(query_images)
                query_results = []
                overall_best_score = None
                overall_match = None

                partial_results = []
                next_segment_ids = []
                completed_payloads: list[dict | None] = []
                for idx in range(total_queries):
                    partial_results.append(
                        {
                            "media_type": "video",
                            "is_match": False,
                            "best_bbox": None,
                            "best_score": None,
                            "candidates": 0,
                            "match": None,
                            "segments": [],
                            "fps": None,
                            "frame_count": None,
                            "duration": None,
                            "query_preview": query_previews[idx],
                            "query_label": query_labels[idx],
                            "ocr": ocr_payload,
                        }
                    )
                    next_segment_ids.append(1)
                    completed_payloads.append(None)

                def on_progress(frame_idx: int, total_frames: int, stage: str = "segmenting") -> None:
                    if state.is_cancelled():
                        raise TaskCancelled("Task canceled")
                    if stage == "keyframe":
                        progress = 0.05 * (frame_idx / max(total_frames, 1))
                        state.set_progress(progress, "keyframe", "关键帧提取中")
                        return
                    progress = 0.05 + 0.65 * (frame_idx / max(total_frames, 1))
                    state.set_progress(progress, "segmenting", "视频关键帧分割中")

                def on_candidate(query_index: int, candidate) -> None:
                    return

                with tempfile.TemporaryDirectory(prefix="video_candidates_") as tmpdir:
                    cache_dir = Path(tmpdir)
                    batch_results = pipeline.detect_video_batch(
                        query_images,
                        target_path,
                        start_time=video_start,
                        end_time=video_end,
                        progress_cb=on_progress,
                        candidate_cb=None,
                        cancel_cb=state.is_cancelled,
                        embedding_threshold=embedding,
                        cache_dir=cache_dir,
                    )
                    state.check_cancelled()
                    state.set_progress(0.75, "matching", "匹配候选目标")
                    for idx, result in enumerate(batch_results, start=1):
                        payload = serialize_video_result(
                            result,
                            query_preview=query_previews[idx - 1],
                            query_label=query_labels[idx - 1],
                            preview_limit=120,
                            media_client=app.state.media,
                        )
                        completed_payloads[idx - 1] = payload
                        query_results.append(
                            {
                                "query_id": query_ids[idx - 1],
                                "query_label": query_labels[idx - 1],
                                "query_preview": query_previews[idx - 1],
                                "result": payload,
                            }
                        )
                        if payload.get("best_score") is not None and (
                            overall_best_score is None
                            or payload["best_score"] > overall_best_score
                        ):
                            overall_best_score = payload["best_score"]
                            overall_match = payload.get("match")

                state.check_cancelled()
                state.set_progress(0.75, "matching", "汇总匹配结果")
                if total_queries == 1:
                    payload = query_results[0]["result"]
                    if ocr_payload is not None:
                        payload["ocr"] = ocr_payload
                else:
                    payload = {
                        "media_type": "multi",
                        "is_match": any(item["result"].get("is_match") for item in query_results),
                        "best_score": overall_best_score,
                        "match": overall_match,
                        "query_results": query_results,
                        "ocr": ocr_payload,
                    }
                state.result = payload
                state.publish("result", {"result": payload})
                app.state.db.update_result(state.task_id, payload)
            else:
                target = target_image_for_ocr or load_image_bytes(
                    target_bytes or b"", target_file.filename
                )
                if algo_tasks is None:
                    state.set_progress(0.2, "segmenting", "分割目标图")
                    state.check_cancelled()
                full_preview = _image_preview(target, max_size=640, media_client=app.state.media)
                full_size = target.size

                query_images = []
                query_labels = []
                query_previews = []
                query_ids = []
                if is_excel:
                    queries = load_excel_queries(query_bytes, query_image.filename)
                    for idx, (row_index, row_text, query_image_obj) in enumerate(
                        queries, start=1
                    ):
                        query_images.append(query_image_obj)
                        query_labels.append(row_text or f"第{row_index}行")
                        query_previews.append(
                            _image_preview(query_image_obj, max_size=256, media_client=app.state.media)
                        )
                        query_ids.append(idx)
                else:
                    query = load_image_bytes(query_bytes, query_image.filename)
                    query_images.append(query)
                    query_labels.append(query_image.filename or "查询图")
                    query_previews.append(
                        _image_preview(query, max_size=256, media_client=app.state.media)
                    )
                    query_ids.append(1)

                if algo_tasks is not None:
                    state.check_cancelled()
                    algo_task_id = algo_tasks.create_task(
                        query_images=query_images,
                        target_bytes=target_bytes or target_path.read_bytes(),
                        target_name=target_name,
                        target_content_type=target_content_type,
                        embedding_threshold=embedding,
                        video_start=None,
                        video_end=None,
                        query_excel_bytes=query_bytes if is_excel else None,
                        query_excel_name=query_image.filename if is_excel else None,
                    )
                    setattr(state, "remote_task_id", algo_task_id)
                    for event in algo_tasks.iter_events(algo_task_id):
                        if state.is_cancelled():
                            algo_tasks.cancel_task(algo_task_id)
                            raise TaskCancelled("Task canceled")
                        event_type = event.get("event")
                        data = event.get("data") or {}
                        if event_type in {None, "keep-alive"}:
                            continue
                        if event_type == "progress":
                            state.set_progress(
                                float(data.get("progress", 0.0)),
                                data.get("stage", ""),
                                data.get("message", ""),
                            )
                            continue
                        if event_type in {"partial", "result"}:
                            raw_result = data.get("result")
                            if not isinstance(raw_result, dict):
                                continue
                            payload = _render_algo_payload(
                                raw_result,
                                query_previews=query_previews,
                                query_labels=query_labels,
                                query_ids=query_ids,
                                full_preview=full_preview,
                                full_size=full_size,
                                full_image=target,
                                media_client=app.state.media,
                                include_best=event_type == "result",
                                preview_limit=None,
                            )
                            if ocr_payload is not None:
                                payload = dict(payload)
                                payload["ocr"] = ocr_payload
                            state.result = payload
                            state.publish(event_type, {"result": payload})
                            if event_type == "result":
                                app.state.db.update_result(state.task_id, payload)
                            continue
                        if event_type == "error":
                            raise RuntimeError(data.get("message") or "Algorithm task failed")
                        if event_type == "canceled":
                            raise TaskCancelled("Task canceled")
                        if event_type == "done":
                            break
                    return

                total_queries = len(query_images)
                query_results = []
                overall_best_score = None
                overall_match = None

                partial_results = []
                match_segment_indices = []
                next_match_ids = []
                completed_payloads: list[dict | None] = []

                for idx in range(total_queries):
                    partial_results.append(
                        {
                            "media_type": "image",
                            "is_match": False,
                            "best_bbox": None,
                            "best_score": None,
                            "candidates": 0,
                            "match": None,
                            "candidate_results": [],
                            "query_preview": query_previews[idx],
                            "query_label": query_labels[idx],
                            "ocr": ocr_payload,
                        }
                    )
                    match_segment_indices.append(0)
                    next_match_ids.append(1)
                    completed_payloads.append(None)

                def on_candidate(query_index, candidate, candidate_idx, total):
                    if total and partial_results[query_index]["candidates"] != total:
                        partial_results[query_index]["candidates"] = total
                    if candidate.match.is_match:
                        if candidate.kind == "full":
                            label = "原图"
                        else:
                            match_segment_indices[query_index] += 1
                            label = f"切分 {match_segment_indices[query_index]}"
                        candidate_payload = serialize_image_candidate(
                            candidate,
                            label,
                            next_match_ids[query_index],
                            full_preview=full_preview,
                            full_size=full_size,
                            full_image=target,
                            media_client=app.state.media,
                        )
                        next_match_ids[query_index] += 1
                        partial_results[query_index]["candidate_results"].append(
                            candidate_payload
                        )
                        if (
                            partial_results[query_index]["best_score"] is None
                            or candidate.score > partial_results[query_index]["best_score"]
                        ):
                            partial_results[query_index]["best_score"] = candidate.score
                            partial_results[query_index]["best_bbox"] = candidate.bbox
                            partial_results[query_index]["match"] = candidate_payload["match"]
                        partial_results[query_index]["is_match"] = True
                        if total_queries == 1:
                            state.result = dict(partial_results[0])
                            state.publish("partial", {"result": state.result})
                            return
                        result_items = []
                        any_match = False
                        for item_idx in range(total_queries):
                            payload = completed_payloads[item_idx]
                            if payload is None and item_idx == query_index:
                                payload = dict(partial_results[item_idx])
                            if payload is None:
                                continue
                            if payload.get("is_match"):
                                any_match = True
                            result_items.append(
                                {
                                    "query_id": query_ids[item_idx],
                                    "query_label": query_labels[item_idx],
                                    "query_preview": query_previews[item_idx],
                                    "result": payload,
                                }
                            )
                        state.result = {
                            "media_type": "multi",
                            "is_match": any_match,
                            "query_results": result_items,
                            "ocr": ocr_payload,
                        }
                        state.publish("partial", {"result": state.result})

                def on_progress(query_idx, candidate_idx, total_queries, total_candidates):
                    span = 0.6 / max(total_queries, 1)
                    progress = 0.2 + span * (query_idx - 1) + span * (
                        candidate_idx / max(total_candidates, 1)
                    )
                    state.set_progress(progress, "matching", "相似度匹配中")

                batch_results = app.state.algorithm.detect_image_batch(
                    query_images,
                    target,
                    progress_cb=on_progress,
                    candidate_cb=on_candidate,
                    cancel_cb=state.is_cancelled,
                    embedding_threshold=embedding,
                )
                state.check_cancelled()

                for idx, result in enumerate(batch_results, start=1):
                    payload = serialize_image_result(
                        result,
                        full_preview=full_preview,
                        full_size=full_size,
                        full_image=target,
                        query_preview=query_previews[idx - 1],
                        query_label=query_labels[idx - 1],
                        media_client=app.state.media,
                    )
                    completed_payloads[idx - 1] = payload
                    query_results.append(
                        {
                            "query_id": query_ids[idx - 1],
                            "query_label": query_labels[idx - 1],
                            "query_preview": query_previews[idx - 1],
                            "result": payload,
                        }
                    )
                    if payload.get("best_score") is not None and (
                        overall_best_score is None or payload["best_score"] > overall_best_score
                    ):
                        overall_best_score = payload["best_score"]
                        overall_match = payload.get("match")

                state.check_cancelled()
                state.set_progress(0.8, "matching", "相似度匹配")
                if total_queries == 1:
                    payload = query_results[0]["result"]
                    if ocr_payload is not None:
                        payload["ocr"] = ocr_payload
                else:
                    payload = {
                        "media_type": "multi",
                        "is_match": any(item["result"].get("is_match") for item in query_results),
                        "best_score": overall_best_score,
                        "match": overall_match,
                        "query_results": query_results,
                        "ocr": ocr_payload,
                    }
                state.result = payload
                state.publish("result", {"result": payload})
                app.state.db.update_result(state.task_id, payload)
        except TaskCancelled:
            app.state.db.update_status(state.task_id, "canceled")
            raise
        except Exception as exc:
            app.state.db.update_error(state.task_id, str(exc))
            raise

    task_state = tasks.create_with_id(task_id, worker)
    return TaskResponse(task_id=task_state.task_id)


@router.get("/v1/tasks/{task_id}", response_model=TaskStatusResponse)
def get_task(task_id: str, request: Request, threshold: float | None = Query(None)):
    tasks = request.app.state.tasks
    state = tasks.get(task_id)
    if state is None:
        record = request.app.state.db.get_task(task_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Task not found")
        result = record.get("result_json")
        parsed = json.loads(result) if result else None
        parsed = _attach_ocr_if_missing(parsed, task_id, request.app)
        hydrated = hydrate_full_preview(parsed, record, request.app.state.media)
        filtered = apply_threshold(hydrated, threshold)
        return TaskStatusResponse(
            task_id=task_id,
            status=record.get("status", "done"),
            progress=1.0 if record.get("status") == "done" else 0.0,
            ocr_progress=None,
            ocr_stage=None,
            ocr_message=None,
            error=record.get("error"),
            result=filtered,
        )
    result = _attach_ocr_if_missing(state.result, task_id, request.app)
    result = apply_threshold(result, threshold)
    return TaskStatusResponse(
        task_id=task_id,
        status=state.status,
        progress=state.progress,
        ocr_progress=state.ocr_progress,
        ocr_stage=state.ocr_stage,
        ocr_message=state.ocr_message,
        error=state.error,
        result=result,
    )


@router.post("/v1/tasks/{task_id}/cancel", response_model=TaskStatusResponse)
def cancel_task(task_id: str, request: Request):
    tasks = request.app.state.tasks
    state = tasks.get(task_id)
    if state is None:
        record = request.app.state.db.get_task(task_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Task not found")
        request.app.state.db.update_status(task_id, "canceled")
        return TaskStatusResponse(
            task_id=task_id,
            status="canceled",
            progress=0.0,
            ocr_progress=None,
            ocr_stage=None,
            ocr_message=None,
            error=record.get("error"),
            result=json.loads(record["result_json"]) if record.get("result_json") else None,
        )
    tasks.cancel(task_id)
    remote_task_id = getattr(state, "remote_task_id", None)
    if remote_task_id and request.app.state.algorithm_tasks is not None:
        request.app.state.algorithm_tasks.cancel_task(remote_task_id)
    request.app.state.db.update_status(task_id, "canceled")
    return TaskStatusResponse(
        task_id=task_id,
        status=state.status,
        progress=state.progress,
        ocr_progress=state.ocr_progress,
        ocr_stage=state.ocr_stage,
        ocr_message=state.ocr_message,
        error=state.error,
        result=state.result,
    )


@router.get("/v1/tasks/{task_id}/events")
def task_events(
    task_id: str,
    request: Request,
    last_event_id: int | None = Query(None),
    threshold: float | None = Query(None),
):
    tasks = request.app.state.tasks
    state = tasks.get(task_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Task not found")
    if last_event_id is None:
        header_id = request.headers.get("last-event-id")
        if header_id and header_id.isdigit():
            last_event_id = int(header_id)

    def transform(event):
        payload = event.payload
        if isinstance(payload, dict) and "result" in payload:
            result = _attach_ocr_if_missing(payload.get("result"), task_id, request.app)
            return {"result": apply_threshold(result, threshold)}
        return payload

    return StreamingResponse(
        sse_event_stream(state, last_event_id=last_event_id, transform=transform),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache"},
    )


@router.get("/v1/tasks/{task_id}/download")
def download_result(task_id: str, request: Request, threshold: float | None = Query(None)):
    tasks = request.app.state.tasks
    state = tasks.get(task_id)
    if state is None or state.result is None:
        record = request.app.state.db.get_task(task_id)
        if record is None or not record.get("result_json"):
            raise HTTPException(status_code=404, detail="Result not found")
        result = json.loads(record["result_json"])
    else:
        result = state.result
    result = _attach_ocr_if_missing(result, task_id, request.app)
    result = apply_threshold(result, threshold)
    filename = f"result-{task_id}.json"
    headers = {"Content-Disposition": f"attachment; filename={filename}"}
    return JSONResponse(content=result, headers=headers)


@router.get("/v1/history")
def list_history(request: Request):
    return {"items": request.app.state.db.list_tasks()}


@router.get("/v1/history/{task_id}")
def get_history(task_id: str, request: Request, threshold: float | None = Query(None)):
    record = request.app.state.db.get_task(task_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Task not found")
    result = record.get("result_json")
    parsed = json.loads(result) if result else None
    parsed = _attach_ocr_if_missing(parsed, task_id, request.app)
    filtered = apply_threshold(parsed, threshold)
    return {
        "task_id": task_id,
        "name": record.get("name"),
        "created_at": record.get("created_at"),
        "status": record.get("status"),
        "media_type": record.get("media_type"),
        "match_threshold": record.get("match_threshold"),
        "result": filtered,
    }


@router.delete("/v1/history/{task_id}")
def delete_history(task_id: str, request: Request):
    deleted = request.app.state.db.delete_task(task_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Task not found")
    return {"status": "ok"}
