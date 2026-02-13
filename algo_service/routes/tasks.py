"""Async task endpoints: create, status, cancel, SSE events."""

from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from PIL import Image

from algo_service.config import DATA_DIR
from algo_service.routes.detect import _normalize_embedding_threshold
from algo_service.services.image_loader import (
    is_excel_upload,
    load_excel_images,
    load_image_bytes,
)
from algo_service.workers import image_worker, video_worker

router = APIRouter()


@router.post("/v1/tasks")
async def create_task(
    request: Request,
    target_file: UploadFile = File(...),
    query_images: list[UploadFile] = File(...),
    video_start: float | None = Form(None),
    video_end: float | None = Form(None),
    embedding_threshold: float | None = Form(None),
    image_mode: str | None = Form(None),
) -> JSONResponse:
    """Create an async detection task for image or video targets."""
    if not query_images:
        raise HTTPException(status_code=400, detail="query_images missing")

    query_list: list[Image.Image] = []
    query_origins: list[dict] | None = None
    if len(query_images) == 1 and is_excel_upload(
        query_images[0].filename, query_images[0].content_type
    ):
        excel_bytes = await query_images[0].read()
        query_list, query_origins = load_excel_images(
            excel_bytes, query_images[0].filename,
        )
    else:
        for item in query_images:
            payload = await item.read()
            query_list.append(load_image_bytes(payload, item.filename))

    content_type = (target_file.content_type or "").lower()
    is_video = content_type.startswith("video/")
    target_name = target_file.filename or ("target.mp4" if is_video else "target.png")
    target_suffix = Path(target_name).suffix or (".mp4" if is_video else ".png")

    task_id = str(uuid.uuid4())
    task_dir = DATA_DIR / task_id
    task_dir.mkdir(parents=True, exist_ok=True)

    target_path = task_dir / f"target{target_suffix}"
    target_image: Image.Image | None = None
    if is_video:
        with open(target_path, "wb") as target_fp:
            while True:
                chunk = await target_file.read(8 * 1024 * 1024)
                if not chunk:
                    break
                target_fp.write(chunk)
    else:
        raw = await target_file.read()
        target_image = load_image_bytes(raw, target_file.filename)

    threshold = _normalize_embedding_threshold(embedding_threshold)
    algorithm = request.app.state.algorithm

    if is_video:
        def worker(state) -> None:
            video_worker(
                state,
                algorithm=algorithm,
                query_list=query_list,
                query_origins=query_origins,
                target_path=target_path,
                task_dir=task_dir,
                threshold=threshold,
                video_start=video_start,
                video_end=video_end,
            )
    else:
        if target_image is None:
            raise HTTPException(status_code=400, detail="target image missing")
        _target = target_image

        def worker(state) -> None:
            image_worker(
                state,
                algorithm=algorithm,
                query_list=query_list,
                query_origins=query_origins,
                target_image=_target,
                task_dir=task_dir,
                threshold=threshold,
            )

    state = request.app.state.tasks.create_with_id(task_id, worker)
    media_type = "video" if is_video else "image"
    return JSONResponse({"task_id": state.task_id, "media_type": media_type})


@router.get("/v1/tasks/{task_id}")
def get_task(request: Request, task_id: str) -> dict:
    """Return current status and result of a task."""
    state = request.app.state.tasks.get(task_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Task not found")
    return {
        "task_id": task_id,
        "status": state.status,
        "progress": state.progress,
        "error": state.error,
        "result": state.result,
    }


@router.post("/v1/tasks/{task_id}/cancel")
def cancel_task(request: Request, task_id: str) -> dict:
    """Cancel a running task."""
    if not request.app.state.tasks.cancel(task_id):
        raise HTTPException(status_code=404, detail="Task not found")
    return {"status": "ok"}


@router.get("/v1/tasks/{task_id}/events")
def task_events(request: Request, task_id: str) -> StreamingResponse:
    """Stream task events via Server-Sent Events."""
    from app.core.tasks import sse_event_stream

    state = request.app.state.tasks.get(task_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Task not found")
    return StreamingResponse(
        sse_event_stream(state),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache"},
    )
