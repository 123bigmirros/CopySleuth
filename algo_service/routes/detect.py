"""Synchronous image detection endpoint."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

from algo_service.config import DATA_DIR
from algo_service.serializers import build_image_result_payload
from algo_service.services.image_loader import (
    is_excel_upload,
    load_excel_images,
    load_image_bytes,
)

router = APIRouter()


def _normalize_embedding_threshold(value: float | None) -> float | None:
    """Validate and normalize an embedding threshold value."""
    if value is None:
        return None
    try:
        threshold = float(value)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="Invalid embedding_threshold") from exc
    if threshold > 1 and threshold <= 100:
        threshold = threshold / 100.0
    if threshold < 0 or threshold > 1:
        raise HTTPException(status_code=400, detail="Invalid embedding_threshold")
    return threshold


def _normalize_image_mode(value: str | None) -> str:
    """Validate image_mode parameter; defaults to ``"path"``."""
    if value is None:
        return "path"
    mode = value.strip().lower()
    if mode in ("b64", "path"):
        return mode
    raise HTTPException(
        status_code=400,
        detail=f"Invalid image_mode: {value!r}, must be 'b64' or 'path'",
    )


@router.post("/v1/detect")
async def detect(
    request: Request,
    target_image: UploadFile = File(...),
    query_images: list[UploadFile] | None = File(None),
    query_image: UploadFile | None = File(None),
    embedding_threshold: float | None = Form(None),
    image_mode: str | None = Form(None),
) -> JSONResponse:
    """Synchronous multi-query image detection."""
    if query_images is None or len(query_images) == 0:
        if query_image is None:
            raise HTTPException(status_code=400, detail="query_image missing")
        query_images = [query_image]

    query_list: list[Image.Image] = []
    query_origins: list[dict] | None = None
    if len(query_images) == 1 and is_excel_upload(
        query_images[0].filename, query_images[0].content_type
    ):
        excel_bytes = await query_images[0].read()
        query_list, query_origins = load_excel_images(excel_bytes, query_images[0].filename)
    else:
        for item in query_images:
            payload = await item.read()
            query_list.append(load_image_bytes(payload, item.filename))
    target_payload = await target_image.read()
    target = load_image_bytes(target_payload, target_image.filename)

    threshold = _normalize_embedding_threshold(embedding_threshold)
    request_id = str(uuid.uuid4())
    task_dir = DATA_DIR / "requests" / request_id
    task_dir.mkdir(parents=True, exist_ok=True)
    results = request.app.state.algorithm.detect_image_batch(
        query_list,
        target,
        embedding_threshold=threshold,
    )
    for img in query_list:
        img.close()
    query_list.clear()
    target.close()

    payloads = []
    for idx, result in enumerate(results):
        p = build_image_result_payload(result, task_dir=task_dir, query_index=idx)
        if query_origins is not None:
            p["excel_origin"] = query_origins[idx]
        payloads.append(p)
    return JSONResponse({"media_type": "image", "results": payloads})
