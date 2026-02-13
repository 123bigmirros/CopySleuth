from __future__ import annotations

import io
import os
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageDraw


MEDIA_STORE_DIR = Path(os.getenv("MEDIA_STORE_DIR", "./media_store")).resolve()
MEDIA_STORE_DIR.mkdir(parents=True, exist_ok=True)
MEDIA_BASE_URL = os.getenv("MEDIA_BASE_URL", "").rstrip("/")


app = FastAPI(title="Media Preprocess Service", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/media", StaticFiles(directory=str(MEDIA_STORE_DIR)), name="media")


def _scale_bbox(
    bbox: tuple[int, int, int, int], scale: float
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = bbox
    return (
        int(round(x0 * scale)),
        int(round(y0 * scale)),
        int(round(x1 * scale)),
        int(round(y1 * scale)),
    )


@app.get("/v1/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/v1/preview")
async def preview(
    request: Request,
    file: UploadFile = File(...),
    max_size: int = Form(0),
    bbox: Optional[str] = Form(None),
) -> JSONResponse:
    payload = await file.read()
    image = Image.open(io.BytesIO(payload)).convert("RGB")
    orig_w, orig_h = image.size
    max_size = int(max_size or 0)
    scale = 1.0
    if max_size > 0:
        longest = max(orig_w, orig_h)
        if longest > max_size:
            scale = max_size / float(longest)
            new_w = max(1, int(round(orig_w * scale)))
            new_h = max(1, int(round(orig_h * scale)))
            if hasattr(Image, "Resampling"):
                resample = Image.Resampling.BILINEAR
            else:
                resample = Image.BILINEAR
            image = image.resize((new_w, new_h), resample=resample)

    if bbox:
        try:
            parts = [int(float(x)) for x in bbox.split(",")]
            if len(parts) == 4:
                x0, y0, x1, y1 = _scale_bbox((parts[0], parts[1], parts[2], parts[3]), scale)
                draw = ImageDraw.Draw(image)
                width = max(2, int(round(4 * scale)))
                for offset in range(width):
                    draw.rectangle(
                        [x0 - offset, y0 - offset, x1 + offset, y1 + offset],
                        outline=(255, 0, 0),
                    )
        except Exception:
            pass

    image_id = uuid.uuid4().hex
    filename = f"{image_id}.png"
    path = MEDIA_STORE_DIR / filename
    image.save(path, format="PNG")

    base = MEDIA_BASE_URL
    if not base:
        base = str(request.base_url).rstrip("/")
    url = f"{base}/media/{filename}"

    return JSONResponse(
        {
            "id": image_id,
            "url": url,
            "width": image.size[0],
            "height": image.size[1],
            "orig_width": orig_w,
            "orig_height": orig_h,
        }
    )
