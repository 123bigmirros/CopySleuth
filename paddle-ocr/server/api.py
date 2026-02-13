import json
import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File, Form

from server.jobs import JobManager
from server.ocr_engine import OCREngine
from server.schemas import JobResponse, JobStatus
from video_ocr_pipeline import run_video_ocr

app = FastAPI(title="PaddleOCR Service", version="1.0")

_ENGINE = OCREngine()
_JOBS = JobManager(max_workers=1)


def _strip_spaces(text: str) -> str:
    return "".join(text.split())


def _clean_texts(texts):
    cleaned = []
    for text in texts:
        if text is None:
            continue
        stripped = _strip_spaces(str(text))
        if stripped:
            cleaned.append(stripped)
    return cleaned


def _clean_lines(lines):
    cleaned = []
    for line in lines:
        if not isinstance(line, dict):
            continue
        text = line.get("text")
        if text is None:
            continue
        stripped = _strip_spaces(str(text))
        if not stripped:
            continue
        updated = dict(line)
        updated["text"] = stripped
        cleaned.append(updated)
    return cleaned


def _log_texts(prefix: str, texts):
    preview = texts[:20]
    suffix = "" if len(texts) <= 20 else f" ...(+{len(texts) - 20})"
    print(f"{prefix} texts={preview}{suffix}", flush=True)


async def _save_upload(upload: UploadFile) -> Path:
    suffix = Path(upload.filename or "").suffix or ".bin"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        while True:
            chunk = await upload.read(8 * 1024 * 1024)
            if not chunk:
                break
            tmp.write(chunk)
    finally:
        tmp.close()
    return Path(tmp.name)


def _cleanup_path(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)  # type: ignore[arg-type]
    except Exception:
        try:
            if path.exists():
                path.unlink()
        except Exception:
            pass


def _parse_keywords(values: list[str]) -> list[str]:
    if not values:
        return []
    if len(values) == 1:
        raw = values[0].strip()
        if raw.startswith("["):
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, list):
                    return [str(item) for item in parsed if str(item).strip()]
            except Exception:
                pass
    return [str(item) for item in values if str(item).strip()]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/v1/ocr/image", response_model=JobResponse)
async def ocr_image(file: UploadFile = File(...)):
    path = await _save_upload(file)

    def _run():
        try:
            lines = _ENGINE.image_lines(str(path))
        except Exception:
            _cleanup_path(path)
            raise
        if lines:
            texts = [line.get("text") for line in lines if line.get("text")]
        else:
            texts = _ENGINE.image_texts(str(path))
        texts = _clean_texts(texts)
        lines = _clean_lines(lines)
        _log_texts("[ocr:image]", texts)
        result = {
            "texts": texts,
            "text": "\n".join(texts).strip(),
            "lines": lines,
        }
        _cleanup_path(path)
        return result

    job_id = _JOBS.submit(_run)
    return JobResponse(job_id=job_id, status="queued")


@app.post("/v1/ocr/video", response_model=JobResponse)
async def ocr_video(
    file: UploadFile = File(...),
    keywords: list[str] = Form(...),
    frame_select: str = Form("scenedetect"),
    scenedetect_threshold: float = Form(27.0),
    scene_threshold: float = Form(8.0),
    select_max_dim: int = Form(160),
    max_dim: int | None = Form(None),
    dedup_hash_threshold: int = Form(8),
    dedup_max_skip_frames: int = Form(0),
):
    path = await _save_upload(file)
    parsed_keywords = _parse_keywords(keywords)
    if not parsed_keywords:
        _cleanup_path(path)
        raise HTTPException(status_code=400, detail="keywords must be non-empty")

    def _run():
        try:
            ocr = _ENGINE.get_ocr()
            return run_video_ocr(
                str(path),
                parsed_keywords,
                device="gpu",
                ocr=ocr,
                show_progress=False,
                frame_stride=1,
                max_dim=max_dim,
                frame_select=frame_select,
                scene_threshold=scene_threshold,
                scenedetect_threshold=scenedetect_threshold,
                dedup_hash_threshold=dedup_hash_threshold,
                dedup_max_skip_frames=dedup_max_skip_frames,
                select_max_dim=select_max_dim,
            )
        finally:
            _cleanup_path(path)

    job_id = _JOBS.submit(_run)
    return JobResponse(job_id=job_id, status="queued")


@app.get("/v1/jobs/{job_id}", response_model=JobStatus)
def job_status(job_id: str):
    job = _JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return JobStatus(job_id=job_id, **job)
