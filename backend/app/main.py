from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import logging
import os

from app.api.routes import router
from app.core.config import load_settings
from app.core.tasks import TaskManager
from app.core.db import TaskDB
from app.core.ocr_store import OCRStore
from app.services.algorithm_api import AlgorithmApiClient, AlgorithmApiConfig, AlgorithmTaskClient
from app.services.media_client import MediaClient
from app.services.paddle_ocr_service import PaddleOCRService


app = FastAPI(title="Copyright Detector", version="0.1.0")
logger = logging.getLogger(__name__)

# Always allow CORS for local dev without env setup.
allow_origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router)


@app.on_event("startup")
def startup() -> None:
    settings = load_settings()
    logger.info(
        "Backend settings: ALGO_SERVICE_URL=%s OCR_API_BASE=%s MEDIA_SERVICE_URL=%s",
        settings.algo_service_url,
        settings.ocr_api_base,
        settings.media_service_url,
    )
    app.state.algorithm_api = None
    app.state.algorithm_tasks = None
    if settings.algo_service_url:
        config = AlgorithmApiConfig(
            base_url=settings.algo_service_url,
            timeout_s=settings.algo_service_timeout_s,
            image_mode=settings.algo_image_mode,
        )
        app.state.algorithm_api = AlgorithmApiClient(config)
        app.state.algorithm_tasks = AlgorithmTaskClient(config)
        app.state.algorithm = None
    else:
        from app.services.local_algorithm import LocalAlgorithmService

        app.state.algorithm = LocalAlgorithmService(settings)
    app.state.settings = settings
    task_workers = max(1, int(os.getenv("TASK_WORKERS", "8")))
    app.state.tasks = TaskManager(worker_count=task_workers)
    data_dir = Path(os.getenv("BACKEND_DATA_DIR", "./data/backend"))
    data_dir.mkdir(parents=True, exist_ok=True)
    app.state.db = TaskDB(data_dir / "app.db")
    app.state.ocr_store = OCRStore(data_dir / "ocr")
    app.state.ocr = PaddleOCRService(settings.paddle_ocr_dir, api_base=settings.ocr_api_base)
    app.state.media = MediaClient(settings.media_service_url)
