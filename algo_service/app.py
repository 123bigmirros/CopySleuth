"""Algorithm Service — FastAPI application entry point.

Thin glue module: creates the app, configures middleware, registers routes,
and manages the application lifespan (startup / shutdown).
"""

from __future__ import annotations

import logging
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.formparsers import MultiPartParser

from algo_service.config import CORS_ORIGINS, DATA_DIR, MAX_FILE_COUNT, TASK_WORKERS

# ---------------------------------------------------------------------------
# Ensure the backend package is importable (it lives one level up).
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]
BACKEND_DIR = ROOT_DIR / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.append(str(BACKEND_DIR))

from app.core.config import load_settings  # noqa: E402
from app.core.tasks import TaskManager  # noqa: E402

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Starlette multipart limit
# ---------------------------------------------------------------------------
MultiPartParser.max_file_count = MAX_FILE_COUNT


# ---------------------------------------------------------------------------
# Lifespan (replaces deprecated @app.on_event)
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    """Initialize algorithm backend and task manager on startup."""
    settings = load_settings()
    from app.services.local_algorithm import LocalAlgorithmService
    application.state.algorithm = LocalAlgorithmService(settings)
    application.state.tasks = TaskManager(worker_count=TASK_WORKERS)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Algorithm service started (workers=%d, data_dir=%s)", TASK_WORKERS, DATA_DIR)
    yield


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------
app = FastAPI(title="Algorithm Service", version="0.2.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register route modules
from algo_service.routes.detect import router as detect_router  # noqa: E402
from algo_service.routes.health import router as health_router  # noqa: E402
from algo_service.routes.tasks import router as tasks_router  # noqa: E402

app.include_router(health_router)
app.include_router(detect_router)
app.include_router(tasks_router)
