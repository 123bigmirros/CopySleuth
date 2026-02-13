"""Algo-service configuration loaded from environment variables."""

from __future__ import annotations

import os
from pathlib import Path


DATA_DIR: Path = Path(os.getenv("ALGO_DATA_DIR", "/data/pic_cmp_algo")).resolve()

TASK_WORKERS: int = max(1, int(os.getenv("ALGO_TASK_WORKERS") or os.getenv("TASK_WORKERS", "4")))

MAX_FILE_COUNT: int = int(os.getenv("ALGO_MAX_FILE_COUNT", "10000"))

CORS_ORIGINS: list[str] = [
    origin.strip()
    for origin in os.getenv("ALGO_CORS_ORIGINS", "*").split(",")
    if origin.strip()
]
