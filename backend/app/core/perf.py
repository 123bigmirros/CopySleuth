from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path


_LOGGER: logging.Logger | None = None


def _get_logger() -> logging.Logger:
    global _LOGGER
    if _LOGGER is not None:
        return _LOGGER
    logger = logging.getLogger("perf")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        log_path = os.getenv("PERF_LOG_PATH", "")
        if not log_path:
            repo_root = Path(__file__).resolve().parents[3]
            log_path = str(repo_root / "logs" / "perf.log")
        path = Path(log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(path)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
    _LOGGER = logger
    return logger


def log_perf(event: str, **fields: object) -> None:
    payload = {"event": event, "ts": time.time()}
    payload.update(fields)
    logger = _get_logger()
    logger.info(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
