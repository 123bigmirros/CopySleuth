from __future__ import annotations

import json
import threading
from pathlib import Path


class OCRStore:
    def __init__(self, base_dir: Path) -> None:
        self._base_dir = base_dir
        self._base_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _path_for(self, task_id: str) -> Path:
        safe_id = "".join(ch for ch in task_id if ch.isalnum() or ch in {"-", "_"})
        return self._base_dir / f"{safe_id}.json"

    def load(self, task_id: str) -> dict | None:
        path = self._path_for(task_id)
        if not path.exists():
            return None
        with self._lock:
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return None

    def save(self, task_id: str, payload: dict) -> Path:
        path = self._path_for(task_id)
        with self._lock:
            path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return path
