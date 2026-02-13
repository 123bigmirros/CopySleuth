from __future__ import annotations

import json
import os
import sqlite3
import time
from pathlib import Path


class TaskDB:
    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS tasks (
                    task_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    status TEXT NOT NULL,
                    media_type TEXT NOT NULL,
                    query_path TEXT NOT NULL,
                    target_path TEXT NOT NULL,
                    match_threshold REAL NOT NULL,
                    embedding_threshold REAL NOT NULL,
                    result_json TEXT,
                    error TEXT
                )
                """
            )
            columns = {
                row["name"]
                for row in conn.execute("PRAGMA table_info(tasks)").fetchall()
            }
            if "embedding_threshold" not in columns:
                conn.execute(
                    "ALTER TABLE tasks ADD COLUMN embedding_threshold REAL NOT NULL DEFAULT 0.5"
                )

    def create_task(
        self,
        task_id: str,
        name: str,
        media_type: str,
        query_path: Path,
        target_path: Path,
        match_threshold: float,
        embedding_threshold: float,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO tasks (
                    task_id, name, created_at, status, media_type,
                    query_path, target_path, match_threshold, embedding_threshold
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    task_id,
                    name,
                    time.time(),
                    "running",
                    media_type,
                    str(query_path),
                    str(target_path),
                    match_threshold,
                    embedding_threshold,
                ),
            )

    def update_result(self, task_id: str, result: dict) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE tasks
                SET result_json = ?, status = ?, error = NULL
                WHERE task_id = ?
                """,
                (json.dumps(result, ensure_ascii=False), "done", task_id),
            )

    def update_error(self, task_id: str, error: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE tasks
                SET status = ?, error = ?
                WHERE task_id = ?
                """,
                ("error", error, task_id),
            )

    def update_status(self, task_id: str, status: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE tasks SET status = ? WHERE task_id = ?",
                (status, task_id),
            )

    def get_task(self, task_id: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM tasks WHERE task_id = ?",
                (task_id,),
            ).fetchone()
        return dict(row) if row else None

    def list_tasks(self) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT task_id, name, created_at, status, media_type, match_threshold, embedding_threshold
                FROM tasks
                ORDER BY created_at DESC
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def delete_task(self, task_id: str) -> bool:
        row = self.get_task(task_id)
        if not row:
            return False
        for path_key in ("query_path", "target_path"):
            path = row.get(path_key)
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass
        with self._connect() as conn:
            conn.execute("DELETE FROM tasks WHERE task_id = ?", (task_id,))
        return True
