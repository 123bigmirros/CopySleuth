from __future__ import annotations

import json
import logging
import os
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable, Iterable

@dataclass
class TaskEvent:
    event_id: int
    event_type: str
    payload: dict
    created_at: float


class TaskCancelled(Exception):
    pass


@dataclass
class TaskState:
    task_id: str
    status: str = "pending"
    progress: float = 0.0
    ocr_progress: float | None = None
    ocr_stage: str | None = None
    ocr_message: str | None = None
    result: dict | None = None
    error: str | None = None
    cancel_requested: bool = False
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    events: queue.Queue[TaskEvent] = field(default_factory=queue.Queue)
    event_log: list[TaskEvent] = field(default_factory=list)
    _event_counter: int = 0

    def publish(self, event_type: str, payload: dict) -> None:
        self.updated_at = time.time()
        self._event_counter += 1
        event = TaskEvent(
            event_id=self._event_counter,
            event_type=event_type,
            payload=payload,
            created_at=self.updated_at,
        )
        # Skip storing partial events in event_log to avoid accumulating
        # large result snapshots in memory; they are still sent via the queue.
        if event_type != "partial":
            self.event_log.append(event)
            if len(self.event_log) > 2000:
                self.event_log = self.event_log[-2000:]
        self.events.put(event)

    def set_progress(self, progress: float, stage: str, message: str = "") -> None:
        self.progress = max(0.0, min(progress, 1.0))
        self.publish(
            "progress",
            {"progress": self.progress, "stage": stage, "message": message},
        )

    def set_ocr_progress(self, progress: float, stage: str, message: str = "") -> None:
        self.ocr_progress = max(0.0, min(progress, 1.0))
        self.ocr_stage = stage
        self.ocr_message = message
        self.publish(
            "ocr_progress",
            {
                "progress": self.ocr_progress,
                "stage": stage,
                "message": message,
            },
        )

    def cancel(self) -> bool:
        if self.status in {"done", "error", "canceled"}:
            return False
        self.cancel_requested = True
        self.status = "canceled"
        self.publish("canceled", {"message": "任务已取消"})
        return True

    def is_cancelled(self) -> bool:
        return self.cancel_requested or self.status == "canceled"

    def check_cancelled(self) -> None:
        if self.is_cancelled():
            raise TaskCancelled("Task canceled")


class TaskManager:
    def __init__(self, worker_count: int = 1) -> None:
        self._tasks: dict[str, TaskState] = {}
        self._lock = threading.Lock()
        self._queue: queue.Queue[tuple[TaskState, Callable[[TaskState], None]]] = queue.Queue()
        self._worker_threads: list[threading.Thread] = []
        self._max_finished_tasks = int(os.environ.get("MAX_FINISHED_TASKS", "20"))
        self._finished_order: list[str] = []
        count = max(1, int(worker_count))
        for _ in range(count):
            thread = threading.Thread(target=self._run_loop, daemon=True)
            thread.start()
            self._worker_threads.append(thread)

    def create(self, worker: Callable[[TaskState], None]) -> TaskState:
        task_id = str(uuid.uuid4())
        return self.create_with_id(task_id, worker)

    def create_with_id(self, task_id: str, worker: Callable[[TaskState], None]) -> TaskState:
        state = TaskState(task_id=task_id)
        with self._lock:
            self._tasks[task_id] = state
        self._queue.put((state, worker))
        return state

    def get(self, task_id: str) -> TaskState | None:
        with self._lock:
            return self._tasks.get(task_id)

    def cancel(self, task_id: str) -> bool:
        with self._lock:
            state = self._tasks.get(task_id)
        if state is None:
            return False
        return state.cancel()

    def _run_loop(self) -> None:
        while True:
            state, worker = self._queue.get()
            try:
                if state.is_cancelled():
                    continue
                if state.is_cancelled():
                    continue
                self._run_worker(state, worker)
            finally:
                self._queue.task_done()
                # Evict old finished tasks to prevent unbounded memory growth
                self._evict_finished(state.task_id)

    def _evict_finished(self, task_id: str) -> None:
        with self._lock:
            self._finished_order.append(task_id)
            while len(self._finished_order) > self._max_finished_tasks:
                old_id = self._finished_order.pop(0)
                old_state = self._tasks.pop(old_id, None)
                if old_state is not None:
                    old_state.result = None
                    old_state.event_log.clear()
                    # Drain the events queue to release any buffered payloads
                    while not old_state.events.empty():
                        try:
                            old_state.events.get_nowait()
                        except queue.Empty:
                            break

    @staticmethod
    def _run_worker(state: TaskState, worker: Callable[[TaskState], None]) -> None:
        logger = logging.getLogger(__name__)
        state.status = "running"
        state.set_progress(0.01, "queued", "任务已开始")
        try:
            state.check_cancelled()
            worker(state)
            if state.status == "canceled":
                return
            if state.status != "error":
                state.status = "done"
                state.set_progress(1.0, "completed", "处理完成")
                state.publish("done", {"status": "ok"})
        except TaskCancelled:
            if state.status != "canceled":
                state.status = "canceled"
                state.publish("canceled", {"message": "任务已取消"})
        except Exception as exc:  # pragma: no cover - safety net
            state.status = "error"
            state.error = str(exc)
            state.publish("error", {"message": state.error})
            logger.exception("Task failed: %s", state.task_id)


def sse_event_stream(
    state: TaskState,
    last_event_id: int | None = None,
    transform: Callable[[TaskEvent], dict] | None = None,
) -> Iterable[bytes]:
    if last_event_id is not None:
        for event in state.event_log:
            if event.event_id <= last_event_id:
                continue
            payload_data = transform(event) if transform else event.payload
            data = json.dumps(payload_data, ensure_ascii=False)
            payload = (
                f"id: {event.event_id}\n"
                f"event: {event.event_type}\n"
                f"data: {data}\n\n"
            )
            yield payload.encode("utf-8")
    while True:
        try:
            event = state.events.get(timeout=1.0)
            if last_event_id is not None and event.event_id <= last_event_id:
                continue
            payload_data = transform(event) if transform else event.payload
            data = json.dumps(payload_data, ensure_ascii=False)
            payload = f"id: {event.event_id}\nevent: {event.event_type}\ndata: {data}\n\n"
            yield payload.encode("utf-8")
            if event.event_type in {"done", "error", "canceled"}:
                break
        except queue.Empty:
            keep_alive = ": keep-alive\n\n"
            yield keep_alive.encode("utf-8")
            if state.status in {"done", "error", "canceled"} and state.events.empty():
                break
