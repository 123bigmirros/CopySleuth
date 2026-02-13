import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor


class JobManager:
    def __init__(self, max_workers=1):
        self._jobs = {}
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

    def submit(self, fn, *args, **kwargs):
        job_id = str(uuid.uuid4())
        now = time.time()
        with self._lock:
            self._jobs[job_id] = {
                "status": "queued",
                "submitted_at": now,
                "started_at": None,
                "finished_at": None,
                "result": None,
                "error": None,
            }

        def _runner():
            with self._lock:
                self._jobs[job_id]["status"] = "running"
                self._jobs[job_id]["started_at"] = time.time()
            try:
                result = fn(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001
                with self._lock:
                    self._jobs[job_id]["status"] = "failed"
                    self._jobs[job_id]["error"] = str(exc)
                    self._jobs[job_id]["finished_at"] = time.time()
                return
            with self._lock:
                self._jobs[job_id]["status"] = "succeeded"
                self._jobs[job_id]["result"] = result
                self._jobs[job_id]["finished_at"] = time.time()

        self._executor.submit(_runner)
        return job_id

    def get(self, job_id):
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            return dict(job)
