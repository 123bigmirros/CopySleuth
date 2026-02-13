from __future__ import annotations

from contextlib import contextmanager
import threading


_CPU_LOCK = threading.RLock()


@contextmanager
def cpu_lock():
    with _CPU_LOCK:
        yield
