from __future__ import annotations

from contextlib import contextmanager


@contextmanager
def gpu_lock():
    yield
