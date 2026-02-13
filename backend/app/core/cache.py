from __future__ import annotations

from collections import OrderedDict
import hashlib
from typing import Generic, TypeVar

from PIL import Image

K = TypeVar("K")
V = TypeVar("V")


class LRUCache(Generic[K, V]):
    def __init__(self, maxsize: int) -> None:
        self._maxsize = max(1, int(maxsize))
        self._data: OrderedDict[K, V] = OrderedDict()

    def get(self, key: K) -> V | None:
        if key not in self._data:
            return None
        value = self._data.pop(key)
        self._data[key] = value
        return value

    def set(self, key: K, value: V) -> None:
        if key in self._data:
            self._data.pop(key)
        elif len(self._data) >= self._maxsize:
            self._data.popitem(last=False)
        self._data[key] = value

    def clear(self) -> None:
        self._data.clear()


def image_cache_key(image: Image.Image) -> str:
    cached = image.info.get("cache_key")
    if cached:
        return str(cached)
    payload = image.tobytes()
    digest = hashlib.sha256(payload).hexdigest()
    key = f"raw:{image.mode}:{image.size[0]}x{image.size[1]}:{digest}"
    image.info["cache_key"] = key
    return key
