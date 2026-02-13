from __future__ import annotations

from typing import Callable, Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from PIL import Image

    from app.pipelines.detection import DetectionResult
    from app.pipelines.video_detection import VideoDetectionResult


class AlgorithmService(Protocol):
    def detect_image(
        self,
        query_image: "Image.Image",
        target_image: "Image.Image",
        progress_cb: Callable | None = None,
        candidate_cb: Callable | None = None,
        cancel_cb: Callable | None = None,
        embedding_threshold: float | None = None,
    ) -> "DetectionResult":
        ...

    def detect_image_batch(
        self,
        query_images: list["Image.Image"],
        target_image: "Image.Image",
        progress_cb: Callable | None = None,
        candidate_cb: Callable | None = None,
        cancel_cb: Callable | None = None,
        embedding_threshold: float | None = None,
    ) -> list["DetectionResult"]:
        ...

    def detect_video_batch(
        self,
        query_images: list["Image.Image"],
        video_path: "Path",
        start_time: float | None = None,
        end_time: float | None = None,
        progress_cb: Callable | None = None,
        candidate_cb: Callable | None = None,
        cancel_cb: Callable | None = None,
        embedding_threshold: float | None = None,
        cache_dir: "Path | None" = None,
    ) -> list["VideoDetectionResult"]:
        ...
