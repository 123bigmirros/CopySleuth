from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import Callable

from PIL import Image
import torch

from app.core.config import Settings
from app.pipelines.detection import DetectionResult, DetectorPipeline
from app.pipelines.video_detection import VideoDetectionResult, VideoDetectorPipeline
from app.services.sam3_service import Sam3Segmenter
from app.services.roma_service import RomaMatcher
from app.services.algorithm import AlgorithmService


class LocalAlgorithmService(AlgorithmService):
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._segmenter = Sam3Segmenter(settings)
        self._matcher = RomaMatcher(settings)
        self._image_pipeline = DetectorPipeline(
            self._segmenter,
            self._matcher,
            candidate_store_mode=settings.candidate_store_mode,
            cache_size=settings.cache_image_lru,
        )
        self._video_pipeline: VideoDetectorPipeline | None = None

    def detect_image(
        self,
        query_image: Image.Image,
        target_image: Image.Image,
        progress_cb: Callable | None = None,
        candidate_cb: Callable | None = None,
        cancel_cb: Callable | None = None,
        embedding_threshold: float | None = None,
    ) -> DetectionResult:
        try:
            return self._image_pipeline.detect(
                query_image,
                target_image,
                progress_cb=progress_cb,
                candidate_cb=candidate_cb,
                cancel_cb=cancel_cb,
                embedding_threshold=embedding_threshold,
            )
        finally:
            self._release_models()

    def detect_image_batch(
        self,
        query_images: list[Image.Image],
        target_image: Image.Image,
        progress_cb: Callable | None = None,
        candidate_cb: Callable | None = None,
        cancel_cb: Callable | None = None,
        embedding_threshold: float | None = None,
    ) -> list[DetectionResult]:
        try:
            return self._image_pipeline.detect_batch_queries(
                query_images,
                target_image,
                progress_cb=progress_cb,
                candidate_cb=candidate_cb,
                cancel_cb=cancel_cb,
                embedding_threshold=embedding_threshold,
            )
        finally:
            self._release_models()

    def detect_video_batch(
        self,
        query_images: list[Image.Image],
        video_path: Path,
        start_time: float | None = None,
        end_time: float | None = None,
        progress_cb: Callable | None = None,
        candidate_cb: Callable | None = None,
        cancel_cb: Callable | None = None,
        embedding_threshold: float | None = None,
        cache_dir: Path | None = None,
    ) -> list[VideoDetectionResult]:
        pipeline = self._get_video_pipeline()
        try:
            return pipeline.detect_batch_queries(
                query_images,
                video_path,
                start_time=start_time,
                end_time=end_time,
                progress_cb=progress_cb,
                candidate_cb=candidate_cb,
                cancel_cb=cancel_cb,
                embedding_threshold=embedding_threshold,
                cache_dir=cache_dir,
            )
        finally:
            self._release_models()

    def _get_video_pipeline(self) -> VideoDetectorPipeline:
        if self._video_pipeline is None:
            self._video_pipeline = VideoDetectorPipeline(
                self._segmenter,
                self._matcher,
                cache_size=self._settings.cache_video_keyframe_lru,
            )
        return self._video_pipeline

    def _release_models(self) -> None:
        pass
