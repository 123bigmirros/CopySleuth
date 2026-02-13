from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

ProgressCallback = Callable[[int, int], None]


@dataclass(frozen=True)
class SceneChangeConfig:
    min_scene_change: float = 0.1
    resize_width: int = 0


@dataclass(frozen=True)
class Keyframe:
    frame_idx: int
    timestamp: float
    score: float
    reason: str


class KeyframeExtractor:
    def __init__(self, config: SceneChangeConfig | None = None) -> None:
        self._config = config or SceneChangeConfig()

    def extract(
        self,
        video_path: Path,
        *,
        start_frame: int,
        end_frame: int,
        fps: float,
        progress_cb: ProgressCallback | None = None,
    ) -> list[Keyframe]:
        if end_frame < start_frame:
            return []

        start_frame = max(0, int(start_frame))
        end_frame = max(start_frame, int(end_frame))
        total_frames = end_frame - start_frame + 1
        if total_frames <= 0:
            return []

        fps = float(fps or 0.0) or 30.0

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open video: {video_path}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        keyframes: list[Keyframe] = []
        prev_hist: np.ndarray | None = None
        processed = 0
        frame_idx = start_frame
        progress_step = max(1, total_frames // 200) if total_frames else 1

        try:
            while processed < total_frames:
                ok, frame = cap.read()
                if not ok:
                    break
                resized = self._resize_for_hist(frame)
                hist = self._compute_hist(resized)
                ts = frame_idx / fps if fps else 0.0
                if prev_hist is None:
                    keyframes.append(
                        Keyframe(
                            frame_idx=frame_idx,
                            timestamp=ts,
                            score=0.0,
                            reason="first",
                        )
                    )
                else:
                    score = float(
                        cv2.compareHist(
                            prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA
                        )
                    )
                    if score >= self._config.min_scene_change:
                        keyframes.append(
                            Keyframe(
                                frame_idx=frame_idx,
                                timestamp=ts,
                                score=score,
                                reason="scene-change",
                            )
                        )

                prev_hist = hist
                frame_idx += 1
                processed += 1
                if progress_cb and processed % progress_step == 0:
                    progress_cb(processed, total_frames)
        finally:
            cap.release()

        return keyframes

    def extract_indices(
        self,
        video_path: Path,
        *,
        start_frame: int,
        end_frame: int,
        fps: float,
        progress_cb: ProgressCallback | None = None,
    ) -> list[int]:
        return [
            keyframe.frame_idx
            for keyframe in self.extract(
                video_path,
                start_frame=start_frame,
                end_frame=end_frame,
                fps=fps,
                progress_cb=progress_cb,
            )
        ]

    def _resize_for_hist(self, frame: np.ndarray) -> np.ndarray:
        if self._config.resize_width <= 0:
            return frame
        height, width = frame.shape[:2]
        if width <= self._config.resize_width:
            return frame
        scale = self._config.resize_width / float(width)
        new_height = max(1, int(height * scale))
        return cv2.resize(
            frame, (self._config.resize_width, new_height), interpolation=cv2.INTER_AREA
        )

    @staticmethod
    def _compute_hist(frame: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [48, 48], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        return hist
