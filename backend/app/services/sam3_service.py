from __future__ import annotations

from pathlib import Path

import cv2
import torch
from PIL import Image

from app.core.config import Settings
from app.core.paths import ensure_repo_on_path
from app.core.tasks import TaskCancelled
from app.services.sam3_auto_segmenter import Sam3AutoSegmenter
from app.services.sam3_common import (
    SegmentedObject,
    SegmentationTree,
    VideoTrack,
    VideoTrackResult,
    resolve_bpe_path,
)


class Sam3Segmenter:
    def __init__(self, settings: Settings) -> None:
        ensure_repo_on_path(settings.sam3_dir)
        self._impl = Sam3AutoSegmenter(settings)

    def segment(
        self, image: Image.Image, confidence_threshold: float | None = None
    ) -> list[SegmentedObject]:
        return self._impl.segment(image)

    def segment_tree(
        self, image: Image.Image, confidence_threshold: float | None = None
    ) -> SegmentationTree:
        return self._impl.segment_tree(image)


class Sam3VideoTracker:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        ensure_repo_on_path(settings.sam3_dir)
        from sam3.model_builder import build_sam3_video_predictor

        if not torch.cuda.is_available():
            raise RuntimeError("SAM3 video tracking requires CUDA-enabled GPU.")

        bpe_path = resolve_bpe_path(settings.sam3_dir)
        if not settings.sam3_checkpoint.exists():
            raise FileNotFoundError(
                f"SAM3 checkpoint not found: {settings.sam3_checkpoint}"
            )

        self._predictor = build_sam3_video_predictor(
            checkpoint_path=str(settings.sam3_checkpoint),
            bpe_path=str(bpe_path),
            video_loader_type="cv2",
        )
        try:
            self._predictor.model.tracker.offload_output_to_cpu_for_eval = False
        except Exception:
            pass

    def track(
        self,
        video_path: Path,
        prompts: list[str],
        progress_cb: callable | None = None,
        cancel_cb: callable | None = None,
    ) -> VideoTrackResult:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open video: {video_path}")

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 30.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

        session = self._predictor.handle_request(
            {
                "type": "start_session",
                "resource_path": str(video_path),
                "offload_video_to_cpu": True,
            }
        )
        session_id = session["session_id"]

        try:
            for prompt in prompts:
                if cancel_cb and cancel_cb():
                    raise TaskCancelled("Task canceled")
                response = self._predictor.handle_request(
                    {
                        "type": "add_prompt",
                        "session_id": session_id,
                        "frame_index": 0,
                        "text": prompt,
                    }
                )
                outputs = response.get("outputs") or {}
                out_obj_ids = outputs.get("out_obj_ids")
                if out_obj_ids is None:
                    continue

            tracks: dict[int, VideoTrack] = {}
            progress_step = max(1, frame_count // 100) if frame_count else 1
            for payload in self._predictor.handle_stream_request(
                {
                    "type": "propagate_in_video",
                    "session_id": session_id,
                    "propagation_direction": "forward",
                    "start_frame_index": 0,
                }
            ):
                if cancel_cb and cancel_cb():
                    raise TaskCancelled("Task canceled")
                frame_idx = int(payload["frame_index"])
                outputs = payload.get("outputs") or {}
                out_obj_ids = outputs.get("out_obj_ids")
                out_masks = outputs.get("out_binary_masks")
                out_boxes = outputs.get("out_boxes_xywh")
                if out_obj_ids is None or out_masks is None or out_boxes is None:
                    continue

                for obj_id, mask, box_xywh in zip(out_obj_ids, out_masks, out_boxes):
                    mask_bool = mask.astype(bool)
                    if not mask_bool.any():
                        continue
                    obj_id_int = int(obj_id)
                    x, y, w, h = box_xywh
                    x0 = int(x * width)
                    y0 = int(y * height)
                    x1 = int((x + w) * width)
                    y1 = int((y + h) * height)
                    bbox = (max(0, x0), max(0, y0), min(width, x1), min(height, y1))

                    if obj_id_int not in tracks:
                        tracks[obj_id_int] = VideoTrack(
                            obj_id=obj_id_int,
                            first_frame_idx=frame_idx,
                            last_frame_idx=frame_idx,
                            first_mask=mask_bool,
                            first_bbox=bbox,
                        )
                    else:
                        tracks[obj_id_int].last_frame_idx = frame_idx

                if progress_cb and frame_idx % progress_step == 0 and frame_count:
                    progress_cb(frame_idx, frame_count)
        finally:
            self._predictor.handle_request({"type": "close_session", "session_id": session_id})
            cap.release()

        return VideoTrackResult(
            tracks=list(tracks.values()),
            fps=fps,
            frame_count=frame_count,
            width=width,
            height=height,
        )
