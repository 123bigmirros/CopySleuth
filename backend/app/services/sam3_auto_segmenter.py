from __future__ import annotations

import numpy as np
from PIL import Image
import torch

from app.core.config import Settings
from app.core.gpu_lock import gpu_lock
from app.core.paths import ensure_repo_on_path
from app.services.sam3_common import (
    SegmentedObject,
    SegmentationNode,
    SegmentationTree,
    pick_device,
    resolve_bpe_path,
)


class Sam3AutoSegmenter:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        ensure_repo_on_path(settings.sam3_dir)
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.auto_segment_image_lib import AutoSegmentConfig, run_auto_segment
        bpe_path = resolve_bpe_path(settings.sam3_dir)
        if not settings.sam3_checkpoint.exists():
            raise FileNotFoundError(
                f"SAM3 checkpoint not found: {settings.sam3_checkpoint}"
            )

        self.device = pick_device(settings.device)
        device_name = self.device.type
        self._config = AutoSegmentConfig(return_full_masks=False)
        self._auto_segment = run_auto_segment
        self._model = build_sam3_image_model(
            bpe_path=str(bpe_path),
            device=device_name,
            eval_mode=True,
            checkpoint_path=str(settings.sam3_checkpoint),
            load_from_HF=False,
            enable_segmentation=True,
            enable_inst_interactivity=True,
        )
        self._processor = Sam3Processor(
            self._model,
            resolution=self._config.resolution,
            device=device_name,
            confidence_threshold=self._config.confidence_threshold,
        )

    def _segment_image_np(self, image: np.ndarray):
        with torch.inference_mode():
            result = self._auto_segment(image, self._processor, self._model, self._config)
        return result

    def segment(self, image: Image.Image) -> list[SegmentedObject]:
        with gpu_lock():
            image = image.convert("RGB")
            image_np = np.array(image)
            result = self._segment_image_np(image_np)
            masks = result.mask_crops or result.masks
            boxes = result.boxes
            scores = result.scores

        if isinstance(masks, np.ndarray):
            if masks.size == 0:
                return []
        else:
            if not masks:
                return []

        results: list[SegmentedObject] = []
        for mask, box, score in zip(masks, boxes, scores):
            x0, y0, x1, y1 = box.astype(int)
            if x1 <= x0 or y1 <= y0:
                continue
            if mask.shape == (y1 - y0, x1 - x0):
                mask_crop = mask
            else:
                mask_crop = mask[y0:y1, x0:x1].copy()
            if mask_crop.size == 0:
                continue
            if mask_crop.dtype != bool:
                mask_crop = mask_crop.astype(bool, copy=False)
            results.append(
                SegmentedObject(
                    mask=mask_crop,
                    score=float(score),
                    bbox=(int(x0), int(y0), int(x1), int(y1)),
                )
            )
        return results

    def segment_tree(self, image: Image.Image) -> SegmentationTree:
        image = image.convert("RGB")
        width, height = image.size
        root_mask = np.ones((height, width), dtype=bool)
        nodes: dict[int, SegmentationNode] = {
            0: SegmentationNode(
                node_id=0,
                mask=root_mask,
                score=1.0,
                bbox=(0, 0, width, height),
                parent_id=None,
                children=[],
                depth=0,
                area=int(root_mask.sum()),
            )
        }

        segments = self.segment(image)
        for segment in segments:
            x0, y0, x1, y1 = segment.bbox
            mask_crop = segment.mask
            expected_shape = (y1 - y0, x1 - x0)
            if mask_crop.shape != expected_shape:
                mask_crop = segment.mask[y0:y1, x0:x1]
            if mask_crop.size == 0:
                continue
            node_id = len(nodes)
            nodes[node_id] = SegmentationNode(
                node_id=node_id,
                mask=mask_crop,
                score=segment.score,
                bbox=segment.bbox,
                parent_id=0,
                children=[],
                depth=1,
                area=int(mask_crop.sum()),
            )
            nodes[0].children.append(node_id)

        return SegmentationTree(root_id=0, nodes=nodes, width=width, height=height)
