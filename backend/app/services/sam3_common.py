from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch


@dataclass
class SegmentedObject:
    mask: np.ndarray  # bool HxW (bbox coordinates for the segment)
    score: float
    bbox: tuple[int, int, int, int]  # x0, y0, x1, y1


@dataclass
class SegmentationNode:
    node_id: int
    mask: np.ndarray  # bool HxW (bbox coordinates for this node)
    score: float
    bbox: tuple[int, int, int, int]  # x0, y0, x1, y1
    parent_id: int | None
    children: list[int] = field(default_factory=list)
    depth: int = 0
    area: int = 0


@dataclass
class SegmentationTree:
    root_id: int
    nodes: dict[int, SegmentationNode]
    width: int
    height: int


@dataclass
class VideoTrack:
    obj_id: int
    first_frame_idx: int
    last_frame_idx: int
    first_mask: np.ndarray  # bool HxW
    first_bbox: tuple[int, int, int, int] | None


@dataclass
class VideoTrackResult:
    tracks: list[VideoTrack]
    fps: float
    frame_count: int
    width: int
    height: int


def pick_device(device: str) -> torch.device:
    if device and device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_bpe_path(sam3_dir: Path) -> Path:
    primary = sam3_dir / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz"
    if primary.exists():
        return primary
    fallback = sam3_dir / "assets" / "bpe_simple_vocab_16e6.txt.gz"
    if fallback.exists():
        return fallback
    raise FileNotFoundError(primary)


def mask_to_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask)
    if ys.size == 0:
        return None
    y0, y1 = int(ys.min()), int(ys.max() + 1)
    x0, x1 = int(xs.min()), int(xs.max() + 1)
    return x0, y0, x1, y1


def compute_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    return float(inter / union)
