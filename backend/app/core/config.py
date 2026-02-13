from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_SAM3_DIR = MODELS_DIR / "sam3"
DEFAULT_ROMA_V2_DIR = MODELS_DIR / "RoMaV2"
DEFAULT_ROMA_V2_WEIGHTS = DEFAULT_ROMA_V2_DIR / "checkpoints" / "romav2.pt"
DEFAULT_DINOV3_DIR = MODELS_DIR / "DINOv3"
DEFAULT_PADDLE_OCR_DIR = PROJECT_ROOT / "paddle-ocr"


@dataclass(frozen=True)
class Settings:
    sam3_dir: Path
    romav2_dir: Path
    romav2_weights: Path
    dinov3_dir: Path
    sam3_checkpoint: Path
    device: str
    confidence_threshold: float
    embedding_threshold: float
    embed_batch_size: int
    embed_max_edge: int
    romav2_compile: bool
    candidate_store_mode: str
    seg_max_depth: int
    seg_min_area: int
    seg_iou_dup_thresh: float
    seg_same_scale_ratio: float
    seg_stride_root: int
    seg_stride_child: int
    seg_rel_score: float
    seg_cap_root: int
    seg_cap_child: int
    seg_white_bg: bool
    seg_offload_every: int
    seg_offload_mb: int
    cache_image_lru: int
    cache_video_keyframe_lru: int
    paddle_ocr_dir: Path
    ocr_api_base: str | None
    media_service_url: str | None
    algo_service_url: str | None
    algo_service_timeout_s: float
    algo_image_mode: str | None


def load_settings() -> Settings:
    sam3_dir = Path(os.getenv("SAM3_DIR", DEFAULT_SAM3_DIR))
    romav2_dir = Path(os.getenv("ROMA_V2_DIR", DEFAULT_ROMA_V2_DIR))
    romav2_weights = Path(os.getenv("ROMA_V2_WEIGHTS", DEFAULT_ROMA_V2_WEIGHTS))
    dinov3_dir = Path(os.getenv("DINOV3_DIR", DEFAULT_DINOV3_DIR))
    paddle_ocr_dir = Path(os.getenv("PADDLE_OCR_DIR", DEFAULT_PADDLE_OCR_DIR))
    ocr_api_base = os.getenv("OCR_API_BASE", "").strip() or None
    media_service_url = os.getenv("MEDIA_SERVICE_URL", "").strip() or None
    sam3_checkpoint = Path(
        os.getenv("SAM3_CHECKPOINT", sam3_dir / "sam3_checkpoints" / "sam3.pt")
    )
    device = os.getenv("DEVICE", "auto")
    confidence_threshold = float(os.getenv("SEG_CONFIDENCE", "0.35"))
    embedding_threshold = float(os.getenv("EMBEDDING_THRESHOLD", "0.6"))
    embed_batch_size = int(os.getenv("EMBED_BATCH_SIZE", "5096"))
    embed_max_edge = int(os.getenv("EMBED_MAX_EDGE", "0"))
    romav2_compile = os.getenv("ROMA_V2_COMPILE", "0") in {"1", "true", "True"}
    candidate_store_mode = os.getenv("CANDIDATE_STORE_MODE", "matches").lower()
    if candidate_store_mode not in {"all", "matches", "best"}:
        candidate_store_mode = "matches"
    seg_max_depth = int(os.getenv("SEG_MAX_DEPTH", "4"))
    seg_min_area = int(os.getenv("SEG_MIN_AREA", "100"))
    seg_iou_dup_thresh = float(os.getenv("SEG_IOU_DUP_THRESH", "0.8"))
    seg_same_scale_ratio = float(os.getenv("SEG_SAME_SCALE_RATIO", "2.0"))
    seg_stride_root = int(os.getenv("SEG_STRIDE_ROOT", "32"))
    seg_stride_child = int(os.getenv("SEG_STRIDE_CHILD", "16"))
    seg_rel_score = float(os.getenv("SEG_REL_SCORE", "0.0"))
    seg_cap_root = max(1, int(os.getenv("SEG_CAP_ROOT", "1000000")))
    seg_cap_child = max(1, int(os.getenv("SEG_CAP_CHILD", "1000000")))
    seg_white_bg = os.getenv("SEG_WHITE_BG", "1") not in {"0", "false", "False"}
    seg_offload_every = int(os.getenv("SEG_OFFLOAD_EVERY", "4"))
    seg_offload_mb = int(os.getenv("SEG_OFFLOAD_MB", "0"))
    cache_image_lru = max(1, int(os.getenv("CACHE_IMAGE_LRU", "16")))
    cache_video_keyframe_lru = max(1, int(os.getenv("CACHE_VIDEO_KEYFRAME_LRU", "32")))
    algo_service_url = os.getenv("ALGO_SERVICE_URL", "").strip() or None
    algo_service_timeout_s = float(os.getenv("ALGO_SERVICE_TIMEOUT_S", "120"))
    algo_image_mode = os.getenv("ALGO_IMAGE_MODE", "path").strip().lower()
    if not algo_image_mode:
        algo_image_mode = None
    if algo_image_mode not in {None, "path", "b64"}:
        algo_image_mode = None
    return Settings(
        sam3_dir=sam3_dir,
        romav2_dir=romav2_dir,
        romav2_weights=romav2_weights,
        dinov3_dir=dinov3_dir,
        sam3_checkpoint=sam3_checkpoint,
        device=device,
        confidence_threshold=confidence_threshold,
        embedding_threshold=embedding_threshold,
        embed_batch_size=embed_batch_size,
        embed_max_edge=embed_max_edge,
        romav2_compile=romav2_compile,
        candidate_store_mode=candidate_store_mode,
        seg_max_depth=seg_max_depth,
        seg_min_area=seg_min_area,
        seg_iou_dup_thresh=seg_iou_dup_thresh,
        seg_same_scale_ratio=seg_same_scale_ratio,
        seg_stride_root=seg_stride_root,
        seg_stride_child=seg_stride_child,
        seg_rel_score=seg_rel_score,
        seg_cap_root=seg_cap_root,
        seg_cap_child=seg_cap_child,
        seg_white_bg=seg_white_bg,
        seg_offload_every=seg_offload_every,
        seg_offload_mb=seg_offload_mb,
        cache_image_lru=cache_image_lru,
        cache_video_keyframe_lru=cache_video_keyframe_lru,
        paddle_ocr_dir=paddle_ocr_dir,
        ocr_api_base=ocr_api_base,
        media_service_url=media_service_url,
        algo_service_url=algo_service_url,
        algo_service_timeout_s=algo_service_timeout_s,
        algo_image_mode=algo_image_mode,
    )
