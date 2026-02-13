from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass
import gc
import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from PIL import Image
import torch

from app.core.tasks import TaskCancelled
from app.core.perf import log_perf
from app.pipelines.keyframe_extract import KeyframeExtractor
from app.services.roma_service import MatchResult, RomaMatcher
from app.services.sam3_service import Sam3Segmenter


logger = logging.getLogger(__name__)

ImagePayload = Image.Image | Path


def _crop_with_mask_np(
    frame_np: np.ndarray,
    mask: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> Image.Image:
    x0, y0, x1, y1 = bbox
    crop = frame_np[y0:y1, x0:x1].copy()
    if mask.shape[0] == (y1 - y0) and mask.shape[1] == (x1 - x0):
        mask_crop = mask
    else:
        mask_crop = mask[y0:y1, x0:x1]
    if mask_crop.ndim != 2:
        mask_crop = mask_crop.squeeze()
    if crop.ndim == 2:
        crop = np.stack([crop] * 3, axis=-1)
    crop[~mask_crop] = 255
    return Image.fromarray(crop)


def _select_candidate_indices(
    sim_row: torch.Tensor,
    *,
    min_sim: float,
) -> list[int]:
    if sim_row.numel() == 0:
        return []
    mask = sim_row >= float(min_sim)
    if not bool(mask.any()):
        return []
    indices = torch.nonzero(mask, as_tuple=False).flatten()
    indices = torch.sort(indices).values
    return indices.tolist()


def _load_image_payload(value: ImagePayload) -> tuple[Image.Image, bool]:
    if isinstance(value, Image.Image):
        return value, False
    with Image.open(value) as image:
        return image.convert("RGB"), True


def _prepare_images(values: list[ImagePayload]) -> tuple[list[Image.Image], list[Image.Image]]:
    images: list[Image.Image] = []
    created: list[Image.Image] = []
    for value in values:
        image, should_close = _load_image_payload(value)
        images.append(image)
        if should_close:
            created.append(image)
    return images, created


@dataclass
class VideoMatchCandidate:
    obj_id: int
    first_frame_idx: int
    last_frame_idx: int
    start_time: float
    end_time: float
    bbox: tuple[int, int, int, int] | None
    match: MatchResult
    score: float
    image: ImagePayload
    full_image: ImagePayload
    kind: str


@dataclass
class VideoDetectionResult:
    is_match: bool
    best_score: float | None
    best_match: MatchResult | None
    candidates: int
    matches: list[VideoMatchCandidate]
    fps: float
    frame_count: int
    duration: float


@dataclass
class _VideoCandidate:
    obj_id: int
    first_frame_idx: int
    last_frame_idx: int
    start_time: float
    end_time: float
    bbox: tuple[int, int, int, int] | None
    image: ImagePayload
    full_image: ImagePayload
    kind: str
    embedding: torch.Tensor | None = None  # pre-computed DINOv3 embedding for reuse


@dataclass
class _TrackedObject:
    """An object tracked across consecutive frames for dedup."""
    obj_id: int
    first_frame_idx: int
    last_frame_idx: int
    start_time: float
    end_time: float
    bbox: tuple[int, int, int, int] | None
    image: ImagePayload
    full_image: ImagePayload
    kind: str
    embedding: torch.Tensor  # (D,) normalised vector


@dataclass
class VideoDetectorPipeline:
    def __init__(
        self,
        segmenter: Sam3Segmenter,
        matcher: RomaMatcher,
        cache_size: int = 32,
    ) -> None:
        self._segmenter = segmenter
        self._matcher = matcher
        self._keyframe_extractor = KeyframeExtractor()

    def clear_cache(self) -> None:
        # Clear keyframe extractor cache if any
        if hasattr(self._keyframe_extractor, 'clear_cache'):
            self._keyframe_extractor.clear_cache()

    def _detect_batch_queries_range(
        self,
        video_path: Path,
        *,
        start_frame: int,
        end_frame: int,
        frame_indices: list[int] | None = None,
        fps: float,
        progress_cb: callable | None = None,
        cancel_cb: callable | None = None,
        next_obj_id: int = 1,
        stats: dict[str, float] | None = None,
        image_store: Callable[[Image.Image, str], Path] | None = None,
        collect_candidates: list[_VideoCandidate] | None = None,
        candidate_batch_cb: Callable[[list[_VideoCandidate]], None] | None = None,
        candidate_batch_limit: int = 1,
        dedup_threshold: float = 0.0,
        intra_dedup_threshold: float = 0.0,
    ) -> tuple[int, int]:
        frame_indices = (
            sorted(
                {
                    int(idx)
                    for idx in frame_indices
                    if start_frame <= idx <= end_frame
                }
            )
            if frame_indices
            else []
        )
        total_frames = len(frame_indices) if frame_indices else end_frame - start_frame + 1
        if total_frames <= 0:
            return next_obj_id, 0

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open video: {video_path}")
        if not frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        pending_candidates: list[_VideoCandidate] = []
        progress_step = max(1, total_frames // 100) if total_frames else 1
        added_candidates = 0
        candidate_batch_limit = max(1, int(candidate_batch_limit))

        # --- Cross-frame dedup pool ---
        dedup_enabled = dedup_threshold > 0.0
        tracked_pool: list[_TrackedObject] = []
        pool_embeddings: torch.Tensor | None = None  # (N, D) on device
        dedup_removed = 0

        # --- Intra-frame dedup ---
        intra_dedup_enabled = intra_dedup_threshold > 0.0
        intra_dedup_removed = 0
        need_embeddings = dedup_enabled or intra_dedup_enabled

        def flush_candidates() -> None:
            if not pending_candidates:
                return
            if candidate_batch_cb is not None:
                candidate_batch_cb(pending_candidates)
                pending_candidates.clear()
                return
            if collect_candidates is not None:
                collect_candidates.extend(pending_candidates)
                pending_candidates.clear()
                return
            raise RuntimeError("candidate_batch_cb or collect_candidates is required")

        def _segment_frame(frame_image: Image.Image):
            """Run segmentation only — safe to call from a worker thread."""
            seg_start = time.perf_counter()
            segmented = self._segmenter.segment(frame_image)
            seg_elapsed = time.perf_counter() - seg_start
            return segmented, seg_elapsed

        def _post_process_frame(
            frame_idx: int,
            frame_image: Image.Image,
            segmented,
            seg_elapsed: float,
        ) -> None:
            nonlocal next_obj_id, added_candidates, pool_embeddings, dedup_removed, intra_dedup_removed
            if stats is not None:
                stats["segment_s"] = stats.get("segment_s", 0.0) + seg_elapsed

            frame_payload: ImagePayload | None = None
            if image_store is not None:
                frame_payload = image_store(frame_image, "frame")

            if not segmented:
                if image_store is not None:
                    try:
                        frame_image.close()
                    except Exception:
                        pass
                return

            frame_np = np.array(frame_image)
            if image_store is None:
                frame_payload = frame_image

            # Build all candidates for this frame first
            frame_candidates: list[_VideoCandidate] = []
            frame_crop_images: list[Image.Image] = []  # PIL crops for dedup embedding
            for segment in sorted(segmented, key=lambda item: item.score, reverse=True):
                crop = _crop_with_mask_np(frame_np, segment.mask, segment.bbox)
                if image_store is not None:
                    crop_payload: ImagePayload = image_store(crop, "crop")
                else:
                    crop_payload = crop
                start_time = frame_idx / fps if fps else 0.0
                end_time = (frame_idx + 1) / fps if fps else 0.0
                frame_candidates.append(
                    _VideoCandidate(
                        obj_id=next_obj_id,
                        first_frame_idx=frame_idx,
                        last_frame_idx=frame_idx,
                        start_time=start_time,
                        end_time=end_time,
                        bbox=segment.bbox,
                        image=crop_payload,
                        full_image=frame_payload,
                        kind="crop",
                    )
                )
                if need_embeddings:
                    frame_crop_images.append(crop)
                elif image_store is not None:
                    # No dedup — close the PIL crop immediately (payload is on disk)
                    try:
                        crop.close()
                    except Exception:
                        pass
                next_obj_id += 1

            if not frame_candidates:
                if image_store is not None:
                    try:
                        frame_image.close()
                    except Exception:
                        pass
                return

            # --- Compute embeddings (for intra-frame and/or cross-frame dedup) ---
            if need_embeddings and frame_candidates:
                emb_start = time.perf_counter()
                cand_embeddings, _ = self._matcher.embed_images(frame_crop_images)
                if stats is not None:
                    stats["dedup_embed_s"] = stats.get("dedup_embed_s", 0.0) + (
                        time.perf_counter() - emb_start
                    )
                if cand_embeddings.numel() > 0:
                    cand_embeddings = cand_embeddings.float()

                    # --- Intra-frame dedup (greedy, score-descending) ---
                    n = len(frame_candidates)
                    if intra_dedup_enabled and n > 1:
                        intra_sim = cand_embeddings @ cand_embeddings.T
                        intra_keep = [True] * n
                        for i in range(n):
                            if not intra_keep[i]:
                                continue
                            for j in range(i + 1, n):
                                if not intra_keep[j]:
                                    continue
                                if float(intra_sim[i, j]) >= intra_dedup_threshold:
                                    intra_keep[j] = False
                                    intra_dedup_removed += 1
                        del intra_sim
                        # Filter candidates, embeddings, and crop images
                        frame_candidates = [c for c, k in zip(frame_candidates, intra_keep) if k]
                        cand_embeddings = cand_embeddings[[i for i, k in enumerate(intra_keep) if k]]
                        frame_crop_images = [img for img, k in zip(frame_crop_images, intra_keep) if k]

                    # --- Cross-frame dedup against active pool ---
                    if dedup_enabled:
                        keep_mask = [True] * len(frame_candidates)
                        if pool_embeddings is not None and pool_embeddings.shape[0] > 0:
                            if cand_embeddings.device != pool_embeddings.device:
                                cand_embeddings = cand_embeddings.to(pool_embeddings.device)
                            sim = cand_embeddings @ pool_embeddings.T  # (M, N)
                            max_sim, max_idx = sim.max(dim=1)  # (M,)
                            max_sim_list = max_sim.cpu().tolist()
                            max_idx_list = max_idx.cpu().tolist()
                            del sim, max_sim, max_idx
                            for i, (s, pidx) in enumerate(zip(max_sim_list, max_idx_list)):
                                if s >= dedup_threshold:
                                    # Duplicate — update tracked object time range
                                    tracked = tracked_pool[pidx]
                                    cand = frame_candidates[i]
                                    tracked.last_frame_idx = cand.last_frame_idx
                                    tracked.end_time = cand.end_time
                                    keep_mask[i] = False
                                    dedup_removed += 1

                        # Add new candidates to pool and attach embeddings for reuse
                        new_embeddings: list[torch.Tensor] = []
                        for i, is_new in enumerate(keep_mask):
                            if is_new:
                                cand = frame_candidates[i]
                                emb_vec = cand_embeddings[i]
                                cand.embedding = emb_vec
                                tracked_pool.append(
                                    _TrackedObject(
                                        obj_id=cand.obj_id,
                                        first_frame_idx=cand.first_frame_idx,
                                        last_frame_idx=cand.last_frame_idx,
                                        start_time=cand.start_time,
                                        end_time=cand.end_time,
                                        bbox=cand.bbox,
                                        image=cand.image,
                                        full_image=cand.full_image,
                                        kind=cand.kind,
                                        embedding=emb_vec,
                                    )
                                )
                                new_embeddings.append(emb_vec)
                        if new_embeddings:
                            new_emb_t = torch.stack(new_embeddings)
                            if pool_embeddings is None:
                                pool_embeddings = new_emb_t
                            else:
                                pool_embeddings = torch.cat([pool_embeddings, new_emb_t], dim=0)
                        del cand_embeddings, new_embeddings

                        # Only emit non-duplicate candidates
                        frame_candidates = [c for c, k in zip(frame_candidates, keep_mask) if k]
                    else:
                        # No cross-frame dedup — attach embeddings for reuse
                        for i, cand in enumerate(frame_candidates):
                            cand.embedding = cand_embeddings[i]
                        del cand_embeddings

            # Close crop images used only for embedding
            if image_store is not None:
                for crop_img in frame_crop_images:
                    try:
                        crop_img.close()
                    except Exception:
                        pass

            # Emit surviving candidates
            for cand in frame_candidates:
                pending_candidates.append(cand)
                added_candidates += 1
                if len(pending_candidates) >= candidate_batch_limit:
                    flush_candidates()

            if image_store is not None:
                try:
                    frame_image.close()
                except Exception:
                    pass

        _SEG_PREFETCH = int(os.getenv("SEG_PREFETCH", "3"))

        processed_idx = 0
        try:
            if frame_indices:
                # Pre-decode all frames, submit segmentation concurrently
                decoded_frames: list[tuple[int, Image.Image]] = []
                for frame_idx in frame_indices:
                    if cancel_cb and cancel_cb():
                        raise TaskCancelled("Task canceled")
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    decode_start = time.perf_counter()
                    ok, frame = cap.read()
                    if not ok:
                        if stats is not None:
                            stats["decode_s"] = stats.get("decode_s", 0.0) + (
                                time.perf_counter() - decode_start
                            )
                        continue
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_image = Image.fromarray(frame_rgb)
                    if stats is not None:
                        stats["decode_s"] = stats.get("decode_s", 0.0) + (
                            time.perf_counter() - decode_start
                        )
                    decoded_frames.append((frame_idx, frame_image))

                # Submit segmentation requests with prefetch
                with ThreadPoolExecutor(max_workers=_SEG_PREFETCH) as seg_pool:
                    futures: list[tuple[int, Image.Image, Future]] = []
                    for frame_idx, frame_image in decoded_frames:
                        fut = seg_pool.submit(_segment_frame, frame_image)
                        futures.append((frame_idx, frame_image, fut))

                    for frame_idx, frame_image, fut in futures:
                        if cancel_cb and cancel_cb():
                            # Cancel remaining futures
                            for _, _, pending_fut in futures:
                                pending_fut.cancel()
                            raise TaskCancelled("Task canceled")
                        segmented, seg_elapsed = fut.result()
                        _post_process_frame(frame_idx, frame_image, segmented, seg_elapsed)
                        processed_idx += 1
                        if progress_cb and processed_idx % progress_step == 0 and total_frames:
                            progress_cb(processed_idx, total_frames)
            else:
                # Sequential range: decode + prefetch segmentation
                frame_idx = start_frame
                with ThreadPoolExecutor(max_workers=_SEG_PREFETCH) as seg_pool:
                    pending: list[tuple[int, Image.Image, Future]] = []

                    def _drain_one() -> None:
                        nonlocal processed_idx
                        fidx, fimg, fut = pending.pop(0)
                        segmented, seg_elapsed = fut.result()
                        _post_process_frame(fidx, fimg, segmented, seg_elapsed)
                        processed_idx += 1
                        if progress_cb and processed_idx % progress_step == 0 and total_frames:
                            progress_cb(processed_idx, total_frames)

                    while processed_idx < total_frames:
                        if cancel_cb and cancel_cb():
                            for _, _, pfut in pending:
                                pfut.cancel()
                            raise TaskCancelled("Task canceled")
                        decode_start = time.perf_counter()
                        ok, frame = cap.read()
                        if not ok:
                            if stats is not None:
                                stats["decode_s"] = stats.get("decode_s", 0.0) + (
                                    time.perf_counter() - decode_start
                                )
                            break
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frame_image = Image.fromarray(frame_rgb)
                        if stats is not None:
                            stats["decode_s"] = stats.get("decode_s", 0.0) + (
                                time.perf_counter() - decode_start
                            )
                        fut = seg_pool.submit(_segment_frame, frame_image)
                        pending.append((frame_idx, frame_image, fut))
                        frame_idx += 1
                        # Drain completed futures to keep prefetch bounded
                        while len(pending) > _SEG_PREFETCH:
                            _drain_one()
                    # Drain remaining
                    while pending:
                        _drain_one()
        finally:
            cap.release()

        flush_candidates()

        # Clean up pool embeddings
        del pool_embeddings
        tracked_pool.clear()

        if stats is not None:
            stats["frames"] = stats.get("frames", 0.0) + float(processed_idx)
            stats["candidates"] = stats.get("candidates", 0.0) + float(added_candidates)
            stats["dedup_removed"] = stats.get("dedup_removed", 0.0) + float(dedup_removed)
            stats["intra_dedup_removed"] = stats.get("intra_dedup_removed", 0.0) + float(intra_dedup_removed)
        return next_obj_id, added_candidates

    def detect(
        self,
        query_image: Image.Image,
        video_path: Path,
        start_time: float | None = None,
        end_time: float | None = None,
        progress_cb: callable | None = None,
        candidate_cb: callable | None = None,
        cancel_cb: callable | None = None,
    ) -> VideoDetectionResult:
        results = self.detect_batch_queries(
            [query_image],
            video_path,
            start_time=start_time,
            end_time=end_time,
            progress_cb=progress_cb,
            candidate_cb=(lambda idx, candidate: candidate_cb(candidate))
            if candidate_cb
            else None,
            cancel_cb=cancel_cb,
        )
        return results[0]

    def detect_batch_queries(
        self,
        query_images: list[Image.Image],
        video_path: Path,
        start_time: float | None = None,
        end_time: float | None = None,
        progress_cb: callable | None = None,
        candidate_cb: callable | None = None,
        cancel_cb: callable | None = None,
        embedding_threshold: float | None = None,
        cache_dir: Path | None = None,
    ) -> list[VideoDetectionResult]:
        if not query_images:
            return []
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open video: {video_path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 30.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = frame_count / fps if fps else 0.0
        cap.release()
        if frame_count <= 0:
            raise ValueError("视频帧数为空，无法处理")

        start_time = 0.0 if start_time is None else float(start_time)
        end_time = duration if end_time is None else float(end_time)
        start_time = max(0.0, start_time)
        end_time = max(0.0, end_time)
        if duration:
            start_time = min(start_time, duration)
            end_time = min(end_time, duration)
        if end_time <= start_time:
            min_span = 1.0 / fps if fps else 0.0
            if duration:
                if start_time >= duration:
                    start_time = max(0.0, duration - min_span)
                    end_time = duration
                else:
                    end_time = min(duration, start_time + min_span)
            else:
                end_time = start_time + min_span

        start_frame = max(0, min(frame_count - 1, int(start_time * fps)))
        end_frame = int(end_time * fps) - 1
        end_frame = max(start_frame, min(frame_count - 1, end_frame))

        keyframe_start = time.perf_counter()
        keyframe_progress = None
        if progress_cb:
            def keyframe_progress(done: int, total: int) -> None:
                progress_cb(done, total, "keyframe")
        frame_indices = self._keyframe_extractor.extract_indices(
            video_path,
            start_frame=start_frame,
            end_frame=end_frame,
            fps=fps,
            progress_cb=keyframe_progress,
        )
        if not frame_indices:
            frame_indices = [start_frame]
        keyframe_s = time.perf_counter() - keyframe_start

        matches_by_query: list[list[VideoMatchCandidate]] = [[] for _ in query_images]
        best_overall_scores: list[float | None] = [None for _ in query_images]
        best_overall_matches: list[MatchResult | None] = [None for _ in query_images]
        candidate_count = 0
        next_obj_id = 1
        stats: dict[str, float] = {"keyframe_s": keyframe_s}
        image_store = None
        image_counter = 0
        temp_dir_ctx: tempfile.TemporaryDirectory[str] | None = None
        if cache_dir is None:
            temp_dir_ctx = tempfile.TemporaryDirectory(prefix="video_candidates_")
            cache_dir = Path(temp_dir_ctx.name)
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)

            def store_image(image: Image.Image, prefix: str) -> Path:
                nonlocal image_counter
                image_counter += 1
                path = cache_dir / f"{prefix}_{image_counter:08d}.png"
                image.save(path)
                return path

            image_store = store_image

        total_start = time.perf_counter()

        try:
            query_embeddings, query_sizes = self._matcher.embed_images(query_images)
            if query_embeddings.numel() == 0:
                self._matcher.release_embed_model()
                total_s = time.perf_counter() - total_start
                stats["total_s"] = total_s
                log_perf(
                    "video_detect",
                    queries=len(query_images),
                    candidates=0,
                    keyframes=len(frame_indices),
                    frames=stats.get("frames", 0.0),
                    segment_s=stats.get("segment_s", 0.0),
                    decode_s=stats.get("decode_s", 0.0),
                    keyframe_s=stats.get("keyframe_s", 0.0),
                    match_s=stats.get("match_s", 0.0),
                    total_s=stats.get("total_s", 0.0),
                    fps=fps,
                    frame_count=frame_count,
                    duration=duration,
                    embedding_threshold=embedding_threshold,
                    intra_dedup_removed=0,
                )
                return [
                    VideoDetectionResult(
                        is_match=False,
                        best_score=None,
                        best_match=None,
                        candidates=0,
                        matches=[],
                        fps=fps,
                        frame_count=frame_count,
                        duration=duration,
                    )
                    for _ in query_images
                ]

            query_embeddings = query_embeddings.float()
            if query_embeddings.device != self._matcher.device:
                query_embeddings = query_embeddings.to(self._matcher.device)
            # Ensure query embeddings stay on device for matching
            query_embeddings_device = query_embeddings
            total_queries = len(query_images)
            match_threshold = (
                embedding_threshold
                if embedding_threshold is not None
                else self._matcher.embedding_threshold
            )
            min_sim = match_threshold
            candidate_batch_limit = max(1, int(os.getenv("VIDEO_CANDIDATE_BATCH_SIZE", "1024")))
            dedup_threshold = float(os.getenv("VIDEO_DEDUP_THRESHOLD", "0.90"))
            intra_dedup_threshold = float(os.getenv("INTRA_DEDUP_THRESHOLD", "0.85"))

            filter_s = 0.0
            # --- Phase 1: Segment + embed + incremental filter ---
            # Instead of accumulating all embeddings, filter each batch
            # immediately and only keep pairs that pass the threshold.
            filtered_pairs: list[tuple[int, float]] = []  # (q_idx, sim)
            filtered_meta: list[dict] = []  # metadata for each passing pair

            def _rel_path(path: Path | None) -> str | None:
                if path is None:
                    return None
                try:
                    return str(path.relative_to(cache_dir))
                except Exception:
                    return str(path)

            def process_candidate_batch(batch_candidates: list[_VideoCandidate]) -> None:
                nonlocal filter_s
                if not batch_candidates:
                    return
                if cancel_cb and cancel_cb():
                    raise TaskCancelled("Task canceled")

                # Check if all candidates have pre-computed embeddings (from dedup)
                precomputed = [c.embedding for c in batch_candidates]
                all_precomputed = all(e is not None for e in precomputed)

                candidate_images = [candidate.image for candidate in batch_candidates]
                batch_images, close_images = _prepare_images(candidate_images)
                batch_start = time.perf_counter()
                try:
                    if all_precomputed:
                        candidate_embeddings = torch.stack(precomputed)
                    else:
                        candidate_embeddings, _ = self._matcher.embed_images(batch_images)
                    if candidate_embeddings.numel() == 0:
                        return
                    candidate_embeddings = candidate_embeddings.float()
                    if candidate_embeddings.device != query_embeddings_device.device:
                        candidate_embeddings = candidate_embeddings.to(query_embeddings_device.device)
                    # Incremental matmul + filter
                    sim = query_embeddings_device @ candidate_embeddings.T
                    del candidate_embeddings
                    qi, ci = torch.where(sim >= min_sim)
                    if qi.numel() > 0:
                        sv = sim[qi, ci]
                        q_list = qi.cpu().tolist()
                        c_list = ci.cpu().tolist()
                        s_list = sv.cpu().tolist()
                        del qi, ci, sv
                        for i in range(len(q_list)):
                            c_local = c_list[i]
                            cand = batch_candidates[c_local]
                            filtered_pairs.append((q_list[i], s_list[i]))
                            filtered_meta.append({
                                "obj": cand.obj_id,
                                "f0": cand.first_frame_idx,
                                "f1": cand.last_frame_idx,
                                "st": cand.start_time,
                                "et": cand.end_time,
                                "bbox": cand.bbox,
                                "img": _rel_path(cand.image)
                                if isinstance(cand.image, Path)
                                else None,
                                "full": _rel_path(cand.full_image)
                                if isinstance(cand.full_image, Path)
                                else None,
                                "kind": cand.kind,
                            })
                    else:
                        del qi, ci
                    del sim
                finally:
                    filter_s += time.perf_counter() - batch_start
                    for image in close_images:
                        try:
                            image.close()
                        except Exception:
                            pass

            segment_progress = None
            if progress_cb:
                def segment_progress(done: int, total: int) -> None:
                    progress_cb(done, total, "segmenting")

            next_obj_id, added = self._detect_batch_queries_range(
                video_path,
                start_frame=start_frame,
                end_frame=end_frame,
                frame_indices=frame_indices,
                fps=fps,
                progress_cb=segment_progress,
                cancel_cb=cancel_cb,
                next_obj_id=next_obj_id,
                stats=stats,
                image_store=image_store,
                candidate_batch_cb=process_candidate_batch,
                candidate_batch_limit=candidate_batch_limit,
                dedup_threshold=dedup_threshold,
                intra_dedup_threshold=intra_dedup_threshold,
            )
            candidate_count += added

            stats["match_s"] = stats.get("match_s", 0.0) + filter_s

            self._matcher.release_embed_model()
            del query_embeddings, query_embeddings_device

            if candidate_count == 0 or not filtered_pairs:
                total_s = time.perf_counter() - total_start
                stats["total_s"] = total_s
                log_perf(
                    "video_detect",
                    queries=len(query_images),
                    candidates=0,
                    keyframes=len(frame_indices),
                    frames=stats.get("frames", 0.0),
                    segment_s=stats.get("segment_s", 0.0),
                    decode_s=stats.get("decode_s", 0.0),
                    keyframe_s=stats.get("keyframe_s", 0.0),
                    match_s=stats.get("match_s", 0.0),
                    total_s=stats.get("total_s", 0.0),
                    fps=fps,
                    frame_count=frame_count,
                    duration=duration,
                    embedding_threshold=embedding_threshold,
                    dedup_removed=int(stats.get("dedup_removed", 0)),
                    dedup_embed_s=stats.get("dedup_embed_s", 0.0),
                    intra_dedup_removed=int(stats.get("intra_dedup_removed", 0)),
                )
                return [
                    VideoDetectionResult(
                        is_match=False,
                        best_score=None,
                        best_match=None,
                        candidates=0,
                        matches=[],
                        fps=fps,
                        frame_count=frame_count,
                        duration=duration,
                    )
                    for _ in query_images
                ]

            # Sort filtered pairs by query index for RoMa cache optimization
            sort_order = sorted(range(len(filtered_pairs)), key=lambda i: filtered_pairs[i][0])
            filtered_pairs = [filtered_pairs[i] for i in sort_order]
            filtered_meta = [filtered_meta[i] for i in sort_order]
            del sort_order
            filtered_pairs_count = len(filtered_pairs)

            if filtered_pairs_count == 0:
                total_s = time.perf_counter() - total_start
                stats["total_s"] = total_s
                results = [
                    VideoDetectionResult(
                        is_match=False,
                        best_score=None,
                        best_match=None,
                        candidates=candidate_count,
                        matches=[],
                        fps=fps,
                        frame_count=frame_count,
                        duration=duration,
                    )
                    for _ in query_images
                ]
                log_perf(
                    "video_detect",
                    queries=len(query_images),
                    candidates=candidate_count,
                    keyframes=len(frame_indices),
                    frames=stats.get("frames", 0.0),
                    segment_s=stats.get("segment_s", 0.0),
                    decode_s=stats.get("decode_s", 0.0),
                    keyframe_s=stats.get("keyframe_s", 0.0),
                    match_s=stats.get("match_s", 0.0),
                    total_s=stats.get("total_s", 0.0),
                    fps=fps,
                    frame_count=frame_count,
                    duration=duration,
                    embedding_threshold=embedding_threshold,
                    dedup_removed=int(stats.get("dedup_removed", 0)),
                    dedup_embed_s=stats.get("dedup_embed_s", 0.0),
                    intra_dedup_removed=int(stats.get("intra_dedup_removed", 0)),
                )
                return results

            # --- Phase 2: RoMa matching only on filtered pairs ---
            self._matcher.prepare_matcher()
            match_start = time.perf_counter()
            match_threshold = (
                embedding_threshold
                if embedding_threshold is not None
                else self._matcher.embedding_threshold
            )
            _has_pipeline = hasattr(self._matcher, '_extract_keypoints')

            def _record_match(pair_idx: int, match_result: MatchResult) -> None:
                q_idx, sim_val = filtered_pairs[pair_idx]
                meta = filtered_meta[pair_idx]
                score = match_result.score
                if (
                    best_overall_scores[q_idx] is None
                    or score > best_overall_scores[q_idx]
                ):
                    best_overall_scores[q_idx] = score
                    best_overall_matches[q_idx] = match_result
                if not match_result.is_match:
                    return
                bbox = meta.get("bbox")
                bbox = tuple(bbox) if bbox is not None else None
                full_value = meta.get("full")
                if full_value is not None:
                    full_path = Path(full_value)
                    if not full_path.is_absolute():
                        full_path = cache_dir / full_path
                else:
                    img_value = meta.get("img")
                    full_path = cache_dir / img_value if img_value else None
                img_value = meta.get("img")
                img_path = Path(img_value) if img_value else None
                if img_path is not None and not img_path.is_absolute():
                    img_path = cache_dir / img_path
                candidate = VideoMatchCandidate(
                    obj_id=int(meta.get("obj", 0)),
                    first_frame_idx=int(meta.get("f0", 0)),
                    last_frame_idx=int(meta.get("f1", 0)),
                    start_time=float(meta.get("st", 0.0)),
                    end_time=float(meta.get("et", 0.0)),
                    bbox=bbox,
                    match=match_result,
                    score=score,
                    image=img_path,
                    full_image=full_path,
                    kind=meta.get("kind", "crop"),
                )
                matches_by_query[q_idx].append(candidate)
                if candidate_cb:
                    candidate_cb(q_idx, candidate)

            if _has_pipeline:
                # GPU/CPU pipeline: same pattern as detection.py
                _ransac_workers = int(os.getenv("RANSAC_WORKERS", "4"))
                _ransac_max_pending = _ransac_workers * 2
                ransac_pool = ThreadPoolExecutor(max_workers=_ransac_workers)
                pending_ransac: list[tuple[int, Future]] = []

                def _build_match_result(
                    sim_val: float, ransac_future: Future
                ) -> MatchResult:
                    ok, rstats = ransac_future.result()
                    inliers = int(rstats["inliers"])
                    inlier_ratio = float(rstats["inlier_ratio"])
                    total_matches = int(rstats["total_matches"])
                    score = inlier_ratio if ok else 0.0
                    is_match = ok and inliers >= self._matcher._min_inliers and score > 0.0
                    return MatchResult(
                        embedding_similarity=sim_val,
                        embedding_pass=sim_val >= match_threshold,
                        ransac_ok=ok,
                        inliers=inliers,
                        total_matches=total_matches,
                        inlier_ratio=inlier_ratio,
                        score=score,
                        is_match=is_match,
                    )

                def _drain_pending(limit: int) -> None:
                    while len(pending_ransac) > limit:
                        pidx, fut = pending_ransac.pop(0)
                        q_idx, sim_val = filtered_pairs[pidx]
                        mr = _build_match_result(sim_val, fut)
                        _record_match(pidx, mr)

                for pair_idx in range(filtered_pairs_count):
                    q_idx, sim_val = filtered_pairs[pair_idx]
                    meta = filtered_meta[pair_idx]
                    if cancel_cb and cancel_cb():
                        raise TaskCancelled("Task canceled")
                    img_value = meta.get("img")
                    if img_value is None:
                        continue
                    img_path = Path(img_value)
                    if not img_path.is_absolute():
                        img_path = cache_dir / img_path
                    with Image.open(img_path) as image:
                        candidate_image = image.convert("RGB")
                    candidate_size = candidate_image.size
                    try:
                        kpts_a, kpts_b, total = self._matcher._extract_keypoints(
                            query_images[q_idx],
                            candidate_image,
                            query_sizes[q_idx],
                            candidate_size,
                        )
                    finally:
                        try:
                            candidate_image.close()
                        except Exception:
                            pass
                    fut = ransac_pool.submit(self._matcher._ransac, kpts_a, kpts_b, total)
                    pending_ransac.append((pair_idx, fut))
                    _drain_pending(_ransac_max_pending)

                _drain_pending(0)
                ransac_pool.shutdown(wait=True)
            else:
                # Remote matcher fallback: sequential calls
                for pair_idx in range(filtered_pairs_count):
                    q_idx, sim_val = filtered_pairs[pair_idx]
                    meta = filtered_meta[pair_idx]
                    if cancel_cb and cancel_cb():
                        raise TaskCancelled("Task canceled")
                    img_value = meta.get("img")
                    if img_value is None:
                        continue
                    img_path = Path(img_value)
                    if not img_path.is_absolute():
                        img_path = cache_dir / img_path
                    with Image.open(img_path) as image:
                        candidate_image = image.convert("RGB")
                    try:
                        candidate_size = candidate_image.size
                        match_result = self._matcher.match_with_similarity(
                            query_images[q_idx],
                            candidate_image,
                            query_sizes[q_idx],
                            candidate_size,
                            sim_val,
                            embedding_threshold=embedding_threshold,
                        )
                    finally:
                        try:
                            candidate_image.close()
                        except Exception:
                            pass
                    _record_match(pair_idx, match_result)

            self._matcher.release_matcher()
            stats["match_s"] = stats.get("match_s", 0.0) + (
                time.perf_counter() - match_start
            )
        finally:
            if temp_dir_ctx is not None:
                temp_dir_ctx.cleanup()

        total_s = time.perf_counter() - total_start
        stats["total_s"] = total_s
        if stats.get("frames", 0.0) > 0:
            logger.info(
                "video_pipeline timing total_s=%.3f keyframe_s=%.3f decode_s=%.3f segment_s=%.3f match_s=%.3f frames=%.0f candidates=%.0f intra_dedup_removed=%.0f",
                stats.get("total_s", 0.0),
                stats.get("keyframe_s", 0.0),
                stats.get("decode_s", 0.0),
                stats.get("segment_s", 0.0),
                stats.get("match_s", 0.0),
                stats.get("frames", 0.0),
                stats.get("candidates", 0.0),
                stats.get("intra_dedup_removed", 0.0),
            )

        results: list[VideoDetectionResult] = []
        for idx, matches in enumerate(matches_by_query):
            is_match = len(matches) > 0
            results.append(
                VideoDetectionResult(
                    is_match=is_match,
                    best_score=best_overall_scores[idx],
                    best_match=best_overall_matches[idx],
                    candidates=candidate_count,
                    matches=matches,
                    fps=fps,
                    frame_count=frame_count,
                    duration=duration,
                )
            )
        log_perf(
            "video_detect",
            queries=len(query_images),
            candidates=candidate_count,
            keyframes=len(frame_indices),
            frames=stats.get("frames", 0.0),
            segment_s=stats.get("segment_s", 0.0),
            decode_s=stats.get("decode_s", 0.0),
            keyframe_s=stats.get("keyframe_s", 0.0),
            match_s=stats.get("match_s", 0.0),
            total_s=stats.get("total_s", 0.0),
            fps=fps,
            frame_count=frame_count,
            duration=duration,
            embedding_threshold=embedding_threshold,
            dedup_removed=int(stats.get("dedup_removed", 0)),
            dedup_embed_s=stats.get("dedup_embed_s", 0.0),
            intra_dedup_removed=int(stats.get("intra_dedup_removed", 0)),
        )
        return results
