from __future__ import annotations

from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, Future
import gc
import logging
import os
import threading
import time

import numpy as np
from PIL import Image
import torch

from app.core.tasks import TaskCancelled
from app.core.gpu_lock import gpu_lock
from app.core.cache import LRUCache, image_cache_key
from app.core.perf import log_perf
from app.services.sam3_service import Sam3Segmenter
from app.services.roma_service import MatchResult, RomaMatcher

logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    is_match: bool
    best_bbox: tuple[int, int, int, int] | None
    best_score: float | None
    best_match: MatchResult | None
    candidates: int
    candidate_results: list["CandidateResult"]


@dataclass
class CandidateResult:
    kind: str
    bbox: tuple[int, int, int, int] | None
    match: MatchResult
    score: float
    image: Image.Image


@dataclass
class _SegmentCandidate:
    kind: str
    bbox: tuple[int, int, int, int] | None
    image: Image.Image


@dataclass
class _CandidateCacheEntry:
    candidates: list[_SegmentCandidate]
    embeddings: torch.Tensor | None
    sizes: list[tuple[int, int]] | None


def _crop_with_mask_np(
    image_np: np.ndarray,
    mask: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> Image.Image:
    x0, y0, x1, y1 = bbox
    crop = image_np[y0:y1, x0:x1].copy()
    if mask.shape[0] == (y1 - y0) and mask.shape[1] == (x1 - x0):
        mask_crop = mask
    else:
        mask_crop = mask[y0:y1, x0:x1]
    if mask_crop.ndim != 2:
        mask_crop = mask_crop.squeeze()
    if crop.ndim == 2:
        crop = np.stack([crop] * 3, axis=-1)
    # White background for non-mask area
    crop[~mask_crop] = 255
    return Image.fromarray(crop)


def _crop_with_mask(
    image: Image.Image, mask: np.ndarray, bbox: tuple[int, int, int, int]
) -> Image.Image:
    return _crop_with_mask_np(np.array(image), mask, bbox)


def _build_segment_candidates(
    segments: list, target_image: Image.Image
) -> list[_SegmentCandidate]:
    candidates: list[_SegmentCandidate] = []
    image_np = np.array(target_image)
    for segment in segments:
        crop = _crop_with_mask_np(image_np, segment.mask, segment.bbox)
        candidates.append(
            _SegmentCandidate(
                kind="segment",
                bbox=segment.bbox,
                image=crop,
            )
        )
    return candidates


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


class DetectorPipeline:
    def __init__(
        self,
        segmenter: Sam3Segmenter,
        matcher: RomaMatcher,
        candidate_store_mode: str = "matches",
        cache_size: int = 16,
    ) -> None:
        self._segmenter = segmenter
        self._matcher = matcher
        self._candidate_store_mode = candidate_store_mode
        self._candidate_cache = LRUCache[str, _CandidateCacheEntry](cache_size)
        self._candidate_cache_lock = threading.Lock()

    def clear_cache(self) -> None:
        with self._candidate_cache_lock:
            self._candidate_cache.clear()

    def detect(
        self,
        query_image: Image.Image,
        target_image: Image.Image,
        progress_cb: callable | None = None,
        candidate_cb: callable | None = None,
        cancel_cb: callable | None = None,
        embedding_threshold: float | None = None,
    ) -> DetectionResult:
        results = self.detect_batch_queries(
            [query_image],
            target_image,
            progress_cb=(lambda _, candidate_idx, __, total: progress_cb(candidate_idx, total))
            if progress_cb
            else None,
            candidate_cb=(lambda _, candidate, candidate_idx, total: candidate_cb(candidate, candidate_idx, total))
            if candidate_cb
            else None,
            cancel_cb=cancel_cb,
            embedding_threshold=embedding_threshold,
        )
        return results[0]

    def detect_batch_queries(
        self,
        query_images: list[Image.Image],
        target_image: Image.Image,
        progress_cb: callable | None = None,
        candidate_cb: callable | None = None,
        cancel_cb: callable | None = None,
        embedding_threshold: float | None = None,
    ) -> list[DetectionResult]:
        start_ts = time.perf_counter()
        segment_s = 0.0
        embed_s = 0.0
        match_s = 0.0
        match_calls = 0
        selected_candidates = 0
        target_key = image_cache_key(target_image)
        cache_entry = None
        candidates: list[_SegmentCandidate] = []
        segment_s = 0.0
        with self._candidate_cache_lock:
            cache_entry = self._candidate_cache.get(target_key)
        if cache_entry is not None:
            candidates = cache_entry.candidates
        else:
            segment_start = time.perf_counter()
            segments = self._segmenter.segment(target_image)
            segment_s = time.perf_counter() - segment_start
            candidates = _build_segment_candidates(segments, target_image)
            cache_entry = _CandidateCacheEntry(
                candidates=candidates,
                embeddings=None,
                sizes=None,
            )
            with self._candidate_cache_lock:
                self._candidate_cache.set(target_key, cache_entry)

        if not candidates or not query_images:
            total_s = time.perf_counter() - start_ts
            log_perf(
                "image_detect",
                queries=len(query_images),
                candidates=len(candidates),
                selected=0,
                match_calls=0,
                segment_s=segment_s,
                embed_s=embed_s,
                match_s=match_s,
                total_s=total_s,

                embedding_threshold=float(
                    embedding_threshold
                    if embedding_threshold is not None
                    else self._matcher.embedding_threshold
                ),
            )
            return [
                DetectionResult(
                    is_match=False,
                    best_bbox=None,
                    best_score=None,
                    best_match=None,
                    candidates=0,
                    candidate_results=[],
                )
                for _ in query_images
            ]

        candidate_images = [candidate.image for candidate in candidates]
        total_queries = len(query_images)
        total_candidates = len(candidates)
        candidate_results_by_query: list[list[CandidateResult]] = [
            [] for _ in query_images
        ]
        best_candidates: list[CandidateResult | None] = [None for _ in query_images]
        store_mode = self._candidate_store_mode

        candidate_embeddings = cache_entry.embeddings
        candidate_sizes = cache_entry.sizes
        embed_start = time.perf_counter()
        # Move cached CPU embeddings to CUDA if needed
        if candidate_embeddings is not None and candidate_embeddings.device != self._matcher.device:
            candidate_embeddings = candidate_embeddings.to(self._matcher.device)
        with gpu_lock():
            if candidate_embeddings is None or candidate_sizes is None:
                candidate_embeddings, candidate_sizes = self._matcher.embed_images(
                    candidate_images
                )
                if candidate_embeddings.numel() == 0:
                    embed_s += time.perf_counter() - embed_start
                    total_s = time.perf_counter() - start_ts
                    log_perf(
                        "image_detect",
                        queries=total_queries,
                        candidates=total_candidates,
                        selected=0,
                        match_calls=0,
                        segment_s=segment_s,
                        embed_s=embed_s,
                        match_s=match_s,
                        total_s=total_s,
        
                        embedding_threshold=float(
                            embedding_threshold
                            if embedding_threshold is not None
                            else self._matcher.embedding_threshold
                        ),
                    )
                    return [
                        DetectionResult(
                            is_match=False,
                            best_bbox=None,
                            best_score=None,
                            best_match=None,
                            candidates=0,
                            candidate_results=[],
                        )
                        for _ in query_images
                    ]
                candidate_embeddings = candidate_embeddings.float()
                # --- Intra-image dedup: remove near-duplicate segments ---
                intra_dedup_threshold = float(os.getenv("INTRA_DEDUP_THRESHOLD", "0.85"))
                if intra_dedup_threshold > 0.0 and candidate_embeddings.shape[0] > 1:
                    intra_sim = candidate_embeddings @ candidate_embeddings.T
                    n = candidate_embeddings.shape[0]
                    intra_keep = [True] * n
                    intra_removed = 0
                    for i in range(n):
                        if not intra_keep[i]:
                            continue
                        for j in range(i + 1, n):
                            if not intra_keep[j]:
                                continue
                            if float(intra_sim[i, j]) >= intra_dedup_threshold:
                                intra_keep[j] = False
                                intra_removed += 1
                    del intra_sim
                    if intra_removed > 0:
                        keep_idx = [i for i, k in enumerate(intra_keep) if k]
                        candidates = [candidates[i] for i in keep_idx]
                        candidate_images = [candidate_images[i] for i in keep_idx]
                        candidate_embeddings = candidate_embeddings[keep_idx]
                        candidate_sizes = [candidate_sizes[i] for i in keep_idx]
                        cache_entry.candidates = candidates
                        total_candidates = len(candidates)
                        logger.info(
                            "intra_dedup removed %d/%d segments (threshold=%.2f)",
                            intra_removed, intra_removed + len(candidates), intra_dedup_threshold,
                        )
                # Store CPU copy in cache to avoid holding CUDA memory between tasks
                cache_embeddings = candidate_embeddings.cpu()
                with self._candidate_cache_lock:
                    cache_entry.embeddings = cache_embeddings
                    cache_entry.sizes = candidate_sizes
                    self._candidate_cache.set(target_key, cache_entry)
                # Keep CUDA version for current task use
                candidate_embeddings = candidate_embeddings.to(self._matcher.device)
        embed_s += time.perf_counter() - embed_start
        match_threshold = (
            self._matcher.embedding_threshold
            if embedding_threshold is None
            else float(embedding_threshold)
        )
        min_sim = match_threshold

        def record_candidate(
            q_idx: int, candidate_idx: int, match_result: MatchResult
        ) -> None:
            candidate = candidates[candidate_idx]
            record = CandidateResult(
                kind=candidate.kind,
                bbox=candidate.bbox,
                match=match_result,
                score=match_result.score,
                image=candidate.image,
            )
            if store_mode == "all":
                candidate_results_by_query[q_idx].append(record)
                if (
                    best_candidates[q_idx] is None
                    or record.score > best_candidates[q_idx].score
                ):
                    best_candidates[q_idx] = record
                if candidate_cb:
                    candidate_cb(q_idx, record, candidate_idx + 1, total_candidates)
                return
            if not match_result.is_match:
                return
            if store_mode == "matches":
                candidate_results_by_query[q_idx].append(record)
            elif store_mode == "best":
                if (
                    best_candidates[q_idx] is None
                    or record.score > best_candidates[q_idx].score
                ):
                    candidate_results_by_query[q_idx] = [record]
                    best_candidates[q_idx] = record
            if store_mode != "best":
                if (
                    best_candidates[q_idx] is None
                    or record.score > best_candidates[q_idx].score
                ):
                    best_candidates[q_idx] = record
                if candidate_cb:
                    candidate_cb(q_idx, record, candidate_idx + 1, total_candidates)

        match_start = time.perf_counter()
        # Embed all queries at once
        with gpu_lock():
            query_embeddings, query_sizes = self._matcher.embed_images(query_images)
            if query_embeddings.numel() == 0:
                embed_s += time.perf_counter() - match_start
                total_s = time.perf_counter() - start_ts
                log_perf(
                    "image_detect",
                    queries=total_queries,
                    candidates=total_candidates,
                    selected=0,
                    match_calls=0,
                    segment_s=segment_s,
                    embed_s=embed_s,
                    match_s=match_s,
                    total_s=total_s,

                    embedding_threshold=float(
                        embedding_threshold
                        if embedding_threshold is not None
                        else self._matcher.embedding_threshold
                    ),
                )
                return [
                    DetectionResult(
                        is_match=False,
                        best_bbox=None,
                        best_score=None,
                        best_match=None,
                        candidates=0,
                        candidate_results=[],
                    )
                    for _ in query_images
                ]
            query_embeddings = query_embeddings.float()
            self._matcher.clear_cache()
            self._matcher.log_stage("embedding_done")
            self._matcher.release_embed_model()
        embed_s += time.perf_counter() - match_start

        # Matmul + vectorized filtering
        device = self._matcher.device
        # Ensure both tensors are on the same device for matmul
        if candidate_embeddings.device != query_embeddings.device:
            candidate_embeddings = candidate_embeddings.to(query_embeddings.device)

        # Estimate similarity matrix size: Q * C * 4 bytes (float32)
        mat_bytes = query_embeddings.shape[0] * candidate_embeddings.shape[0] * 4
        max_mat_bytes = int(os.getenv("SIM_MAX_BYTES", str(512 * 1024 * 1024)))

        if mat_bytes <= max_mat_bytes:
            # Small enough — single matmul on GPU, no chunking
            sim = query_embeddings @ candidate_embeddings.T
            qi, ci = torch.where(sim >= min_sim)
            sv = sim[qi, ci]
            del sim
        else:
            # Large matrix — chunk along query axis to cap memory
            chunk_rows = max(1, max_mat_bytes // (candidate_embeddings.shape[0] * 4))
            qi_parts: list[torch.Tensor] = []
            ci_parts: list[torch.Tensor] = []
            sv_parts: list[torch.Tensor] = []
            for q_start in range(0, query_embeddings.shape[0], chunk_rows):
                q_end = min(query_embeddings.shape[0], q_start + chunk_rows)
                sim_chunk = query_embeddings[q_start:q_end] @ candidate_embeddings.T
                q_local, c_local = torch.where(sim_chunk >= min_sim)
                if q_local.numel() > 0:
                    sv_parts.append(sim_chunk[q_local, c_local])
                    qi_parts.append(q_local + q_start)
                    ci_parts.append(c_local)
                del sim_chunk
            if qi_parts:
                qi = torch.cat(qi_parts)
                ci = torch.cat(ci_parts)
                sv = torch.cat(sv_parts)
            else:
                qi = torch.empty(0, dtype=torch.long, device=device)
                ci = torch.empty(0, dtype=torch.long, device=device)
                sv = torch.empty(0, device=device)
            del qi_parts, ci_parts, sv_parts

        del candidate_embeddings

        # Sort by query index so RoMa's query preprocessing cache hits
        if qi.numel() > 0:
            sort_idx = torch.argsort(qi, stable=True)
            qi = qi[sort_idx]
            ci = ci[sort_idx]
            sv = sv[sort_idx]
            del sort_idx
            q_list = qi.cpu().tolist()
            c_list = ci.cpu().tolist()
            s_list = sv.cpu().tolist()
        else:
            q_list = []
            c_list = []
            s_list = []
        del qi, ci, sv

        selected_candidates = len(q_list)

        # RoMa matching only on filtered pairs — pipeline GPU forward / CPU RANSAC
        with gpu_lock():
            self._matcher.prepare_matcher()
            try:
                _ransac_workers = int(os.getenv("RANSAC_WORKERS", "4"))
                _ransac_max_pending = _ransac_workers * 2
                ransac_pool = ThreadPoolExecutor(max_workers=_ransac_workers)
                pending_ransac: list[tuple[int, int, float, Future]] = []

                def _build_match_result(
                    sim_val: float, ransac_future: Future
                ) -> MatchResult:
                    ok, stats = ransac_future.result()
                    inliers = int(stats["inliers"])
                    inlier_ratio = float(stats["inlier_ratio"])
                    total_matches = int(stats["total_matches"])
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
                    """Drain completed RANSAC futures to keep memory bounded."""
                    while len(pending_ransac) > limit:
                        q_idx, candidate_idx, sim_val, fut = pending_ransac.pop(0)
                        match_result = _build_match_result(sim_val, fut)
                        record_candidate(q_idx, candidate_idx, match_result)

                for pair_idx in range(len(q_list)):
                    q_idx = q_list[pair_idx]
                    candidate_idx = c_list[pair_idx]
                    sim_val = s_list[pair_idx]
                    if cancel_cb and cancel_cb():
                        raise TaskCancelled("Task canceled")
                    match_calls += 1

                    # GPU: extract keypoints
                    kpts_a, kpts_b, total = self._matcher._extract_keypoints(
                        query_images[q_idx],
                        candidate_images[candidate_idx],
                        query_sizes[q_idx],
                        candidate_sizes[candidate_idx],
                    )
                    # Submit CPU RANSAC to thread pool (overlaps with next GPU forward)
                    fut = ransac_pool.submit(self._matcher._ransac, kpts_a, kpts_b, total)
                    pending_ransac.append((q_idx, candidate_idx, sim_val, fut))
                    # Drain oldest futures when queue is full to bound memory
                    _drain_pending(_ransac_max_pending)

                # Collect remaining RANSAC results
                _drain_pending(0)

                ransac_pool.shutdown(wait=True)

                if progress_cb:
                    progress_cb(total_queries, len(q_list), total_queries, total_candidates)
            finally:
                self._matcher.release_matcher()
                self._matcher.log_stage("matching_done")
        del query_embeddings, query_sizes
        match_s = time.perf_counter() - match_start

        results: list[DetectionResult] = []
        for q_idx in range(total_queries):
            candidate_results = candidate_results_by_query[q_idx]
            best_candidate = best_candidates[q_idx]
            is_match = any(candidate.match.is_match for candidate in candidate_results)
            best_bbox = best_candidate.bbox if best_candidate else None
            best_score = best_candidate.score if best_candidate else None
            best_match = best_candidate.match if best_candidate else None
            results.append(
                DetectionResult(
                    is_match=is_match,
                    best_bbox=best_bbox,
                    best_score=best_score,
                    best_match=best_match,
                    candidates=len(candidate_results),
                    candidate_results=candidate_results,
                )
            )

        total_s = time.perf_counter() - start_ts
        log_perf(
            "image_detect",
            queries=total_queries,
            candidates=total_candidates,
            selected=selected_candidates,
            match_calls=match_calls,
            segment_s=segment_s,
            embed_s=embed_s,
            match_s=match_s,
            total_s=total_s,
            embedding_threshold=float(
                embedding_threshold
                if embedding_threshold is not None
                else self._matcher.embedding_threshold
            ),
        )
        return results
