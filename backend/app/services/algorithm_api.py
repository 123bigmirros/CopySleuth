from __future__ import annotations

import io
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib import request as urlrequest
from urllib import error as urlerror

from PIL import Image

from app.services.algorithm import AlgorithmService


@dataclass(frozen=True)
class AlgorithmApiConfig:
    base_url: str
    timeout_s: float = 120.0
    image_mode: str | None = None


@dataclass(frozen=True)
class AlgoMatchResult:
    embedding_similarity: float
    embedding_pass: bool
    ransac_ok: bool
    inliers: int
    total_matches: int
    inlier_ratio: float
    score: float
    is_match: bool


@dataclass(frozen=True)
class AlgoCandidateResult:
    kind: str
    bbox: tuple[int, int, int, int] | None
    match: AlgoMatchResult
    score: float
    image: Image.Image | Path | None


@dataclass(frozen=True)
class AlgoDetectionResult:
    is_match: bool
    best_bbox: tuple[int, int, int, int] | None
    best_score: float | None
    best_match: AlgoMatchResult | None
    candidates: int
    candidate_results: list[AlgoCandidateResult]


@dataclass(frozen=True)
class AlgoVideoMatchCandidate:
    obj_id: int
    first_frame_idx: int
    last_frame_idx: int
    start_time: float
    end_time: float
    bbox: tuple[int, int, int, int] | None
    match: AlgoMatchResult
    score: float
    image: Image.Image | Path | None
    full_image: Image.Image | Path | None
    kind: str


@dataclass(frozen=True)
class AlgoVideoDetectionResult:
    is_match: bool
    best_score: float | None
    best_match: AlgoMatchResult | None
    candidates: int
    matches: list[AlgoVideoMatchCandidate]
    fps: float
    frame_count: int
    duration: float


class AlgorithmApiClient(AlgorithmService):
    def __init__(self, config: AlgorithmApiConfig) -> None:
        self._base_url = config.base_url.rstrip("/")
        self._timeout_s = float(config.timeout_s)
        image_mode = (config.image_mode or "").strip().lower()
        self._image_mode = image_mode if image_mode in {"path", "b64"} else None

    def detect_image(
        self,
        query_image: Image.Image,
        target_image: Image.Image,
        progress_cb=None,
        candidate_cb=None,
        cancel_cb=None,
        embedding_threshold: float | None = None,
    ) -> AlgoDetectionResult:
        results = self.detect_image_batch(
            [query_image],
            target_image,
            embedding_threshold=embedding_threshold,
        )
        return results[0]

    def detect_image_batch(
        self,
        query_images: list[Image.Image],
        target_image: Image.Image,
        progress_cb=None,
        candidate_cb=None,
        cancel_cb=None,
        embedding_threshold: float | None = None,
    ) -> list[AlgoDetectionResult]:
        if not query_images:
            return []
        files = []
        for idx, image in enumerate(query_images, start=1):
            files.append(
                ("query_images", f"query_{idx:03d}.png", "image/png", _image_to_bytes(image))
            )
        files.append(
            ("target_image", "target.png", "image/png", _image_to_bytes(target_image))
        )
        fields = []
        if embedding_threshold is not None:
            fields.append(("embedding_threshold", str(float(embedding_threshold))))
        if self._image_mode:
            fields.append(("image_mode", self._image_mode))
        body, content_type = _encode_multipart(fields=fields, files=files)
        payload = self._post_json("/v1/detect", body, content_type)
        results_payload = payload.get("results") or []
        return [parse_image_result(item) for item in results_payload]

    def detect_video_batch(
        self,
        query_images: list[Image.Image],
        video_path: Path,
        start_time: float | None = None,
        end_time: float | None = None,
        progress_cb=None,
        candidate_cb=None,
        cancel_cb=None,
        embedding_threshold: float | None = None,
        cache_dir: Path | None = None,
    ) -> list[AlgoVideoDetectionResult]:
        raise RuntimeError("Remote algorithm client does not support detect_video_batch")

    def _post_json(self, path: str, body: bytes, content_type: str) -> dict:
        url = f"{self._base_url}{path}"
        req = urlrequest.Request(
            url,
            data=body,
            headers={"Content-Type": content_type},
            method="POST",
        )
        try:
            with urlrequest.urlopen(req, timeout=self._timeout_s) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (TimeoutError, urlerror.URLError):
            raise


class AlgorithmTaskClient:
    def __init__(self, config: AlgorithmApiConfig) -> None:
        self._base_url = config.base_url.rstrip("/")
        self._timeout_s = float(config.timeout_s)

    def create_task(
        self,
        *,
        query_images: list[Image.Image],
        target_bytes: bytes,
        target_name: str,
        target_content_type: str,
        embedding_threshold: float | None = None,
        video_start: float | None = None,
        video_end: float | None = None,
        query_excel_bytes: bytes | None = None,
        query_excel_name: str | None = None,
    ) -> str:
        files = []
        if query_excel_bytes is not None:
            excel_name = query_excel_name or "query.xlsx"
            excel_ct = (
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
            files.append(("query_images", excel_name, excel_ct, query_excel_bytes))
        else:
            for idx, image in enumerate(query_images, start=1):
                files.append(
                    ("query_images", f"query_{idx:03d}.png", "image/png", _image_to_bytes(image))
                )
        files.append(("target_file", target_name, target_content_type, target_bytes))
        fields = []
        if embedding_threshold is not None:
            fields.append(("embedding_threshold", str(float(embedding_threshold))))
        if video_start is not None:
            fields.append(("video_start", str(float(video_start))))
        if video_end is not None:
            fields.append(("video_end", str(float(video_end))))
        body, content_type = _encode_multipart(fields=fields, files=files)
        payload = self._post_json("/v1/tasks", body, content_type)
        task_id = payload.get("task_id")
        if not task_id:
            raise RuntimeError("algorithm task_id missing")
        return str(task_id)

    def cancel_task(self, task_id: str) -> None:
        url = f"{self._base_url}/v1/tasks/{task_id}/cancel"
        req = urlrequest.Request(url, method="POST")
        try:
            with urlrequest.urlopen(req, timeout=self._timeout_s):
                return
        except Exception:
            return

    def iter_events(self, task_id: str):
        url = f"{self._base_url}/v1/tasks/{task_id}/events"
        req = urlrequest.Request(url, headers={"Accept": "text/event-stream"})
        with urlrequest.urlopen(req, timeout=self._timeout_s) as resp:
            event_type = None
            data_lines: list[str] = []
            for raw_line in resp:
                line = raw_line.decode("utf-8").rstrip("\n")
                if not line:
                    if data_lines:
                        payload_text = "\n".join(data_lines)
                        data = json.loads(payload_text) if payload_text else None
                        yield {"event": event_type or "message", "data": data}
                    event_type = None
                    data_lines = []
                    continue
                if line.startswith(":"):
                    yield {"event": "keep-alive", "data": None}
                    continue
                if line.startswith("event:"):
                    event_type = line[len("event:") :].strip()
                    continue
                if line.startswith("data:"):
                    data_lines.append(line[len("data:") :].lstrip())

    def _post_json(self, path: str, body: bytes, content_type: str) -> dict:
        url = f"{self._base_url}{path}"
        req = urlrequest.Request(
            url,
            data=body,
            headers={"Content-Type": content_type},
            method="POST",
        )
        try:
            with urlrequest.urlopen(req, timeout=self._timeout_s) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (TimeoutError, urlerror.URLError):
            raise


def parse_image_result(payload: dict) -> AlgoDetectionResult:
    candidates_payload = payload.get("candidate_results") or []
    candidate_results: list[AlgoCandidateResult] = []
    for item in candidates_payload:
        match = _match_from_payload(item.get("match"))
        candidate_results.append(
            AlgoCandidateResult(
                kind=item.get("kind", "segment"),
                bbox=_bbox_from_payload(item.get("bbox")),
                match=match or _empty_match(),
                score=float(item.get("score", 0.0)),
                image=None,
            )
        )
    best_match = _match_from_payload(payload.get("best_match"))
    if best_match is None and candidate_results:
        best_candidate = max(candidate_results, key=lambda c: c.score, default=None)
        best_match = best_candidate.match if best_candidate else None
    return AlgoDetectionResult(
        is_match=bool(payload.get("is_match", bool(candidate_results))),
        best_bbox=_bbox_from_payload(payload.get("best_bbox")),
        best_score=_float_or_none(payload.get("best_score")),
        best_match=best_match,
        candidates=int(payload.get("candidates", len(candidate_results))),
        candidate_results=candidate_results,
    )


def parse_video_result(payload: dict) -> AlgoVideoDetectionResult:
    segments = payload.get("segments") or payload.get("matches") or []
    matches: list[AlgoVideoMatchCandidate] = []
    for item in segments:
        match = _match_from_payload(item.get("match"))
        matches.append(
            AlgoVideoMatchCandidate(
                obj_id=int(item.get("obj_id", 0)),
                first_frame_idx=int(item.get("first_frame_index", 0)),
                last_frame_idx=int(item.get("last_frame_index", 0)),
                start_time=float(item.get("start_time", 0.0)),
                end_time=float(item.get("end_time", 0.0)),
                bbox=_bbox_from_payload(item.get("bbox")),
                match=match or _empty_match(),
                score=float(item.get("score", 0.0)),
                image=None,
                full_image=None,
                kind=item.get("kind", "crop"),
            )
        )
    best_match = _match_from_payload(payload.get("best_match"))
    if best_match is None and matches:
        best_candidate = max(matches, key=lambda c: c.score, default=None)
        best_match = best_candidate.match if best_candidate else None
    return AlgoVideoDetectionResult(
        is_match=bool(payload.get("is_match", bool(matches))),
        best_score=_float_or_none(payload.get("best_score")),
        best_match=best_match,
        candidates=int(payload.get("candidates", len(matches))),
        matches=matches,
        fps=float(payload.get("fps", 0.0)),
        frame_count=int(payload.get("frame_count", 0)),
        duration=float(payload.get("duration", 0.0)),
    )


def _image_to_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def _encode_multipart(
    fields: Iterable[tuple[str, str]],
    *,
    files: Iterable[tuple[str, str, str, bytes]],
) -> tuple[bytes, str]:
    boundary = "----algo-" + str(os.urandom(8).hex())
    body = bytearray()
    for name, value in fields:
        body.extend(f"--{boundary}\r\n".encode("utf-8"))
        body.extend(
            f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("utf-8")
        )
        body.extend(str(value).encode("utf-8"))
        body.extend(b"\r\n")
    for name, filename, content_type, data in files:
        body.extend(f"--{boundary}\r\n".encode("utf-8"))
        body.extend(
            (
                f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'
                f"Content-Type: {content_type}\r\n\r\n"
            ).encode("utf-8")
        )
        body.extend(data)
        body.extend(b"\r\n")
    body.extend(f"--{boundary}--\r\n".encode("utf-8"))
    return bytes(body), f"multipart/form-data; boundary={boundary}"


def _match_from_payload(payload: dict | None) -> AlgoMatchResult | None:
    if not payload:
        return None
    return AlgoMatchResult(
        embedding_similarity=float(payload.get("embedding_similarity", 0.0)),
        embedding_pass=bool(payload.get("embedding_pass", False)),
        ransac_ok=bool(payload.get("ransac_ok", False)),
        inliers=int(payload.get("inliers", 0)),
        total_matches=int(payload.get("total_matches", 0)),
        inlier_ratio=float(payload.get("inlier_ratio", 0.0)),
        score=float(payload.get("score", 0.0)),
        is_match=bool(payload.get("is_match", False)),
    )


def _empty_match() -> AlgoMatchResult:
    return AlgoMatchResult(
        embedding_similarity=0.0,
        embedding_pass=False,
        ransac_ok=False,
        inliers=0,
        total_matches=0,
        inlier_ratio=0.0,
        score=0.0,
        is_match=False,
    )


def _bbox_from_payload(value) -> tuple[int, int, int, int] | None:
    if not value:
        return None
    if isinstance(value, (list, tuple)) and len(value) == 4:
        return tuple(int(v) for v in value)
    return None


def _float_or_none(value) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
