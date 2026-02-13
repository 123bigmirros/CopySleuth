from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class MatchStats(BaseModel):
    embedding_similarity: float
    embedding_pass: bool
    ransac_ok: bool
    inliers: int
    total_matches: int
    inlier_ratio: float
    score: float
    is_match: bool


class CandidateResult(BaseModel):
    id: int
    label: str
    kind: str
    bbox: tuple[int, int, int, int] | None
    score: float
    match: MatchStats
    preview: str
    full_preview: str | None = None
    full_width: int | None = None
    full_height: int | None = None


class VideoSegmentResult(BaseModel):
    id: int
    obj_id: int
    kind: str | None = None
    start_time: float
    end_time: float
    first_frame_index: int
    last_frame_index: int
    bbox: tuple[int, int, int, int] | None
    score: float
    match: MatchStats
    preview: str
    full_preview: str | None = None
    full_width: int | None = None
    full_height: int | None = None


class OCRResult(BaseModel):
    enabled: bool = True
    keywords: list[str] = []
    keyword_count: int = 0
    texts: list[str] = []
    text: str | None = None
    matches: list[str] = []
    match_count: int = 0
    is_match: bool = False
    positions: dict | None = None
    video: dict | None = None
    error: str | None = None


class OcrOnlyResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ocr: OCRResult | None = None


class DetectionResponse(BaseModel):
    media_type: str
    is_match: bool
    best_bbox: tuple[int, int, int, int] | None
    best_score: float | None
    candidates: int
    match: MatchStats | None
    query_preview: str | None = None
    query_label: str | None = None
    candidate_results: list[CandidateResult] = []
    segments: list[VideoSegmentResult] = []
    fps: float | None = None
    frame_count: int | None = None
    duration: float | None = None
    ocr: OCRResult | None = None


class QueryResult(BaseModel):
    query_id: int
    query_label: str | None = None
    query_preview: str | None = None
    result: DetectionResponse


class MultiDetectionResponse(BaseModel):
    media_type: str
    is_match: bool
    best_score: float | None = None
    match: MatchStats | None = None
    query_results: list[QueryResult] = []
    ocr: OCRResult | None = None


class TaskResponse(BaseModel):
    task_id: str


class TaskStatusResponse(BaseModel):
    task_id: str
    status: str
    progress: float
    ocr_progress: float | None = None
    ocr_stage: str | None = None
    ocr_message: str | None = None
    error: str | None = None
    result: DetectionResponse | MultiDetectionResponse | OcrOnlyResponse | None = None
