from typing import List, Optional

from pydantic import BaseModel, Field


class ImageRequest(BaseModel):
    path: str = Field(..., description="Absolute path to image on the host")


class VideoRequest(BaseModel):
    path: str = Field(..., description="Absolute path to video on the host")
    keywords: List[str] = Field(..., description="Keywords to match in OCR text")
    frame_select: str = Field("scenedetect", description="Frame selection mode")
    scenedetect_threshold: float = Field(27.0, description="PySceneDetect threshold")
    scene_threshold: float = Field(8.0, description="Adaptive/scene diff threshold")
    select_max_dim: int = Field(160, description="Max dim for frame selection")
    max_dim: Optional[int] = Field(None, description="Max dim for OCR frames")
    dedup_hash_threshold: int = Field(8, description="dHash distance threshold")
    dedup_max_skip_frames: int = Field(0, description="0 disables forced keep")


class JobResponse(BaseModel):
    job_id: str
    status: str


class JobStatus(BaseModel):
    job_id: str
    status: str
    submitted_at: float
    started_at: Optional[float]
    finished_at: Optional[float]
    result: Optional[dict]
    error: Optional[str]
