from __future__ import annotations

import json
import mimetypes
import uuid
import threading
import time
from pathlib import Path
from urllib.request import Request, urlopen

from app.core.paths import ensure_repo_on_path


def _normalize_text(text: str) -> str:
    return "".join(text.split()).lower()


def _dedupe_preserve(items: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        output.append(item)
    return output


def _strip_spaces(text: str) -> str:
    return "".join(text.split())


def _clean_texts(texts: list[str]) -> list[str]:
    cleaned = []
    for text in texts:
        if text is None:
            continue
        stripped = _strip_spaces(str(text))
        if stripped:
            cleaned.append(stripped)
    return cleaned


def _clean_lines(lines: list[dict]) -> list[dict]:
    cleaned = []
    for line in lines:
        if not isinstance(line, dict):
            continue
        text = line.get("text")
        if text is None:
            continue
        stripped = _strip_spaces(str(text))
        if not stripped:
            continue
        updated = dict(line)
        updated["text"] = stripped
        cleaned.append(updated)
    return cleaned


def _log_texts(prefix: str, texts: list[str]) -> None:
    preview = texts[:20]
    suffix = "" if len(texts) <= 20 else f" ...(+{len(texts) - 20})"
    print(f"{prefix} texts={preview}{suffix}", flush=True)


def _log_video_summary(prefix: str, output: dict) -> None:
    keywords = output.get("keywords") or []
    summary = []
    for entry in keywords:
        keyword = entry.get("keyword")
        matches = entry.get("matches")
        if keyword:
            summary.append((keyword, matches))
    print(f"{prefix} video_keywords={summary}", flush=True)


def _line_matches_keyword(text: str, keyword: str) -> bool:
    normalized_text = _normalize_text(text)
    normalized_keyword = _normalize_text(keyword)
    if not normalized_keyword:
        return False
    return normalized_keyword in normalized_text


def _filter_lines_by_keywords(lines: list[dict], keywords: list[str]) -> list[dict]:
    if not lines or not keywords:
        return []
    output = []
    for line in lines:
        text = str(line.get("text") or "")
        if not text:
            continue
        if any(_line_matches_keyword(text, keyword) for keyword in keywords):
            output.append(line)
    return output


class PaddleOCRService:
    def __init__(self, repo_dir: Path, api_base: str | None = None) -> None:
        self._repo_dir = repo_dir
        self._engine = None
        self._lock = threading.Lock()
        self._api_base = api_base.rstrip("/") if api_base else None

    def _get_engine(self):
        with self._lock:
            if self._engine is None:
                ensure_repo_on_path(self._repo_dir)
                from server.ocr_engine import OCREngine  # type: ignore

                self._engine = OCREngine()
            return self._engine

    def _http_post(self, path: str, payload: dict, timeout: float = 10.0) -> dict:
        if not self._api_base:
            raise RuntimeError("OCR API base not configured")
        url = f"{self._api_base}{path}"
        data = json.dumps(payload).encode("utf-8")
        req = Request(url, data=data, headers={"Content-Type": "application/json"})
        with urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _http_post_file(
        self,
        path: str,
        file_path: Path,
        *,
        fields: dict[str, str] | None = None,
        timeout: float = 30.0,
    ) -> dict:
        if not self._api_base:
            raise RuntimeError("OCR API base not configured")
        fields = fields or {}
        content_type, _ = mimetypes.guess_type(file_path.name)
        content_type = content_type or "application/octet-stream"
        boundary = uuid.uuid4().hex
        body = bytearray()
        for name, value in fields.items():
            body.extend(f"--{boundary}\r\n".encode("utf-8"))
            body.extend(
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("utf-8")
            )
            body.extend(str(value).encode("utf-8"))
            body.extend(b"\r\n")
        body.extend(f"--{boundary}\r\n".encode("utf-8"))
        body.extend(
            (
                f'Content-Disposition: form-data; name="file"; filename="{file_path.name}"\r\n'
                f"Content-Type: {content_type}\r\n\r\n"
            ).encode("utf-8")
        )
        body.extend(file_path.read_bytes())
        body.extend(b"\r\n")
        body.extend(f"--{boundary}--\r\n".encode("utf-8"))
        url = f"{self._api_base}{path}"
        req = Request(
            url,
            data=bytes(body),
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        with urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _http_get(self, path: str, timeout: float = 10.0) -> dict:
        if not self._api_base:
            raise RuntimeError("OCR API base not configured")
        url = f"{self._api_base}{path}"
        req = Request(url, headers={"Content-Type": "application/json"})
        with urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _wait_job(self, job_id: str, timeout: float = 900.0) -> dict:
        deadline = time.time() + timeout
        while True:
            payload = self._http_get(f"/v1/jobs/{job_id}")
            status = payload.get("status")
            if status == "succeeded":
                return payload.get("result") or {}
            if status == "failed":
                raise RuntimeError(payload.get("error") or "OCR job failed")
            if time.time() > deadline:
                raise TimeoutError("OCR job timed out")
            time.sleep(0.5)

    def _match_texts(self, texts: list[str], keywords: list[str]) -> tuple[str, list[str]]:
        full_text = "\n".join(texts).strip()
        normalized_full = _normalize_text(full_text)
        matched = []
        for keyword in keywords:
            normalized_keyword = _normalize_text(keyword)
            if not normalized_keyword:
                continue
            if normalized_keyword in normalized_full:
                matched.append(keyword)
        matched = _dedupe_preserve(matched)
        return full_text, matched

    def image_texts(self, image_path: Path) -> list[str]:
        engine = self._get_engine()
        return engine.image_texts(str(image_path))

    def image_lines(self, image_path: Path) -> list[dict]:
        engine = self._get_engine()
        return engine.image_lines(str(image_path))

    def match_image_keywords(self, image_path: Path, keywords: list[str]) -> dict:
        if self._api_base:
            result = self._http_post_file("/v1/ocr/image", image_path)
            job_id = result.get("job_id")
            if not job_id:
                raise RuntimeError("OCR image job_id missing")
            job_result = self._wait_job(job_id)
            lines = job_result.get("lines") or []
            raw_texts = job_result.get("texts") or [
                line.get("text") for line in lines if line.get("text")
            ]
            _full_text, matched = self._match_texts(raw_texts, keywords)
            cleaned_texts = _clean_texts(raw_texts)
            cleaned_lines = _clean_lines(lines)
            matched_lines = _filter_lines_by_keywords(cleaned_lines, matched)
            positions = {"lines": matched_lines} if matched else None
            _log_texts("[ocr:image]", cleaned_texts)
            return {
                "enabled": True,
                "keywords": keywords,
                "keyword_count": len(keywords),
                "texts": cleaned_texts,
                "text": "\n".join(cleaned_texts).strip(),
                "matches": matched,
                "match_count": len(matched),
                "is_match": bool(matched),
                "positions": positions,
                "video": None,
                "error": None,
            }

        lines = self.image_lines(image_path)
        raw_texts = [
            line.get("text") for line in lines if line.get("text")
        ] or self.image_texts(image_path)
        _full_text, matched = self._match_texts(raw_texts, keywords)
        cleaned_texts = _clean_texts(raw_texts)
        cleaned_lines = _clean_lines(lines)
        matched_lines = _filter_lines_by_keywords(cleaned_lines, matched)
        positions = {"lines": matched_lines} if matched else None
        _log_texts("[ocr:image]", cleaned_texts)
        return {
            "enabled": True,
            "keywords": keywords,
            "keyword_count": len(keywords),
            "texts": cleaned_texts,
            "text": "\n".join(cleaned_texts).strip(),
            "matches": matched,
            "match_count": len(matched),
            "is_match": bool(matched),
            "positions": positions,
            "video": None,
            "error": None,
        }

    def match_video_keywords(self, video_path: Path, keywords: list[str]) -> dict:
        if self._api_base:
            result = self._http_post_file(
                "/v1/ocr/video",
                video_path,
                fields={"keywords": json.dumps(keywords)},
                timeout=300.0,
            )
            job_id = result.get("job_id")
            if not job_id:
                raise RuntimeError("OCR video job_id missing")
            output = self._wait_job(job_id)
        else:
            ensure_repo_on_path(self._repo_dir)
            from video_ocr_pipeline import run_video_ocr  # type: ignore

            output = run_video_ocr(str(video_path), keywords, show_progress=False)

        keyword_entries = output.get("keywords") or []
        for entry in keyword_entries:
            for segment in entry.get("segments") or []:
                positions = segment.get("first_frame_positions") or []
                segment["first_frame_positions"] = _clean_lines(positions)
        matched_keywords = [
            entry.get("keyword")
            for entry in keyword_entries
            if entry.get("matches", 0) > 0
        ]
        matched_keywords = [kw for kw in matched_keywords if kw]
        matched_keywords = _dedupe_preserve(matched_keywords)
        _log_video_summary("[ocr:video]", output)
        return {
            "enabled": True,
            "keywords": keywords,
            "keyword_count": len(keywords),
            "texts": [],
            "text": "",
            "matches": matched_keywords,
            "match_count": len(matched_keywords),
            "is_match": bool(matched_keywords),
            "video": output,
            "error": None,
        }
