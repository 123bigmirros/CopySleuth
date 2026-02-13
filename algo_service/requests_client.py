from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
from pathlib import Path
import time
from typing import Iterable

import requests


DEFAULT_BASE_URL = os.getenv("ALGO_BASE_URL", "http://127.0.0.1:8001")
DEFAULT_OCR_BASE_URL = os.getenv("OCR_BASE_URL", "http://127.0.0.1:8002")
DEFAULT_IMAGE_MODE = os.getenv("ALGO_IMAGE_MODE", "b64")


def _guess_type(path: Path) -> str:
    content_type, _ = mimetypes.guess_type(path.name)
    return content_type or "application/octet-stream"


def _is_excel_path(path: Path) -> bool:
    return path.suffix.lower() in {".xlsx", ".xlsm"}


def _validate_query_paths(query_paths: Iterable[Path]) -> None:
    for path in query_paths:
        if _is_excel_path(path):
            raise RuntimeError(
                "Excel queries are not supported by the algorithm service. "
                "Please upload images only."
            )


def _open_files(paths: Iterable[Path], field: str) -> tuple[list[tuple[str, tuple[str, bytes, str]]], list]:
    files = []
    handles = []
    for idx, path in enumerate(paths, start=1):
        fp = path.open("rb")
        handles.append(fp)
        files.append(
            (
                field,
                (
                    path.name or f"file_{idx:03d}",
                    fp,
                    _guess_type(path),
                ),
            )
        )
    return files, handles


def _close_handles(handles: list) -> None:
    for handle in handles:
        try:
            handle.close()
        except Exception:
            pass


def _save_b64_image(payload: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = base64.b64decode(payload)
    out_path.write_bytes(data)


def request_health(base_url: str) -> dict:
    resp = requests.get(f"{base_url}/v1/health", timeout=10)
    resp.raise_for_status()
    return resp.json()


def request_detect(
    base_url: str,
    query_paths: list[Path],
    target_path: Path,
    embedding_threshold: float | None = None,
    image_mode: str | None = None,
    out_json: Path | None = None,
    save_candidates: Path | None = None,
) -> dict:
    _validate_query_paths(query_paths)
    query_files, query_handles = _open_files(query_paths, "query_images")
    target_files, target_handles = _open_files([target_path], "target_image")
    files = query_files + target_files
    data = {}
    if embedding_threshold is not None:
        data["embedding_threshold"] = str(float(embedding_threshold))
    mode = (image_mode or DEFAULT_IMAGE_MODE).strip().lower()
    if mode:
        data["image_mode"] = mode
    try:
        resp = requests.post(
            f"{base_url}/v1/detect",
            files=files,
            data=data,
            timeout=120,
        )
        resp.raise_for_status()
        payload = resp.json()
    finally:
        _close_handles(query_handles + target_handles)

    if save_candidates is not None:
        save_candidates.mkdir(parents=True, exist_ok=True)
        for q_idx, result in enumerate(payload.get("results", []), start=1):
            for c_idx, candidate in enumerate(result.get("candidate_results", []), start=1):
                image_b64 = candidate.get("image_b64")
                if not image_b64:
                    continue
                out_path = save_candidates / f"q{q_idx:02d}_cand_{c_idx:06d}.png"
                _save_b64_image(image_b64, out_path)

    if out_json is not None:
        out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def _iter_sse(response: requests.Response):
    event_type = None
    data_lines: list[str] = []
    for raw_line in response.iter_lines(decode_unicode=True):
        if raw_line is None:
            continue
        line = raw_line.rstrip("\r")
        if not line:
            if data_lines:
                data_text = "\n".join(data_lines)
                data = json.loads(data_text) if data_text else None
                yield {"event": event_type or "message", "data": data}
            event_type = None
            data_lines = []
            continue
        if line.startswith(":"):
            continue
        if line.startswith("event:"):
            event_type = line[len("event:") :].strip()
            continue
        if line.startswith("data:"):
            data_lines.append(line[len("data:") :].lstrip())


def request_task(
    base_url: str,
    query_paths: list[Path],
    target_path: Path,
    embedding_threshold: float | None = None,
    video_start: float | None = None,
    video_end: float | None = None,
    image_mode: str | None = None,
    out_json: Path | None = None,
) -> dict:
    _validate_query_paths(query_paths)
    query_files, query_handles = _open_files(query_paths, "query_images")
    target_files, target_handles = _open_files([target_path], "target_file")
    files = query_files + target_files
    data = {}
    if embedding_threshold is not None:
        data["embedding_threshold"] = str(float(embedding_threshold))
    if video_start is not None:
        data["video_start"] = str(float(video_start))
    if video_end is not None:
        data["video_end"] = str(float(video_end))
    mode = (image_mode or DEFAULT_IMAGE_MODE).strip().lower()
    if mode:
        data["image_mode"] = mode
    try:
        resp = requests.post(
            f"{base_url}/v1/tasks",
            files=files,
            data=data,
            timeout=120,
        )
        resp.raise_for_status()
        payload = resp.json()
    finally:
        _close_handles(query_handles + target_handles)

    task_id = payload.get("task_id")
    if not task_id:
        raise RuntimeError("task_id missing from response")

    last_result = None
    with requests.get(
        f"{base_url}/v1/tasks/{task_id}/events",
        headers={"Accept": "text/event-stream"},
        stream=True,
        timeout=300,
    ) as resp:
        resp.raise_for_status()
        for event in _iter_sse(resp):
            event_type = event.get("event")
            data = event.get("data") or {}
            if event_type == "progress":
                print(f"[progress] {data.get('progress', 0):.2f} {data.get('stage', '')} {data.get('message', '')}")
            elif event_type in {"partial", "result"}:
                last_result = data.get("result")
                print(f"[{event_type}] received")
            elif event_type in {"error", "canceled"}:
                raise RuntimeError(data.get("message") or f"task {event_type}")
            elif event_type == "done":
                break

    if out_json is not None and last_result is not None:
        out_json.write_text(json.dumps(last_result, ensure_ascii=False, indent=2), encoding="utf-8")
    return last_result or {}


def request_task_status(base_url: str, task_id: str) -> dict:
    resp = requests.get(f"{base_url}/v1/tasks/{task_id}", timeout=30)
    resp.raise_for_status()
    return resp.json()


def request_task_cancel(base_url: str, task_id: str) -> dict:
    resp = requests.post(f"{base_url}/v1/tasks/{task_id}/cancel", timeout=30)
    resp.raise_for_status()
    return resp.json()


def request_ocr_image(ocr_base_url: str, image_path: Path) -> dict:
    with image_path.open("rb") as handle:
        files = {"file": (image_path.name, handle, _guess_type(image_path))}
        resp = requests.post(f"{ocr_base_url}/v1/ocr/image", files=files, timeout=60)
        resp.raise_for_status()
        return resp.json()


def request_ocr_video(
    ocr_base_url: str,
    video_path: Path,
    keywords: list[str],
    frame_select: str | None = None,
    scenedetect_threshold: float | None = None,
    scene_threshold: float | None = None,
    select_max_dim: int | None = None,
    max_dim: int | None = None,
    dedup_hash_threshold: int | None = None,
    dedup_max_skip_frames: int | None = None,
) -> dict:
    data = [("keywords", keyword) for keyword in keywords]
    if frame_select is not None:
        data.append(("frame_select", frame_select))
    if scenedetect_threshold is not None:
        data.append(("scenedetect_threshold", str(float(scenedetect_threshold))))
    if scene_threshold is not None:
        data.append(("scene_threshold", str(float(scene_threshold))))
    if select_max_dim is not None:
        data.append(("select_max_dim", str(int(select_max_dim))))
    if max_dim is not None:
        data.append(("max_dim", str(int(max_dim))))
    if dedup_hash_threshold is not None:
        data.append(("dedup_hash_threshold", str(int(dedup_hash_threshold))))
    if dedup_max_skip_frames is not None:
        data.append(("dedup_max_skip_frames", str(int(dedup_max_skip_frames))))
    with video_path.open("rb") as handle:
        files = {"file": (video_path.name, handle, _guess_type(video_path))}
        resp = requests.post(
            f"{ocr_base_url}/v1/ocr/video",
            data=data,
            files=files,
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()


def request_ocr_status(ocr_base_url: str, job_id: str) -> dict:
    resp = requests.get(f"{ocr_base_url}/v1/jobs/{job_id}", timeout=30)
    resp.raise_for_status()
    return resp.json()


def wait_ocr_job(
    ocr_base_url: str,
    job_id: str,
    poll_interval: float = 1.0,
    timeout_s: float = 600.0,
) -> dict:
    start = time.time()
    while True:
        status = request_ocr_status(ocr_base_url, job_id)
        state = (status.get("status") or "").lower()
        if state in {"succeeded", "failed"}:
            return status
        if time.time() - start > timeout_s:
            raise RuntimeError("OCR job timeout")
        time.sleep(poll_interval)


def main() -> None:
    parser = argparse.ArgumentParser(description="Algorithm service client (requests)")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("health", help="Call /v1/health")

    detect_parser = subparsers.add_parser("detect", help="Call /v1/detect")
    detect_parser.add_argument(
        "--query",
        action="append",
        required=True,
        help="Query image path (repeatable; Excel not supported)",
    )
    detect_parser.add_argument("--target", required=True, help="Target image path")
    detect_parser.add_argument("--embedding-threshold", type=float)
    detect_parser.add_argument(
        "--image-mode",
        choices=["b64", "path"],
        default=DEFAULT_IMAGE_MODE,
        help="Candidate image payload: b64 or server-local path",
    )
    detect_parser.add_argument("--out-json", help="Save response JSON")
    detect_parser.add_argument("--save-candidates", help="Directory to save candidate crops")

    task_parser = subparsers.add_parser("task", help="Call /v1/tasks and stream SSE")
    task_parser.add_argument(
        "--query",
        action="append",
        required=True,
        help="Query image path (repeatable; Excel not supported)",
    )
    task_parser.add_argument("--target", required=True, help="Target image/video path")
    task_parser.add_argument("--embedding-threshold", type=float)
    task_parser.add_argument("--video-start", type=float)
    task_parser.add_argument("--video-end", type=float)
    task_parser.add_argument("--out-json", help="Save final result JSON")
    task_parser.add_argument(
        "--image-mode",
        choices=["b64", "path"],
        default=DEFAULT_IMAGE_MODE,
        help="Candidate image payload: b64 or server-local path",
    )

    status_parser = subparsers.add_parser("status", help="Call /v1/tasks/{task_id}")
    status_parser.add_argument("--task-id", required=True)

    cancel_parser = subparsers.add_parser("cancel", help="Call /v1/tasks/{task_id}/cancel")
    cancel_parser.add_argument("--task-id", required=True)

    ocr_image_parser = subparsers.add_parser("ocr-image", help="Call OCR /v1/ocr/image")
    ocr_image_parser.add_argument("--ocr-base-url", default=DEFAULT_OCR_BASE_URL)
    ocr_image_parser.add_argument("--path", required=True, help="Local image path")
    ocr_image_parser.add_argument("--wait", action="store_true", help="Wait for OCR job")
    ocr_image_parser.add_argument("--poll-interval", type=float, default=1.0)
    ocr_image_parser.add_argument("--timeout", type=float, default=600.0)

    ocr_video_parser = subparsers.add_parser("ocr-video", help="Call OCR /v1/ocr/video")
    ocr_video_parser.add_argument("--ocr-base-url", default=DEFAULT_OCR_BASE_URL)
    ocr_video_parser.add_argument("--path", required=True, help="Local video path")
    ocr_video_parser.add_argument("--keyword", action="append", required=True)
    ocr_video_parser.add_argument("--frame-select")
    ocr_video_parser.add_argument("--scenedetect-threshold", type=float)
    ocr_video_parser.add_argument("--scene-threshold", type=float)
    ocr_video_parser.add_argument("--select-max-dim", type=int)
    ocr_video_parser.add_argument("--max-dim", type=int)
    ocr_video_parser.add_argument("--dedup-hash-threshold", type=int)
    ocr_video_parser.add_argument("--dedup-max-skip-frames", type=int)
    ocr_video_parser.add_argument("--wait", action="store_true", help="Wait for OCR job")
    ocr_video_parser.add_argument("--poll-interval", type=float, default=1.0)
    ocr_video_parser.add_argument("--timeout", type=float, default=600.0)

    ocr_status_parser = subparsers.add_parser("ocr-status", help="Call OCR /v1/jobs/{job_id}")
    ocr_status_parser.add_argument("--ocr-base-url", default=DEFAULT_OCR_BASE_URL)
    ocr_status_parser.add_argument("--job-id", required=True)

    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    if args.command == "health":
        payload = request_health(base_url)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    if args.command == "detect":
        payload = request_detect(
            base_url,
            query_paths=[Path(p) for p in args.query],
            target_path=Path(args.target),
            embedding_threshold=args.embedding_threshold,
            image_mode=args.image_mode,
            out_json=Path(args.out_json) if args.out_json else None,
            save_candidates=Path(args.save_candidates) if args.save_candidates else None,
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    if args.command == "status":
        payload = request_task_status(base_url, args.task_id)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    if args.command == "cancel":
        payload = request_task_cancel(base_url, args.task_id)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    if args.command == "ocr-image":
        job = request_ocr_image(args.ocr_base_url.rstrip("/"), Path(args.path))
        if args.wait:
            payload = wait_ocr_job(
                args.ocr_base_url.rstrip("/"),
                job_id=job.get("job_id", ""),
                poll_interval=args.poll_interval,
                timeout_s=args.timeout,
            )
        else:
            payload = job
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    if args.command == "ocr-video":
        job = request_ocr_video(
            args.ocr_base_url.rstrip("/"),
            Path(args.path),
            keywords=args.keyword,
            frame_select=args.frame_select,
            scenedetect_threshold=args.scenedetect_threshold,
            scene_threshold=args.scene_threshold,
            select_max_dim=args.select_max_dim,
            max_dim=args.max_dim,
            dedup_hash_threshold=args.dedup_hash_threshold,
            dedup_max_skip_frames=args.dedup_max_skip_frames,
        )
        if args.wait:
            payload = wait_ocr_job(
                args.ocr_base_url.rstrip("/"),
                job_id=job.get("job_id", ""),
                poll_interval=args.poll_interval,
                timeout_s=args.timeout,
            )
        else:
            payload = job
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    if args.command == "task":
        payload = request_task(
            base_url,
            query_paths=[Path(p) for p in args.query],
            target_path=Path(args.target),
            embedding_threshold=args.embedding_threshold,
            video_start=args.video_start,
            video_end=args.video_end,
            image_mode=args.image_mode,
            out_json=Path(args.out_json) if args.out_json else None,
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    if args.command == "ocr-status":
        payload = request_ocr_status(args.ocr_base_url.rstrip("/"), args.job_id)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return


if __name__ == "__main__":
    main()
