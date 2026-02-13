#!/usr/bin/env python3
import argparse
import base64
import json
import os
import sys
import site
import ctypes
import math

import cv2
try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None

_ROOT = os.path.dirname(os.path.abspath(__file__))
_PADDLEX_HOME = os.path.join(_ROOT, ".paddlex")
os.environ.setdefault("PADDLEX_HOME", _PADDLEX_HOME)
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
os.makedirs(_PADDLEX_HOME, exist_ok=True)


def _augment_ld_library_path():
    if os.name != "posix":
        return
    lib_dirs = []
    conda_lib = os.path.join(sys.prefix, "lib")
    if os.path.isdir(conda_lib):
        lib_dirs.append(conda_lib)
    for base in site.getsitepackages():
        nvidia_root = os.path.join(base, "nvidia")
        if not os.path.isdir(nvidia_root):
            continue
        for root, dirs, _ in os.walk(nvidia_root):
            if os.path.basename(root) == "lib":
                lib_dirs.append(root)
    if not lib_dirs:
        return
    current = os.environ.get("LD_LIBRARY_PATH", "")
    parts = [p for p in current.split(":") if p]
    for lib_dir in lib_dirs:
        if lib_dir not in parts:
            parts.insert(0, lib_dir)
    os.environ["LD_LIBRARY_PATH"] = ":".join(parts)


def _preload_openmp():
    if os.name != "posix":
        return
    lib_path = os.path.join(sys.prefix, "lib", "libgomp.so.1")
    if os.path.exists(lib_path):
        try:
            ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
        except OSError:
            pass


_augment_ld_library_path()
_preload_openmp()

from paddleocr import PaddleOCR


def _patch_paddlex_pp_option():
    try:
        import inspect
        from paddlex.inference.utils.pp_option import PaddlePredictorOption as _PPO
        import paddleocr._common_args as _common_args
    except Exception:
        return

    sig = inspect.signature(_PPO.__init__)
    params = list(sig.parameters.values())
    if len(params) == 2 and params[1].kind == inspect.Parameter.VAR_KEYWORD:
        original_init = _PPO.__init__

        def _compat_init(self, *args, **kwargs):
            # PaddleOCR 3.0.3 passes model_name as a positional arg; ignore it for newer PaddleX.
            if args:
                if len(args) > 1:
                    raise TypeError("PaddlePredictorOption() takes at most 1 positional argument")
            return original_init(self, **kwargs)

        _PPO.__init__ = _compat_init
        _common_args.PaddlePredictorOption = _PPO


def _to_json(res):
    if hasattr(res, "json"):
        data = res.json
        return data() if callable(data) else data
    if hasattr(res, "to_json"):
        data = res.to_json()
        if isinstance(data, str):
            try:
                return json.loads(data)
            except json.JSONDecodeError:
                return None
        return data
    if isinstance(res, dict):
        return res
    return None


def _extract_texts(data):
    if not isinstance(data, dict):
        return []
    if "rec_texts" in data and isinstance(data["rec_texts"], list):
        return [t for t in data["rec_texts"] if t]
    if "res" in data and isinstance(data["res"], dict):
        if "rec_texts" in data["res"] and isinstance(data["res"]["rec_texts"], list):
            return [t for t in data["res"]["rec_texts"] if t]
    if "result" in data and isinstance(data["result"], dict):
        if "rec_texts" in data["result"] and isinstance(data["result"]["rec_texts"], list):
            return [t for t in data["result"]["rec_texts"] if t]
    return []


def _normalize_polygon(poly):
    if not isinstance(poly, (list, tuple)):
        return None
    if len(poly) == 4 and all(isinstance(p, (list, tuple)) and len(p) >= 2 for p in poly):
        points = []
        for point in poly:
            points.append([float(point[0]), float(point[1])])
        return points
    if len(poly) == 8 and all(isinstance(v, (int, float)) for v in poly):
        points = []
        for idx in range(0, 8, 2):
            points.append([float(poly[idx]), float(poly[idx + 1])])
        return points
    return None


def _bbox_from_points(points):
    if not points:
        return None
    xs = [pt[0] for pt in points]
    ys = [pt[1] for pt in points]
    return [min(xs), min(ys), max(xs), max(ys)]


def _normalize_bbox(box):
    if not isinstance(box, (list, tuple)):
        return None
    if len(box) == 4 and all(isinstance(v, (int, float)) for v in box):
        return [float(box[0]), float(box[1]), float(box[2]), float(box[3])]
    points = _normalize_polygon(box)
    if points:
        return _bbox_from_points(points)
    return None


def _extract_lines(data):
    if not isinstance(data, dict):
        return []
    candidates = [data]
    for key in ("res", "result"):
        if isinstance(data.get(key), dict):
            candidates.append(data[key])
    list_candidates = []
    if isinstance(data.get("results"), list):
        list_candidates.append(data["results"])

    for container in candidates:
        texts = container.get("rec_texts")
        polys = container.get("rec_polys")
        boxes = (
            container.get("rec_boxes")
            or container.get("dt_boxes")
            or container.get("det_boxes")
            or container.get("boxes")
        )
        if isinstance(texts, list):
            if isinstance(polys, list) and len(polys) == len(texts):
                lines = []
                for text, poly in zip(texts, polys):
                    points = _normalize_polygon(poly)
                    bbox = _bbox_from_points(points) if points else None
                    lines.append({"text": text, "bbox": bbox, "polygon": points})
                return lines
            if isinstance(boxes, list) and len(boxes) == len(texts):
                lines = []
                for text, box in zip(texts, boxes):
                    bbox = _normalize_bbox(box)
                    polygon = _normalize_polygon(box)
                    lines.append({"text": text, "bbox": bbox, "polygon": polygon})
                return lines

    for items in list_candidates:
        lines = []
        for item in items:
            if not isinstance(item, dict):
                continue
            text = item.get("text") or item.get("rec_text")
            if not text:
                continue
            polygon = _normalize_polygon(item.get("polygon") or item.get("poly"))
            bbox = _normalize_bbox(item.get("bbox") or item.get("box"))
            if polygon and not bbox:
                bbox = _bbox_from_points(polygon)
            lines.append({"text": text, "bbox": bbox, "polygon": polygon})
        if lines:
            return lines
    return []


def _frame_to_data_url(frame, max_dim=640):
    if max_dim is not None:
        frame = _resize_frame(frame, max_dim)
    ok, buffer = cv2.imencode(".png", frame)
    if not ok:
        return None
    encoded = base64.b64encode(buffer.tobytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _frame_preview_with_scale(frame, max_dim=640):
    original_h, original_w = frame.shape[:2]
    preview = _resize_frame(frame, max_dim) if max_dim is not None else frame
    preview_h, preview_w = preview.shape[:2]
    scale_x = preview_w / float(original_w) if original_w else 1.0
    scale_y = preview_h / float(original_h) if original_h else 1.0
    ok, buffer = cv2.imencode(".png", preview)
    if not ok:
        return None, None, None, 1.0, 1.0
    encoded = base64.b64encode(buffer.tobytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}", preview_w, preview_h, scale_x, scale_y


def _parse_keywords(raw, repeated):
    keywords = []
    if raw:
        raw = raw.strip()
        if raw.startswith("["):
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON keywords list: {exc}") from exc
            if not isinstance(parsed, list):
                raise ValueError("JSON keywords must be a list")
            keywords.extend([str(k).strip() for k in parsed])
        else:
            keywords.extend([part.strip() for part in raw.split(",")])
    if repeated:
        keywords.extend([k.strip() for k in repeated])
    keywords = [k for k in keywords if k]
    seen = set()
    ordered = []
    for k in keywords:
        if k not in seen:
            seen.add(k)
            ordered.append(k)
    return ordered


def _build_segments(entries, max_gap_frames):
    segments = []
    current = None
    for entry in entries:
        frame = entry["frame"]
        time_s = entry["time"]
        if current is None:
            current = {
                "start_frame": frame,
                "end_frame": frame,
                "start_time": time_s,
                "end_time": time_s,
                "frames": [frame],
                "first_frame_positions": entry.get("positions") or [],
                "first_frame_preview": entry.get("frame_preview"),
                "first_frame_width": entry.get("frame_width"),
                "first_frame_height": entry.get("frame_height"),
            }
            continue
        if frame - current["end_frame"] <= max_gap_frames:
            current["end_frame"] = frame
            current["end_time"] = time_s
            current["frames"].append(frame)
        else:
            segments.append(current)
            current = {
                "start_frame": frame,
                "end_frame": frame,
                "start_time": time_s,
                "end_time": time_s,
                "frames": [frame],
                "first_frame_positions": entry.get("positions") or [],
                "first_frame_preview": entry.get("frame_preview"),
                "first_frame_width": entry.get("frame_width"),
                "first_frame_height": entry.get("frame_height"),
            }
    if current is not None:
        segments.append(current)
    return segments


def _frame_has_keyword(texts, keyword):
    for text in texts:
        if keyword in text:
            return True
    return False


def _scale_lines(lines, scale_x, scale_y):
    if not lines:
        return []
    output = []
    for line in lines:
        if not isinstance(line, dict):
            continue
        updated = dict(line)
        bbox = line.get("bbox")
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            updated["bbox"] = [
                bbox[0] * scale_x,
                bbox[1] * scale_y,
                bbox[2] * scale_x,
                bbox[3] * scale_y,
            ]
        polygon = line.get("polygon")
        if isinstance(polygon, (list, tuple)):
            scaled_polygon = []
            ok = True
            for point in polygon:
                if not isinstance(point, (list, tuple)) or len(point) < 2:
                    ok = False
                    break
                scaled_polygon.append([point[0] * scale_x, point[1] * scale_y])
            if ok:
                updated["polygon"] = scaled_polygon
        output.append(updated)
    return output


def _resize_frame(frame, max_dim):
    if max_dim is None:
        return frame
    max_dim = int(max_dim)
    if max_dim <= 0:
        return frame
    height, width = frame.shape[:2]
    current_max = max(height, width)
    if current_max <= max_dim:
        return frame
    scale = max_dim / float(current_max)
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _prepare_select_frame(frame, max_dim):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return _resize_frame(gray, max_dim)


def _frame_diff_mean(prev_gray, curr_gray):
    diff = cv2.absdiff(prev_gray, curr_gray)
    return float(diff.mean())


def _dhash(gray, hash_size=8):
    resized = cv2.resize(gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    diff = resized[:, 1:] > resized[:, :-1]
    bits = 0
    for value in diff.flatten():
        bits = (bits << 1) | int(value)
    return bits


def _hamming_distance(a, b):
    return (a ^ b).bit_count()


def _scenedetect_scenes(video_path, threshold):
    try:
        from scenedetect import open_video, SceneManager
        from scenedetect.detectors import ContentDetector
    except Exception as exc:
        raise RuntimeError(
            "PySceneDetect is required for frame_select=scenedetect. "
            "Install it with: pip install scenedetect"
        ) from exc

    try:
        video = open_video(video_path)
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector(threshold=threshold))
        scene_manager.detect_scenes(video)
        scene_list = scene_manager.get_scene_list()
    except Exception:
        # Fallback to older PySceneDetect API.
        from scenedetect import VideoManager

        video_manager = VideoManager([video_path])
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector(threshold=threshold))
        video_manager.start()
        scene_manager.detect_scenes(frame_source=video_manager)
        scene_list = scene_manager.get_scene_list()
        video_manager.release()

    scenes = []
    for start, end in scene_list:
        start_frame = int(start.get_frames())
        end_frame = int(end.get_frames()) - 1
        if end_frame < start_frame:
            end_frame = start_frame
        scenes.append((start_frame, end_frame))
    return scenes


def run_video_ocr(
    video_path,
    keywords,
    det_model_dir=os.path.join(_ROOT, "PP-OCRv5_server_det_infer"),
    rec_model_dir=os.path.join(_ROOT, "PP-OCRv5_server_rec_infer"),
    textline_model_dir=os.path.join(_ROOT, "PP-LCNet_x0_25_textline_ori_infer"),
    device="gpu",
    lang=None,
    use_textline_orientation=False,
    max_gap_frames=3,
    show_progress=True,
    batch_size=8,
    rec_batch_size=None,
    frame_stride=1,
    max_dim=None,
    frame_select="scenedetect",
    scene_threshold=8.0,
    scenedetect_threshold=27.0,
    dedup_hash_threshold=8,
    dedup_max_skip_frames=0,
    max_skip_frames=60,
    select_max_dim=160,
    ocr=None,
):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not keywords:
        raise ValueError("No keywords provided.")
    frame_stride = int(frame_stride)
    if frame_stride < 1:
        raise ValueError("frame_stride must be >= 1")
    if frame_select not in {"stride", "adaptive", "scene", "scenedetect"}:
        raise ValueError("frame_select must be one of: stride, adaptive, scene, scenedetect")
    if dedup_max_skip_frames is not None:
        dedup_max_skip_frames = int(dedup_max_skip_frames)
        if dedup_max_skip_frames < 0:
            raise ValueError("dedup_max_skip_frames must be >= 0")
    if max_skip_frames is not None:
        max_skip_frames = int(max_skip_frames)
        if max_skip_frames < 1:
            raise ValueError("max_skip_frames must be >= 1")

    if ocr is None:
        ocr_kwargs = {
            "device": device,
            "text_detection_model_name": "PP-OCRv5_server_det",
            "text_recognition_model_name": "PP-OCRv5_server_rec",
            "text_detection_model_dir": det_model_dir,
            "text_recognition_model_dir": rec_model_dir,
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": use_textline_orientation,
        }
        if rec_batch_size is not None:
            ocr_kwargs["text_recognition_batch_size"] = rec_batch_size
        if lang:
            ocr_kwargs["lang"] = lang
        if use_textline_orientation and os.path.isdir(textline_model_dir):
            ocr_kwargs["textline_orientation_model_dir"] = textline_model_dir

        try:
            _patch_paddlex_pp_option()
            ocr = PaddleOCR(**ocr_kwargs)
        except Exception:  # noqa: BLE001
            if device == "gpu":
                print("GPU init failed; try --device cpu", file=sys.stderr)
            raise

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 1.0
        print("[warn] FPS not available; using 1.0 for timestamps.", file=sys.stderr)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    pbar = None
    if show_progress and tqdm is not None:
        total_for_progress = None
        if total_frames > 0:
            if frame_select == "stride":
                total_for_progress = int(math.ceil(total_frames / float(frame_stride)))
            else:
                total_for_progress = total_frames
        pbar = tqdm(
            total=total_for_progress,
            unit="frame",
            desc="OCR",
        )

    matches = {kw: [] for kw in keywords}
    frame_idx = 0
    batch_frames = []
    batch_indices = []
    batch_size = max(1, int(batch_size))
    last_select_frame = None
    skipped_since_keep = 0
    keep_frames = None
    last_dhash = None
    dedup_skipped = 0
    if frame_select == "scenedetect":
        scenes = _scenedetect_scenes(video_path, scenedetect_threshold)
        keep_frames = set()
        step = max(1, int(frame_stride))
        for start_frame, end_frame in scenes:
            keep_frames.add(start_frame)
            keep_frames.add(end_frame)
            for idx in range(start_frame, end_frame + 1, step):
                keep_frames.add(idx)
        if total_frames > 0:
            keep_frames = {idx for idx in keep_frames if 0 <= idx < total_frames}

    def _process_batch(frames, indices):
        if not frames:
            return
        results = ocr.predict(frames)
        for frame, idx, res in zip(frames, indices, results):
            texts = []
            data = _to_json(res)
            texts.extend(_extract_texts(data))
            lines = _extract_lines(data)
            if not texts and not lines:
                continue
            frame_preview = None
            frame_width = None
            frame_height = None
            scale_x = 1.0
            scale_y = 1.0
            for kw in keywords:
                if _frame_has_keyword(texts, kw) or any(
                    kw in (line.get("text") or "") for line in lines
                ):
                    if frame_preview is None:
                        (
                            frame_preview,
                            frame_width,
                            frame_height,
                            scale_x,
                            scale_y,
                        ) = _frame_preview_with_scale(frame)
                    positions = [
                        line
                        for line in lines
                        if kw in (line.get("text") or "")
                    ]
                    if scale_x != 1.0 or scale_y != 1.0:
                        positions = _scale_lines(positions, scale_x, scale_y)
                    time_s = idx / fps
                    matches[kw].append(
                        {
                            "frame": idx,
                            "time": time_s,
                            "positions": positions,
                            "frame_preview": frame_preview,
                            "frame_width": frame_width,
                            "frame_height": frame_height,
                        }
                    )

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        keep = True
        if frame_select in {"adaptive", "scene"}:
            select_frame = _prepare_select_frame(frame, select_max_dim)
            if last_select_frame is None:
                keep = True
            else:
                diff = _frame_diff_mean(last_select_frame, select_frame)
                keep = diff >= scene_threshold
                if not keep and max_skip_frames is not None and skipped_since_keep >= max_skip_frames:
                    keep = True
            if keep:
                last_select_frame = select_frame
                skipped_since_keep = 0
            else:
                skipped_since_keep += 1
        elif frame_select == "scenedetect":
            keep = frame_idx in keep_frames
            if keep:
                select_frame = _prepare_select_frame(frame, select_max_dim)
                current_hash = _dhash(select_frame)
                if last_dhash is None:
                    keep = True
                else:
                    distance = _hamming_distance(last_dhash, current_hash)
                    keep = distance >= dedup_hash_threshold
                    if not keep and dedup_max_skip_frames and dedup_skipped >= dedup_max_skip_frames:
                        keep = True
                if keep:
                    last_dhash = current_hash
                    dedup_skipped = 0
                else:
                    dedup_skipped += 1
        if keep:
            batch_frames.append(_resize_frame(frame, max_dim))
            batch_indices.append(frame_idx)
            if len(batch_frames) >= batch_size:
                _process_batch(batch_frames, batch_indices)
                batch_frames = []
                batch_indices = []
        frame_idx += 1
        if pbar is not None:
            pbar.update(1)
        if frame_select == "stride" and frame_stride > 1:
            skipped = 0
            while skipped < frame_stride - 1:
                ok = cap.grab()
                if not ok:
                    break
                frame_idx += 1
                skipped += 1
                if pbar is not None:
                    pbar.update(1)
            if skipped < frame_stride - 1:
                break

    if batch_frames:
        _process_batch(batch_frames, batch_indices)

    cap.release()
    if pbar is not None:
        pbar.close()

    output = {
        "video": video_path,
        "fps": fps,
        "frame_count": frame_idx,
        "frame_stride": frame_stride,
        "max_dim": max_dim,
        "frame_select": frame_select,
        "scene_threshold": scene_threshold,
        "scenedetect_threshold": scenedetect_threshold,
        "dedup_hash_threshold": dedup_hash_threshold,
        "dedup_max_skip_frames": dedup_max_skip_frames,
        "max_skip_frames": max_skip_frames,
        "select_max_dim": select_max_dim,
        "keywords": [],
    }

    for kw in keywords:
        entries = matches[kw]
        segments = _build_segments(entries, max_gap_frames)
        output["keywords"].append(
            {
                "keyword": kw,
                "matches": len(entries),
                "segments": segments,
            }
        )

    return output


def main():
    parser = argparse.ArgumentParser(
        description="Run PaddleOCR on every video frame and match keywords."
    )
    parser.add_argument("video", help="Path to input video")
    parser.add_argument(
        "--keywords",
        required=False,
        help="Comma-separated keywords or JSON list string, e.g. 'foo,bar' or '[\"foo\",\"bar\"]'",
    )
    parser.add_argument(
        "--keyword",
        action="append",
        default=[],
        help="Repeatable keyword flag (e.g., --keyword foo --keyword bar)",
    )
    parser.add_argument(
        "--det-model-dir",
        default=os.path.join(_ROOT, "PP-OCRv5_server_det_infer"),
        help="Path to detection model directory",
    )
    parser.add_argument(
        "--rec-model-dir",
        default=os.path.join(_ROOT, "PP-OCRv5_server_rec_infer"),
        help="Path to recognition model directory",
    )
    parser.add_argument(
        "--textline-model-dir",
        default=os.path.join(_ROOT, "PP-LCNet_x0_25_textline_ori_infer"),
        help="Path to textline orientation model directory (if enabled)",
    )
    parser.add_argument("--device", default="gpu", choices=["gpu", "cpu"], help="Inference device")
    parser.add_argument("--lang", default=None, help="Language hint (e.g., 'en', 'ch')")
    parser.add_argument("--use-textline-orientation", action="store_true", help="Enable text line orientation")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Frames per OCR batch (default: 8)",
    )
    parser.add_argument(
        "--rec-batch-size",
        type=int,
        default=None,
        help="Text recognition batch size (optional, e.g., 16 or 32)",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="Process every Nth frame (default: 1 = all frames)",
    )
    parser.add_argument(
        "--max-dim",
        type=int,
        default=None,
        help="Resize frames so the longest side is <= this value before OCR",
    )
    parser.add_argument(
        "--frame-select",
        default="scenedetect",
        choices=["stride", "adaptive", "scene", "scenedetect"],
        help="Frame selection mode: stride, adaptive, scene, or scenedetect (default: scenedetect)",
    )
    parser.add_argument(
        "--scene-threshold",
        type=float,
        default=8.0,
        help="Scene/adaptive selection threshold; higher keeps fewer frames (default: 8.0)",
    )
    parser.add_argument(
        "--scenedetect-threshold",
        type=float,
        default=27.0,
        help="PySceneDetect content threshold; lower = more cuts (default: 27.0)",
    )
    parser.add_argument(
        "--dedup-hash-threshold",
        type=int,
        default=8,
        help="dHash distance threshold for near-duplicate removal in scenedetect mode (default: 8)",
    )
    parser.add_argument(
        "--dedup-max-skip-frames",
        type=int,
        default=0,
        help="Force keep at least every N frames in scenedetect dedup; 0 disables (default: 0)",
    )
    parser.add_argument(
        "--max-skip-frames",
        type=int,
        default=60,
        help="Force keep at least every N frames in adaptive/scene mode (default: 60)",
    )
    parser.add_argument(
        "--select-max-dim",
        type=int,
        default=160,
        help="Downscale frames to this max dim for selection (default: 160)",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bar",
    )
    parser.add_argument(
        "--max-gap-frames",
        type=int,
        default=3,
        help="Maximum frame gap to merge matches into one segment (default: 3)",
    )
    parser.add_argument("--out", default=None, help="Optional path to save JSON output")
    args = parser.parse_args()

    try:
        keywords = _parse_keywords(args.keywords, args.keyword)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if not keywords:
        print("No keywords provided. Use --keywords or --keyword.", file=sys.stderr)
        return 2

    try:
        output = run_video_ocr(
            args.video,
            keywords,
            det_model_dir=args.det_model_dir,
            rec_model_dir=args.rec_model_dir,
            textline_model_dir=args.textline_model_dir,
            device=args.device,
            lang=args.lang,
            use_textline_orientation=args.use_textline_orientation,
            max_gap_frames=args.max_gap_frames,
            show_progress=not args.no_progress,
            batch_size=args.batch_size,
            rec_batch_size=args.rec_batch_size,
            frame_stride=args.frame_stride,
            max_dim=args.max_dim,
            frame_select=args.frame_select,
            scene_threshold=args.scene_threshold,
            scenedetect_threshold=args.scenedetect_threshold,
            dedup_hash_threshold=args.dedup_hash_threshold,
            dedup_max_skip_frames=args.dedup_max_skip_frames,
            max_skip_frames=args.max_skip_frames,
            select_max_dim=args.select_max_dim,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        return 2

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=True, indent=2)
            f.write("\n")
    else:
        json.dump(output, sys.stdout, ensure_ascii=True, indent=2)
        sys.stdout.write("\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
