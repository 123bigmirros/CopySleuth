import ctypes
import json
import os
import site
import sys
import threading

from paddleocr import PaddleOCR

_ROOT = os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir)))
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
        for root, _, _ in os.walk(nvidia_root):
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


class OCREngine:
    def __init__(self):
        self._lock = threading.Lock()
        self._ocr = None

    def _init_ocr(self):
        if self._ocr is not None:
            return
        _patch_paddlex_pp_option()
        ocr_kwargs = {
            "device": "gpu",
            "text_detection_model_name": "PP-OCRv5_server_det",
            "text_recognition_model_name": "PP-OCRv5_server_rec",
            "text_detection_model_dir": os.path.join(_ROOT, "PP-OCRv5_server_det_infer"),
            "text_recognition_model_dir": os.path.join(_ROOT, "PP-OCRv5_server_rec_infer"),
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": False,
        }
        textline_dir = os.path.join(_ROOT, "PP-LCNet_x0_25_textline_ori_infer")
        if os.path.isdir(textline_dir):
            ocr_kwargs["textline_orientation_model_dir"] = textline_dir
        self._ocr = PaddleOCR(**ocr_kwargs)

    def get_ocr(self):
        with self._lock:
            self._init_ocr()
            return self._ocr

    def image_texts(self, image_path):
        with self._lock:
            self._init_ocr()
            results = self._ocr.predict(image_path)
        texts = []
        for res in results:
            data = _to_json(res)
            extracted = _extract_texts(data)
            if extracted:
                texts.extend(extracted)
        return texts

    def image_lines(self, image_path):
        with self._lock:
            self._init_ocr()
            results = self._ocr.predict(image_path)
        lines = []
        for res in results:
            data = _to_json(res)
            extracted = _extract_lines(data)
            if extracted:
                lines.extend(extracted)
        return lines
