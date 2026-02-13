#!/usr/bin/env python3
import argparse
import json
import os
import sys
import site
import ctypes

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


def main():
    parser = argparse.ArgumentParser(description="Run PaddleOCR 3.0.x on a single image and print recognized text.")
    parser.add_argument("image", help="Path to input image")
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
    parser.add_argument("--out", default=None, help="Optional path to save recognized text")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"Image not found: {args.image}", file=sys.stderr)
        return 2

    ocr_kwargs = {
        "device": args.device,
        "text_detection_model_name": "PP-OCRv5_server_det",
        "text_recognition_model_name": "PP-OCRv5_server_rec",
        "text_detection_model_dir": args.det_model_dir,
        "text_recognition_model_dir": args.rec_model_dir,
        "use_doc_orientation_classify": False,
        "use_doc_unwarping": False,
        "use_textline_orientation": args.use_textline_orientation,
    }
    if args.lang:
        ocr_kwargs["lang"] = args.lang
    if args.use_textline_orientation and os.path.isdir(args.textline_model_dir):
        ocr_kwargs["textline_orientation_model_dir"] = args.textline_model_dir

    try:
        _patch_paddlex_pp_option()
        ocr = PaddleOCR(**ocr_kwargs)
    except Exception as exc:  # noqa: BLE001
        if args.device == "gpu":
            print("GPU init failed; try --device cpu", file=sys.stderr)
        raise

    results = ocr.predict(args.image)
    all_texts = []
    for res in results:
        data = _to_json(res)
        texts = _extract_texts(data)
        if texts:
            all_texts.extend(texts)
        else:
            # Fallback: print full result object for visibility
            res.print()

    output_text = "\n".join(all_texts).strip()
    if output_text:
        print(output_text)
    else:
        print("[warn] No text extracted from JSON output; see printed result above.")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(output_text + ("\n" if output_text else ""))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
