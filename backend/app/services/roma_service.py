from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import gc
import logging
import os
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModel

from app.core.config import Settings
from app.core.paths import ensure_repo_on_path


logger = logging.getLogger(__name__)

ImageInput = Image.Image | Path | str | np.ndarray | torch.Tensor


@dataclass
class MatchResult:
    embedding_similarity: float
    embedding_pass: bool
    ransac_ok: bool
    inliers: int
    total_matches: int
    inlier_ratio: float
    score: float
    is_match: bool


def _pick_device(device: str) -> torch.device:
    if device and device != "auto":
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("DEVICE is set to CUDA but torch.cuda.is_available() is False")
        if device == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("DEVICE is set to MPS but torch.backends.mps.is_available() is False")
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _resize_for_embedding(image: Image.Image, max_edge: int) -> Image.Image:
    if max_edge <= 0:
        return image
    width, height = image.size
    longest = max(width, height)
    if longest <= max_edge:
        return image
    scale = max_edge / float(longest)
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))
    if hasattr(Image, "Resampling"):
        resample = Image.Resampling.BILINEAR
    else:
        resample = Image.BILINEAR
    return image.resize((new_w, new_h), resample=resample)


def _maybe_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def _log_stage(stage: str, device: torch.device) -> None:
    if device.type == "cuda":
        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        logger.info(
            "stage=%s device=cuda allocated_mb=%.1f reserved_mb=%.1f",
            stage,
            allocated / (1024 * 1024),
            reserved / (1024 * 1024),
        )
    elif device.type == "mps":
        logger.info("stage=%s device=mps", stage)
    else:
        logger.info("stage=%s device=cpu", stage)


def _clear_nested(obj):
    """Recursively clear dicts/lists/tuples to drop all tensor references."""
    if isinstance(obj, dict):
        for v in obj.values():
            _clear_nested(v)
        obj.clear()
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _clear_nested(v)


class RomaMatcher:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        ensure_repo_on_path(settings.romav2_dir)
        torch.set_float32_matmul_precision("highest")
        self.device = _pick_device(settings.device)
        if self.device.type != "cuda":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                raise RuntimeError("DINOv3 requires CUDA; torch.cuda.is_available() is False")
        self._roma_device = torch.device("cpu")
        self._set_romav2_device(self._roma_device)
        from romav2 import RoMaV2
        self._embedding_threshold = settings.embedding_threshold
        self._min_inliers = 10
        self._num_matches = int(os.getenv("ROMA_NUM_MATCHES", "1500"))
        self._ransac_thresh = float(os.getenv("ROMA_RANSAC_THRESH", "0.2"))
        self._ransac_conf = float(os.getenv("ROMA_RANSAC_CONF", "0.999"))
        self._ransac_iters = int(os.getenv("ROMA_RANSAC_ITERS", "2000"))

        # Cache for query image preprocessing in RoMa matching
        self._query_cache_key: object = None
        self._query_cache_lr: torch.Tensor | None = None
        self._query_cache_hr: torch.Tensor | None = None

        # Pre-import romav2 utilities
        from romav2.device import device as _roma_dev
        from romav2.geometry import prec_mat_from_prec_params as _prec_fn
        self._roma_dev_ref = lambda: __import__("romav2.device", fromlist=["device"]).device
        self._prec_mat_fn = _prec_fn

        self._dinov3_dir = settings.dinov3_dir
        if not self._dinov3_dir.exists():
            raise FileNotFoundError(f"DINOv3 model dir not found: {self._dinov3_dir}")
        self._processor = AutoImageProcessor.from_pretrained(self._dinov3_dir)
        self._embed_model = AutoModel.from_pretrained(self._dinov3_dir).to(self.device)
        self._embed_model.eval()
        self._use_fp16 = self.device.type == "cuda"
        self._embed_dtype = self._infer_embed_dtype()
        self._autocast_dtype = self._pick_autocast_dtype()
        if self.device.type == "cuda":
            # Ensure all parameters share a single dtype to avoid mixed-type conv errors.
            self._embed_model = self._embed_model.to(dtype=self._embed_dtype)
        self._embed_batch_size = settings.embed_batch_size
        self._embed_max_edge = settings.embed_max_edge
        self._cached_embedding_path: Path | None = None
        self._cached_embedding: torch.Tensor | None = None
        self._cached_size: tuple[int, int] | None = None

        self._ensure_weights_cached(settings.romav2_weights)
        self._set_romav2_device(self._roma_device)
        self._model = RoMaV2(RoMaV2.Cfg(compile=settings.romav2_compile))
        self._model.apply_setting("precise")
        self._set_romav2_device(self.device)
        self._model.to(self.device)
        self._roma_device = self.device
        self._embed_on_device = True
        self._roma_on_device = True

    def _set_romav2_device(self, device: torch.device | None = None) -> None:
        if device is None:
            device = self.device
        try:
            import romav2.device as roma_device
        except Exception:
            return
        roma_device.device = device
        for name, module in list(sys.modules.items()):
            if not name.startswith("romav2."):
                continue
            if not hasattr(module, "device"):
                continue
            try:
                value = getattr(module, "device")
            except Exception:
                continue
            if isinstance(value, torch.device):
                setattr(module, "device", device)

    def _ensure_roma_on_device(self) -> None:
        try:
            model_device = next(self._model.parameters()).device
        except StopIteration:
            model_device = torch.device("cpu")
        if model_device != self.device:
            self._model.to(self.device)
            model_device = self.device
        self._roma_device = model_device
        self._roma_on_device = model_device == self.device

    @staticmethod
    def _ensure_weights_cached(local_weights: Path) -> Path:
        if not local_weights.exists():
            raise FileNotFoundError(f"RoMaV2 weights not found: {local_weights}")
        hub_dir = Path(torch.hub.get_dir())
        cache_path = hub_dir / "checkpoints" / local_weights.name
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        if not cache_path.exists():
            cache_path.write_bytes(local_weights.read_bytes())
        return cache_path

    @staticmethod
    def _load_image(path: Path) -> Image.Image:
        return Image.open(path).convert("RGB")

    def _infer_embed_dtype(self) -> torch.dtype:
        param = next(self._embed_model.parameters(), None)
        return param.dtype if param is not None else torch.float32

    def _pick_autocast_dtype(self) -> torch.dtype:
        if self.device.type != "cuda":
            return torch.float32
        if self._embed_dtype == torch.bfloat16:
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
            self._embed_model = self._embed_model.to(dtype=torch.float16)
            self._embed_dtype = torch.float16
            return torch.float16
        if self._embed_dtype == torch.float16:
            return torch.float16
        return torch.float16

    def _get_embedding(self, image: Image.Image) -> torch.Tensor:
        self._ensure_embed_on_device()
        image = _resize_for_embedding(image, self._embed_max_edge)
        inputs = self._processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        if "pixel_values" in inputs and inputs["pixel_values"].dtype != self._embed_dtype:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype=self._embed_dtype)
        with torch.inference_mode():
            if self._use_fp16:
                with torch.autocast(device_type="cuda", enabled=True, dtype=self._autocast_dtype):
                    outputs = self._embed_model(**inputs)
            else:
                outputs = self._embed_model(**inputs)
            pooled = getattr(outputs, "pooler_output", None)
            if pooled is not None:
                emb = pooled
            else:
                emb = outputs.last_hidden_state[:, 0]
        embedding = torch.nn.functional.normalize(emb, dim=-1)
        del inputs, outputs, emb, pooled
        return embedding

    def _auto_gpu_batch(self) -> int:
        """Estimate a good embedding batch size based on free GPU memory."""
        env_val = os.getenv("EMBED_GPU_BATCH")
        if env_val:
            return int(env_val)
        if self.device.type != "cuda":
            return 32
        try:
            free, total = torch.cuda.mem_get_info(self.device)
        except Exception:
            return 64
        # Reserve 2GB headroom for RoMa / other ops; use at most 50% of free
        usable = max(0, free - 2 * 1024**3) * 0.5
        # ~80MB per image in batch (224x224 ViT forward + activations)
        per_image_bytes = 80 * 1024 * 1024
        batch = max(16, min(256, int(usable / per_image_bytes)))
        return batch

    def embed_images(self, images: list[Image.Image]) -> tuple[torch.Tensor, list[tuple[int, int]]]:
        """Embed images with pipelined CPU preprocess / GPU inference.

        A producer thread pool preprocesses images and feeds a bounded queue;
        the main thread drains the queue into GPU batches.  Embeddings stay on
        GPU throughout — no CPU round-trip.
        """
        if not images:
            return torch.empty((0, 0), device=self.device), []
        self._ensure_embed_on_device()
        sizes = [image.size for image in images]
        max_edge = self._embed_max_edge
        gpu_batch = self._auto_gpu_batch()
        preprocess_workers = int(os.getenv("EMBED_PREPROCESS_WORKERS", "4"))

        import queue as _queue

        def _preprocess(img: Image.Image) -> torch.Tensor:
            img = _resize_for_embedding(img, max_edge)
            inputs = self._processor(images=img, return_tensors="pt")
            pv = inputs["pixel_values"]
            if pv.dtype != self._embed_dtype:
                pv = pv.to(dtype=self._embed_dtype)
            return pv.squeeze(0)

        # Bounded queue: at most 2 GPU-batches worth of preprocessed tensors
        q: _queue.Queue[torch.Tensor | None] = _queue.Queue(maxsize=gpu_batch * 2)

        def _producer() -> None:
            with ThreadPoolExecutor(max_workers=preprocess_workers) as pool:
                for tensor in pool.map(_preprocess, images):
                    q.put(tensor)
            q.put(None)  # sentinel

        import threading as _threading
        prod_thread = _threading.Thread(target=_producer, daemon=True)
        prod_thread.start()

        embeddings: list[torch.Tensor] = []
        batch_buf: list[torch.Tensor] = []

        def _flush_batch() -> None:
            if not batch_buf:
                return
            batch_t = torch.stack(batch_buf).to(self.device)
            batch_buf.clear()
            with torch.inference_mode():
                if self._use_fp16:
                    with torch.autocast(device_type="cuda", enabled=True, dtype=self._autocast_dtype):
                        outputs = self._embed_model(pixel_values=batch_t)
                else:
                    outputs = self._embed_model(pixel_values=batch_t)
                pooled = getattr(outputs, "pooler_output", None)
                if pooled is not None:
                    emb = pooled
                else:
                    emb = outputs.last_hidden_state[:, 0]
            # Keep on GPU — no .cpu() round-trip
            embeddings.append(torch.nn.functional.normalize(emb, dim=-1))
            del batch_t, outputs, emb, pooled

        while True:
            item = q.get()
            if item is None:
                break
            batch_buf.append(item)
            if len(batch_buf) >= gpu_batch:
                _flush_batch()
        _flush_batch()
        prod_thread.join()

        result = torch.cat(embeddings, dim=0)
        embeddings.clear()
        return result, sizes

    def clear_cache(self) -> None:
        return

    def log_stage(self, stage: str) -> None:
        _log_stage(stage, self.device)

    @property
    def embedding_threshold(self) -> float:
        return self._embedding_threshold

    def _ensure_embed_on_device(self) -> None:
        if self.device.type == "cpu":
            return
        if self._embed_on_device:
            return
        self._embed_model.to(self.device)
        self._embed_on_device = True
        _log_stage("dino_embedded_load", self.device)

    def release_embed_model(self) -> None:
        return

    def prepare_matcher(self) -> None:
        if self.device.type == "cpu":
            return
        if self._roma_on_device:
            return
        self._set_romav2_device(self.device)
        self._model.to(self.device)
        self._roma_device = self.device
        self._roma_on_device = True
        _log_stage("romav2_load", self.device)

    def release_matcher(self) -> None:
        return

    def _embed_path(self, image_path: Path, use_cache: bool) -> tuple[torch.Tensor, tuple[int, int]]:
        if use_cache and self._cached_embedding_path == image_path:
            if self._cached_embedding is not None and self._cached_size is not None:
                return self._cached_embedding, self._cached_size

        image = self._load_image(image_path)
        embedding = self._get_embedding(image)
        size = image.size
        if use_cache:
            self._cached_embedding_path = image_path
            self._cached_embedding = embedding
            self._cached_size = size
        return embedding, size

    def _get_query_preprocessed(
        self, img_a: ImageInput,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return cached (lr, hr) tensors for query image on roma device."""
        cache_key = id(img_a) if not isinstance(img_a, Path) else img_a
        if cache_key == self._query_cache_key and self._query_cache_lr is not None:
            return self._query_cache_lr, self._query_cache_hr

        model = self._model
        cpu_device = torch.device("cpu")
        img_A = model._load_image(img_a, device_override=cpu_device)
        roma_device = self._roma_dev_ref()

        img_A_lr = torch.nn.functional.interpolate(
            img_A, size=(model.H_lr, model.W_lr),
            mode="bicubic", align_corners=False, antialias=True,
        )
        if model.H_hr is not None and model.W_hr is not None:
            img_A_hr = torch.nn.functional.interpolate(
                img_A, size=(model.H_hr, model.W_hr),
                mode="bicubic", align_corners=False, antialias=True,
            )
        else:
            img_A_hr = None
        del img_A

        if roma_device.type != "cpu":
            img_A_lr = img_A_lr.to(roma_device)
            if img_A_hr is not None:
                img_A_hr = img_A_hr.to(roma_device)

        self._query_cache_key = cache_key
        self._query_cache_lr = img_A_lr
        self._query_cache_hr = img_A_hr
        return img_A_lr, img_A_hr

    def invalidate_query_cache(self) -> None:
        self._query_cache_key = None
        self._query_cache_lr = None
        self._query_cache_hr = None

    def _extract_keypoints(
        self,
        img_a: ImageInput,
        img_b: ImageInput,
        size_a: tuple[int, int],
        size_b: tuple[int, int],
    ) -> tuple[np.ndarray | None, np.ndarray | None, int]:
        """GPU phase: RoMa forward + sample keypoints. Returns (kpts_a, kpts_b, total_matches)."""
        model = self._model
        prec_mat_from_prec_params = self._prec_mat_fn
        roma_device = self._roma_dev_ref()
        _raw_forward = getattr(model.forward, "__wrapped__", None)

        with torch.no_grad():
            img_A_lr, img_A_hr = self._get_query_preprocessed(img_a)

            cpu_device = torch.device("cpu")
            img_B = model._load_image(img_b, device_override=cpu_device)
            img_B_lr = torch.nn.functional.interpolate(
                img_B, size=(model.H_lr, model.W_lr),
                mode="bicubic", align_corners=False, antialias=True,
            )
            if model.H_hr is not None and model.W_hr is not None:
                img_B_hr = torch.nn.functional.interpolate(
                    img_B, size=(model.H_hr, model.W_hr),
                    mode="bicubic", align_corners=False, antialias=True,
                )
            else:
                img_B_hr = None
            del img_B

            if roma_device.type != "cpu":
                img_B_lr = img_B_lr.to(roma_device)
                if img_B_hr is not None:
                    img_B_hr = img_B_hr.to(roma_device)

            if _raw_forward is not None:
                predictions = _raw_forward(model, img_A_lr, img_B_lr, img_A_hr=img_A_hr, img_B_hr=img_B_hr)
            else:
                predictions = model(img_A_lr, img_B_lr, img_A_hr=img_A_hr, img_B_hr=img_B_hr)

            del img_B_lr, img_B_hr

            warp_AB = predictions["warp_AB"]
            confidence_AB = predictions["confidence_AB"]

            overlap_AB = confidence_AB[..., :1].sigmoid()
            if model.threshold is not None:
                overlap_AB[overlap_AB > model.threshold] = 1.0
            precision_AB = prec_mat_from_prec_params(confidence_AB[..., 1:4])

            preds = {
                "warp_AB": warp_AB,
                "overlap_AB": overlap_AB,
                "precision_AB": precision_AB,
            }

            if model.bidirectional:
                warp_BA = predictions["warp_BA"]
                confidence_BA = predictions["confidence_BA"]
                overlap_BA = confidence_BA[..., :1].sigmoid()
                if model.threshold is not None:
                    overlap_BA[overlap_BA > model.threshold] = 1.0
                precision_BA = prec_mat_from_prec_params(confidence_BA[..., 1:4])
                preds["overlap_BA"] = overlap_BA
                preds["warp_BA"] = warp_BA
                preds["precision_BA"] = precision_BA
                del warp_BA, confidence_BA, overlap_BA, precision_BA

            del warp_AB, confidence_AB, overlap_AB, precision_AB
            _clear_nested(predictions)
            del predictions

            sample_out = model.sample(preds, self._num_matches)
            matches = sample_out[0]
            total_matches = int(matches.shape[0])
            if total_matches < 8:
                del preds, matches, sample_out
                return None, None, total_matches

            h_a, w_a = size_a[1], size_a[0]
            h_b, w_b = size_b[1], size_b[0]
            kpts_a, kpts_b = model.to_pixel_coordinates(matches, h_a, w_a, h_b, w_b)

            kpts_a_np = kpts_a.cpu().numpy().astype(np.float32)
            kpts_b_np = kpts_b.cpu().numpy().astype(np.float32)
            del preds, matches, sample_out, kpts_a, kpts_b

        return kpts_a_np, kpts_b_np, total_matches

    def _ransac(
        self,
        kpts_a_np: np.ndarray | None,
        kpts_b_np: np.ndarray | None,
        total_matches: int,
    ) -> tuple[bool, dict[str, float]]:
        """CPU phase: RANSAC on extracted keypoints."""
        _fail = {"inliers": 0, "inlier_ratio": 0.0, "total_matches": total_matches}
        if kpts_a_np is None or kpts_b_np is None:
            return False, _fail
        if kpts_a_np.shape[0] < 8 or kpts_b_np.shape[0] < 8:
            return False, _fail
        if not (np.isfinite(kpts_a_np).all() and np.isfinite(kpts_b_np).all()):
            return False, _fail

        try:
            F, mask = cv2.findFundamentalMat(
                    kpts_a_np,
                    kpts_b_np,
                    method=cv2.USAC_MAGSAC,
                    ransacReprojThreshold=self._ransac_thresh,
                    confidence=self._ransac_conf,
                    maxIters=self._ransac_iters,
                    )
        except cv2.error:
            return False, _fail
        if F is None or mask is None:
            return False, _fail

        inliers = int(mask.ravel().sum())
        inlier_ratio = float(inliers / max(1, len(mask)))
        return True, {
                "inliers": inliers,
                "inlier_ratio": inlier_ratio,
                "total_matches": total_matches,
                }

    def _estimate_match(
        self,
        img_a: ImageInput,
        img_b: ImageInput,
        size_a: tuple[int, int],
        size_b: tuple[int, int],
    ) -> tuple[bool, dict[str, float]]:
        kpts_a, kpts_b, total = self._extract_keypoints(img_a, img_b, size_a, size_b)
        return self._ransac(kpts_a, kpts_b, total)

    def match_with_similarity(
        self,
        image_a: ImageInput,
        image_b: ImageInput,
        size_a: tuple[int, int],
        size_b: tuple[int, int],
        embedding_similarity: float,
        *,
        embedding_threshold: float | None = None,
    ) -> MatchResult:
        self.prepare_matcher()
        threshold = (
            self._embedding_threshold
            if embedding_threshold is None
            else float(embedding_threshold)
        )
        if embedding_similarity < threshold:
            return MatchResult(
                embedding_similarity=embedding_similarity,
                embedding_pass=False,
                ransac_ok=False,
                inliers=0,
                total_matches=0,
                inlier_ratio=0.0,
                score=0.0,
                is_match=False,
            )
        ok, stats = self._estimate_match(image_a, image_b, size_a, size_b)
        inliers = int(stats["inliers"])
        inlier_ratio = float(stats["inlier_ratio"])
        total_matches = int(stats["total_matches"])
        score = inlier_ratio if ok else 0.0
        is_match = ok and inliers >= self._min_inliers and score > 0.0
        return MatchResult(
                embedding_similarity=embedding_similarity,
                embedding_pass=True,
                ransac_ok=ok,
                inliers=inliers,
                total_matches=total_matches,
                inlier_ratio=inlier_ratio,
                score=score,
                is_match=is_match,
                )

    def match(self, image_path_a: Path, image_path_b: Path) -> MatchResult:
        emb_a, size_a = self._embed_path(image_path_a, use_cache=True)
        emb_b, size_b = self._embed_path(image_path_b, use_cache=False)
        embedding_similarity = float(
                torch.nn.functional.cosine_similarity(emb_a, emb_b).item()
                )
        return self.match_with_similarity(
                image_path_a, image_path_b, size_a, size_b, embedding_similarity
                )

    @staticmethod
    def save_image(image: Image.Image, path: Path) -> None:
        image.save(path)
