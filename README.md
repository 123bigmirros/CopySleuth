# CopySleuth

A copyright detection system built on SAM3, RoMaV2, and DINOv3. Given query images, it locates and determines whether the same content appears in target images or videos.

## Architecture

```
┌────────────┐     ┌──────────────┐     ┌──────────────┐
│  Frontend   │────▶│   Backend    │────▶│ Algo Service  │
│  (Vue 3)    │     │  (FastAPI)   │     │  (FastAPI)    │
│  :5178      │     │  :8003       │     │  :8001        │
└────────────┘     └──────┬───────┘     └──────────────┘
                          │
                   ┌──────┴───────┐
                   │ OCR Service  │
                   │ (PaddleOCR)  │
                   │ :8002        │
                   └──────────────┘
                          │
                   ┌──────┴───────┐
                   │Media Service │
                   │  :8012       │
                   └──────────────┘
```

| Service | Port | Description |
|---|---|---|
| Algo Service | 8001 | Core algorithm: SAM3 segmentation + RoMaV2/DINOv3 matching |
| OCR Service | 8002 | PaddleOCR text detection and recognition |
| Backend | 8003 | Main API: orchestrates algo, OCR, media, and result assembly |
| Media Service | 8012 | Preview image generation and storage |
| Frontend | 5178 | Vue 3 + Vite web UI for upload and result display |

## Prerequisites

- Python 3.12+
- CUDA 12.1+ (GPU inference)
- Node.js 18+ (frontend only)
- ~8-12 GB VRAM (SAM3 + RoMaV2 + DINOv3)

## Quick Start

### 1. Clone

```bash
git clone https://github.com/123bigmirros/CopySleuth.git
cd CopySleuth
```

### 2. Setup Environment

```bash
conda create -n pic_cmp python=3.12
conda activate pic_cmp
./scripts/setup.sh
```

This single script handles everything: installs pip dependencies, downloads models (SAM3 and DINOv3 via ModelScope, RoMaV2 via GitHub), and installs model packages.

### 3. Start Services

**Algorithm service only** (for API/script usage):

```bash
./scripts/run_algo.sh
```

**All services** (algo + backend + OCR + media + frontend):

```bash
./scripts/run_all.sh
```

Stop all services:

```bash
./scripts/stop_all.sh
```

### 4. Verify

```bash
./scripts/check_health.sh --wait
```

Frontend is available at http://localhost:5178 after all services are started.

## Usage Examples

### Image-to-Image Detection

```bash
# Using the test script
./scripts/test_image.sh query.png target.png

# Or directly via the CLI client
python -m algo_service.requests_client detect \
  --query query.png --target target.png --out-json result.json
```

### Image-to-Video Detection

```bash
# Using the test script
./scripts/test_video.sh query.png target.mp4

# Or directly via the CLI client
python -m algo_service.requests_client task \
  --query query.png --target target.mp4 --out-json result.json

# With time range (seconds)
python -m algo_service.requests_client task \
  --query query.png --target target.mp4 \
  --video-start 10 --video-end 60 --out-json result.json
```

## API Reference

### Backend (port 8003)

| Method | Path | Description |
|---|---|---|
| `POST` | `/v1/tasks` | Create async detection task (image/video) |
| `GET` | `/v1/tasks/{task_id}` | Get task status and result |
| `GET` | `/v1/tasks/{task_id}/events` | SSE event stream (progress, partial, result) |
| `POST` | `/v1/tasks/{task_id}/cancel` | Cancel task |
| `GET` | `/v1/tasks/{task_id}/download` | Download result as JSON |
| `POST` | `/v1/detect` | Synchronous image detection (legacy) |
| `GET` | `/v1/history` | Task history list |

### Algo Service (port 8001, internal)

| Method | Path | Description |
|---|---|---|
| `GET` | `/v1/health` | Health check |
| `POST` | `/v1/detect` | Synchronous image detection (multi-query) |
| `POST` | `/v1/tasks` | Async task (image/video) |
| `GET` | `/v1/tasks/{task_id}` | Task status |
| `GET` | `/v1/tasks/{task_id}/events` | SSE event stream |
| `POST` | `/v1/tasks/{task_id}/cancel` | Cancel task |

## Development

### Run Tests

```bash
conda activate pic_cmp
pytest tests/ -v
```

Tests run with mocked algorithm services — no GPU or model weights needed.

### Lint

```bash
pip install ruff
ruff check algo_service/ backend/ tests/
```

## Troubleshooting

### CUDA out of memory

SAM3 + RoMaV2 + DINOv3 requires ~8-12 GB VRAM. If you run out of memory:
- Lower `EMBED_MAX_EDGE` (default 518, try 364)
- Use CPU mode: set `DEVICE=cpu` in `.env` (much slower)

### Services not responding

```bash
./scripts/check_health.sh

# Check individual logs
tail -f logs/algo.log
tail -f logs/backend.log
tail -f logs/ocr.log
```

## License

[Apache License 2.0](LICENSE)

## Acknowledgments

- [SAM3](https://github.com/facebookresearch/sam3) — Meta Segment Anything Model
- [RoMaV2](https://github.com/Parskatt/RoMaV2) — Robust Dense Feature Matching v2
- [DINOv3](https://github.com/facebookresearch/dinov3) — Self-supervised Vision Transformer
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) — OCR toolkit
