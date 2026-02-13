# Algorithm Service / 算法服务

核心算法服务，负责 SAM3 分割 + RoMaV2/DINOv3 匹配，返回原始候选结果（不含预览和 OCR）。

Core algorithm microservice handling SAM3 segmentation and RoMaV2/DINOv3 matching. Returns raw candidate results without previews or OCR.

## 模块结构 / Module Structure

```
algo_service/
├── app.py              # FastAPI 入口 + lifespan
├── config.py           # 服务配置 (DATA_DIR, workers, CORS)
├── routes/
│   ├── health.py       # GET /v1/health
│   ├── detect.py       # POST /v1/detect (同步检测)
│   └── tasks.py        # /v1/tasks/* (异步任务)
├── services/
│   └── image_loader.py # 图片/Excel 加载与校验
├── serializers.py      # 结果序列化 (candidate payload 等)
├── workers.py          # 异步任务 worker (image/video)
└── requests_client.py  # CLI 测试客户端
```

## 运行 / Run

```bash
conda activate pic_cmp
export SAM3_DIR=./models/sam3
export ROMA_V2_DIR=./models/RoMaV2
export DINOV3_DIR=./models/DINOv3
export SAM3_CHECKPOINT=./models/sam3/sam3_checkpoints/sam3.pt
export ALGO_DATA_DIR=./data/algo
uvicorn algo_service.app:app --host 127.0.0.1 --port 8001
```

## 环境变量 / Environment Variables

| 变量 | 默认值 | 说明 |
|---|---|---|
| `ALGO_DATA_DIR` | `/data/pic_cmp_algo` | 任务数据存储目录 |
| `ALGO_TASK_WORKERS` | `4` | 异步任务并发 worker 数 |
| `ALGO_MAX_FILE_COUNT` | `10000` | multipart 上传文件数上限 |
| `ALGO_CORS_ORIGINS` | `*` | CORS 允许的 origin (逗号分隔) |

## API

| 方法 | 路径 | 说明 |
|---|---|---|
| `GET` | `/v1/health` | 健康检查 |
| `POST` | `/v1/detect` | 同步图片检测 (multipart) |
| `POST` | `/v1/tasks` | 创建异步任务 (图片/视频) |
| `GET` | `/v1/tasks/{task_id}` | 查询任务状态 |
| `GET` | `/v1/tasks/{task_id}/events` | SSE 事件流 |
| `POST` | `/v1/tasks/{task_id}/cancel` | 取消任务 |

### POST /v1/detect

multipart/form-data 参数：

- `query_images` (multiple) 或 `query_image` — 查询图片
- `target_image` — 目标图片
- `embedding_threshold` — 嵌入相似度阈值 (0-1 或 0-100)
- `image_mode` — 候选图返回模式: `b64` 或 `path`

### POST /v1/tasks

multipart/form-data 参数：

- `query_images` (multiple) — 查询图片 (支持 Excel)
- `target_file` — 目标图片或视频
- `video_start` / `video_end` — 视频时间范围 (秒)
- `embedding_threshold` — 嵌入相似度阈值
- `image_mode` — 候选图返回模式

## 测试 / Tests

```bash
pytest tests/ -v
```

测试通过 mock 算法后端运行，不需要 GPU 或模型权重。
