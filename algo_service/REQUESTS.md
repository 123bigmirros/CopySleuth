# Algorithm Service Requests (Python requests)

这份说明面向“直接调用算法服务”的开发者，覆盖算法服务全部公开接口与返回结构。

## 依赖

```bash
pip install requests
```

## 服务地址

默认：`http://127.0.0.1:8001`（脚本可通过环境变量 `ALGO_BASE_URL` 覆盖）

## 能力范围与约束

- 算法服务只做 **分割 + 匹配**，返回原始结果。
- **不做 OCR**，也不解析 Excel。
- 支持 **多张查询图**（`query_images` 重复字段）。
- `/v1/detect` 仅支持 **图片目标**；`/v1/tasks` 支持 **图片/视频目标**。
- `embedding_threshold` 支持 `0~1` 或 `0~100`，>1 且 <=100 会自动换算到 0~1。
- `image_mode` 支持 `b64` 或 `path`：
  - `b64`：返回 `image_b64` / `full_image_b64`（PNG Base64）
  - `path`：返回 `image_path` / `full_image_path`（服务端本地路径）
- `image_mode` 默认值来自环境变量 `ALGO_IMAGE_MODE`（未设置时为 `b64`）。
- `image_mode=path` 时，文件保存在 `ALGO_DATA_DIR`（默认 `/data/pic_cmp_algo`）下。

## 1) 同步图片匹配 `/v1/detect`

- 方法：`POST`
- 类型：`multipart/form-data`
- 参数：
  - `query_images`：可重复字段，多张查询图
  - `query_image`：单张查询图（兼容字段，二选一）
  - `target_image`：目标图
  - `embedding_threshold`：可选，0~1 或 0~100
  - `image_mode`：可选，`b64` / `path`

### 请求示例

```bash
curl -X POST "http://127.0.0.1:8001/v1/detect" \
  -F "query_images=@/path/to/q1.png" \
  -F "query_images=@/path/to/q2.png" \
  -F "target_image=@/path/to/target.png" \
  -F "embedding_threshold=0.5" \
  -F "image_mode=b64"
```

### 响应结构（简化）

> `results` 数组顺序与 `query_images` 上传顺序一致。

```json
{
  "media_type": "image",
  "results": [
    {
      "is_match": true,
      "best_bbox": [x0, y0, x1, y1],
      "best_score": 0.97,
      "best_match": {"score": 0.97, "is_match": true, ...},
      "candidates": 12,
      "candidate_results": [
        {
          "kind": "segment",
          "bbox": [x0, y0, x1, y1],
          "score": 0.97,
          "match": {"score": 0.97, "is_match": true, ...},
          "image_b64": "..."
        }
      ]
    }
  ]
}
```

### 字段说明

- `best_match` / `match` 字段包含：
  - `embedding_similarity`, `embedding_pass`, `ransac_ok`, `inliers`,
    `total_matches`, `inlier_ratio`, `score`, `is_match`
- `candidate_results[].kind`：候选类型（如 `segment` / `full`）。
- `image_mode=path` 时，`candidate_results[]` 中返回 `image_path`（服务端路径）。

## 2) 异步图片/视频任务 `/v1/tasks`

- 方法：`POST`
- 类型：`multipart/form-data`
- 参数：
  - `query_images`：可重复字段，多张查询图（必填）
  - `target_file`：目标图或视频
  - `video_start` / `video_end`：可选，视频起止秒
  - `embedding_threshold`：可选
  - `image_mode`：可选，`b64` / `path`

> 目标是否为视频根据 `target_file.content_type` 是否以 `video/` 开头判断。

### 响应

```json
{"task_id": "...", "media_type": "image"}
```

### SSE 事件订阅

- `GET /v1/tasks/{task_id}/events`
- 事件类型：`progress` / `partial` / `result` / `done` / `error` / `canceled`
- `partial` 只会在 **图片任务** 中出现（用于流式返回候选）。

### 结果结构（图片任务）

- 单张查询：`result` 直接是图片结果（与 `/v1/detect` 单条一致）。
- 多张查询：`media_type=multi`，返回 `query_results`。

```json
{
  "media_type": "multi",
  "is_match": true,
  "best_score": 0.92,
  "match": {"score": 0.92, "is_match": true, ...},
  "query_results": [
    {"query_index": 1, "result": {"media_type": "image", ...}},
    {"query_index": 2, "result": {"media_type": "image", ...}}
  ]
}
```

### 结果结构（视频任务）

- 单张查询：`media_type=video`，包含 `segments`。
- 多张查询：`media_type=multi`，每个 `result` 为视频结果。

```json
{
  "media_type": "video",
  "is_match": true,
  "best_score": 0.91,
  "best_match": {"score": 0.91, "is_match": true, ...},
  "candidates": 8,
  "segments": [
    {
      "obj_id": 1,
      "kind": "segment",
      "start_time": 1.23,
      "end_time": 2.34,
      "first_frame_index": 10,
      "last_frame_index": 42,
      "bbox": [x0, y0, x1, y1],
      "score": 0.91,
      "match": {"score": 0.91, "is_match": true, ...},
      "image_b64": "...",
      "full_image_b64": "..."
    }
  ],
  "fps": 25.0,
  "frame_count": 900,
  "duration": 36.0
}
```

`image_mode=path` 时，`segments` 返回 `image_path` / `full_image_path`。

## 3) 任务状态 / 取消

- `GET /v1/tasks/{task_id}`
  - 返回：`task_id`, `status`, `progress`, `error`, `result`
  - `status` 取值：`pending|running|done|error|canceled`
- `POST /v1/tasks/{task_id}/cancel`

## 4) OCR 服务（可选，独立服务）

OCR 服务默认地址：`http://127.0.0.1:8002`（脚本可通过环境变量 `OCR_BASE_URL` 覆盖）

- OCR 接口使用 **multipart/form-data** 直接上传文件。
- 注意：算法服务本身不调用 OCR，这里仅提供独立 OCR API 的使用方式。

- `POST /v1/ocr/image` (multipart)
  - `file`: 图片文件
- `POST /v1/ocr/video` (multipart)
  - `file`: 视频文件
  - `keywords`: 关键词（可重复字段）
- `GET /v1/jobs/{job_id}`

## 5) Python requests 脚本

脚本位置：`algo_service/requests_client.py`

### detect 示例（多查询图）

```bash
python algo_service/requests_client.py detect \
  --base-url http://127.0.0.1:8001 \
  --query /path/to/q1.png \
  --query /path/to/q2.png \
  --target /path/to/target.png \
  --embedding-threshold 0.5 \
  --image-mode b64 \
  --out-json /tmp/detect.json \
  --save-candidates /tmp/candidates
```

> `--save-candidates` 仅在 `image_mode=b64` 时生效。

### task + SSE 示例（视频）

```bash
python algo_service/requests_client.py task \
  --base-url http://127.0.0.1:8001 \
  --query /path/to/q1.png \
  --query /path/to/q2.png \
  --target /path/to/video.mp4 \
  --video-start 5 --video-end 30 \
  --embedding-threshold 0.5 \
  --image-mode b64 \
  --out-json /tmp/task_result.json
```

### status / cancel 示例

```bash
python algo_service/requests_client.py status --base-url http://127.0.0.1:8001 --task-id <task_id>
python algo_service/requests_client.py cancel --base-url http://127.0.0.1:8001 --task-id <task_id>
```

### OCR 示例

```bash
python algo_service/requests_client.py ocr-image \
  --ocr-base-url http://127.0.0.1:8002 \
  --path /path/to/image.png \
  --wait

python algo_service/requests_client.py ocr-video \
  --ocr-base-url http://127.0.0.1:8002 \
  --path /path/to/video.mp4 \
  --keyword keyword1 --keyword keyword2 \
  --wait

python algo_service/requests_client.py ocr-status \
  --ocr-base-url http://127.0.0.1:8002 \
  --job-id <job_id>
```

## 6) 最简 Python 片段

```python
import requests

base_url = "http://127.0.0.1:8001"
files = [
    ("query_images", ("q1.png", open("q1.png", "rb"), "image/png")),
    ("query_images", ("q2.png", open("q2.png", "rb"), "image/png")),
    ("target_image", ("t.png", open("t.png", "rb"), "image/png")),
]
resp = requests.post(
    f"{base_url}/v1/detect",
    files=files,
    data={"embedding_threshold": "0.5", "image_mode": "b64"},
)
print(resp.json())
```

SSE 订阅示例：

```python
import json
import requests

resp = requests.get(f"{base_url}/v1/tasks/{task_id}/events", stream=True)
for line in resp.iter_lines(decode_unicode=True):
    if line.startswith("data:"):
        payload = json.loads(line[len("data:"):].strip())
        print(payload)
```
