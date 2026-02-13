# Backend (MVP)

## Run

```bash
pip install -r backend/requirements.txt
export SAM3_DIR=./models/sam3
export SAM3_CHECKPOINT=./models/sam3/sam3_checkpoints/sam3.pt
export MEDIA_SERVICE_URL=http://127.0.0.1:8012
# 算法服务（推荐）：后端通过 HTTP 调用算法服务执行匹配
# export ALGO_SERVICE_URL=http://127.0.0.1:8001
# 算法候选图返回模式（默认 b64）
# export ALGO_IMAGE_MODE=b64
# OCR 默认本地执行（建议 OCR 服务端口 8002），如需远端 OCR 服务再设置：
# export OCR_API_BASE=http://127.0.0.1:8002
uvicorn app.main:app --app-dir backend --host 0.0.0.0 --port 8003
```

## API

### `GET /health`

响应:

```json
{"status":"ok"}
```

### `POST /v1/detect` (同步图片匹配，兼容旧版)

请求 `multipart/form-data`:

- `query_image`: 查询图片文件（或 Excel：A 列文字、B 列图片）
- `target_image`: 目标图片文件
- `use_ocr`: 可选，`true/false`，仅当查询为 Excel 时生效

示例：

```bash
curl -X POST "http://127.0.0.1:8003/v1/detect" \
  -F "query_image=@/path/to/query.xlsx" \
  -F "target_image=@/path/to/target.png" \
  -F "use_ocr=true"
```

响应 `DetectionResponse` (图片结果):

- `media_type`: `"image"`
- `is_match`: 是否存在命中
- `best_bbox`: 最佳候选框 `(x0,y0,x1,y1)` 或 `null`
- `best_score`: 最佳分数或 `null`
- `candidates`: 候选总数
- `match`: 匹配统计或 `null`
- `candidate_results[]`: 候选列表
  - `id`, `label`, `kind`, `bbox`, `score`
  - `match`: `embedding_similarity`, `embedding_pass`, `ransac_ok`, `inliers`,
    `total_matches`, `inlier_ratio`, `score`, `is_match`
  - `preview`: base64 data URL 或远端 URL（配置 `MEDIA_SERVICE_URL` 时）
  - `full_preview`, `full_width`, `full_height` (可选)
- `ocr`: 可选，OCR 文字匹配结果（仅当 `use_ocr=true` 且 Excel 查询）

### `POST /v1/tasks` (异步任务：图片/视频)

请求 `multipart/form-data`:

- `query_image`: 查询图片或 Excel（含多图；A 列文字、B 列图片）
- `target_file`: 目标图片或视频文件
- `video_start`: 可选，视频起始秒数
- `video_end`: 可选，视频结束秒数
- `match_threshold`: 可选，0~1 浮点
- `use_ocr`: 可选，`true/false`，仅当查询为 Excel 时生效
- `task_name`: 可选，自定义任务名

响应:

```json
{"task_id":"..."}
```

示例（图片 + Excel + OCR）：

```bash
curl -X POST "http://127.0.0.1:8003/v1/tasks" \
  -F "query_image=@/path/to/query.xlsx" \
  -F "target_file=@/path/to/target.png" \
  -F "match_threshold=0.95" \
  -F "embedding_threshold=0.50" \
  -F "use_ocr=true" \
  -F "task_name=demo-ocr"
```

示例（视频）：

```bash
curl -X POST "http://127.0.0.1:8003/v1/tasks" \
  -F "query_image=@/path/to/query.png" \
  -F "target_file=@/path/to/target.mp4" \
  -F "video_start=5" \
  -F "video_end=30" \
  -F "match_threshold=0.90" \
  -F "embedding_threshold=0.45" \
  -F "task_name=demo-video"
```

### `GET /v1/tasks/{task_id}` (任务状态/结果)

查询参数:

- `threshold`: 可选，0~1 或 0~100（百分比），用于过滤结果

响应 `TaskStatusResponse`:

- `task_id`, `status` (`pending|running|done|error|canceled`), `progress`, `error`
- `ocr_progress`, `ocr_stage`, `ocr_message`：OCR 进度与提示（未启用时为 `null`）
- `result`: `DetectionResponse` 或 `MultiDetectionResponse`

示例：

```bash
curl "http://127.0.0.1:8003/v1/tasks/{task_id}?threshold=0.95"
```

`MultiDetectionResponse`:

- `media_type`: `"multi"`
- `is_match`, `best_score`, `match`
- `query_results[]`: `{query_id, query_label, query_preview, result}`
- `ocr`: 可选，OCR 文字匹配结果

`DetectionResponse` (视频结果):

- `media_type`: `"video"`
- `segments[]`: `VideoSegmentResult` 列表（含 `preview`/`full_preview`）
- `fps`, `frame_count`, `duration`
- `ocr`: 可选，OCR 文字匹配结果（仅当 Excel 查询）

`OCRResult`:

- `enabled`: 是否启用
- `keywords`, `keyword_count`
- `texts`, `text`
- `matches`, `match_count`, `is_match`
- `positions`: 可选，命中文字的位置（图片返回 `lines` + `preview` + 尺寸信息）
- `video`: 视频 OCR 结果（含每个关键词的时间段/帧段）
- `error`: OCR 失败信息（如有）

视频 OCR 返回示例（`result.ocr.video.keywords`）：

```json
{
  "keyword": "示例词",
  "matches": 3,
  "segments": [
    {"start_time": 2.4, "end_time": 3.1, "start_frame": 60, "end_frame": 78, "frames": [60, 65, 70, 78]}
  ]
}
```

### `POST /v1/tasks/{task_id}/cancel`

响应同 `TaskStatusResponse`，`status` 为 `canceled`。

### `GET /v1/tasks/{task_id}/events` (SSE)

查询参数:

- `last_event_id`: 可选，也可用请求头 `Last-Event-Id`

事件类型:

- `progress`: `{progress, stage, message}`
- `ocr_progress`: `{progress, stage, message}`
- `partial`: `{result: ...}`
- `result`: `{result: ...}`
- `done` / `error` / `canceled`

示例：

```bash
curl -N "http://127.0.0.1:8003/v1/tasks/{task_id}/events?threshold=0.95"
```

### `GET /v1/tasks/{task_id}/download`

查询参数:

- `threshold`: 可选，0~1 或 0~100（百分比）

响应: JSON 文件下载（`Content-Disposition: attachment`）。

### `GET /v1/history`

响应:

```json
{"items":[{"task_id":"...","name":"...","created_at":0,"status":"done","media_type":"image","match_threshold":0.95}]}
```

### `GET /v1/history/{task_id}`

查询参数:

- `threshold`: 可选，0~1 或 0~100（百分比）

响应包含: `task_id`, `name`, `created_at`, `status`, `media_type`, `match_threshold`, `result`。

### `DELETE /v1/history/{task_id}`

响应:

```json
{"status":"ok"}
```

## Architecture Note

Backend now depends on an algorithm service interface. The default implementation
is a local adapter that wraps SAM3/RoMaV2 pipelines; this keeps API responses
unchanged while allowing the algorithm layer to be swapped later without
changing backend routes.
