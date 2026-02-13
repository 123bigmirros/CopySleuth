# 使用说明（视频 OCR 关键词管线）

## 前置
先下载模型：
```bash
bash scripts/download_models.sh
```

## 使用：关键词来自 Excel（推荐）
Excel 每个表的第一列会作为关键词列表。
```bash
python video_ocr_from_excel.py \
  /home/dymiao/pic_cmp/video/20250214_104446.mp4 \
  /home/dymiao/pic_cmp/video/demo.xlsx \
  --out /home/dymiao/pic_cmp/video/matches.json
```
可选：如果第一行是表头，加 `--drop-header`。

## 使用：命令行传关键词
```bash
python video_ocr_pipeline.py /path/to/video.mp4 --keywords "foo,bar"
# 或
python video_ocr_pipeline.py /path/to/video.mp4 --keyword foo --keyword bar --out /tmp/matches.json
```

## 关键规则
- 默认使用 `frame_select=scenedetect` 做抽帧 + 去重优化（可切回 `stride` 处理全帧）。
- 关键词匹配规则：`keyword in text`。
- 时间段合并：命中帧间隔 ≤ 3 帧视为同一段（可用 `--max-gap-frames` 调整）。
- 默认使用 GPU（`--device gpu`）。
- 性能参数：`--batch-size`（默认 8），可配合 `--rec-batch-size 16/32` 提升吞吐。

## 输出
JSON 内包含每个关键词的命中帧号与时间段。

新增字段（用于前端定位）：
- `keywords[].segments[].first_frame_positions`: 命中段第一帧的文字位置（含 `bbox`/`polygon`）
- `keywords[].segments[].first_frame_preview`: 第一帧预览（base64 data URL）
- `keywords[].segments[].first_frame_width` / `first_frame_height`: 第一帧尺寸
