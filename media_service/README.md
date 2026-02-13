# Media Preprocess Service

轻量服务：生成预览图并存储到本地目录，通过 `/media/*` 静态访问。

## 启动

```bash
pip install -r media_service/requirements.txt
chmod +x media_service/run.sh
MEDIA_STORE_DIR=/data/pic_cmp_media \
MEDIA_BASE_URL=http://10.246.52.103:8012 \
media_service/run.sh
```

环境变量：
- `MEDIA_STORE_DIR`: 预览图存储目录（默认 `./media_store`）
- `MEDIA_BASE_URL`: 生成预览 URL 的基地址（可选，默认请求地址）
- `HOST`/`PORT`: 服务监听地址（默认 `0.0.0.0:8012`）

## API

`POST /v1/preview` (multipart/form-data)
- `file`: 图片文件
- `max_size`: 最大边长（可选）
- `bbox`: `x0,y0,x1,y1`（可选，绘制框）

响应：
```json
{"url":"http://10.246.52.103:8012/media/xxx.png","width":320,"height":180}
```
