<template>
  <div class="page">
    <header class="hero">
      <div>
        <p class="eyebrow">Pic-Cmp / 版权检测</p>
        <h1>同画面内容检索</h1>
        <p class="lead">
          上传查询图与目标图/视频，系统将自动分割目标并进行相似内容比对。
        </p>
      </div>
      <div class="stats">
        <div class="stat">
          <span>模型</span>
          <strong>SAM3 + RoMa</strong>
        </div>
        <div class="stat">
          <span>接口</span>
            <strong>/v1/tasks</strong>
          </div>
        </div>
      </header>

    <main class="grid">
      <section class="card upload-preview">
        <div class="upload-header">
          <div>
            <h2>上传与预览</h2>
            <p class="hint">查询图支持图片或 Excel，目标图支持图片或视频。</p>
          </div>
          <div class="actions">
            <button class="primary" :disabled="loading || !queryFile || !targetFile" @click="onSubmit">
              {{ loading ? "检测中..." : "开始检测" }}
            </button>
            <button class="ghost" :disabled="!loading || !taskId" @click="onCancel">
              取消任务
            </button>
          </div>
        </div>
        <div class="name-row">
          <label>任务命名</label>
          <input v-model="taskName" type="text" placeholder="默认使用文件名+时间" />
        </div>
        <div class="option-row">
          <label>文字匹配</label>
          <div class="option-control">
            <label class="switch">
              <input type="checkbox" v-model="useOcr" :disabled="!ocrAvailable" />
              <span></span>
            </label>
            <span class="option-hint">启用 OCR（仅 Excel）</span>
          </div>
        </div>
        <div v-if="loading" class="progress">
          <div class="progress-bar" :style="{ width: `${Math.round(progress * 100)}%` }"></div>
          <div class="progress-meta">
            <span>{{ Math.round(progress * 100) }}%</span>
            <span>{{ stage || "处理中" }}</span>
          </div>
        </div>
        <div v-if="ocrProgress !== null" class="progress ocr-progress">
          <div
            class="progress-bar"
            :style="{ width: `${Math.round((ocrProgress || 0) * 100)}%` }"
          ></div>
          <div class="progress-meta">
            <span>OCR {{ Math.round((ocrProgress || 0) * 100) }}%</span>
            <span>{{ ocrStage || "OCR处理中" }}</span>
          </div>
          <div v-if="ocrMessage" class="progress-message">{{ ocrMessage }}</div>
        </div>
        <div class="preview-grid">
          <div>
            <p>查询图（待核验）</p>
            <input
              ref="queryInput"
              type="file"
              accept="image/*,.xlsx,.xlsm"
              class="file-input"
              @change="onQueryChange"
            />
            <div class="frame clickable" @click="triggerQuery">
              <img v-if="queryPreview" :src="queryPreview" alt="query preview" />
              <div v-else-if="queryIsExcel" class="excel-placeholder">
                <strong>Excel 已上传</strong>
                <span>{{ queryFileName || "查询表格" }}</span>
              </div>
              <span v-else>点击上传</span>
            </div>
          </div>
          <div>
            <p>目标图/视频（被检素材）</p>
            <input
              ref="targetInput"
              type="file"
              accept="image/*,video/*"
              class="file-input"
              @change="onTargetChange"
            />
            <div class="frame clickable" @click="triggerTarget">
              <template v-if="targetPreview && !isTargetVideo">
                <img :src="targetPreview" alt="target preview" />
              </template>
              <template v-else-if="targetVideoUrl && isTargetVideo">
                <video
                  ref="targetVideoRef"
                  :src="targetVideoUrl"
                  controls
                  @loadedmetadata="onTargetVideoMetadata"
                />
              </template>
              <span v-else>点击上传</span>
            </div>
            <div v-if="isTargetVideo" class="range-panel">
              <div class="range-row">
                <label>处理范围</label>
                <div class="range-value">
                  <div class="time-field">
                    <input
                      v-model.number="startMinutes"
                      type="number"
                      min="0"
                      step="1"
                      class="time-input minutes"
                      @change="onTimePartsChange('start')"
                    />
                    <span class="time-sep">:</span>
                    <input
                      v-model.number="startSecondsPart"
                      type="number"
                      min="0"
                      max="59"
                      step="1"
                      class="time-input seconds"
                      @change="onTimePartsChange('start')"
                    />
                  </div>
                  <span class="range-sep">-</span>
                  <div class="time-field">
                    <input
                      v-model.number="endMinutes"
                      type="number"
                      min="0"
                      step="1"
                      class="time-input minutes"
                      @change="onTimePartsChange('end')"
                    />
                    <span class="time-sep">:</span>
                    <input
                      v-model.number="endSecondsPart"
                      type="number"
                      min="0"
                      max="59"
                      step="1"
                      class="time-input seconds"
                      @change="onTimePartsChange('end')"
                    />
                  </div>
                </div>
              </div>
              <div class="range-slider">
                <input
                  type="range"
                  :min="0"
                  :max="maxVideoSeconds"
                  step="1"
                  v-model.number="videoStartSeconds"
                  @input="onVideoStartInput"
                />
                <input
                  type="range"
                  :min="0"
                  :max="maxVideoSeconds"
                  step="1"
                  v-model.number="videoEndSeconds"
                  @input="onVideoEndInput"
                />
              </div>
              <div class="range-hint">
                <span>默认全片</span>
                <span v-if="targetVideoDuration">视频时长 {{ formatTime(targetVideoDuration) }}</span>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section class="card result">
        <h2>检测结果</h2>
        <div class="threshold-panel">
          <div class="threshold-row">
            <label>匹配阈值</label>
            <div class="threshold-value">
              <input
                v-model.number="matchThreshold"
                type="number"
                min="0"
                max="100"
                step="0.01"
                class="threshold-input"
              />
              <span class="threshold-unit">%</span>
            </div>
          </div>
          <input
            type="range"
            min="0"
            max="100"
            step="0.1"
            v-model.number="matchThreshold"
          />
          <div class="threshold-row">
            <label>Embedding 阈值</label>
            <div class="threshold-value">
              <input
                v-model.number="embeddingThreshold"
                type="number"
                min="0"
                max="100"
                step="0.01"
                class="threshold-input"
              />
              <span class="threshold-unit">%</span>
            </div>
          </div>
          <input
            type="range"
            min="0"
            max="100"
            step="0.1"
            v-model.number="embeddingThreshold"
          />
          <div class="threshold-hint">调整 embedding 阈值后需重新提交任务</div>
        </div>
        <div class="history-panel">
          <div class="history-header">
            <span>历史记录</span>
            <div class="history-header-actions">
              <button class="ghost small" @click="loadHistory">刷新</button>
              <button class="ghost small" @click="historyExpanded = !historyExpanded">
                {{ historyExpanded ? "收起" : "展开" }}
              </button>
            </div>
          </div>
          <div v-if="historyItems.length === 0" class="history-empty">暂无记录</div>
          <div v-else>
            <div class="history-compact">
              <div class="history-select">
                <select v-model="selectedHistoryId" @change="handleHistorySelect">
                  <option value="">选择历史记录</option>
                  <option v-for="item in historyItems" :key="item.task_id" :value="item.task_id">
                    {{ item.name }} · {{ formatDate(item.created_at) }}
                  </option>
                </select>
              </div>
              <div class="history-compact-actions">
                <button
                  class="ghost tiny danger"
                  :disabled="!selectedHistoryId"
                  @click="deleteHistory(selectedHistoryId)"
                >
                  删除
                </button>
              </div>
            </div>
            <div v-if="historyExpanded" class="history-list">
              <div
                v-for="item in historyItems"
                :key="item.task_id"
                class="history-item"
                :class="item.task_id === taskId ? 'active' : ''"
                @click="selectHistory(item.task_id)"
              >
                <div>
                  <div class="history-title">{{ item.name }}</div>
                  <div class="history-meta">
                    <span>{{ formatDate(item.created_at) }}</span>
                    <span>{{ item.media_type === "video" ? "视频" : "图片" }}</span>
                  </div>
                </div>
                <button class="ghost tiny" @click.stop="deleteHistory(item.task_id)">删除</button>
              </div>
            </div>
          </div>
        </div>
        <div v-if="error" class="notice error">{{ error }}</div>
        <div v-else-if="!rawResult" class="notice">等待检测结果</div>
        <div v-else class="result-body">
          <div class="badge" :class="overallStatus.badge">
            {{ overallStatus.label }}
          </div>

          <div v-if="normalizedResults.length" class="subcard result-group">
            <div class="result-group-header">
              <h3>匹配结果</h3>
              <div class="result-group-meta">
                <span class="result-time">
                  耗时 {{ formatDuration(matchDurationSeconds) }}
                </span>
                <span class="group-status" :class="overallStatus.group">
                  {{ overallStatus.label }}
                </span>
              </div>
            </div>
            <div v-if="flatMatches.length" class="result-scroll">
              <div class="result-grid">
                <div v-for="item in flatMatches" :key="item.key" class="result-item">
                <div class="result-cells">
                  <div class="result-cell">
                    <span class="result-label">查询图</span>
                    <div v-if="item.query_text" class="result-thumb text">
                      {{ item.query_text }}
                    </div>
                    <a
                      v-else-if="item.query_preview"
                      :href="item.query_preview"
                      :download="downloadName('query', item.query, item)"
                      class="result-thumb link"
                    >
                      <img :src="item.query_preview" alt="query preview" />
                    </a>
                    <div v-else class="result-thumb placeholder">无预览</div>
                  </div>
                  <div class="result-cell">
                    <span class="result-label">匹配内容</span>
                    <div v-if="item.match_text" class="result-thumb text">
                      {{ item.match_text }}
                    </div>
                    <a
                      v-else-if="item.preview"
                      :href="item.preview"
                      :download="downloadName('match', item.query, item)"
                      class="result-thumb link"
                      :class="{ overlay: hasOverlay(item) }"
                    >
                      <img :src="item.preview" :alt="item.label" />
                      <svg
                        v-if="hasOverlay(item)"
                        class="overlay-boxes"
                        :viewBox="`0 0 ${item.overlay_width} ${item.overlay_height}`"
                      >
                        <rect
                          v-for="(box, idx) in item.overlay_boxes"
                          :key="`box-${item.key}-${idx}`"
                          :x="box[0]"
                          :y="box[1]"
                          :width="box[2] - box[0]"
                          :height="box[3] - box[1]"
                        />
                      </svg>
                    </a>
                    <div v-else class="result-thumb placeholder">无预览</div>
                  </div>
                  <div class="result-cell">
                    <span class="result-label">所在帧</span>
                    <a
                      v-if="item.full_preview"
                      :href="item.full_preview"
                      :download="downloadName('frame', item.query, item)"
                      class="result-thumb link frame-thumb"
                      :class="{ overlay: hasOverlay(item) }"
                    >
                      <img :src="item.full_preview" :alt="item.label" />
                      <svg
                        v-if="hasOverlay(item)"
                        class="overlay-boxes"
                        :viewBox="`0 0 ${item.overlay_width} ${item.overlay_height}`"
                      >
                        <rect
                          v-for="(box, idx) in item.overlay_boxes"
                          :key="`box-full-${item.key}-${idx}`"
                          :x="box[0]"
                          :y="box[1]"
                          :width="box[2] - box[0]"
                          :height="box[3] - box[1]"
                        />
                      </svg>
                    </a>
                    <div v-else class="result-thumb placeholder">无预览</div>
                  </div>
                </div>
                <div class="result-meta">
                  <span>{{ item.query_label }}</span>
                  <span>匹配度 {{ formatScore(item.score) }}</span>
                </div>
                <div v-if="item.start_time !== undefined" class="result-meta">
                  <span>
                    {{ formatTime(item.start_time) }} - {{ formatTime(item.end_time) }}
                  </span>
                  <span>首帧 {{ item.first_frame_index }}</span>
                  <span>
                    {{ item.media_type === "ocr" ? "OCR" : item.kind === "frame" ? "帧" : "片段" }}
                    {{ item.id }}
                  </span>
                </div>
              </div>
              </div>
            </div>
            <div v-else class="notice">{{ loading ? "检测中" : "暂无匹配" }}</div>
          </div>

          <div v-if="taskId && rawResult" class="download">
            <a class="primary" :href="downloadUrl" target="_blank" rel="noopener">下载结果</a>
          </div>
        </div>
      </section>
    </main>
  </div>
</template>

<script setup>
import { ref, computed, onBeforeUnmount, onMounted, watch } from "vue";

const resolveApiBase = () => {
  const envBase = import.meta.env.VITE_API_BASE;
  if (envBase) {
    return envBase;
  }
  return "http://127.0.0.1:8003";
};

const apiBase = resolveApiBase();

const queryFile = ref(null);
const targetFile = ref(null);
const queryPreview = ref("");
const queryFileName = ref("");
const queryIsExcel = ref(false);
const targetPreview = ref("");
const targetVideoUrl = ref("");
const targetVideoDuration = ref(null);
const targetVideoRef = ref(null);
const videoStartSeconds = ref(0);
const videoEndSeconds = ref(0);
const startMinutes = ref(0);
const startSecondsPart = ref(0);
const endMinutes = ref(0);
const endSecondsPart = ref(0);
const matchThreshold = ref(95.0);
const embeddingThreshold = ref(50.0);
const useOcr = ref(false);
const loading = ref(false);
const rawResult = ref(null);
const error = ref("");
const queryInput = ref(null);
const targetInput = ref(null);
const progress = ref(0);
const stage = ref("");
const ocrProgress = ref(null);
const ocrStage = ref("");
const ocrMessage = ref("");
const taskId = ref("");
const taskName = ref("");
const historyItems = ref([]);
const historyExpanded = ref(false);
const selectedHistoryId = ref("");
const matchStartMs = ref(null);
const matchEndMs = ref(null);
const elapsedTicker = ref(0);
const targetImage = ref(null);
const targetImageSize = ref(null);
const useLocalPreviews = ref(false);
let elapsedTimer = null;
let eventSource = null;
let pollTimer = null;
let pollInFlight = false;
let pollingActive = false;
let lastEventId = "";
const previewCache = new Map();
const fullPreviewCache = new Map();

const PREVIEW_MAX = 256;
const FULL_PREVIEW_MAX = 640;

function resetPreviewCache() {
  previewCache.clear();
  fullPreviewCache.clear();
}

function loadImageFromDataUrl(dataUrl) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = (err) => reject(err);
    img.src = dataUrl;
  });
}

function clampBbox(bbox, width, height) {
  if (!Array.isArray(bbox) || bbox.length !== 4) return null;
  let [x0, y0, x1, y1] = bbox.map((v) => Number(v));
  if (![x0, y0, x1, y1].every(Number.isFinite)) return null;
  x0 = Math.max(0, Math.min(width - 1, Math.floor(x0)));
  y0 = Math.max(0, Math.min(height - 1, Math.floor(y0)));
  x1 = Math.max(0, Math.min(width, Math.ceil(x1)));
  y1 = Math.max(0, Math.min(height, Math.ceil(y1)));
  if (x1 < x0) [x0, x1] = [x1, x0];
  if (y1 < y0) [y0, y1] = [y1, y0];
  const w = Math.max(1, x1 - x0);
  const h = Math.max(1, y1 - y0);
  return { x0, y0, x1, y1, w, h };
}

function drawCroppedPreview(bbox) {
  if (!targetImage.value || !targetImageSize.value) return "";
  const key = `crop:${bbox?.join?.(",") || ""}`;
  if (previewCache.has(key)) return previewCache.get(key);
  const { width, height } = targetImageSize.value;
  const rect = clampBbox(bbox, width, height);
  if (!rect) return "";
  const scale = Math.min(1, PREVIEW_MAX / Math.max(rect.w, rect.h));
  const outW = Math.max(1, Math.round(rect.w * scale));
  const outH = Math.max(1, Math.round(rect.h * scale));
  const canvas = document.createElement("canvas");
  canvas.width = outW;
  canvas.height = outH;
  const ctx = canvas.getContext("2d");
  if (!ctx) return "";
  ctx.drawImage(
    targetImage.value,
    rect.x0,
    rect.y0,
    rect.w,
    rect.h,
    0,
    0,
    outW,
    outH
  );
  const dataUrl = canvas.toDataURL("image/png");
  previewCache.set(key, dataUrl);
  return dataUrl;
}

function drawFullPreview(bbox) {
  if (!targetImage.value || !targetImageSize.value) return targetPreview.value || "";
  const key = `full:${bbox?.join?.(",") || ""}`;
  if (fullPreviewCache.has(key)) return fullPreviewCache.get(key);
  const { width, height } = targetImageSize.value;
  const scale = Math.min(1, FULL_PREVIEW_MAX / Math.max(width, height));
  const outW = Math.max(1, Math.round(width * scale));
  const outH = Math.max(1, Math.round(height * scale));
  const canvas = document.createElement("canvas");
  canvas.width = outW;
  canvas.height = outH;
  const ctx = canvas.getContext("2d");
  if (!ctx) return targetPreview.value || "";
  ctx.drawImage(targetImage.value, 0, 0, outW, outH);
  const rect = clampBbox(bbox, width, height);
  if (rect) {
    const x0 = rect.x0 * scale;
    const y0 = rect.y0 * scale;
    const w = rect.w * scale;
    const h = rect.h * scale;
    ctx.strokeStyle = "#ff0000";
    ctx.lineWidth = Math.max(2, Math.round(6 * scale));
    ctx.strokeRect(x0, y0, w, h);
  }
  const dataUrl = canvas.toDataURL("image/png");
  fullPreviewCache.set(key, dataUrl);
  return dataUrl;
}

const normalizedResults = computed(() => {
  if (!rawResult.value) return [];
  const fallbackQueryPreview =
    useLocalPreviews.value && !queryIsExcel.value ? (queryPreview.value || "") : "";
  if (Array.isArray(rawResult.value.query_results)) {
    return rawResult.value.query_results.map((item, index) => ({
      query_id: item.query_id ?? index + 1,
      query_label: item.query_label || `查询 ${index + 1}`,
      query_preview: item.query_preview || item.result?.query_preview || fallbackQueryPreview,
      media_type: item.result?.media_type,
      is_match: item.result?.is_match,
      candidate_results: item.result?.candidate_results || [],
      segments: item.result?.segments || [],
    }));
  }
  return [
    {
      query_id: 1,
      query_label: rawResult.value.query_label || "查询图",
      query_preview: rawResult.value.query_preview || fallbackQueryPreview,
      media_type: rawResult.value.media_type,
      is_match: rawResult.value.is_match,
      candidate_results: rawResult.value.candidate_results || [],
      segments: rawResult.value.segments || [],
    },
  ];
});

const overallStatus = computed(() => {
  if (!rawResult.value) {
    return { label: "", badge: "", group: "" };
  }
  if (rawResult.value.is_match) {
    return { label: "疑似匹配", badge: "success", group: "hit" };
  }
  if (loading.value) {
    return { label: "检测中", badge: "pending", group: "pending" };
  }
  return { label: "未匹配", badge: "fail", group: "miss" };
});

const matchDurationSeconds = computed(() => {
  if (!matchStartMs.value) return null;
  const endMs = matchEndMs.value || elapsedTicker.value || Date.now();
  const seconds = Math.max(0, (endMs - matchStartMs.value) / 1000);
  return seconds;
});

const isTargetVideo = computed(() => {
  return targetFile.value?.type?.startsWith("video/") || false;
});

const ocrAvailable = computed(() => {
  return queryIsExcel.value;
});

const ocrResult = computed(() => {
  if (!rawResult.value) return null;
  return rawResult.value.ocr || null;
});

const ocrMatchItems = computed(() => {
  const result = ocrResult.value;
  if (!result || !result.is_match) return [];
  const items = [];
  if (result.video) {
    const keywordEntries = result.video.keywords || [];
    keywordEntries.forEach((entry, entryIndex) => {
      const keyword = entry.keyword || `关键词${entryIndex + 1}`;
      const segments = entry.segments || [];
      segments.forEach((seg, segIndex) => {
        const positions = seg.first_frame_positions || [];
        const matchedLines = positions.filter((line) =>
          lineMatchesKeyword(line?.text, keyword)
        );
        const lineTexts = matchedLines.map((line) => line?.text).filter(Boolean);
        const matchTextRaw = lineTexts.join(" / ");
        const matchText =
          matchTextRaw.length > 160 ? `${matchTextRaw.slice(0, 160)}...` : matchTextRaw;
        const boxesSource = matchedLines.length ? matchedLines : positions;
        const boxes = boxesSource
          .map((line) => line?.bbox)
          .filter((box) => Array.isArray(box) && box.length === 4);
        items.push({
          key: `ocr-video-${entryIndex}-${segIndex}`,
          media_type: "ocr",
          query_text: keyword,
          query_label: keyword,
          match_text: matchText || keyword,
          preview: seg.first_frame_preview || "",
          full_preview: seg.first_frame_preview || "",
          overlay_boxes: boxes,
          overlay_width: seg.first_frame_width,
          overlay_height: seg.first_frame_height,
          score: 1,
          start_time: seg.start_time,
          end_time: seg.end_time,
          first_frame_index: seg.start_frame,
          id: segIndex + 1,
          kind: "segment",
        });
      });
    });
    return items;
  }

  const positions = result.positions || {};
  const lines = positions.lines || [];
  const preview =
    positions.preview ||
    (useLocalPreviews.value ? targetPreview.value || "" : "");
  const width = positions.width ?? targetImageSize.value?.width;
  const height = positions.height ?? targetImageSize.value?.height;
  const matched = result.matches || [];
  matched.forEach((keyword, idx) => {
    const matchedLines = lines.filter((line) => lineMatchesKeyword(line?.text, keyword));
    const lineTexts = matchedLines.map((line) => line?.text).filter(Boolean);
    const matchTextRaw = lineTexts.join(" / ");
    const matchText =
      matchTextRaw.length > 160 ? `${matchTextRaw.slice(0, 160)}...` : matchTextRaw;
    const boxesSource = matchedLines.length ? matchedLines : lines;
    const boxes = boxesSource
      .map((line) => line?.bbox)
      .filter((box) => Array.isArray(box) && box.length === 4);
    items.push({
      key: `ocr-image-${idx}`,
      media_type: "ocr",
      query_text: keyword,
      query_label: keyword,
      match_text: matchText || keyword,
      preview,
      full_preview: preview,
      overlay_boxes: boxes,
      overlay_width: width,
      overlay_height: height,
      score: 1,
      id: idx + 1,
      kind: "image",
    });
  });
  return items;
});

const flatMatches = computed(() => {
  const output = [];
  normalizedResults.value.forEach((query) => {
    if (query.media_type === "image") {
      sortByScore(query.candidate_results || []).forEach((item) => {
        const canUseLocal = useLocalPreviews.value && targetImage.value && targetImageSize.value;
        const preview = item.preview || (canUseLocal ? drawCroppedPreview(item.bbox) : "");
        const fullPreview = item.full_preview || (canUseLocal ? drawFullPreview(item.bbox) : "");
        output.push({
          ...item,
          key: `image-${query.query_id}-${item.id}`,
          media_type: "image",
          query,
          query_label: query.query_label,
          query_preview: query.query_preview,
          preview,
          full_preview: fullPreview,
        });
      });
    } else if (query.media_type === "video") {
      sortByScore(query.segments || []).forEach((item) => {
        output.push({
          ...item,
          key: `video-${query.query_id}-${item.id}`,
          media_type: "video",
          query,
          query_label: query.query_label,
          query_preview: query.query_preview,
        });
      });
    }
  });
  if (ocrMatchItems.value.length) {
    output.push(...ocrMatchItems.value);
  }
  return output;
});

const downloadUrl = computed(() => {
  if (!taskId.value) return "";
  const threshold = (matchThreshold.value / 100).toFixed(4);
  return `${apiBase}/v1/tasks/${taskId.value}/download?threshold=${threshold}`;
});

function fileToDataUrl(file) {
  return new Promise((resolve) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.readAsDataURL(file);
  });
}

function isExcelFile(file) {
  if (!file) return false;
  const name = file.name?.toLowerCase() || "";
  if (name.endsWith(".xlsx") || name.endsWith(".xlsm")) return true;
  return (
    file.type.includes("spreadsheetml") ||
    file.type.includes("ms-excel")
  );
}

async function onQueryChange(event) {
  const file = event.target.files[0];
  if (!file) return;
  queryFile.value = file;
  queryFileName.value = file.name || "";
  queryIsExcel.value = isExcelFile(file);
  if (queryIsExcel.value) {
    queryPreview.value = "";
  } else {
    queryPreview.value = await fileToDataUrl(file);
  }
}

async function onTargetChange(event) {
  const file = event.target.files[0];
  if (!file) return;
  if (targetVideoUrl.value) {
    URL.revokeObjectURL(targetVideoUrl.value);
  }
  targetVideoDuration.value = null;
  videoStartSeconds.value = 0;
  videoEndSeconds.value = 0;
  startMinutes.value = 0;
  startSecondsPart.value = 0;
  endMinutes.value = 0;
  endSecondsPart.value = 0;
  const stamp = new Date().toISOString().replace(/[-:T]/g, "").slice(0, 14);
  taskName.value = `${file.name}-${stamp}`;
  targetFile.value = file;
  if (file.type.startsWith("video/")) {
    targetPreview.value = "";
    targetVideoUrl.value = URL.createObjectURL(file);
    targetImage.value = null;
    targetImageSize.value = null;
    resetPreviewCache();
  } else {
    targetVideoUrl.value = "";
    targetPreview.value = await fileToDataUrl(file);
    try {
      const img = await loadImageFromDataUrl(targetPreview.value);
      targetImage.value = img;
      targetImageSize.value = { width: img.naturalWidth, height: img.naturalHeight };
      resetPreviewCache();
    } catch {
      targetImage.value = null;
      targetImageSize.value = null;
      resetPreviewCache();
    }
  }
}

function triggerQuery() {
  queryInput.value?.click();
}

function triggerTarget() {
  targetInput.value?.click();
}

function currentThresholdParam() {
  return (matchThreshold.value / 100).toFixed(4);
}

function openEventSource(reset = false) {
  if (!taskId.value) return;
  if (eventSource) {
    eventSource.close();
    eventSource = null;
  }
  if (reset) {
    lastEventId = "";
  }
  const params = new URLSearchParams();
  params.set("threshold", currentThresholdParam());
  if (lastEventId) {
    params.set("last_event_id", lastEventId);
  }
  eventSource = new EventSource(`${apiBase}/v1/tasks/${taskId.value}/events?${params.toString()}`);
  eventSource.addEventListener("progress", (event) => {
    if (event.lastEventId) {
      lastEventId = event.lastEventId;
    }
    const data = JSON.parse(event.data);
    progress.value = data.progress;
    stage.value = data.stage;
  });
  eventSource.addEventListener("ocr_progress", (event) => {
    if (event.lastEventId) {
      lastEventId = event.lastEventId;
    }
    const data = JSON.parse(event.data);
    ocrProgress.value = typeof data.progress === "number" ? data.progress : ocrProgress.value;
    ocrStage.value = data.stage || "";
    ocrMessage.value = data.message || "";
  });
  eventSource.addEventListener("partial", (event) => {
    if (event.lastEventId) {
      lastEventId = event.lastEventId;
    }
    const data = JSON.parse(event.data);
    rawResult.value = data.result;
  });
  eventSource.addEventListener("result", (event) => {
    if (event.lastEventId) {
      lastEventId = event.lastEventId;
    }
    const data = JSON.parse(event.data);
    rawResult.value = data.result;
  });
  eventSource.addEventListener("canceled", () => {
    error.value = "已取消";
    loading.value = false;
    if (matchEndMs.value === null && matchStartMs.value !== null) {
      matchEndMs.value = Date.now();
    }
    eventSource?.close();
    stopPolling();
  });
  eventSource.addEventListener("error", (event) => {
    if (event?.lastEventId) {
      lastEventId = event.lastEventId;
    }
    if (event?.data) {
      try {
        const data = JSON.parse(event.data);
        error.value = data.message || "处理失败";
      } catch {
        error.value = "处理失败";
      }
      loading.value = false;
      if (matchEndMs.value === null && matchStartMs.value !== null) {
        matchEndMs.value = Date.now();
      }
      eventSource?.close();
      stopPolling();
      return;
    }
    eventSource?.close();
    startPolling();
  });
  eventSource.addEventListener("done", () => {
    loading.value = false;
    if (matchEndMs.value === null && matchStartMs.value !== null) {
      matchEndMs.value = Date.now();
    }
    eventSource?.close();
    stopPolling();
    loadHistory();
  });
}

async function refreshResultWithThreshold() {
  if (!taskId.value) return;
  try {
    const response = await fetch(
      `${apiBase}/v1/tasks/${taskId.value}?threshold=${currentThresholdParam()}`
    );
    if (!response.ok) {
      throw new Error("获取结果失败");
    }
    const data = await response.json();
    rawResult.value = data.result || null;
  } catch (err) {
    error.value = err.message || "获取结果失败";
  }
}

async function loadHistory() {
  try {
    const response = await fetch(`${apiBase}/v1/history`);
    if (!response.ok) {
      throw new Error("获取历史失败");
    }
    const data = await response.json();
    historyItems.value = data.items || [];
  } catch (err) {
    error.value = err.message || "获取历史失败";
  }
}

function handleHistorySelect() {
  if (!selectedHistoryId.value) return;
  selectHistory(selectedHistoryId.value);
}

async function selectHistory(id) {
  try {
    stopPolling();
    if (eventSource) {
      eventSource.close();
      eventSource = null;
    }
    useLocalPreviews.value = false;
    matchStartMs.value = null;
    matchEndMs.value = null;
    const response = await fetch(
      `${apiBase}/v1/tasks/${id}?threshold=${currentThresholdParam()}`
    );
    if (!response.ok) {
      throw new Error("获取历史详情失败");
    }
    const data = await response.json();
    taskId.value = data.task_id;
    selectedHistoryId.value = data.task_id;
    rawResult.value = data.result || null;
    ocrProgress.value = typeof data.ocr_progress === "number" ? data.ocr_progress : null;
    ocrStage.value = data.ocr_stage || "";
    ocrMessage.value = data.ocr_message || "";
    loading.value = false;
    error.value = "";
    stage.value = "";
  } catch (err) {
    error.value = err.message || "获取历史详情失败";
  }
}

async function deleteHistory(id) {
  try {
    if (!id) return;
    const response = await fetch(`${apiBase}/v1/history/${id}`, { method: "DELETE" });
    if (!response.ok) {
      throw new Error("删除失败");
    }
    if (taskId.value === id) {
      taskId.value = "";
      rawResult.value = null;
      matchStartMs.value = null;
      matchEndMs.value = null;
    }
    if (selectedHistoryId.value === id) {
      selectedHistoryId.value = "";
    }
    await loadHistory();
  } catch (err) {
    error.value = err.message || "删除失败";
  }
}

async function onSubmit() {
  if (!queryFile.value || !targetFile.value) return;
  loading.value = true;
  error.value = "";
  rawResult.value = null;
  progress.value = 0;
  stage.value = "";
  ocrProgress.value = useOcr.value ? 0 : null;
  ocrStage.value = "";
  ocrMessage.value = "";
  taskId.value = "";
  useLocalPreviews.value = true;
  matchStartMs.value = Date.now();
  matchEndMs.value = null;
  lastEventId = "";
  stopPolling();
  if (eventSource) {
    eventSource.close();
    eventSource = null;
  }

  try {
    const formData = new FormData();
    formData.append("query_image", queryFile.value);
    formData.append("target_file", targetFile.value);
    formData.append("match_threshold", (matchThreshold.value / 100).toFixed(4));
    formData.append("embedding_threshold", (embeddingThreshold.value / 100).toFixed(4));
    formData.append("use_ocr", useOcr.value && ocrAvailable.value ? "true" : "false");
    if (taskName.value) {
      formData.append("task_name", taskName.value);
    }
    if (isTargetVideo.value) {
      formData.append("video_start", Math.round(videoStartSeconds.value).toString());
      formData.append("video_end", Math.round(videoEndSeconds.value).toString());
    }

    const response = await fetch(`${apiBase}/v1/tasks`, {
      method: "POST",
      body: formData
    });

    if (!response.ok) {
      const payload = await response.json().catch(() => ({}));
      throw new Error(payload.detail || "请求失败");
    }

    const payload = await response.json();
    taskId.value = payload.task_id;
    if ("EventSource" in window) {
      openEventSource(true);
    } else {
      startPolling();
    }
  } catch (err) {
    error.value = err.message || "请求失败";
    loading.value = false;
    if (matchEndMs.value === null && matchStartMs.value !== null) {
      matchEndMs.value = Date.now();
    }
  } finally {
    if (!eventSource) {
      loading.value = false;
    }
  }
}

async function pollOnce() {
  if (!pollingActive || !taskId.value || pollInFlight) return;
  pollInFlight = true;
  try {
    const response = await fetch(
      `${apiBase}/v1/tasks/${taskId.value}?threshold=${currentThresholdParam()}`
    );
    if (!response.ok) {
      throw new Error("轮询失败");
    }
    const data = await response.json();
    if (typeof data.progress === "number") {
      progress.value = data.progress;
    }
    if (!stage.value && data.status === "running") {
      stage.value = "处理中";
    }
    if (typeof data.ocr_progress === "number") {
      ocrProgress.value = data.ocr_progress;
    }
    if (data.ocr_stage) {
      ocrStage.value = data.ocr_stage;
    }
    if (data.ocr_message) {
      ocrMessage.value = data.ocr_message;
    }
    if (data.result) {
      rawResult.value = data.result;
    }
    if (data.error) {
      error.value = data.error;
    }
    if (data.status === "done") {
      loading.value = false;
      if (matchEndMs.value === null && matchStartMs.value !== null) {
        matchEndMs.value = Date.now();
      }
      stopPolling();
      loadHistory();
    } else if (data.status === "error") {
      loading.value = false;
      if (matchEndMs.value === null && matchStartMs.value !== null) {
        matchEndMs.value = Date.now();
      }
      stopPolling();
    } else if (data.status === "canceled") {
      loading.value = false;
      error.value = "已取消";
      if (matchEndMs.value === null && matchStartMs.value !== null) {
        matchEndMs.value = Date.now();
      }
      stopPolling();
    }
  } catch (err) {
    error.value = err.message || "轮询失败";
    loading.value = false;
    stopPolling();
  } finally {
    pollInFlight = false;
    if (pollingActive) {
      pollTimer = window.setTimeout(pollOnce, 1500);
    }
  }
}

function startPolling() {
  if (pollingActive) return;
  pollingActive = true;
  pollOnce();
}

function stopPolling() {
  pollingActive = false;
  if (pollTimer) {
    clearTimeout(pollTimer);
    pollTimer = null;
  }
  pollInFlight = false;
}

async function onCancel() {
  if (!taskId.value) return;
  try {
    await fetch(`${apiBase}/v1/tasks/${taskId.value}/cancel`, { method: "POST" });
  } catch {
    // Ignore network errors; UI will update via polling/SSE if possible.
  } finally {
    loading.value = false;
    error.value = "已取消";
    stage.value = "";
    ocrProgress.value = null;
    ocrStage.value = "";
    ocrMessage.value = "";
    stopPolling();
    if (eventSource) {
      eventSource.close();
      eventSource = null;
    }
  }
}

function onTargetVideoMetadata(event) {
  const duration = event?.target?.duration;
  if (typeof duration === "number" && !Number.isNaN(duration)) {
    targetVideoDuration.value = duration;
    videoStartSeconds.value = 0;
    videoEndSeconds.value = Math.round(duration);
  }
}

const maxVideoSeconds = computed(() =>
  targetVideoDuration.value ? Math.round(targetVideoDuration.value) : 0
);

function onVideoStartInput() {
  if (videoStartSeconds.value > videoEndSeconds.value) {
    videoEndSeconds.value = videoStartSeconds.value;
  }
}

function onVideoEndInput() {
  if (videoEndSeconds.value < videoStartSeconds.value) {
    videoStartSeconds.value = videoEndSeconds.value;
  }
}

function onTimePartsChange(target) {
  const maxSeconds = maxVideoSeconds.value || 0;
  const minutes = Math.max(0, target === "start" ? startMinutes.value : endMinutes.value);
  const seconds = Math.min(
    59,
    Math.max(0, Math.round(target === "start" ? startSecondsPart.value : endSecondsPart.value))
  );
  const total = Math.min(maxSeconds, minutes * 60 + seconds);
  if (target === "start") {
    videoStartSeconds.value = Math.min(total, videoEndSeconds.value);
  } else {
    videoEndSeconds.value = Math.max(total, videoStartSeconds.value);
  }
}

watch(videoStartSeconds, (value) => {
  const clamped = Math.max(0, Math.round(value));
  const mins = Math.floor(clamped / 60);
  const secs = Math.max(0, clamped - mins * 60);
  startMinutes.value = mins;
  startSecondsPart.value = Math.round(secs);
});

watch(videoEndSeconds, (value) => {
  const clamped = Math.max(0, Math.round(value));
  const mins = Math.floor(clamped / 60);
  const secs = Math.max(0, clamped - mins * 60);
  endMinutes.value = mins;
  endSecondsPart.value = Math.round(secs);
});

function formatScore(score) {
  if (score === null || score === undefined) return "-";
  return `${(score * 100).toFixed(2)}%`;
}

function formatRatio(ratio) {
  if (ratio === null || ratio === undefined) return "-";
  return `${(ratio * 100).toFixed(2)}%`;
}

function formatTime(seconds) {
  if (seconds === null || seconds === undefined || Number.isNaN(seconds)) return "-";
  const total = Math.max(0, seconds);
  const mins = Math.floor(total / 60);
  const secs = (total % 60).toFixed(2).padStart(5, "0");
  return `${mins}:${secs}`;
}

function formatDate(ts) {
  if (!ts) return "-";
  const date = new Date(ts * 1000);
  return date.toLocaleString();
}

function formatDuration(seconds) {
  if (seconds === null || seconds === undefined || Number.isNaN(seconds)) return "-";
  const total = Math.max(0, seconds);
  const mins = Math.floor(total / 60);
  const secs = (total % 60).toFixed(1).padStart(4, "0");
  return `${mins}:${secs}`;
}

function normalizeText(text) {
  return String(text || "").replace(/\s+/g, "").toLowerCase();
}

function lineMatchesKeyword(text, keyword) {
  const normalizedText = normalizeText(text);
  const normalizedKeyword = normalizeText(keyword);
  if (!normalizedKeyword) return false;
  return normalizedText.includes(normalizedKeyword);
}

function sortByScore(items) {
  return [...items].sort((a, b) => (b.score || 0) - (a.score || 0));
}

function hasOverlay(item) {
  return (
    item &&
    Array.isArray(item.overlay_boxes) &&
    item.overlay_boxes.length > 0 &&
    Number.isFinite(item.overlay_width) &&
    Number.isFinite(item.overlay_height)
  );
}

function downloadName(kind, query, item) {
  const label = query?.query_label || item?.query_label || "query";
  const safeLabel = label.replace(/[^\w.-]+/g, "_") || "query";
  const id = item?.id ? `_${item.id}` : "";
  return `${safeLabel}_${kind}${id}.png`;
}

onBeforeUnmount(() => {
  stopPolling();
  if (elapsedTimer) {
    clearInterval(elapsedTimer);
    elapsedTimer = null;
  }
  if (eventSource) {
    eventSource.close();
    eventSource = null;
  }
  if (targetVideoUrl.value) {
    URL.revokeObjectURL(targetVideoUrl.value);
  }
});

onMounted(() => {
  loadHistory();
});

watch(loading, (isLoading) => {
  if (isLoading) {
    if (!elapsedTimer) {
      elapsedTicker.value = Date.now();
      elapsedTimer = setInterval(() => {
        elapsedTicker.value = Date.now();
      }, 1000);
    }
  } else if (elapsedTimer) {
    clearInterval(elapsedTimer);
    elapsedTimer = null;
  }
});

watch([historyItems, taskId], () => {
  if (taskId.value && historyItems.value.some((item) => item.task_id === taskId.value)) {
    selectedHistoryId.value = taskId.value;
    return;
  }
  if (
    selectedHistoryId.value &&
    historyItems.value.some((item) => item.task_id === selectedHistoryId.value)
  ) {
    return;
  }
  selectedHistoryId.value = "";
});

watch(matchThreshold, () => {
  if (!taskId.value) return;
  refreshResultWithThreshold();
  if (loading.value && "EventSource" in window) {
    openEventSource(true);
  }
});

watch(ocrAvailable, (available) => {
  if (!available) {
    useOcr.value = false;
    ocrProgress.value = null;
    ocrStage.value = "";
    ocrMessage.value = "";
  }
});
</script>
