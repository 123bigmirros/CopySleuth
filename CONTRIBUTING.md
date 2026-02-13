# Contributing / 贡献指南

感谢你对 pic_cmp 的关注！欢迎提交 Issue 和 Pull Request。

Thanks for your interest in pic_cmp! Issues and pull requests are welcome.

## 开发环境 / Development Setup

```bash
# 创建 conda 环境 (首次)
conda env create -f envs/pic_cmp.yml

# 激活环境并安装测试依赖
conda activate pic_cmp
pip install -e ".[test]"
pip install ruff
```

> `backend/` 有独立的 `requirements.txt`（包含 torch 等重型依赖），`pyproject.toml` 只声明轻量共享依赖。
> 如需完整后端依赖：`pip install -r backend/requirements.txt`

## 服务说明 / Services

| 服务 | 环境 | 说明 |
|------|------|------|
| algo_service | pic_cmp | 算法服务 (SAM3 + RoMaV2 + DINOv3) |
| backend | pic_cmp | 主后端 (协调算法、OCR、媒体服务) |
| media_service | pic_cmp | 预览图生成 (仅需 fastapi + pillow) |
| paddle-ocr | pic_cmp | OCR 服务 (PaddlePaddle) |
| frontend | — | Vue 3 + Vite 前端 |

## 代码规范 / Code Style

- Python 代码使用 [Ruff](https://docs.astral.sh/ruff/) 检查，配置见 `pyproject.toml`
- 提交前请运行 `ruff check` 和 `pytest tests/`
- 类型标注：公开函数应有完整的类型标注
- 日志：使用 `logging` 模块，不要使用 `print`

## 提交 PR / Pull Request

1. Fork 仓库并创建分支
2. 确保测试通过：`pytest tests/ -v`
3. 确保代码检查通过：`ruff check algo_service/ backend/ tests/`
4. 提交 PR 并描述改动内容

## 报告问题 / Report Issues

请在 Issue 中提供：
- 复现步骤
- 期望行为与实际行为
- 环境信息 (Python 版本、CUDA 版本、操作系统)
