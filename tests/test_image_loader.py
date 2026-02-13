"""Unit tests for algo_service.services.image_loader."""

import sys
from pathlib import Path

import pytest
from PIL import Image

# Ensure imports work
ROOT_DIR = Path(__file__).resolve().parents[1]
BACKEND_DIR = ROOT_DIR / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from algo_service.services.image_loader import (
    is_excel_upload,
    load_image_bytes,
)
from tests.conftest import make_test_image


class TestLoadImageBytes:
    def test_valid_png(self):
        data = make_test_image()
        img = load_image_bytes(data, "test.png")
        assert isinstance(img, Image.Image)
        assert img.mode == "RGB"

    def test_invalid_bytes_raises_http_400(self):
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc_info:
            load_image_bytes(b"not-an-image", "bad.png")
        assert exc_info.value.status_code == 400
        assert "Invalid image" in exc_info.value.detail

    def test_none_filename_uses_default(self):
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc_info:
            load_image_bytes(b"garbage", None)
        assert "image" in exc_info.value.detail


class TestIsExcelUpload:
    def test_xlsx_extension(self):
        assert is_excel_upload("data.xlsx", None) is True

    def test_xlsm_extension(self):
        assert is_excel_upload("data.xlsm", None) is True

    def test_png_extension(self):
        assert is_excel_upload("photo.png", None) is False

    def test_spreadsheet_content_type(self):
        assert is_excel_upload(None, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet") is True

    def test_none_inputs(self):
        assert is_excel_upload(None, None) is False
