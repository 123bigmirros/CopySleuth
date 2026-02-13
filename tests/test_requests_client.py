"""Unit tests for algo_service.requests_client."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from algo_service.requests_client import (
    _is_excel_path,
    _validate_query_paths,
)


class TestIsExcelPath:
    def test_xlsx(self):
        assert _is_excel_path(Path("data.xlsx")) is True

    def test_xlsm(self):
        assert _is_excel_path(Path("data.XLSM")) is True

    def test_png(self):
        assert _is_excel_path(Path("image.png")) is False


class TestValidateQueryPaths:
    def test_images_pass(self):
        _validate_query_paths([Path("a.png"), Path("b.jpg")])

    def test_excel_raises(self):
        import pytest
        with pytest.raises(RuntimeError, match="Excel"):
            _validate_query_paths([Path("data.xlsx")])


class TestMainCommandRouting:
    """Verify the CLI dispatches to the correct request function."""

    @patch("algo_service.requests_client.request_health")
    def test_health_command(self, mock_health):
        mock_health.return_value = {"status": "ok"}
        from algo_service.requests_client import main
        with patch("sys.argv", ["prog", "health"]):
            main()
        mock_health.assert_called_once()

    @patch("algo_service.requests_client.request_task_status")
    def test_status_command(self, mock_status):
        mock_status.return_value = {"status": "done"}
        from algo_service.requests_client import main
        with patch("sys.argv", ["prog", "status", "--task-id", "abc"]):
            main()
        mock_status.assert_called_once()

    @patch("algo_service.requests_client.request_task_cancel")
    def test_cancel_command(self, mock_cancel):
        mock_cancel.return_value = {"status": "ok"}
        from algo_service.requests_client import main
        with patch("sys.argv", ["prog", "cancel", "--task-id", "abc"]):
            main()
        mock_cancel.assert_called_once()
