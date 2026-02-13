"""Image and Excel loading / validation utilities."""

from __future__ import annotations

import hashlib
import io
import logging

from fastapi import HTTPException
from PIL import Image

logger = logging.getLogger(__name__)


def load_image_bytes(payload: bytes, filename: str | None = None) -> Image.Image:
    """Open raw bytes as an RGB PIL image, raising HTTP 400 on failure."""
    try:
        image = Image.open(io.BytesIO(payload)).convert("RGB")
    except Exception as exc:
        name = filename or "image"
        raise HTTPException(status_code=400, detail=f"Invalid image: {name}") from exc
    return image


def is_excel_upload(filename: str | None, content_type: str | None) -> bool:
    """Return True if the upload looks like an Excel file."""
    name = (filename or "").lower()
    content = (content_type or "").lower()
    if name.endswith((".xlsx", ".xlsm")):
        return True
    return "spreadsheetml" in content or "ms-excel" in content


def load_excel_images(
    payload: bytes, filename: str | None = None,
) -> tuple[list[Image.Image], list[dict]]:
    """Extract images from column B of all sheets, skipping header rows.

    Returns ``(images, origins)`` where each origin is
    ``{"sheet": <name>, "sheet_index": <0-based>, "row": <1-based>}``.
    """
    from openpyxl import load_workbook

    try:
        workbook = load_workbook(io.BytesIO(payload), data_only=True)
    except Exception as exc:
        raise HTTPException(
            status_code=400, detail=f"Invalid excel: {filename}"
        ) from exc

    images: list[Image.Image] = []
    origins: list[dict] = []
    for sheet_idx, worksheet in enumerate(workbook.worksheets):
        sheet_images_by_row: dict[int, list[Image.Image]] = {}
        for img in getattr(worksheet, "_images", []):
            anchor = getattr(img, "anchor", None)
            start = getattr(anchor, "_from", None)
            if start is None:
                continue
            col_index = getattr(start, "col", None)
            if col_index is None or col_index != 1:
                continue
            row_index = getattr(start, "row", None)
            if row_index is None or row_index < 0:
                continue
            data_loader = getattr(img, "_data", None)
            if data_loader is None:
                continue
            try:
                image_bytes = data_loader()
                pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                digest = hashlib.sha256(image_bytes).hexdigest()
                pil_image.info["cache_key"] = f"sha256:{digest}"
            except Exception:
                continue
            sheet_images_by_row.setdefault(row_index + 1, []).append(pil_image)

        for row_idx in sorted(sheet_images_by_row):
            if row_idx <= 1:
                continue
            for pil_img in sheet_images_by_row[row_idx]:
                images.append(pil_img)
                origins.append({
                    "sheet": worksheet.title,
                    "sheet_index": sheet_idx,
                    "row": row_idx,
                })

    logger.info(
        "[excel] %s: %d sheet(s), %d image(s) extracted",
        filename, len(workbook.worksheets), len(images),
    )
    if not images:
        raise HTTPException(
            status_code=400,
            detail="Excel contains no images in column B",
        )
    return images, origins
