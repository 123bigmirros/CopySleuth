from __future__ import annotations

import json
import mimetypes
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib import request as urlrequest

from PIL import Image


@dataclass(frozen=True)
class MediaPreview:
    url: str
    width: int | None = None
    height: int | None = None


class MediaClient:
    def __init__(self, base_url: str | None, timeout_s: float = 10.0) -> None:
        self._base_url = (base_url or "").rstrip("/")
        self._timeout_s = float(timeout_s)

    @property
    def enabled(self) -> bool:
        return bool(self._base_url)

    def create_preview(
        self,
        image: Image.Image | Path | str | None,
        *,
        max_size: int,
        bbox: tuple[int, int, int, int] | None = None,
    ) -> MediaPreview | None:
        if not self.enabled or image is None:
            return None
        file_name, content_type, data = self._read_image_payload(image)
        if data is None:
            return None
        fields: list[tuple[str, str]] = [("max_size", str(int(max_size or 0)))]
        if bbox:
            fields.append(("bbox", ",".join(str(int(v)) for v in bbox)))
        body, content_type_header = _encode_multipart(
            fields,
            files=[("file", file_name, content_type, data)],
        )
        url = f"{self._base_url}/v1/preview"
        req = urlrequest.Request(
            url,
            data=body,
            headers={"Content-Type": content_type_header},
            method="POST",
        )
        try:
            with urlrequest.urlopen(req, timeout=self._timeout_s) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except Exception:
            return None
        url = payload.get("url")
        if not url:
            return None
        return MediaPreview(
            url=url,
            width=payload.get("width"),
            height=payload.get("height"),
        )

    @staticmethod
    def _read_image_payload(
        image: Image.Image | Path | str,
    ) -> tuple[str, str, bytes | None]:
        if isinstance(image, Image.Image):
            buffer = _image_to_bytes(image)
            return f"{uuid.uuid4().hex}.png", "image/png", buffer
        path = Path(image)
        if not path.exists():
            return path.name, "application/octet-stream", None
        content_type, _ = mimetypes.guess_type(path.name)
        content_type = content_type or "application/octet-stream"
        return path.name, content_type, path.read_bytes()


def _image_to_bytes(image: Image.Image) -> bytes:
    from io import BytesIO

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _encode_multipart(
    fields: Iterable[tuple[str, str]],
    *,
    files: Iterable[tuple[str, str, str, bytes]],
) -> tuple[bytes, str]:
    boundary = uuid.uuid4().hex
    body = bytearray()
    for name, value in fields:
        body.extend(f"--{boundary}\r\n".encode("utf-8"))
        body.extend(
            f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("utf-8")
        )
        body.extend(str(value).encode("utf-8"))
        body.extend(b"\r\n")
    for name, filename, content_type, data in files:
        body.extend(f"--{boundary}\r\n".encode("utf-8"))
        body.extend(
            (
                f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'
                f"Content-Type: {content_type}\r\n\r\n"
            ).encode("utf-8")
        )
        body.extend(data)
        body.extend(b"\r\n")
    body.extend(f"--{boundary}--\r\n".encode("utf-8"))
    return bytes(body), f"multipart/form-data; boundary={boundary}"
