from __future__ import annotations

import base64
import io
import re
import urllib.request
from pathlib import Path
from typing import Any

_LOCAL_PATH_RE = re.compile(r"Local path:\s*(?P<path>[^\n]+)")


def decode_base64_image(value: str) -> bytes:
    data = value.strip()
    if data.startswith("data:image") and "," in data:
        data = data.split(",", 1)[1]
    return base64.b64decode(data)


def first_json_image(
    payload: dict[str, Any],
    *,
    opener: Any = urllib.request.urlopen,
) -> bytes | None:
    for key in ("image", "b64_json", "base64", "image_base64"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return decode_base64_image(value)

    images = payload.get("images")
    if isinstance(images, list) and images:
        first = images[0]
        if isinstance(first, str):
            return decode_base64_image(first)
        if isinstance(first, dict):
            nested = first_json_image(first, opener=opener)
            if nested is not None:
                return nested

    data = payload.get("data")
    if isinstance(data, list) and data:
        first = data[0]
        if isinstance(first, dict):
            nested = first_json_image(first, opener=opener)
            if nested is not None:
                return nested
            url = first.get("url")
            if isinstance(url, str) and url:
                return download_url(url, opener=opener)
    return None


def download_url(
    url: str,
    timeout: float = 120.0,
    *,
    opener: Any = urllib.request.urlopen,
) -> bytes:
    with opener(url, timeout=timeout) as response:
        return response.read()


def reference_image_urls(
    references: list[str],
    max_reference_size: int = 2048,
    *,
    resize_error_cls: type[Exception] = RuntimeError,
) -> list[str]:
    image_urls: list[str] = []
    for reference in references:
        if not isinstance(reference, str):
            continue
        candidates = [reference.strip()]
        match = _LOCAL_PATH_RE.search(reference)
        if match:
            candidates.insert(0, match.group("path").strip().rstrip(").,;]"))
        for candidate in candidates:
            path = Path(candidate)
            if not path.is_file():
                continue
            mime = mime_type(path)
            image_bytes = resize_reference_image(
                path.read_bytes(),
                max_reference_size,
                mime,
                error_cls=resize_error_cls,
            )
            encoded = base64.b64encode(image_bytes).decode("utf-8")
            image_urls.append(f"data:{mime};base64,{encoded}")
            break
    return image_urls


def resize_reference_image(
    image_bytes: bytes,
    max_size: int,
    mime: str,
    *,
    error_cls: type[Exception] = RuntimeError,
) -> bytes:
    if max_size <= 0:
        return image_bytes
    try:
        from PIL import Image
    except ImportError as exc:
        raise error_cls(
            "Pillow is required to resize reference images before upload. "
            "Install pillow or set max_reference_size=0 to disable resizing."
        ) from exc

    with Image.open(io.BytesIO(image_bytes)) as image:
        width, height = image.size
        if width <= max_size and height <= max_size:
            return image_bytes
        scale = min(max_size / float(width), max_size / float(height))
        resized_size = (max(1, int(width * scale)), max(1, int(height * scale)))
        resized = image.resize(resized_size, Image.Resampling.LANCZOS)
        output = io.BytesIO()
        if mime == "image/png":
            resized.save(output, format="PNG")
        elif mime == "image/webp":
            resized.save(output, format="WEBP")
        else:
            if resized.mode not in {"RGB", "L"}:
                resized = resized.convert("RGB")
            resized.save(output, format="JPEG", quality=92)
        return output.getvalue()


def mime_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".png":
        return "image/png"
    if suffix == ".webp":
        return "image/webp"
    if suffix == ".gif":
        return "image/gif"
    return "image/jpeg"


def image_media_type(image: bytes) -> str:
    if image.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if image.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if image.startswith(b"GIF87a") or image.startswith(b"GIF89a"):
        return "image/gif"
    if image.startswith(b"RIFF") and image[8:12] == b"WEBP":
        return "image/webp"
    return "image/png"
