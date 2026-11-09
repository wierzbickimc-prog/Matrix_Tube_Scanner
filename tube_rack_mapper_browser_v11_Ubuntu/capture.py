"""Validate browser stills and identify the exact image/settings behind a result."""
from __future__ import annotations

import base64
import binascii
import hashlib
import io
import json

from PIL import Image

MAX_CAPTURE_BYTES = 20 * 1024 * 1024


def camera_image(value: object) -> bytes | None:
    if value is None:
        return None
    if not isinstance(value, dict) or value.get("event") != "capture":
        raise ValueError("Invalid camera capture. Please retake the photo.")
    data = value.get("data", "")
    prefix = "data:image/jpeg;base64,"
    if not isinstance(data, str) or not data.startswith(prefix):
        raise ValueError("The camera did not return a JPEG. Please retake the photo.")
    encoded = data[len(prefix):]
    if len(encoded) > 4 * ((MAX_CAPTURE_BYTES + 2) // 3):
        raise ValueError("Camera image is too large. Please upload a smaller photo.")
    try:
        image_bytes = base64.b64decode(encoded, validate=True)
        if not image_bytes or len(image_bytes) > MAX_CAPTURE_BYTES:
            raise ValueError("Camera image is empty or too large.")
        with Image.open(io.BytesIO(image_bytes)) as image:
            if image.format != "JPEG":
                raise ValueError("Camera image must be a JPEG.")
            image.verify()
    except (binascii.Error, OSError, Image.DecompressionBombError) as exc:
        raise ValueError("Camera image could not be opened. Please retake the photo.") from exc
    return image_bytes


def source_key(image: bytes | None, *settings: str) -> str | None:
    if image is None:
        return None
    digest = hashlib.sha256(image)
    digest.update(json.dumps(settings, ensure_ascii=True).encode("utf-8"))
    return digest.hexdigest()
