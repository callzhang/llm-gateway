"""Shrink oversized inline images on the CPU before a request reaches the engine.

The engine caps an image at max_pixels anyway, but it resizes on the GPU, so a
very large source image takes VRAM for its full size first (1.44 GiB seen on the
A100, 2026-11), and the engine runs out of memory and exits.  Shrinking here
gives the engine the size it would have used.  Only data: URLs are handled; an
image given by http URL passes through unchanged.
"""
from __future__ import annotations

import base64
import binascii
import io
import math

from PIL import Image

_IMAGE_PART_TYPES = ("image_url", "input_image")


class ImageTooLarge(ValueError):
    """An image Pillow refuses to decode (over ~179 million pixels)."""


def _shrunk_data_url(url: str, max_pixels: int) -> "str | None":
    """A smaller data: URL, or None to leave `url` as it is (already small
    enough, or not something we can read: the engine reports that itself)."""
    header, _, payload = url.partition(",")
    if not header.startswith("data:") or not header.endswith(";base64"):
        return None
    try:
        with Image.open(io.BytesIO(base64.b64decode(payload))) as image:
            width, height = image.size
            if width * height <= max_pixels:
                return None
            source_format = image.format
            keep = {k: image.info[k] for k in ("exif", "icc_profile") if image.info.get(k)}
            scale = math.sqrt(max_pixels / (width * height))
            size = (max(1, int(width * scale)), max(1, int(height * scale)))
            if source_format == "JPEG":
                image.draft(image.mode, size)   # decode at reduced scale: less host memory
            if image.mode in ("P", "1"):
                image = image.convert("RGBA" if image.mode == "P" else "L")
            small = image.resize(size, Image.Resampling.BICUBIC)
        out = io.BytesIO()
        if source_format == "JPEG":
            small.save(out, "JPEG", quality=95, **keep)
            mime = "image/jpeg"
        else:
            small.save(out, "PNG", compress_level=1, **keep)
            mime = "image/png"
    except Image.DecompressionBombError as exc:
        raise ImageTooLarge(str(exc)) from None
    except (OSError, ValueError, SyntaxError, binascii.Error):
        return None
    return f"data:{mime};base64," + base64.b64encode(out.getvalue()).decode("ascii")


def _image_parts(node):
    if isinstance(node, dict):
        if node.get("type") in _IMAGE_PART_TYPES:
            yield node
            return
        node = list(node.values())
    if isinstance(node, list):
        for child in node:
            yield from _image_parts(child)


def has_images(body: dict) -> bool:
    """Whether a Chat Completions or Responses body carries any image, inline or by URL."""
    return any(_image_parts([body.get("messages"), body.get("input")]))


def unsafe_image_url(body: dict) -> bool:
    """True if an image names anything but data: or http(s):.  SGLang opens a
    "/..." or file:// URL from the server's own disk."""
    for part in _image_parts([body.get("messages"), body.get("input")]):
        image = part.get("image_url")
        url = image.get("url") if isinstance(image, dict) else image
        if url is not None and not (
            isinstance(url, str) and url.lower().startswith(("data:", "http://", "https://"))
        ):
            return True
    return False


def fit_images(body: dict, max_pixels: int) -> int:
    """Shrink, in place, every inline image over max_pixels in a Chat Completions
    (`messages`) or Responses (`input`) body.  Returns how many were replaced."""
    replaced = 0
    for part in _image_parts([body.get("messages"), body.get("input")]):
        image = part.get("image_url")
        holder, key = (image, "url") if isinstance(image, dict) else (part, "image_url")
        url = holder.get(key)
        if isinstance(url, str):
            small = _shrunk_data_url(url, max_pixels)
            if small is not None:
                holder[key] = small
                replaced += 1
    return replaced
