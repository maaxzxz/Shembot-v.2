"""Shared image preprocessing for CPU evaluation and prediction."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class LetterboxInfo:
    scale: float
    pad_x: int
    pad_y: int
    original_width: int
    original_height: int


def letterbox_bgr(
    image: np.ndarray,
    size: tuple[int, int] = (640, 640),
    fill: int = 114,
) -> tuple[np.ndarray, LetterboxInfo]:
    """Resize without distortion and pad to the detector input size."""
    if image is None or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected a non-empty BGR image with three channels")
    target_width, target_height = size
    if target_width <= 0 or target_height <= 0:
        raise ValueError("Target dimensions must be positive")

    original_height, original_width = image.shape[:2]
    scale = min(target_width / original_width, target_height / original_height)
    resized_width = max(1, int(round(original_width * scale)))
    resized_height = max(1, int(round(original_height * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=interpolation)

    pad_x = (target_width - resized_width) // 2
    pad_y = (target_height - resized_height) // 2
    canvas = np.full((target_height, target_width, 3), fill, dtype=np.uint8)
    canvas[pad_y : pad_y + resized_height, pad_x : pad_x + resized_width] = resized
    return canvas, LetterboxInfo(
        scale=scale,
        pad_x=pad_x,
        pad_y=pad_y,
        original_width=original_width,
        original_height=original_height,
    )


def prepare_yolox_blob(
    image: np.ndarray,
    size: tuple[int, int] = (640, 640),
) -> tuple[np.ndarray, LetterboxInfo]:
    """Apply the exact color, resize, padding, layout, and dtype used by YOLOX."""
    letterboxed, info = letterbox_bgr(image, size=size)
    rgb = cv2.cvtColor(letterboxed, cv2.COLOR_BGR2RGB)
    blob = np.transpose(rgb.astype(np.float32), (2, 0, 1))[np.newaxis, :, :, :]
    return blob, info


def restore_box(
    box_xywh: tuple[float, float, float, float],
    info: LetterboxInfo,
) -> tuple[int, int, int, int]:
    """Map a detector-space x/y/width/height box back to source coordinates."""
    x, y, width, height = box_xywh
    x1 = int(round((x - info.pad_x) / info.scale))
    y1 = int(round((y - info.pad_y) / info.scale))
    x2 = int(round((x + width - info.pad_x) / info.scale))
    y2 = int(round((y + height - info.pad_y) / info.scale))
    x1 = min(max(x1, 0), info.original_width - 1)
    y1 = min(max(y1, 0), info.original_height - 1)
    x2 = min(max(x2, x1 + 1), info.original_width)
    y2 = min(max(y2, y1 + 1), info.original_height)
    return x1, y1, x2 - x1, y2 - y1

