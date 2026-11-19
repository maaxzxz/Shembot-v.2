"""Create a reproducible visual strip of SHEMBOT browser preprocessing."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


PANEL_SIZE = 512
HEADER_HEIGHT = 58


def fit_panel(image: np.ndarray, size: int = PANEL_SIZE) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(size / width, size / height)
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=interpolation)
    panel = np.full((size, size, 3), 28, dtype=np.uint8)
    x = (size - resized_width) // 2
    y = (size - resized_height) // 2
    panel[y : y + resized_height, x : x + resized_width] = resized
    return panel


def add_header(panel: np.ndarray, title: str, subtitle: str) -> np.ndarray:
    output = np.full((panel.shape[0] + HEADER_HEIGHT, panel.shape[1], 3), 255, dtype=np.uint8)
    output[HEADER_HEIGHT:] = panel
    cv2.putText(output, title, (14, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (22, 50, 77), 2, cv2.LINE_AA)
    cv2.putText(output, subtitle, (14, 47), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (91, 101, 115), 1, cv2.LINE_AA)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    image = cv2.imread(str(args.input), cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"Could not read input image: {args.input}")

    height, width = image.shape[:2]
    full_prepared = cv2.resize(image, (PANEL_SIZE, PANEL_SIZE), interpolation=cv2.INTER_AREA)

    # Mirrors one of the four overlapping browser scans: 58% of the selected
    # area, then resized for a closer detector view.
    crop_width = max(2, int(round(width * 0.58)))
    crop_height = max(2, int(round(height * 0.58)))
    zoom_crop = image[:crop_height, :crop_width]
    zoom_prepared = cv2.resize(zoom_crop, (PANEL_SIZE, PANEL_SIZE), interpolation=cv2.INTER_LINEAR)

    panels = [
        add_header(fit_panel(image), "Original validation frame", f"{width} x {height} source image"),
        add_header(full_prepared, "Full-frame inference input", "Downscaled to 512-pixel maximum side"),
        add_header(zoom_prepared, "Overlapping zoom scan", "58% top-left crop resized to 512 x 512"),
    ]
    strip = np.concatenate(panels, axis=1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), strip, [cv2.IMWRITE_JPEG_QUALITY, 92]):
        raise SystemExit(f"Could not write output image: {args.output}")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

