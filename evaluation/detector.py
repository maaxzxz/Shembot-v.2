"""CPU YOLOX person detector used by the checkpoint prototype."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from preprocess import prepare_yolox_blob, restore_box


@dataclass(frozen=True)
class Detection:
    box: tuple[int, int, int, int]
    score: float


class YoloXPersonDetector:
    """Run YOLOX-S and retain COCO class 0 (`person`) only."""

    def __init__(
        self,
        model_path: Path,
        confidence_threshold: float = 0.35,
        nms_threshold: float = 0.5,
    ) -> None:
        if not model_path.is_file():
            raise FileNotFoundError(f"Model not found: {model_path}")
        self.input_size = (640, 640)
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.net = cv2.dnn.readNet(str(model_path))
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.grids, self.expanded_strides = self._anchors()

    def _anchors(self) -> tuple[np.ndarray, np.ndarray]:
        grids: list[np.ndarray] = []
        strides: list[np.ndarray] = []
        for stride in (8, 16, 32):
            height = self.input_size[1] // stride
            width = self.input_size[0] // stride
            x_values, y_values = np.meshgrid(np.arange(width), np.arange(height))
            grid = np.stack((x_values, y_values), axis=2).reshape(1, -1, 2)
            grids.append(grid)
            strides.append(np.full((*grid.shape[:2], 1), stride))
        return np.concatenate(grids, axis=1), np.concatenate(strides, axis=1)

    def detect(self, image: np.ndarray) -> list[Detection]:
        blob, info = prepare_yolox_blob(image, self.input_size)
        self.net.setInput(blob)
        raw = self.net.forward(self.net.getUnconnectedOutLayersNames())[0][0].copy()
        raw[:, :2] = (raw[:, :2] + self.grids[0]) * self.expanded_strides[0]
        raw[:, 2:4] = np.exp(raw[:, 2:4]) * self.expanded_strides[0]

        # YOLOX confidence is objectness multiplied by the person-class score.
        scores = raw[:, 4] * raw[:, 5]
        mask = scores >= self.confidence_threshold
        if not np.any(mask):
            return []

        candidates = raw[mask, :4]
        scores = scores[mask]
        boxes = np.empty_like(candidates)
        boxes[:, 0] = candidates[:, 0] - candidates[:, 2] / 2.0
        boxes[:, 1] = candidates[:, 1] - candidates[:, 3] / 2.0
        boxes[:, 2:] = candidates[:, 2:]
        keep = cv2.dnn.NMSBoxes(
            boxes.tolist(),
            scores.tolist(),
            self.confidence_threshold,
            self.nms_threshold,
        )
        if len(keep) == 0:
            return []
        return [
            Detection(restore_box(tuple(boxes[index]), info), float(scores[index]))
            for index in np.asarray(keep).reshape(-1)
        ]

