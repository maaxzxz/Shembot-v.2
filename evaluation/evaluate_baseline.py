"""Evaluate the included CPU YOLOX person baseline at IoU 0.50."""

from __future__ import annotations

import argparse
import csv
import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2

from detector import Detection, YoloXPersonDetector


EVALUATION_ROOT = Path(__file__).resolve().parent


def parse_annotation(path: Path) -> list[tuple[int, int, int, int]]:
    root = ET.parse(path).getroot()
    boxes: list[tuple[int, int, int, int]] = []
    for item in root.findall("object"):
        if item.findtext("name", "").strip().lower() != "person":
            continue
        box = item.find("bndbox")
        if box is None:
            continue
        x1 = int(round(float(box.findtext("xmin", "0"))))
        y1 = int(round(float(box.findtext("ymin", "0"))))
        x2 = int(round(float(box.findtext("xmax", "0"))))
        y2 = int(round(float(box.findtext("ymax", "0"))))
        if x2 > x1 and y2 > y1:
            boxes.append((x1, y1, x2 - x1, y2 - y1))
    return boxes


def iou(first: tuple[int, int, int, int], second: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = first
    bx, by, bw, bh = second
    left = max(ax, bx)
    top = max(ay, by)
    right = min(ax + aw, bx + bw)
    bottom = min(ay + ah, by + bh)
    intersection = max(0, right - left) * max(0, bottom - top)
    union = aw * ah + bw * bh - intersection
    return intersection / union if union > 0 else 0.0


def match_detections(
    detections: list[Detection],
    ground_truth: list[tuple[int, int, int, int]],
    iou_threshold: float,
) -> tuple[int, int, int]:
    unmatched = set(range(len(ground_truth)))
    true_positives = 0
    for detection in sorted(detections, key=lambda item: item.score, reverse=True):
        candidates = [
            (iou(detection.box, ground_truth[index]), index)
            for index in unmatched
        ]
        best_iou, best_index = max(candidates, default=(0.0, -1))
        if best_iou >= iou_threshold:
            true_positives += 1
            unmatched.remove(best_index)
    false_positives = len(detections) - true_positives
    false_negatives = len(unmatched)
    return true_positives, false_positives, false_negatives


def annotate(
    image,
    detections: list[Detection],
    ground_truth: list[tuple[int, int, int, int]],
):
    output = image.copy()
    for x, y, width, height in ground_truth:
        cv2.rectangle(output, (x, y), (x + width, y + height), (255, 170, 0), 2)
    for detection in detections:
        x, y, width, height = detection.box
        cv2.rectangle(output, (x, y), (x + width, y + height), (59, 211, 143), 2)
    cv2.rectangle(output, (0, 0), (output.shape[1], 56), (0, 0, 0), -1)
    cv2.putText(
        output,
        f"Ground truth: {len(ground_truth)}  Predicted: {len(detections)}",
        (12, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return output


def evaluate(args: argparse.Namespace) -> dict[str, object]:
    image_paths = sorted(args.images.glob("*.jpg"))
    if len(image_paths) < 30:
        raise ValueError(f"Expected at least 30 validation images; found {len(image_paths)}")
    detector = YoloXPersonDetector(args.model, args.confidence, args.nms)
    args.output.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    total_tp = total_fp = total_fn = 0
    total_inference_ms = 0.0
    total_absolute_count_error = 0

    for position, image_path in enumerate(image_paths, start=1):
        annotation_path = args.annotations / f"{image_path.stem}.xml"
        if not annotation_path.is_file():
            raise FileNotFoundError(f"Missing annotation: {annotation_path}")
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        ground_truth = parse_annotation(annotation_path)
        started = time.perf_counter()
        detections = detector.detect(image)
        inference_ms = (time.perf_counter() - started) * 1000.0
        tp, fp, fn = match_detections(detections, ground_truth, args.iou)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_inference_ms += inference_ms
        absolute_count_error = abs(len(detections) - len(ground_truth))
        total_absolute_count_error += absolute_count_error
        rows.append(
            {
                "image": image_path.name,
                "ground_truth": len(ground_truth),
                "predicted": len(detections),
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
                "absolute_count_error": absolute_count_error,
                "inference_ms": round(inference_ms, 3),
            }
        )
        if position == 1:
            cv2.imwrite(str(args.output / "baseline_preview.jpg"), annotate(image, detections, ground_truth))
        print(
            f"[{position:02}/{len(image_paths)}] {image_path.name}: "
            f"gt={len(ground_truth)}, pred={len(detections)}, tp={tp}, fp={fp}, fn={fn}"
        )

    precision = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
    recall = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    mean_ms = total_inference_ms / len(rows)
    summary: dict[str, object] = {
        "detector": "OpenCV Zoo YOLOX-S INT8, COCO person class only",
        "split": "source-grouped validation sample",
        "images_evaluated": len(rows),
        "unique_source_groups": len(rows),
        "confidence_threshold": args.confidence,
        "nms_threshold": args.nms,
        "iou_threshold": args.iou,
        "true_positives": total_tp,
        "false_positives": total_fp,
        "false_negatives": total_fn,
        "precision_at_iou_0_5": round(precision, 6),
        "recall_at_iou_0_5": round(recall, 6),
        "f1_at_iou_0_5": round(f1, 6),
        "count_mae": round(total_absolute_count_error / len(rows), 6),
        "mean_inference_ms": round(mean_ms, 3),
        "mean_fps": round(1000.0 / mean_ms, 3),
    }
    with (args.output / "baseline_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output / "baseline_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print("\n" + json.dumps(summary, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, default=EVALUATION_ROOT / "data" / "images")
    parser.add_argument("--annotations", type=Path, default=EVALUATION_ROOT / "data" / "annotations")
    parser.add_argument("--model", type=Path, default=EVALUATION_ROOT / "models" / "object_detection_yolox_2022nov_int8.onnx")
    parser.add_argument("--output", type=Path, default=EVALUATION_ROOT / "results")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--nms", type=float, default=0.50)
    parser.add_argument("--iou", type=float, default=0.50)
    args = parser.parse_args()
    for value, name in ((args.confidence, "confidence"), (args.nms, "nms"), (args.iou, "iou")):
        if not 0.0 <= value <= 1.0:
            parser.error(f"--{name} must be between 0 and 1")
    return args


if __name__ == "__main__":
    evaluate(parse_args())

