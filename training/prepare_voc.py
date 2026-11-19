"""Clean a Roboflow Pascal VOC export for one-class person detection.

The source dataset is never modified. All augmented versions of the same
source image are assigned to the same split to prevent train/validation
leakage.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path


SOURCE_SPLITS = ("train", "valid", "test")
OUTPUT_SPLITS = ("train", "valid", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def source_id(path: Path) -> str:
    return path.stem.split(".rf.", 1)[0]


def assigned_split(group_name: str) -> str:
    value = int(hashlib.sha256(group_name.encode("utf-8")).hexdigest()[:8], 16)
    bucket = value / 0xFFFFFFFF
    if bucket < 0.80:
        return "train"
    if bucket < 0.90:
        return "valid"
    return "test"


def clamp(value: int, minimum: int, maximum: int) -> int:
    return min(maximum, max(minimum, value))


def prepare_annotation(xml_path: Path, image_name: str, split: str) -> tuple[ET.ElementTree, int, int]:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    width = int(float(root.findtext("size/width", "0")))
    height = int(float(root.findtext("size/height", "0")))
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image size in {xml_path}")

    corrected = 0
    removed = 0
    for obj in list(root.findall("object")):
        name = obj.find("name")
        if name is None:
            name = ET.SubElement(obj, "name")
        name.text = "person"

        box = obj.find("bndbox")
        if box is None:
            root.remove(obj)
            removed += 1
            continue

        original = {
            key: int(round(float(box.findtext(key, "0"))))
            for key in ("xmin", "ymin", "xmax", "ymax")
        }
        fixed = {
            "xmin": clamp(original["xmin"], 0, width),
            "ymin": clamp(original["ymin"], 0, height),
            "xmax": clamp(original["xmax"], 0, width),
            "ymax": clamp(original["ymax"], 0, height),
        }
        if fixed["xmax"] <= fixed["xmin"] or fixed["ymax"] <= fixed["ymin"]:
            root.remove(obj)
            removed += 1
            continue
        if fixed != original:
            corrected += 1
        for key, value in fixed.items():
            node = box.find(key)
            if node is None:
                node = ET.SubElement(box, key)
            node.text = str(value)

    for key, value in (
        ("folder", split),
        ("filename", image_name),
        ("path", image_name),
    ):
        node = root.find(key)
        if node is None:
            node = ET.SubElement(root, key)
        node.text = value

    return tree, corrected, removed


def main() -> None:
    args = parse_args()
    input_root = args.input.resolve()
    output_root = args.output.resolve()
    if not input_root.is_dir():
        raise SystemExit(f"Input directory does not exist: {input_root}")
    if output_root.exists() and any(output_root.iterdir()):
        raise SystemExit(f"Output directory must be empty or absent: {output_root}")

    records: list[tuple[Path, Path, str]] = []
    missing_pairs: list[str] = []
    for original_split in SOURCE_SPLITS:
        split_dir = input_root / original_split
        for xml_path in sorted(split_dir.glob("*.xml")):
            image_path = xml_path.with_suffix(".jpg")
            if not image_path.exists():
                missing_pairs.append(str(xml_path))
                continue
            records.append((image_path, xml_path, original_split))

    groups: dict[str, list[tuple[Path, Path, str]]] = defaultdict(list)
    for record in records:
        groups[source_id(record[0])].append(record)

    for split in OUTPUT_SPLITS:
        (output_root / split).mkdir(parents=True, exist_ok=True)

    image_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    box_counts: Counter[str] = Counter()
    corrected_counts: Counter[str] = Counter()
    removed_counts: Counter[str] = Counter()
    manifest_rows: list[dict[str, str | int]] = []

    for group_name, group_records in sorted(groups.items()):
        split = assigned_split(group_name)
        source_counts[split] += 1
        for image_path, xml_path, original_split in group_records:
            output_image = output_root / split / image_path.name
            output_xml = output_root / split / xml_path.name
            shutil.copy2(image_path, output_image)
            tree, corrected, removed = prepare_annotation(
                xml_path, output_image.name, split
            )
            # Model Maker 0.4.3 reads VOC files as Unicode strings. Avoid an
            # encoding declaration, which lxml rejects for Unicode input.
            tree.write(output_xml, encoding="utf-8", xml_declaration=False)
            objects = len(tree.getroot().findall("object"))
            image_counts[split] += 1
            box_counts[split] += objects
            corrected_counts[split] += corrected
            removed_counts[split] += removed
            manifest_rows.append(
                {
                    "image": output_image.name,
                    "source_id": group_name,
                    "source_split": original_split,
                    "split": split,
                    "boxes": objects,
                }
            )

    with (output_root / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("image", "source_id", "source_split", "split", "boxes"),
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    report = {
        "input": str(input_root),
        "output": str(output_root),
        "images": dict(image_counts),
        "source_groups": dict(source_counts),
        "boxes": dict(box_counts),
        "corrected_boxes": dict(corrected_counts),
        "removed_boxes": dict(removed_counts),
        "missing_image_pairs": missing_pairs,
        "class_name": "person",
    }
    (output_root / "audit.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
