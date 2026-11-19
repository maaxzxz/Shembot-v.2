"""Train and export a person-only EfficientDet-Lite0 TFLite detector.

This script targets the cleaned Pascal VOC dataset produced by prepare_voc.py.
It keeps augmented versions grouped during dataset preparation, trains from the
official TensorFlow Hub EfficientDet-Lite0 feature vector, evaluates on the
held-out test split, and exports a metadata-bearing TFLite model for MediaPipe.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
TRAINING_ROOT = SCRIPT_DIR.parent
os.environ.setdefault("MPLCONFIGDIR", str(TRAINING_ROOT / ".matplotlib"))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

import tensorflow as tf  # noqa: E402
from tflite_model_maker import object_detector  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=TRAINING_ROOT / "cctv_indoor_clean_v2",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=TRAINING_ROOT / "models" / "cctv_person_efficientdet_lite0",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-train", type=int)
    parser.add_argument("--max-valid", type=int)
    parser.add_argument("--max-test", type=int)
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--continue-training", action="store_true")
    parser.add_argument("--train-whole-model", action="store_true")
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument(
        "--load-checkpoint",
        type=Path,
        help="Load detector weights and skip training (for best-checkpoint export).",
    )
    return parser.parse_args()


def selected_annotations(split_dir: Path, maximum: int | None) -> list[str] | None:
    if maximum is None:
        return None
    return [path.stem for path in sorted(split_dir.glob("*.xml"))[:maximum]]


def load_split(
    dataset_root: Path,
    split: str,
    maximum: int | None,
    cache_root: Path,
) -> object_detector.DataLoader:
    split_dir = dataset_root / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Missing dataset split: {split_dir}")
    annotation_names = selected_annotations(split_dir, maximum)
    suffix = "all" if maximum is None else str(maximum)
    return object_detector.DataLoader.from_pascal_voc(
        images_dir=str(split_dir),
        annotations_dir=str(split_dir),
        label_map=["person"],
        annotation_filenames=annotation_names,
        num_shards=10 if maximum is None else 1,
        cache_dir=str(cache_root / split),
        cache_prefix_filename=f"{split}_{suffix}",
    )


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset.resolve()
    output_root = args.output.resolve()
    cache_root = output_root / "cache"
    checkpoint_root = output_root / "checkpoints"
    export_root = output_root / "export"
    for directory in (cache_root, checkpoint_root, export_root):
        directory.mkdir(parents=True, exist_ok=True)

    print(f"TensorFlow: {tf.__version__}")
    print(f"Devices: {tf.config.list_physical_devices()}")
    print(f"Dataset: {dataset_root}")

    train_data = load_split(
        dataset_root, "train", args.max_train, cache_root
    )
    valid_data = load_split(
        dataset_root, "valid", args.max_valid, cache_root
    )
    test_data = None
    if not args.skip_eval:
        test_data = load_split(
            dataset_root, "test", args.max_test, cache_root
        )

    run = {
        "dataset": str(dataset_root),
        "output": str(output_root),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "train_images": len(train_data),
        "valid_images": len(valid_data),
        "test_images": len(test_data) if test_data else 0,
        "class": "person",
        "base_model": "efficientdet-lite0",
        "train_whole_model": args.train_whole_model,
        "learning_rate": args.learning_rate,
        "continue_training": args.continue_training,
        "loaded_checkpoint": (
            str(args.load_checkpoint.resolve()) if args.load_checkpoint else None
        ),
    }
    (output_root / "run.json").write_text(
        json.dumps(run, indent=2), encoding="utf-8"
    )
    print(json.dumps(run, indent=2))

    hparams = {}
    if args.learning_rate is not None:
        hparams["learning_rate"] = args.learning_rate
        hparams["lr_warmup_init"] = args.learning_rate / 10.0
    spec = object_detector.EfficientDetLite0Spec(
        hparams=hparams,
        model_dir=str(checkpoint_root),
        epochs=args.epochs,
        batch_size=args.batch_size,
        tflite_max_detections=50,
        verbose=1,
    )
    if args.load_checkpoint:
        checkpoint = args.load_checkpoint.resolve()
        if not checkpoint.with_suffix(".index").exists():
            raise FileNotFoundError(f"Missing checkpoint: {checkpoint}.index")
        model = object_detector.create(
            train_data=train_data,
            model_spec=spec,
            validation_data=valid_data,
            batch_size=args.batch_size,
            do_train=False,
        )
        model.model.build((None, *spec.config.image_size, 3))
        restore_status = model.model.load_weights(str(checkpoint))
        restore_status.assert_nontrivial_match()
        restore_status.expect_partial()
        print(f"Loaded checkpoint: {checkpoint}")
        # Restore with the same frozen-backbone structure used by phase one,
        # then unfreeze variables for phase-two optimization. Changing this
        # before restoration prevents the checkpoint from matching correctly.
        if args.train_whole_model:
            spec.config.var_freeze_expr = None
        if args.continue_training:
            model.train(
                train_data,
                validation_data=valid_data,
                epochs=args.epochs,
                batch_size=args.batch_size,
            )
    else:
        model = object_detector.create(
            train_data=train_data,
            model_spec=spec,
            validation_data=valid_data,
            epochs=args.epochs,
            batch_size=args.batch_size,
            train_whole_model=args.train_whole_model,
        )

    if test_data is not None:
        metrics = json_safe(model.evaluate(test_data, batch_size=args.batch_size))
        (output_root / "test_metrics.json").write_text(
            json.dumps(metrics, indent=2), encoding="utf-8"
        )
        print("Test metrics:")
        print(json.dumps(metrics, indent=2))

    if not args.skip_export:
        model.export(
            export_dir=str(export_root),
            tflite_filename="person_detector.tflite",
            export_metadata_json_file=True,
        )
        print(f"Exported: {export_root / 'person_detector.tflite'}")


if __name__ == "__main__":
    main()
