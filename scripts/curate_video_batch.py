from __future__ import annotations

import argparse
import random
from pathlib import Path

from traffic_cv_closed_loop.config import ProjectConfig
from traffic_cv_closed_loop.io_utils import copy_file, ensure_dir, read_jsonl, write_json, write_text
from traffic_cv_closed_loop.models import AnnotationRecord


def split_counts(total: int, train_ratio: float, val_ratio: float, test_ratio: float) -> tuple[int, int, int]:
    if total <= 0:
        return (0, 0, 0)
    if total == 1:
        return (1, 0, 0)
    if total == 2:
        return (1, 1, 0)

    train_count = max(1, round(total * train_ratio))
    val_count = max(1, round(total * val_ratio))
    if train_count + val_count >= total:
        train_count = max(1, total - 2)
        val_count = 1
    test_count = total - train_count - val_count
    if test_count <= 0:
        test_count = 1
        if train_count > val_count and train_count > 1:
            train_count -= 1
        elif val_count > 1:
            val_count -= 1
    return (train_count, val_count, total - train_count - val_count)


def build_dataset_yaml(config: ProjectConfig, dataset_dir: Path, split_count_map: dict[str, int]) -> str:
    name_lines = [f"  {index}: {label}" for index, label in enumerate(config.labels)]
    train_path = "images/train"
    val_path = "images/val" if split_count_map.get("val", 0) > 0 else train_path
    test_path = "images/test" if split_count_map.get("test", 0) > 0 else val_path
    return "\n".join(
        [
            f"path: {dataset_dir}",
            f"train: {train_path}",
            f"val: {val_path}",
            f"test: {test_path}",
            "names:",
            *name_lines,
            "",
        ]
    )


def frame_quality(record: AnnotationRecord) -> dict[str, object]:
    detections = record.qwen_detections or record.candidate.detections
    confidences = [item.confidence for item in detections]
    min_conf = min(confidences) if confidences else 0.0
    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
    review_reasons = list(record.review_reasons)
    has_error = "annotation_error" in review_reasons
    diff = record.diff or {}
    return {
        "frame": record.candidate.relative_path,
        "source_id": record.candidate.source_id,
        "label_count": len(detections),
        "min_conf": round(min_conf, 4),
        "avg_conf": round(avg_conf, 4),
        "added": int(diff.get("added", 0)),
        "removed": int(diff.get("removed", 0)),
        "adjusted": int(diff.get("adjusted", 0)),
        "has_error": has_error,
        "review_reasons": review_reasons,
        "review_required": record.review_required,
        "label_path": record.label_path,
        "image_path": str(record.candidate.extracted),
    }


def keep_record(quality: dict[str, object], args: argparse.Namespace) -> tuple[bool, list[str]]:
    rejection_reasons: list[str] = []
    if bool(quality["has_error"]):
        rejection_reasons.append("annotation_error")
    if int(quality["added"]) > args.max_added:
        rejection_reasons.append(f"added_gt_{args.max_added}")
    if int(quality["removed"]) > args.max_removed:
        rejection_reasons.append(f"removed_gt_{args.max_removed}")
    if int(quality["adjusted"]) > args.max_adjusted:
        rejection_reasons.append(f"adjusted_gt_{args.max_adjusted}")
    if float(quality["min_conf"]) < args.min_confidence:
        rejection_reasons.append(f"min_conf_lt_{args.min_confidence}")
    if int(quality["label_count"]) < args.min_labels:
        rejection_reasons.append(f"label_count_lt_{args.min_labels}")
    return (not rejection_reasons, rejection_reasons)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Curate noisy video-frame annotations into a cleaner YOLO dataset.")
    parser.add_argument("--config", required=True, help="Path to project config JSON.")
    parser.add_argument("--annotations", help="Override annotations jsonl path.")
    parser.add_argument("--dataset-name", help="Output dataset directory name.")
    parser.add_argument("--random-seed", type=int, help="Override split random seed.")
    parser.add_argument("--min-confidence", type=float, default=0.25)
    parser.add_argument("--max-added", type=int, default=0)
    parser.add_argument("--max-removed", type=int, default=10)
    parser.add_argument("--max-adjusted", type=int, default=2)
    parser.add_argument("--min-labels", type=int, default=1)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = ProjectConfig.load(Path(args.config))
    annotations_path = Path(args.annotations) if args.annotations else config.annotations_path
    default_dataset_name = f"{config.config_path.stem.replace('project.', '')}_curated"
    dataset_name = args.dataset_name or default_dataset_name
    random_seed = args.random_seed if args.random_seed is not None else config.random_seed

    records = [AnnotationRecord.from_dict(item) for item in read_jsonl(annotations_path)]
    if not records:
        raise FileNotFoundError(f"No records found in {annotations_path}")

    qualities: list[dict[str, object]] = []
    accepted_records: list[AnnotationRecord] = []
    rejected_rows: list[dict[str, object]] = []

    for record in records:
        quality = frame_quality(record)
        keep, rejection_reasons = keep_record(quality, args)
        quality["accepted"] = keep
        quality["filter_reasons"] = rejection_reasons
        qualities.append(quality)
        if keep:
            accepted_records.append(record)
        else:
            rejected_rows.append(quality)

    if not accepted_records:
        raise RuntimeError("All frames were filtered out. Loosen thresholds and try again.")

    rng = random.Random(random_seed)
    shuffled = list(accepted_records)
    rng.shuffle(shuffled)
    train_count, val_count, test_count = split_counts(
        len(shuffled),
        config.train_split,
        config.val_split,
        config.test_split,
    )
    split_map: dict[str, list[AnnotationRecord]] = {
        "train": shuffled[:train_count],
        "val": shuffled[train_count : train_count + val_count],
        "test": shuffled[train_count + val_count : train_count + val_count + test_count],
    }

    dataset_dir = config.datasets_dir / dataset_name
    for split_name in split_map:
        ensure_dir(dataset_dir / "images" / split_name)
        ensure_dir(dataset_dir / "labels" / split_name)

    manifest: list[dict[str, object]] = []
    class_histogram: dict[str, int] = {}
    for split_name, split_records in split_map.items():
        for record in split_records:
            relative_path = Path(record.candidate.relative_path)
            image_target = dataset_dir / "images" / split_name / relative_path.name
            label_target = dataset_dir / "labels" / split_name / relative_path.with_suffix(".txt").name
            copy_file(record.candidate.extracted, image_target)
            copy_file(Path(record.label_path), label_target)
            detections = record.qwen_detections or record.candidate.detections
            for detection in detections:
                class_histogram[detection.label] = class_histogram.get(detection.label, 0) + 1
            manifest.append(
                {
                    "split": split_name,
                    "relative_path": record.candidate.relative_path,
                    "image_path": str(image_target.resolve()),
                    "label_path": str(label_target.resolve()),
                    "review_required": record.review_required,
                    "review_reasons": record.review_reasons,
                    "diff": record.diff,
                }
            )

    split_count_map = {key: len(value) for key, value in split_map.items()}
    data_yaml_path = dataset_dir / "data.yaml"
    write_text(data_yaml_path, build_dataset_yaml(config, dataset_dir, split_count_map))
    write_json(dataset_dir / "manifest.json", manifest)
    write_json(dataset_dir / "class_histogram.json", class_histogram)
    write_json(dataset_dir / "quality_report.json", qualities)
    write_json(
        dataset_dir / "curation_summary.json",
        {
            "dataset_name": dataset_name,
            "annotations_path": str(annotations_path.resolve()),
            "accepted_count": len(accepted_records),
            "rejected_count": len(rejected_rows),
            "split_counts": split_count_map,
            "thresholds": {
                "min_confidence": args.min_confidence,
                "max_added": args.max_added,
                "max_removed": args.max_removed,
                "max_adjusted": args.max_adjusted,
                "min_labels": args.min_labels,
            },
            "accepted_frames": [record.candidate.relative_path for record in accepted_records],
            "rejected_frames": [
                {
                    "frame": str(item["frame"]),
                    "filter_reasons": item["filter_reasons"],
                }
                for item in rejected_rows
            ],
            "data_yaml_path": str(data_yaml_path.resolve()),
        },
    )

    print(f"Curated dataset created: {dataset_dir}")
    print(f"Accepted {len(accepted_records)} / {len(records)} frames")
    print(f"Splits: train={train_count}, val={val_count}, test={test_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
