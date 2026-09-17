from __future__ import annotations

import random
from pathlib import Path

from traffic_cv_closed_loop.adapters.yolo import build_trainer
from traffic_cv_closed_loop.config import ProjectConfig
from traffic_cv_closed_loop.dataset_media import materialize_dataset_image
from traffic_cv_closed_loop.io_utils import copy_file, ensure_dir, read_jsonl, timestamp_token, write_json, write_jsonl, write_text
from traffic_cv_closed_loop.models import AnnotationRecord


def _split_counts(total: int, train_ratio: float, val_ratio: float, test_ratio: float) -> tuple[int, int, int]:
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


def _build_dataset_yaml(config: ProjectConfig, dataset_dir: Path, split_counts: dict[str, int]) -> str:
    name_lines = [f"  {index}: {label}" for index, label in enumerate(config.labels)]
    train_path = "images/train"
    val_path = "images/val" if split_counts.get("val", 0) > 0 else train_path
    test_path = "images/test" if split_counts.get("test", 0) > 0 else val_path
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


def _build_next_cycle_tasks(
    dataset_version: str,
    hard_examples_count: int,
    review_queue_count: int,
    trainer_status: str,
    fallback_used: bool,
) -> str:
    lines = [
        "# 下一轮闭环任务建议",
        "",
        f"- 数据集版本：`{dataset_version}`",
        f"- 困难样本数：`{hard_examples_count}`",
        f"- 人工复核队列：`{review_queue_count}`",
        f"- 训练状态：`{trainer_status}`",
    ]
    if fallback_used:
        lines.append("- 本轮由于无自动通过样本，已回退为使用全部样本构建演示数据集。")
    lines.extend(
        [
            "",
            "建议动作：",
            "1. 优先处理新增目标、删除目标和低置信度样本。",
            "2. 单独归档夜间、雨雪、逆光、遮挡和远距离样本。",
            "3. 完成人工复核后重新执行闭环，比较新旧模型指标变化。",
        ]
    )
    return "\n".join(lines)


def run_training(config: ProjectConfig, execute_trainer: bool = False) -> dict[str, object]:
    records = [AnnotationRecord.from_dict(item) for item in read_jsonl(config.annotations_path)]
    if not records:
        raise FileNotFoundError("No annotation records found. Run annotate stage first.")

    accepted_records = [item for item in records if item.dataset_ready]
    fallback_used = False
    if not accepted_records:
        accepted_records = list(records)
        fallback_used = True

    generator = random.Random(config.random_seed)
    generator.shuffle(accepted_records)

    train_count, val_count, test_count = _split_counts(
        len(accepted_records),
        config.train_split,
        config.val_split,
        config.test_split,
    )
    dataset_version = f"v{timestamp_token()}"
    dataset_dir = config.datasets_dir / dataset_version
    split_map: dict[str, list[AnnotationRecord]] = {
        "train": accepted_records[:train_count],
        "val": accepted_records[train_count : train_count + val_count],
        "test": accepted_records[train_count + val_count : train_count + val_count + test_count],
    }

    for split_name in split_map:
        ensure_dir(dataset_dir / "images" / split_name)
        ensure_dir(dataset_dir / "labels" / split_name)

    dataset_manifest: list[dict[str, object]] = []
    class_histogram: dict[str, int] = {}

    for split_name, split_records in split_map.items():
        images_split_dir = dataset_dir / "images" / split_name
        labels_split_dir = dataset_dir / "labels" / split_name
        for record in split_records:
            relative_path = Path(record.candidate.relative_path)
            # Guarantee a Ultralytics-readable image extension; the label file
            # is renamed in lock-step so image and label stay paired.
            image_relative = materialize_dataset_image(
                record.candidate.extracted,
                images_split_dir,
                relative_path,
            )
            image_target = images_split_dir / image_relative
            label_target = labels_split_dir / image_relative.with_suffix(".txt")
            copy_file(Path(record.label_path), label_target)
            for detection in record.qwen_detections:
                class_histogram[detection.label] = class_histogram.get(detection.label, 0) + 1
            dataset_manifest.append(
                {
                    "split": split_name,
                    "relative_path": record.candidate.relative_path,
                    "image_path": str(image_target.resolve()),
                    "label_path": str(label_target.resolve()),
                    "review_required": record.review_required,
                    "review_reasons": record.review_reasons,
                }
            )

    split_counts = {key: len(value) for key, value in split_map.items()}

    data_yaml_path = dataset_dir / "data.yaml"
    write_text(data_yaml_path, _build_dataset_yaml(config, dataset_dir, split_counts))
    write_json(dataset_dir / "manifest.json", dataset_manifest)
    write_json(dataset_dir / "class_histogram.json", class_histogram)

    trainer = build_trainer(config)
    run_dir = config.runs_dir / dataset_version
    ensure_dir(run_dir)

    review_queue_count = sum(1 for item in records if item.review_required)
    dataset_summary = {
        "dataset_version": dataset_version,
        "accepted_images": len(accepted_records),
        "split_counts": split_counts,
        "class_histogram": class_histogram,
        "review_queue_count": review_queue_count,
        "fallback_used": fallback_used,
    }
    trainer_result = trainer.train(
        config=config,
        dataset_yaml_path=data_yaml_path,
        run_dir=run_dir,
        dataset_summary=dataset_summary,
        execute=execute_trainer,
    )

    hard_examples = [
        item
        for item in records
        if item.review_required or any(value > 0 for value in item.diff.values())
    ]
    hard_examples_path = config.analysis_dir / "hard_examples.jsonl"
    next_cycle_path = config.analysis_dir / "next_cycle_tasks.md"
    write_jsonl(hard_examples_path, [item.to_dict() for item in hard_examples])
    write_text(
        next_cycle_path,
        _build_next_cycle_tasks(
            dataset_version=dataset_version,
            hard_examples_count=len(hard_examples),
            review_queue_count=review_queue_count,
            trainer_status=str(trainer_result.get("status", "")),
            fallback_used=fallback_used,
        ),
    )

    report = {
        "project_name": config.name,
        "dataset_version": dataset_version,
        "detector_mode": config.yolo_detector,
        "trainer_mode": config.yolo_trainer,
        "qwen_provider": config.qwen_provider,
        "dataset_dir": str(dataset_dir.resolve()),
        "data_yaml_path": str(data_yaml_path.resolve()),
        "run_dir": str(run_dir.resolve()),
        "accepted_images": len(accepted_records),
        "split_counts": dataset_summary["split_counts"],
        "review_queue_count": review_queue_count,
        "class_histogram": class_histogram,
        "hard_examples_path": str(hard_examples_path.resolve()),
        "next_cycle_path": str(next_cycle_path.resolve()),
        "trainer": trainer_result,
    }
    write_json(run_dir / "report.json", report)
    write_json(config.latest_run_path, report)
    write_json(config.status_dir / "train.json", report)
    return report
