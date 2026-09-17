"""Stage 4b — dataset build + YOLO training + hard-example feedback loop.

Reuses the traffic dataset split / ``data.yaml`` / trainer helpers, then extends
the hard-example set with channel-B high-severity frames (``critical`` / ``high``)
in addition to the channel-A review queue.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from hazard_annotation.models import DualChannelRecord
from traffic_cv_closed_loop.adapters.yolo import build_trainer
from traffic_cv_closed_loop.config import ProjectConfig
from traffic_cv_closed_loop.dataset_media import materialize_dataset_image
from traffic_cv_closed_loop.io_utils import (
    copy_file,
    ensure_dir,
    read_jsonl,
    timestamp_token,
    write_json,
    write_jsonl,
    write_text,
)
from traffic_cv_closed_loop.pipeline.training import _build_dataset_yaml, _split_counts

_HIGH_SEVERITY = {"critical", "high"}


def _collect_hard_examples(records: list[DualChannelRecord]) -> list[DualChannelRecord]:
    hard: list[DualChannelRecord] = []
    for record in records:
        has_high_severity = any(
            hazard.severity in _HIGH_SEVERITY for hazard in record.hazard_report.hazards
        )
        has_channel_a_diff = any(value > 0 for value in record.diff.values())
        if record.review_required or has_channel_a_diff or has_high_severity:
            hard.append(record)
    return hard


def _build_next_cycle_tasks(
    dataset_version: str,
    hard_examples_count: int,
    review_queue_count: int,
    high_severity_count: int,
    trainer_status: str,
    fallback_used: bool,
) -> str:
    lines = [
        "# 下一轮闭环任务建议（施工安全隐患）",
        "",
        f"- 数据集版本：`{dataset_version}`",
        f"- 困难样本数（通道 A 复核 + 通道 B 高危）：`{hard_examples_count}`",
        f"- 通道 B 高危帧（critical/high）：`{high_severity_count}`",
        f"- 人工复核队列：`{review_queue_count}`",
        f"- 训练状态：`{trainer_status}`",
    ]
    if fallback_used:
        lines.append("- 本轮无自动通过样本，已回退为使用全部样本构建演示数据集。")
    lines.extend(
        [
            "",
            "建议动作：",
            "1. 优先复核通道 A 的 added / removed / adjusted 目标与低置信度样本。",
            "2. 对通道 B 标记 critical/high 的关键帧优先补充标注并纳入下一轮训练。",
            "3. 单独归档夜间、逆光、遮挡与远距离样本，评估是否需要场景专用数据。",
            "4. 完成复核后重跑闭环，比较新旧 mAP50 / P / R 与隐患命中率变化。",
        ]
    )
    return "\n".join(lines)


def run_training(config: ProjectConfig, execute_trainer: bool = False) -> dict[str, Any]:
    records = [DualChannelRecord.from_dict(item) for item in read_jsonl(config.annotations_path)]
    if not records:
        raise FileNotFoundError("No annotation records found. Run the annotate stage first.")

    accepted = [record for record in records if record.dataset_ready]
    fallback_used = False
    if not accepted:
        accepted = list(records)
        fallback_used = True

    generator = random.Random(config.random_seed)
    generator.shuffle(accepted)

    train_count, val_count, test_count = _split_counts(
        len(accepted), config.train_split, config.val_split, config.test_split
    )
    dataset_version = f"v{timestamp_token()}"
    dataset_dir = config.datasets_dir / dataset_version
    split_map = {
        "train": accepted[:train_count],
        "val": accepted[train_count : train_count + val_count],
        "test": accepted[train_count + val_count : train_count + val_count + test_count],
    }

    for split_name in split_map:
        ensure_dir(dataset_dir / "images" / split_name)
        ensure_dir(dataset_dir / "labels" / split_name)

    manifest: list[dict[str, Any]] = []
    class_histogram: dict[str, int] = {}
    for split_name, split_records in split_map.items():
        images_split_dir = dataset_dir / "images" / split_name
        labels_split_dir = dataset_dir / "labels" / split_name
        for record in split_records:
            relative_path = Path(record.keyframe.relative_path)
            # Guarantee a Ultralytics-readable image extension; the label file
            # is renamed in lock-step so image and label stay paired.
            image_relative = materialize_dataset_image(
                Path(record.keyframe.extracted_path),
                images_split_dir,
                relative_path,
            )
            image_target = images_split_dir / image_relative
            label_target = labels_split_dir / image_relative.with_suffix(".txt")
            copy_file(Path(record.label_path), label_target)
            for detection in record.objects:
                class_histogram[detection.label] = class_histogram.get(detection.label, 0) + 1
            manifest.append(
                {
                    "split": split_name,
                    "relative_path": record.keyframe.relative_path,
                    "image_path": str(image_target.resolve()),
                    "label_path": str(label_target.resolve()),
                    "review_required": record.review_required,
                    "review_reasons": record.review_reasons,
                    "hazard_ids": [hazard.id for hazard in record.hazard_report.hazards],
                }
            )

    split_counts = {name: len(items) for name, items in split_map.items()}
    data_yaml_path = dataset_dir / "data.yaml"
    write_text(data_yaml_path, _build_dataset_yaml(config, dataset_dir, split_counts))
    write_json(dataset_dir / "manifest.json", manifest)
    write_json(dataset_dir / "class_histogram.json", class_histogram)

    trainer = build_trainer(config)
    run_dir = config.runs_dir / dataset_version
    ensure_dir(run_dir)
    review_queue_count = sum(1 for record in records if record.review_required)
    dataset_summary = {
        "dataset_version": dataset_version,
        "accepted_images": len(accepted),
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

    hard_examples = _collect_hard_examples(records)
    high_severity_count = sum(
        1
        for record in records
        if any(hazard.severity in _HIGH_SEVERITY for hazard in record.hazard_report.hazards)
    )
    hard_examples_path = config.analysis_dir / "hard_examples.jsonl"
    next_cycle_path = config.analysis_dir / "next_cycle_tasks.md"
    write_jsonl(hard_examples_path, [record.to_dict() for record in hard_examples])
    write_text(
        next_cycle_path,
        _build_next_cycle_tasks(
            dataset_version=dataset_version,
            hard_examples_count=len(hard_examples),
            review_queue_count=review_queue_count,
            high_severity_count=high_severity_count,
            trainer_status=str(trainer_result.get("status", "")),
            fallback_used=fallback_used,
        ),
    )

    report = {
        "project_name": config.name,
        "mode": config.run_mode,
        "dataset_version": dataset_version,
        "detector_mode": config.yolo_detector,
        "trainer_mode": config.yolo_trainer,
        "qwen_provider": config.qwen_provider,
        "dataset_dir": str(dataset_dir.resolve()),
        "data_yaml_path": str(data_yaml_path.resolve()),
        "run_dir": str(run_dir.resolve()),
        "accepted_images": len(accepted),
        "split_counts": split_counts,
        "review_queue_count": review_queue_count,
        "class_histogram": class_histogram,
        "hard_examples_count": len(hard_examples),
        "high_severity_count": high_severity_count,
        "hard_examples_path": str(hard_examples_path.resolve()),
        "next_cycle_path": str(next_cycle_path.resolve()),
        "trainer": trainer_result,
    }
    write_json(run_dir / "report.json", report)
    write_json(config.latest_run_path, report)
    write_json(config.status_dir / "train.json", report)
    return report
