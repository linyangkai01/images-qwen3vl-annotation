from __future__ import annotations

import csv
import shutil
import subprocess
from pathlib import Path
from typing import Any

from traffic_cv_closed_loop.config import ProjectConfig
from traffic_cv_closed_loop.io_utils import ensure_dir, read_json, stem_sidecar, timestamp_token, write_json, write_text
from traffic_cv_closed_loop.models import Detection


def _xywh_to_xyxy(
    x_center: float,
    y_center: float,
    width: float,
    height: float,
) -> tuple[float, float, float, float]:
    x1 = x_center - (width / 2)
    y1 = y_center - (height / 2)
    x2 = x_center + (width / 2)
    y2 = y_center + (height / 2)
    return (x1, y1, x2, y2)


def _read_results_csv(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        return {}

    final_row = rows[-1]
    key_map = {
        "metrics/precision(B)": "precision",
        "metrics/recall(B)": "recall",
        "metrics/mAP50(B)": "mAP50",
        "metrics/mAP50-95(B)": "mAP50_95",
        "fitness": "fitness",
        "train/box_loss": "train_box_loss",
        "train/cls_loss": "train_cls_loss",
        "val/box_loss": "val_box_loss",
        "val/cls_loss": "val_cls_loss",
    }
    metrics: dict[str, float] = {}
    for source_key, target_key in key_map.items():
        raw_value = final_row.get(source_key)
        if raw_value is None or raw_value == "":
            continue
        try:
            metrics[target_key] = round(float(raw_value), 6)
        except ValueError:
            continue
    return metrics


def _decode_process_output(payload: bytes | str | None) -> str:
    if payload is None:
        return ""
    if isinstance(payload, str):
        return payload
    for encoding in ("utf-8", "gbk"):
        try:
            return payload.decode(encoding)
        except UnicodeDecodeError:
            continue
    return payload.decode("utf-8", errors="replace")


class SidecarDetector:
    def __init__(self, config: ProjectConfig) -> None:
        self.config = config

    def detect(self, image_path: Path) -> list[Detection]:
        payload = read_json(stem_sidecar(image_path, "detections"), default={"objects": []})
        objects = payload.get("objects", []) if isinstance(payload, dict) else payload
        detections: list[Detection] = []
        for item in objects:
            detection = Detection.from_dict(item, default_source="yolo-sidecar")
            if detection.label not in self.config.labels:
                continue
            detections.append(detection)
        return detections

    def detect_many(self, image_paths: list[Path]) -> dict[Path, list[Detection]]:
        return {path.resolve(): self.detect(path) for path in image_paths}


class UltralyticsDetector:
    def __init__(self, config: ProjectConfig) -> None:
        self.config = config

    def _build_predict_command(self, source_dir: Path, run_dir: Path) -> list[str]:
        return [
            self.config.yolo_binary,
            self.config.yolo_task,
            "predict",
            f"model={self.config.yolo_detector_weights}",
            f"source={source_dir}",
            f"conf={self.config.yolo_detector_confidence}",
            f"iou={self.config.yolo_detector_iou}",
            f"imgsz={self.config.yolo_detector_image_size}",
            f"project={run_dir.parent}",
            f"name={run_dir.name}",
            "save=False",
            "save_txt=True",
            "save_conf=True",
            "exist_ok=True",
            "verbose=False",
        ]

    def _parse_label_file(self, label_path: Path) -> list[Detection]:
        if not label_path.exists():
            return []
        detections: list[Detection] = []
        for raw_line in label_path.read_text(encoding="utf-8").splitlines():
            parts = raw_line.strip().split()
            if len(parts) < 5:
                continue
            class_index = int(float(parts[0]))
            if class_index < 0 or class_index >= len(self.config.labels):
                continue
            x_center, y_center, width, height = map(float, parts[1:5])
            confidence = float(parts[5]) if len(parts) >= 6 else 1.0
            detection = Detection(
                label=self.config.labels[class_index],
                bbox=_xywh_to_xyxy(x_center, y_center, width, height),
                confidence=confidence,
                source="yolo-ultralytics",
            )
            detections.append(detection)
        return detections

    def detect_many(self, image_paths: list[Path]) -> dict[Path, list[Detection]]:
        if not image_paths:
            return {}
        if shutil.which(self.config.yolo_binary) is None:
            raise FileNotFoundError(
                f"YOLO binary not found in PATH: {self.config.yolo_binary}"
            )

        resolved_paths = [path.resolve() for path in image_paths]
        parent_dirs = {path.parent for path in resolved_paths}
        if len(parent_dirs) != 1:
            merged: dict[Path, list[Detection]] = {}
            grouped: dict[Path, list[Path]] = {}
            for path in resolved_paths:
                grouped.setdefault(path.parent, []).append(path)
            for grouped_paths in grouped.values():
                merged.update(self.detect_many(grouped_paths))
            return merged

        source_dir = next(iter(parent_dirs))
        run_dir = self.config.detector_runs_dir / f"{timestamp_token()}_{source_dir.name}"
        ensure_dir(run_dir.parent)
        command = self._build_predict_command(source_dir=source_dir, run_dir=run_dir)
        write_text(run_dir / "command.txt", " ".join(command))

        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=False,
        )
        stdout_text = _decode_process_output(completed.stdout)
        stderr_text = _decode_process_output(completed.stderr)
        write_text(run_dir / "stdout.log", stdout_text)
        write_text(run_dir / "stderr.log", stderr_text)
        if completed.returncode != 0:
            raise RuntimeError(
                f"Ultralytics detector failed with exit code {completed.returncode}. "
                f"See {run_dir / 'stderr.log'} for details."
            )

        labels_dir = run_dir / "labels"
        detections_by_path: dict[Path, list[Detection]] = {}
        for image_path in resolved_paths:
            detections_by_path[image_path] = self._parse_label_file(labels_dir / f"{image_path.stem}.txt")

        write_json(
            run_dir / "summary.json",
            {
                "source_dir": str(source_dir),
                "image_count": len(resolved_paths),
                "command": command,
                "labels_dir": str(labels_dir),
            },
        )
        return detections_by_path


class MockTrainer:
    def train(
        self,
        config: ProjectConfig,
        dataset_yaml_path: Path,
        run_dir: Path,
        dataset_summary: dict[str, Any],
        execute: bool = False,
    ) -> dict[str, Any]:
        image_count = int(dataset_summary.get("accepted_images", 0))
        review_queue_count = int(dataset_summary.get("review_queue_count", 0))
        class_histogram = dict(dataset_summary.get("class_histogram", {}))
        class_coverage = len(class_histogram) / max(1, len(config.labels))
        review_pressure = review_queue_count / max(1, image_count)

        map50 = min(0.95, 0.45 + 0.04 * min(8, image_count) + 0.12 * class_coverage - 0.08 * review_pressure)
        precision = min(0.98, map50 + 0.05)
        recall = max(0.35, min(0.96, map50 + 0.01 - 0.05 * review_pressure))
        f1_score = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)

        metrics = {
            "mAP50": round(map50, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1_score, 4),
            "epochs": config.yolo_epochs,
            "trainer_mode": "mock",
        }
        write_json(run_dir / "metrics.json", metrics)
        write_text(
            run_dir / "trainer_summary.md",
            "\n".join(
                [
                    "# Mock YOLO Training Summary",
                    "",
                    f"- Dataset config: `{dataset_yaml_path}`",
                    f"- Accepted images: `{image_count}`",
                    f"- Review queue size: `{review_queue_count}`",
                    f"- Class coverage: `{len(class_histogram)}` / `{len(config.labels)}`",
                    f"- mAP50: `{metrics['mAP50']}`",
                    f"- Precision: `{metrics['precision']}`",
                    f"- Recall: `{metrics['recall']}`",
                    "",
                    "This is a dry-run training summary for pipeline validation only.",
                ]
            ),
        )
        return {
            "mode": "mock",
            "status": "completed",
            "command": "",
            "metrics": metrics,
            "executed": execute,
        }


class UltralyticsTrainer:
    def _build_train_command(
        self,
        config: ProjectConfig,
        dataset_yaml_path: Path,
        run_dir: Path,
    ) -> list[str]:
        command = [
            config.yolo_binary,
            config.yolo_task,
            "train",
            f"data={dataset_yaml_path}",
            f"model={config.yolo_weights}",
            f"epochs={config.yolo_epochs}",
            f"imgsz={config.yolo_image_size}",
            f"device={config.yolo_device}",
            f"batch={config.yolo_batch}",
            f"workers={config.yolo_workers}",
            f"patience={config.yolo_patience}",
            f"cache={str(config.yolo_cache).lower()}",
            f"close_mosaic={config.yolo_close_mosaic}",
            f"save_period={config.yolo_save_period}",
            f"project={run_dir.parent}",
            f"name={run_dir.name}",
            "exist_ok=True",
        ]
        if config.yolo_resume:
            command.append("resume=True")
        return command

    def train(
        self,
        config: ProjectConfig,
        dataset_yaml_path: Path,
        run_dir: Path,
        dataset_summary: dict[str, Any],
        execute: bool = False,
    ) -> dict[str, Any]:
        command = self._build_train_command(
            config=config,
            dataset_yaml_path=dataset_yaml_path,
            run_dir=run_dir,
        )
        command_text = " ".join(command)
        write_text(run_dir / "command.txt", command_text)

        if not execute:
            return {
                "mode": "ultralytics",
                "status": "prepared",
                "command": command_text,
                "metrics": {},
                "executed": False,
                "binary_available": shutil.which(config.yolo_binary) is not None,
            }

        if shutil.which(config.yolo_binary) is None:
            return {
                "mode": "ultralytics",
                "status": "missing_binary",
                "command": command_text,
                "metrics": {},
                "executed": False,
            }

        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=False,
        )
        stdout_text = _decode_process_output(completed.stdout)
        stderr_text = _decode_process_output(completed.stderr)
        write_text(run_dir / "stdout.log", stdout_text)
        write_text(run_dir / "stderr.log", stderr_text)

        metrics = _read_results_csv(run_dir / "results.csv")
        artifacts = {
            "best_weights": str((run_dir / "weights" / "best.pt").resolve()) if (run_dir / "weights" / "best.pt").exists() else "",
            "last_weights": str((run_dir / "weights" / "last.pt").resolve()) if (run_dir / "weights" / "last.pt").exists() else "",
            "results_csv": str((run_dir / "results.csv").resolve()) if (run_dir / "results.csv").exists() else "",
        }
        if metrics:
            write_json(run_dir / "metrics.json", metrics)
        write_text(
            run_dir / "trainer_summary.md",
            "\n".join(
                [
                    "# Ultralytics Training Summary",
                    "",
                    f"- Status: `{'completed' if completed.returncode == 0 else 'failed'}`",
                    f"- Dataset config: `{dataset_yaml_path}`",
                    f"- Command: `{command_text}`",
                    f"- Best weights: `{artifacts['best_weights'] or 'not_found'}`",
                    f"- Results CSV: `{artifacts['results_csv'] or 'not_found'}`",
                ]
                + (
                    ["", "## Metrics"]
                    + [f"- {key}: `{value}`" for key, value in metrics.items()]
                    if metrics
                    else ["", "No parsed metrics found in results.csv."]
                )
            ),
        )

        return {
            "mode": "ultralytics",
            "status": "completed" if completed.returncode == 0 else "failed",
            "command": command_text,
            "metrics": metrics,
            "artifacts": artifacts,
            "executed": True,
            "return_code": completed.returncode,
        }


def build_detector(config: ProjectConfig) -> SidecarDetector | UltralyticsDetector:
    mode = config.yolo_detector.lower()
    if mode == "sidecar":
        return SidecarDetector(config)
    if mode == "ultralytics":
        return UltralyticsDetector(config)
    raise ValueError(f"Unsupported yolo detector mode: {config.yolo_detector}")


def build_trainer(config: ProjectConfig) -> MockTrainer | UltralyticsTrainer:
    mode = config.yolo_trainer.lower()
    if mode == "mock":
        return MockTrainer()
    if mode == "ultralytics":
        return UltralyticsTrainer()
    raise ValueError(f"Unsupported yolo trainer mode: {config.yolo_trainer}")
