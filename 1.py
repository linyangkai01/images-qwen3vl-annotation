from __future__ import annotations

import argparse
import json
from typing import Any

from traffic_cv_closed_loop.config import ProjectConfig
from traffic_cv_closed_loop.diagnostics import build_doctor_report, probe_detector, probe_qwen_service
from traffic_cv_closed_loop.io_utils import read_json
from traffic_cv_closed_loop.pipeline.annotation import run_annotation
from traffic_cv_closed_loop.pipeline.extraction import run_extraction
from traffic_cv_closed_loop.pipeline.training import run_training


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Qwen3VL -> YOLO 闭环训练主程序，可直接调用抽帧、标注、训练三个阶段。",
    )
    parser.add_argument(
        "--config",
        default="examples/production_project/config/project.production.json",
        help="项目配置文件路径，默认使用 production 配置。",
    )
    parser.add_argument(
        "--stage",
        choices=["extract", "annotate", "train", "all", "status", "doctor", "probe-qwen", "probe-detector"],
        default="all",
        help="要执行的阶段，默认 all。",
    )
    parser.add_argument(
        "--execute-trainer",
        action="store_true",
        help="在 train 或 all 时真正执行 YOLO 训练。",
    )
    parser.add_argument(
        "--image",
        help="probe-qwen / probe-detector 时可指定测试图片。",
    )
    return parser


def print_payload(payload: Any) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def run_stage(config: ProjectConfig, args: argparse.Namespace) -> dict[str, Any]:
    if args.stage == "extract":
        return {"extract": run_extraction(config)}
    if args.stage == "annotate":
        return {"annotate": run_annotation(config)}
    if args.stage == "train":
        return {"train": run_training(config, execute_trainer=args.execute_trainer)}
    if args.stage == "all":
        return {
            "extract": run_extraction(config),
            "annotate": run_annotation(config),
            "train": run_training(config, execute_trainer=args.execute_trainer),
        }
    if args.stage == "status":
        return {
            "config": config.to_dict(),
            "extract": read_json(config.status_dir / "extract.json", default={}),
            "annotate": read_json(config.status_dir / "annotate.json", default={}),
            "train": read_json(config.status_dir / "train.json", default={}),
        }
    if args.stage == "doctor":
        return build_doctor_report(config)
    if args.stage == "probe-qwen":
        return probe_qwen_service(config, image_path=args.image)
    if args.stage == "probe-detector":
        return probe_detector(config, image_path=args.image)
    raise ValueError(f"Unsupported stage: {args.stage}")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    config = ProjectConfig.load(args.config)
    payload = run_stage(config, args)
    print_payload(payload)
    if args.stage == "doctor":
        return 0 if payload.get("status") == "ready" else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
