"""Command-line entry point for the construction-hazard annotation pipeline.

Usage::

    python -m hazard_annotation <stage> [--mode quick|full] [--mock] [--config PATH]

Stages: ``extract`` / ``annotate`` / ``train`` / ``aggregate`` / ``visualize`` /
``all`` / ``status`` / ``doctor`` / ``probe-qwen`` / ``demo``.

Pipeline modules are imported lazily inside each handler so that lightweight
stages (``doctor`` / ``status``) keep working even if an optional dependency for
a heavier stage is unavailable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from hazard_annotation.config import load_hazard_config

DEFAULT_CONFIG = "examples/hazard_mock_project/config/project.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hazard_annotation",
        description="工业施工安全隐患自动标注系统（双通道：目标框 + 隐患事件）。",
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help="项目配置文件路径，默认 examples/hazard_mock_project/config/project.json。",
    )
    stage_choices = [
        "extract",
        "annotate",
        "train",
        "aggregate",
        "visualize",
        "all",
        "status",
        "doctor",
        "probe-qwen",
        "probe-inference",
        "demo",
    ]
    parser.add_argument(
        "command",
        nargs="?",
        choices=stage_choices,
        default=None,
        help="要执行的阶段（位置参数），如 demo / all / doctor。",
    )
    parser.add_argument(
        "--stage",
        dest="stage_option",
        choices=stage_choices,
        default=None,
        help="要执行的阶段（等价于位置参数）。",
    )
    parser.add_argument(
        "--mode",
        choices=["quick", "full"],
        default=None,
        help="执行档位，覆盖配置文件中的 run_mode。",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="强制 mock 模式（provider=mock / detector=sidecar / trainer=mock）。",
    )
    parser.add_argument(
        "--execute-trainer",
        action="store_true",
        help="train / all 阶段真正调用外部 yolo CLI 训练。",
    )
    parser.add_argument(
        "--no-resume",
        dest="no_resume",
        action="store_true",
        help=(
            "关闭抽帧断点续跑，强制全量重算。"
            "默认开启续跑：输入与抽帧配置都没变的 source 会直接复用已有结果。"
        ),
    )
    parser.add_argument(
        "--image",
        help="probe-qwen / probe-inference 阶段可指定测试图片。",
    )
    parser.add_argument(
        "--send-request",
        dest="send_request",
        action="store_true",
        help=(
            "probe-inference 阶段真正发一次请求。默认只做 TCP 与无凭据 HTTP 探测；"
            "开启后使用本地合成图（不涉及任何真实素材），需先设置 token 环境变量。"
        ),
    )
    return parser


def _apply_mock(config: Any) -> None:
    config.qwen_provider = "mock"
    config.qwen_require_api_key = False
    config.yolo_detector = "sidecar"
    config.yolo_trainer = "mock"


def _print_payload(payload: Any) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))


def _run_status(config: Any) -> dict[str, Any]:
    from traffic_cv_closed_loop.io_utils import read_json

    return {
        "config": config.to_dict(),
        "extract": read_json(config.status_dir / "extract.json", default={}),
        "annotate": read_json(config.status_dir / "annotate.json", default={}),
        "aggregate": read_json(config.status_dir / "aggregate.json", default={}),
        "train": read_json(config.status_dir / "train.json", default={}),
        "visualize": read_json(config.status_dir / "visualize.json", default={}),
    }


def run_stage(config: Any, args: argparse.Namespace, stage: str) -> dict[str, Any]:
    resume = not bool(getattr(args, "no_resume", False))
    if stage == "extract":
        from hazard_annotation.pipeline.extraction import run_extraction

        return {"extract": run_extraction(config, resume=resume)}
    if stage == "annotate":
        from hazard_annotation.pipeline.annotation import run_annotation

        return {"annotate": run_annotation(config)}
    if stage == "aggregate":
        from hazard_annotation.pipeline.aggregate import run_aggregate

        return {"aggregate": run_aggregate(config)}
    if stage == "train":
        from hazard_annotation.pipeline.training import run_training

        return {"train": run_training(config, execute_trainer=args.execute_trainer)}
    if stage == "visualize":
        from hazard_annotation.pipeline.visualization import run_visualize

        return {"visualize": run_visualize(config)}
    if stage == "all":
        from hazard_annotation.pipeline.aggregate import run_aggregate
        from hazard_annotation.pipeline.annotation import run_annotation
        from hazard_annotation.pipeline.extraction import run_extraction
        from hazard_annotation.pipeline.training import run_training
        from hazard_annotation.pipeline.visualization import run_visualize

        payload: dict[str, Any] = {}
        payload["extract"] = run_extraction(config, resume=resume)
        payload["annotate"] = run_annotation(config)
        payload["aggregate"] = run_aggregate(config)
        payload["train"] = run_training(config, execute_trainer=args.execute_trainer)
        payload["visualize"] = run_visualize(config)
        return payload
    if stage == "status":
        return _run_status(config)
    if stage == "doctor":
        from hazard_annotation.diagnostics import build_doctor_report

        return build_doctor_report(config)
    if stage == "probe-qwen":
        from hazard_annotation.diagnostics import probe_qwen_service

        return probe_qwen_service(config, image_path=args.image)
    if stage == "probe-inference":
        from hazard_annotation.inference import (
            probe_inference_endpoint,
            render_probe_report,
        )

        report = probe_inference_endpoint(
            config,
            image_path=args.image,
            send_request=bool(getattr(args, "send_request", False)),
        )
        print(render_probe_report(report))
        return report
    if stage == "demo":
        from hazard_annotation.demo import run_demo

        return run_demo(config)
    raise ValueError(f"Unsupported stage: {stage}")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    config = load_hazard_config(config_path, mode=args.mode)
    if args.mock:
        _apply_mock(config)

    stage = args.stage_option or args.command or "all"

    # In mock mode a fresh project ships no frames; provision the synthetic
    # demo assets so `all --mock` / `extract --mock` work with no video/model.
    if stage in {"all", "extract"}:
        from hazard_annotation.demo import ensure_mock_assets

        ensure_mock_assets(config)

    payload = run_stage(config, args, stage)
    _print_payload(payload)

    if stage == "doctor":
        return 0 if payload.get("status") == "ready" else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
