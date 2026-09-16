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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="traffic-cv-loop",
        description="Qwen3VL to YOLO closed-loop training workflow for traffic CV projects.",
    )
    parser.add_argument(
        "--config",
        default="examples/mock_project/config/project.json",
        help=argparse.SUPPRESS,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_config_argument(command_parser: argparse.ArgumentParser) -> None:
        # ``argparse.SUPPRESS`` prevents the subparser default from clobbering a
        # ``--config`` supplied before the subcommand; the top-level ``--config``
        # argument already provides the default value.
        command_parser.add_argument(
            "--config",
            default=argparse.SUPPRESS,
            help="Path to the project config JSON file.",
        )

    extract_parser = subparsers.add_parser("extract", help="Run frame extraction and candidate selection.")
    add_config_argument(extract_parser)

    annotate_parser = subparsers.add_parser("annotate", help="Run Qwen3VL annotation refinement.")
    add_config_argument(annotate_parser)

    train_parser = subparsers.add_parser("train", help="Build dataset and run YOLO training.")
    add_config_argument(train_parser)
    train_parser.add_argument(
        "--execute-trainer",
        action="store_true",
        help="Execute the YOLO trainer when the provider supports it.",
    )

    run_loop_parser = subparsers.add_parser("run-loop", help="Run the full closed-loop pipeline.")
    add_config_argument(run_loop_parser)
    run_loop_parser.add_argument(
        "--execute-trainer",
        action="store_true",
        help="Execute the YOLO trainer when the provider supports it.",
    )

    status_parser = subparsers.add_parser("status", help="Show the latest extracted, annotated and training summaries.")
    add_config_argument(status_parser)

    doctor_parser = subparsers.add_parser("doctor", help="Validate production config, env vars and local binaries.")
    add_config_argument(doctor_parser)

    probe_qwen_parser = subparsers.add_parser("probe-qwen", help="Send a live probe request to the configured Qwen service.")
    add_config_argument(probe_qwen_parser)
    probe_qwen_parser.add_argument(
        "--image",
        help="Optional image path for probing the Qwen service.",
    )

    probe_detector_parser = subparsers.add_parser("probe-detector", help="Run one detector probe on a sample image.")
    add_config_argument(probe_detector_parser)
    probe_detector_parser.add_argument(
        "--image",
        help="Optional image path for probing the detector.",
    )

    return parser


def _print_payload(payload: Any) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    config = ProjectConfig.load(args.config)

    if args.command == "extract":
        _print_payload(run_extraction(config))
        return 0
    if args.command == "annotate":
        _print_payload(run_annotation(config))
        return 0
    if args.command == "train":
        _print_payload(run_training(config, execute_trainer=args.execute_trainer))
        return 0
    if args.command == "run-loop":
        payload = {
            "extract": run_extraction(config),
            "annotate": run_annotation(config),
            "train": run_training(config, execute_trainer=args.execute_trainer),
        }
        _print_payload(payload)
        return 0
    if args.command == "status":
        payload = {
            "config": config.to_dict(),
            "extract": read_json(config.status_dir / "extract.json", default={}),
            "annotate": read_json(config.status_dir / "annotate.json", default={}),
            "train": read_json(config.status_dir / "train.json", default={}),
        }
        _print_payload(payload)
        return 0
    if args.command == "doctor":
        report = build_doctor_report(config)
        _print_payload(report)
        return 0 if report["status"] == "ready" else 1
    if args.command == "probe-qwen":
        report = probe_qwen_service(config, image_path=args.image)
        _print_payload(report)
        return 0
    if args.command == "probe-detector":
        report = probe_detector(config, image_path=args.image)
        _print_payload(report)
        return 0

    parser.error(f"Unsupported command: {args.command}")
    return 2
