from __future__ import annotations

import os
import shutil
import time
from pathlib import Path
from typing import Any

from traffic_cv_closed_loop.adapters.qwen import build_qwen_annotator
from traffic_cv_closed_loop.adapters.yolo import build_detector
from traffic_cv_closed_loop.config import ProjectConfig
from traffic_cv_closed_loop.io_utils import iter_image_files, read_text, write_json
from traffic_cv_closed_loop.models import CandidateFrame
from traffic_cv_closed_loop.prompting import render_annotation_prompt


def _describe_model_reference(project_dir: Path, value: str) -> dict[str, Any]:
    if not value:
        return {"reference": value, "kind": "missing", "exists": False}
    if value.startswith("http://") or value.startswith("https://"):
        return {"reference": value, "kind": "remote_url", "exists": True}

    path = Path(value)
    if path.is_absolute() or "/" in value or "\\" in value or value.startswith("."):
        resolved = path if path.is_absolute() else (project_dir / path).resolve()
        return {
            "reference": value,
            "resolved_path": str(resolved),
            "kind": "local_path",
            "exists": resolved.exists(),
        }
    return {"reference": value, "kind": "registry_name", "exists": True}


def _first_probe_image(config: ProjectConfig, image_path: str | None = None) -> Path:
    if image_path:
        path = Path(image_path)
        if not path.is_absolute():
            path = (config.project_dir / path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Probe image not found: {path}")
        return path

    images = iter_image_files(config.input_frames_dir)
    if not images:
        raise FileNotFoundError(f"No images found under {config.input_frames_dir}")
    return images[0]


def build_doctor_report(config: ProjectConfig) -> dict[str, Any]:
    images = iter_image_files(config.input_frames_dir)
    issues: list[str] = []

    prompt_exists = config.prompt_template_path.exists()
    if not prompt_exists:
        issues.append("prompt_template_missing")

    if not config.input_frames_dir.exists():
        issues.append("input_frames_dir_missing")
    elif not images:
        issues.append("input_images_missing")

    qwen_api_key_present = bool(os.environ.get(config.qwen_api_key_env, ""))
    qwen_header_env_status = {
        header_name: {
            "env": env_name,
            "present": bool(os.environ.get(env_name, "")),
        }
        for header_name, env_name in config.qwen_header_env.items()
    }
    if config.qwen_provider.lower() == "openai_compatible":
        if not config.qwen_chat_endpoint:
            issues.append("qwen_endpoint_missing")
        if config.qwen_require_api_key and not qwen_api_key_present:
            issues.append("qwen_api_key_missing")
        for header_name, state in qwen_header_env_status.items():
            if not state["present"]:
                issues.append(f"qwen_header_env_missing:{header_name}")

    yolo_binary_available = shutil.which(config.yolo_binary) is not None
    if (
        config.yolo_detector.lower() == "ultralytics"
        or config.yolo_trainer.lower() == "ultralytics"
    ) and not yolo_binary_available:
        issues.append("yolo_binary_missing")

    detector_weights = _describe_model_reference(config.project_dir, config.yolo_detector_weights)
    trainer_weights = _describe_model_reference(config.project_dir, config.yolo_weights)
    if config.yolo_detector.lower() == "ultralytics" and detector_weights["kind"] == "local_path" and not detector_weights["exists"]:
        issues.append("yolo_detector_weights_missing")
    if config.yolo_trainer.lower() == "ultralytics" and trainer_weights["kind"] == "local_path" and not trainer_weights["exists"]:
        issues.append("yolo_training_weights_missing")

    return {
        "status": "ready" if not issues else "issues_found",
        "issues": issues,
        "config_path": str(config.config_path),
        "project_name": config.name,
        "input": {
            "frames_dir": str(config.input_frames_dir),
            "frames_dir_exists": config.input_frames_dir.exists(),
            "image_count": len(images),
        },
        "prompt_template": {
            "path": str(config.prompt_template_path),
            "exists": prompt_exists,
        },
        "qwen": {
            "provider": config.qwen_provider,
            "chat_endpoint": config.qwen_chat_endpoint,
            "model": config.qwen_model,
            "api_key_env": config.qwen_api_key_env,
            "api_key_present": qwen_api_key_present,
            "header_env": qwen_header_env_status,
            "timeout_seconds": config.qwen_timeout_seconds,
            "max_retries": config.qwen_max_retries,
        },
        "yolo": {
            "binary": config.yolo_binary,
            "binary_available": yolo_binary_available,
            "detector": config.yolo_detector,
            "trainer": config.yolo_trainer,
            "detector_weights": detector_weights,
            "training_weights": trainer_weights,
            "device": config.yolo_device,
        },
    }


def probe_qwen_service(config: ProjectConfig, image_path: str | None = None) -> dict[str, Any]:
    selected_image = _first_probe_image(config, image_path=image_path)
    template = read_text(config.prompt_template_path)
    if not template.strip():
        raise FileNotFoundError(f"Prompt template is empty or missing: {config.prompt_template_path}")

    try:
        relative_path = str(selected_image.relative_to(config.input_frames_dir))
    except ValueError:
        relative_path = selected_image.name

    candidate = CandidateFrame(
        source_id="probe",
        relative_path=relative_path,
        original_path=str(selected_image.resolve()),
        extracted_path=str(selected_image.resolve()),
        detections=[],
        scene=config.scene,
        metadata={"probe": True},
    )
    prompt_text = render_annotation_prompt(config, candidate, template)
    annotator = build_qwen_annotator(config)

    started = time.perf_counter()
    detections, payload = annotator.annotate(candidate, prompt_text)
    elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
    report = {
        "status": "ok",
        "provider": config.qwen_provider,
        "image_path": str(selected_image),
        "elapsed_ms": elapsed_ms,
        "detection_count": len(detections),
        "labels": [item.label for item in detections],
        "response": payload,
    }
    write_json(config.probes_dir / "qwen_probe.json", report)
    return report


def probe_detector(config: ProjectConfig, image_path: str | None = None) -> dict[str, Any]:
    selected_image = _first_probe_image(config, image_path=image_path)
    detector = build_detector(config)

    started = time.perf_counter()
    detections_map = detector.detect_many([selected_image])
    elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
    detections = detections_map.get(selected_image.resolve(), detections_map.get(selected_image, []))
    report = {
        "status": "ok",
        "detector": config.yolo_detector,
        "image_path": str(selected_image),
        "elapsed_ms": elapsed_ms,
        "detection_count": len(detections),
        "detections": [item.to_dict() for item in detections],
    }
    write_json(config.probes_dir / "detector_probe.json", report)
    return report
