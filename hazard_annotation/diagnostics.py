"""Diagnostics for the construction-hazard chain (``doctor`` + probes).

``build_doctor_report`` wraps the traffic doctor and augments it with the three
construction-specific sections required by the architecture:

* ``video``     — configured path / existence / ffmpeg availability.
* ``keyframe``  — six-step keyframe configuration and scene-detect backend.
* ``knowledge`` — knowledge-base existence, hazard count and scene ids.

``probe_qwen_service`` sends a single combined (objects + hazard) request through
the configured provider and reports connectivity, model name and per-frame
latency, so a 4090D deployment can be validated before a full run.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from hazard_annotation.ffmpeg import FfmpegRunner
from hazard_annotation.knowledge import KnowledgeBase, KnowledgeCompressor
from hazard_annotation.models import HazardReport
from traffic_cv_closed_loop.adapters.qwen import build_qwen_annotator
from traffic_cv_closed_loop.config import ProjectConfig
from traffic_cv_closed_loop.diagnostics import build_doctor_report as _traffic_doctor_report
from traffic_cv_closed_loop.diagnostics import probe_detector as _traffic_probe_detector
from traffic_cv_closed_loop.io_utils import iter_image_files, read_text, write_json
from traffic_cv_closed_loop.models import CandidateFrame

_PROMPT_DIR = Path(__file__).resolve().parent / "prompts"
_FALLBACK_PROMPT = "请标注图中施工安全隐患，仅返回 JSON。"


def _video_section(config: ProjectConfig) -> tuple[dict[str, Any], list[str], list[str]]:
    issues: list[str] = []
    warnings: list[str] = []
    runner = FfmpegRunner(config.video.ffmpeg_binary)
    ffmpeg_available = runner.available()
    ffmpeg_version = runner.version() if ffmpeg_available else None
    ffmpeg_version_text = ".".join(str(part) for part in ffmpeg_version) if ffmpeg_version else ""
    ffmpeg_min_version_ok = runner.check_min_version() if ffmpeg_available else False
    video_path = config.video_path
    configured = video_path is not None
    exists = bool(video_path and video_path.exists())

    section = {
        "configured": configured,
        "video_path": str(video_path) if video_path else "",
        "video_exists": exists,
        "decode_backend": config.video.decode_backend,
        "sample_fps": config.video.sample_fps,
        "scale_width": config.video.scale_width,
        "analysis_width": config.video.analysis_width,
        "ffmpeg_binary": config.video.ffmpeg_binary,
        "ffmpeg_available": ffmpeg_available,
        "ffmpeg_version": ffmpeg_version_text,
        "ffmpeg_min_version_ok": ffmpeg_min_version_ok,
    }
    if configured and not exists:
        issues.append("video_path_missing")
    if configured and not ffmpeg_available:
        issues.append("ffmpeg_missing")
    if not configured and not ffmpeg_available:
        warnings.append("ffmpeg_missing_image_dir_mode_only")
    # Version problems are a *warning*, never an issue: an outdated/unparseable
    # ffmpeg breaks the video path (``-vsync`` was removed in ffmpeg 9.x and its
    # replacement ``-fps_mode`` needs >= 5.1) but must not fail `doctor` overall.
    if ffmpeg_available:
        if ffmpeg_version is None:
            warnings.append(
                "ffmpeg_version_unparseable: could not parse `ffmpeg -version`; "
                "`-fps_mode passthrough` requires ffmpeg >= 5.1 (the legacy "
                "`-vsync` option was removed in ffmpeg 9.x)"
            )
        elif not ffmpeg_min_version_ok:
            warnings.append(
                f"ffmpeg_version_too_old:{ffmpeg_version_text} < 5.1; "
                "`-fps_mode passthrough` requires ffmpeg >= 5.1 (the legacy "
                "`-vsync` option was removed in ffmpeg 9.x)"
            )
    return section, issues, warnings


def _keyframe_section(
    config: ProjectConfig, video_configured: bool, ffmpeg_available: bool
) -> tuple[dict[str, Any], list[str], list[str]]:
    issues: list[str] = []
    warnings: list[str] = []
    keyframe = config.keyframe
    section = {
        "enable": keyframe.enable,
        "motion_min_ratio": keyframe.motion_min_ratio,
        "pixel_delta_threshold": keyframe.pixel_delta_threshold,
        "dedup_hamming_threshold": keyframe.dedup_hamming_threshold,
        "scene_change_threshold": keyframe.scene_change_threshold,
        "scene_detect_mode": keyframe.scene_detect_mode,
        "max_keyframes": keyframe.max_keyframes,
        "min_gap_seconds": keyframe.min_gap_seconds,
        "force_keep_labels": list(keyframe.force_keep_labels),
    }
    if (
        keyframe.enable
        and keyframe.scene_detect_mode == "ffmpeg"
        and not ffmpeg_available
    ):
        if video_configured:
            issues.append("keyframe_scene_detect_needs_ffmpeg")
        else:
            warnings.append("keyframe_scene_detect_falls_back_to_histogram")
    return section, issues, warnings


def _knowledge_section(config: ProjectConfig) -> tuple[dict[str, Any], list[str], list[str]]:
    issues: list[str] = []
    warnings: list[str] = []
    kb_path = config.kb_path
    exists = kb_path.exists()
    section: dict[str, Any] = {
        "kb_path": str(kb_path),
        "exists": exists,
        "kb_mode": config.knowledge.kb_mode,
        "kb_max_chars": config.knowledge.kb_max_chars,
        "scene_routing": config.knowledge.scene_routing,
        "hazard_count": 0,
        "scene_ids": [],
        "meta_hazard_count": None,
        "meta_count_stale": False,
    }
    if not exists:
        issues.append("knowledge_base_missing")
        return section, issues, warnings

    try:
        kb = KnowledgeBase.load(kb_path)
    except (ValueError, KeyError) as error:
        issues.append(f"knowledge_base_invalid:{error}")
        return section, issues, warnings

    section["hazard_count"] = kb.total
    section["scene_ids"] = kb.scene_ids()
    meta_count = kb.meta.get("hazard_count") or kb.meta.get("total_hazards")
    section["meta_hazard_count"] = meta_count
    if kb.total == 0:
        issues.append("knowledge_base_empty")
    if meta_count is not None:
        try:
            if int(meta_count) != kb.total:
                section["meta_count_stale"] = True
                warnings.append(
                    f"knowledge_base_meta_count_stale:meta={meta_count},actual={kb.total}"
                )
        except (TypeError, ValueError):
            warnings.append("knowledge_base_meta_count_unreadable")
    return section, issues, warnings


def build_doctor_report(config: ProjectConfig) -> dict[str, Any]:
    """Return an augmented doctor report covering the hazard-annotation chain."""
    base = _traffic_doctor_report(config)
    issues = list(base.get("issues", []))
    warnings: list[str] = []

    video_section, video_issues, video_warnings = _video_section(config)
    keyframe_section, keyframe_issues, keyframe_warnings = _keyframe_section(
        config,
        video_configured=bool(video_section["configured"]),
        ffmpeg_available=bool(video_section["ffmpeg_available"]),
    )
    knowledge_section, knowledge_issues, knowledge_warnings = _knowledge_section(config)

    issues.extend(video_issues)
    issues.extend(keyframe_issues)
    issues.extend(knowledge_issues)
    warnings.extend(video_warnings)
    warnings.extend(keyframe_warnings)
    warnings.extend(knowledge_warnings)

    # In mock mode the synthetic frames/sidecars are generated on demand, so a
    # missing input directory is expected rather than a blocking issue.
    if config.qwen_provider.lower() == "mock":
        issues = [
            item
            for item in issues
            if item not in {"input_frames_dir_missing", "input_images_missing"}
        ]
        warnings.append("mock_mode_assets_autogenerated")

    base.update(
        {
            "project_name": config.name,
            "mode": config.run_mode,
            "video": video_section,
            "keyframe": keyframe_section,
            "knowledge": knowledge_section,
            "warnings": warnings,
            "issues": issues,
            "status": "ready" if not issues else "issues_found",
        }
    )
    return base


def _resolve_combined_prompt(config: ProjectConfig) -> str:
    for candidate in (config.prompt_template_path, _PROMPT_DIR / "combined_zh.md"):
        if candidate and Path(candidate).exists():
            text = read_text(Path(candidate))
            if text.strip():
                return text
    return _FALLBACK_PROMPT


def _first_probe_image(config: ProjectConfig, image_path: str | None = None) -> Path:
    if image_path:
        path = Path(image_path)
        if not path.is_absolute():
            path = (config.project_dir / path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Probe image not found: {path}")
        return path

    images = iter_image_files(config.input_frames_dir)
    if not images and config.qwen_provider.lower() == "mock":
        from hazard_annotation.demo import build_mock_assets

        build_mock_assets(config.project_dir)
        images = iter_image_files(config.input_frames_dir)
    if not images:
        raise FileNotFoundError(
            f"No images found under {config.input_frames_dir}. "
            "Run `python -m hazard_annotation demo` or configure a video source."
        )
    return images[0]


def _render_probe_prompt(config: ProjectConfig, template: str, kb_brief: str, relative_path: str) -> str:
    replacements = {
        "{scene}": config.scene,
        "{labels}": "、".join(config.labels),
        "{candidate_path}": relative_path,
        "{prior_detections}": "[]",
        "{kb_brief}": kb_brief,
        "{scene_ids}": "未确定",
    }
    text = template
    for key, value in replacements.items():
        text = text.replace(key, value)
    return text


def probe_qwen_service(config: ProjectConfig, image_path: str | None = None) -> dict[str, Any]:
    """Send one combined hazard request and report connectivity + latency."""
    selected_image = _first_probe_image(config, image_path=image_path)
    template = _resolve_combined_prompt(config)
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

    kb = KnowledgeBase.load(config.kb_path)
    compressor = KnowledgeCompressor(
        kb,
        include_ten_no_lift=config.knowledge.include_ten_no_lift,
        include_crane_six_forbidden=config.knowledge.include_crane_six_forbidden,
    )
    kb_brief = compressor.build_brief([], config.knowledge.kb_max_chars)
    prompt = _render_probe_prompt(config, template, kb_brief, relative_path)
    annotator = build_qwen_annotator(config)

    started = time.perf_counter()
    try:
        objects, hazard_dict, payload = annotator.annotate_combined(candidate, prompt)
    except Exception as error:  # noqa: BLE001 - surface connectivity failures as data
        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        report = {
            "status": "error",
            "connectivity": "failed",
            "provider": config.qwen_provider,
            "model": config.qwen_model,
            "chat_endpoint": config.qwen_chat_endpoint,
            "image_path": str(selected_image),
            "elapsed_ms": elapsed_ms,
            "error": str(error),
        }
        write_json(config.probes_dir / "hazard_probe.json", report)
        return report

    elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
    hazard_report = HazardReport.from_dict(hazard_dict)
    compressor.backfill(hazard_report)
    report = {
        "status": "ok",
        "connectivity": "ok",
        "provider": config.qwen_provider,
        "model": config.qwen_model,
        "chat_endpoint": config.qwen_chat_endpoint,
        "image_path": str(selected_image),
        "elapsed_ms": elapsed_ms,
        "object_count": len(objects),
        "labels": [item.label for item in objects],
        "hazard_ids": [hazard.id for hazard in hazard_report.hazards],
        "invalid_ids": [hazard.id for hazard in hazard_report.hazards if hazard.invalid_id],
        "overall_risk": hazard_report.max_severity,
        "response": payload,
    }
    write_json(config.probes_dir / "hazard_probe.json", report)
    return report


def probe_detector(config: ProjectConfig, image_path: str | None = None) -> dict[str, Any]:
    """Delegate to the traffic detector probe (channel A shares the detector)."""
    return _traffic_probe_detector(config, image_path=image_path)
