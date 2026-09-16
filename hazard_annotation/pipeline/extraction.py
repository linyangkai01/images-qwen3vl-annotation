"""Stage 1 — keyframe extraction.

``run_extraction`` prefers the configured video and falls back to an image
directory when no video is configured (or when ffmpeg is unavailable but the
directory contains frames). Output: ``extracted/candidates.jsonl`` and
``extracted/summary.json``.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from hazard_annotation.inference import (
    check_video_duration,
    render_duration_notice,
    split_duration_segments,
)
from hazard_annotation.keyframes import KeyframeExtractor, KeyframeSource
from traffic_cv_closed_loop.adapters.yolo import build_detector
from traffic_cv_closed_loop.config import ProjectConfig
from traffic_cv_closed_loop.io_utils import ensure_dir, iter_image_files, write_json, write_jsonl


def _discover_sources(config: ProjectConfig) -> tuple[list[KeyframeSource], list[str]]:
    """Return the list of inputs plus a list of non-fatal warnings."""
    warnings: list[str] = []
    video_path = config.video_path
    if video_path is not None:
        if video_path.exists():
            return [KeyframeSource(source_id=video_path.stem, video_path=video_path)], warnings
        warnings.append(f"video_path_not_found:{video_path}")

    grouped: dict[Path, list[Path]] = defaultdict(list)
    for image_path in iter_image_files(config.input_frames_dir):
        grouped[image_path.parent].append(image_path)

    sources = [
        KeyframeSource(source_id=parent.name, frames_dir=parent)
        for parent in sorted(grouped)
    ]
    if not sources:
        warnings.append(f"no_images_found:{config.input_frames_dir}")
    return sources, warnings


def run_extraction(config: ProjectConfig) -> dict[str, object]:
    ensure_dir(config.extracted_frames_dir)
    ensure_dir(config.status_dir)

    detector = build_detector(config)
    extractor = KeyframeExtractor(config, detector=detector)
    sources, warnings = _discover_sources(config)

    # --- 视频时长校验（私有多模态模型算力限制） -------------------------- #
    duration_report = None
    for source in sources:
        if source.video_path is None:
            continue
        report = check_video_duration(source.video_path, config)
        print(render_duration_notice(report))
        warnings.extend(f"{source.source_id}:{code}" for code in report.warnings)
        if report.action == "segment":
            segments = split_duration_segments(
                report.duration_seconds, config.inference_segment_seconds
            )
            print(
                f"          建议分段（共 {len(segments)} 段）："
                + "、".join(f"{s:.0f}-{e:.0f}s" for s, e in segments[:5])
                + (" …" if len(segments) > 5 else "")
            )
        duration_report = report

    records = []
    per_source: dict[str, dict[str, int]] = {}
    for source in sources:
        source_records = extractor.extract(source)
        per_source[source.source_id] = dict(extractor.counts)
        warnings.extend(
            f"{source.source_id}:{message}" for message in extractor.warnings
        )
        records.extend(source_records)

    write_jsonl(config.candidates_path, [record.to_dict() for record in records])

    totals = {
        "total_frames": sum(item.get("total_frames", 0) for item in per_source.values()),
        "sampled_frames": sum(item.get("sampled_frames", 0) for item in per_source.values()),
        "motion_removed": sum(item.get("motion_removed", 0) for item in per_source.values()),
        "dedup_removed": sum(item.get("dedup_removed", 0) for item in per_source.values()),
        "scene_cuts": sum(item.get("scene_cuts", 0) for item in per_source.values()),
        "force_kept": sum(item.get("force_kept", 0) for item in per_source.values()),
        "protected_dropped": sum(item.get("protected_dropped", 0) for item in per_source.values()),
        "display_missing": sum(item.get("display_missing", 0) for item in per_source.values()),
        "final_keyframes": len(records),
    }
    summary = {
        "project_name": config.name,
        "mode": config.run_mode,
        "video_path": config.video.video_path,
        "input_frames_dir": str(config.input_frames_dir),
        "detector_mode": config.yolo_detector,
        "inference_backend": config.inference_backend,
        "video_duration": duration_report.to_dict() if duration_report else None,
        "source_count": len(sources),
        "per_source": per_source,
        "totals": totals,
        "warnings": warnings,
        "candidates_path": str(config.candidates_path),
    }
    write_json(config.extraction_summary_path, summary)
    write_json(config.status_dir / "extract.json", summary)
    return summary
