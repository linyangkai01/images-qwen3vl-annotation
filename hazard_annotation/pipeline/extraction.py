"""Stage 1 — keyframe extraction.

``run_extraction`` prefers the configured video and falls back to an image
directory when no video is configured (or when ffmpeg is unavailable but the
directory contains frames). Output: ``extracted/candidates.jsonl`` and
``extracted/summary.json``.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from hazard_annotation.checkpoint import ExtractionCheckpoint
from hazard_annotation.inference import (
    VideoDurationRejected,
    check_video_duration,
    format_segment_intervals,
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


def run_extraction(config: ProjectConfig, resume: bool = True) -> dict[str, object]:
    ensure_dir(config.extracted_frames_dir)
    ensure_dir(config.status_dir)

    detector = build_detector(config)
    extractor = KeyframeExtractor(config, detector=detector)
    sources, warnings = _discover_sources(config)

    # --- 断点续跑：按 source 粒度复用已抽好的结果 ---
    checkpoint = ExtractionCheckpoint(config, enabled=resume)
    checkpoint.load()
    if not resume:
        print("[断点续跑] 已关闭（--no-resume），本次全量重算")
    elif checkpoint.invalidation_reason:
        print(
            f"[断点续跑] 断点已作废（{checkpoint.invalidation_reason}），本次全量重算"
        )

    # --- 视频时长校验（私有多模态模型算力限制） -------------------------- #
    duration_report = None
    for source in sources:
        if source.video_path is None:
            continue
        try:
            report = check_video_duration(source.video_path, config)
        except VideoDurationRejected as rejected:
            # 默认策略：拒绝处理该视频并给出人工截取指引。
            # 打印清晰提示后原样抛出，终止本次抽帧（不静默放行、不自动裁剪）。
            print(render_duration_notice(rejected.report))
            raise
        print(render_duration_notice(report))
        warnings.extend(f"{source.source_id}:{code}" for code in report.warnings)
        if report.action == "segment":
            segments = split_duration_segments(
                report.duration_seconds, config.inference_segment_seconds
            )
            print(
                f"          建议分段（共 {len(segments)} 段）："
                + format_segment_intervals(segments)
            )
        duration_report = report

    records = []
    per_source: dict[str, dict[str, int]] = {}
    for source in sources:
        if checkpoint.is_reusable(source):
            source_records = checkpoint.read_records(source.source_id)
            progress = checkpoint.progress_of(source.source_id)
            counts = dict(progress.counts) if progress else {}
            source_warnings = list(progress.warnings) if progress else []
            checkpoint.reused_sources.append(source.source_id)
            print(
                f"[断点续跑] 复用 {source.source_id}："
                f"{len(source_records)} 张关键帧（输入与配置均未变）"
            )
        else:
            source_records = extractor.extract(source)
            counts = dict(extractor.counts)
            source_warnings = list(extractor.warnings)
            checkpoint.mark_done(source, source_records, counts, source_warnings)
            checkpoint.recomputed_sources.append(source.source_id)

        per_source[source.source_id] = counts
        warnings.extend(f"{source.source_id}:{message}" for message in source_warnings)
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
        "checkpoint": checkpoint.summary(),
        "source_count": len(sources),
        "per_source": per_source,
        "totals": totals,
        "warnings": warnings,
        "candidates_path": str(config.candidates_path),
    }
    write_json(config.extraction_summary_path, summary)
    write_json(config.status_dir / "extract.json", summary)
    return summary
