"""Stage 3 — dual-channel annotation (channel A objects + channel B hazards).

Per keyframe: scene routing -> compressed knowledge brief -> combined (or split)
Qwen call -> robust JSON parsing -> local back-fill -> channel split ->
consistency check -> dual persistence -> review queue. Supports ``qwen_concurrency``
and breakpoint resume (already-completed frames are skipped).
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from hazard_annotation.knowledge import KnowledgeBase, KnowledgeCompressor, SceneRouter
from hazard_annotation.models import DualChannelRecord, HazardReport, KeyframeRecord, severity_rank
from traffic_cv_closed_loop.adapters.qwen import _AnnotationParseError, build_qwen_annotator
from traffic_cv_closed_loop.config import ProjectConfig
from traffic_cv_closed_loop.io_utils import (
    ensure_dir,
    read_jsonl,
    read_text,
    write_json,
    write_jsonl,
)
from traffic_cv_closed_loop.pipeline.annotation import _compare_detections

_PROMPT_DIR = Path(__file__).resolve().parents[1] / "prompts"

#: Hazard ids whose report implies a PPE item should be *absent* on the frame.
_PPE_CONFLICTS: dict[str, str] = {
    "D-5-1": "helmet",
    "G-2-1": "helmet",
    "G-2-2": "harness",
    "G-2-3": "harness",
}


def _resolve_prompt(config: ProjectConfig, filename: str, override_path: Any = None) -> str:
    if override_path:
        path = Path(str(override_path))
        if path.exists():
            text = read_text(path)
            if text.strip():
                return text
    project_prompt = config.project_dir / "assets" / "prompts" / filename
    if project_prompt.exists():
        text = read_text(project_prompt)
        if text.strip():
            return text
    return read_text(_PROMPT_DIR / filename)


def _render_prompt(
    template: str,
    config: ProjectConfig,
    record: KeyframeRecord,
    kb_brief: str,
    scene_ids: list[str],
) -> str:
    prior = json.dumps([item.to_dict() for item in record.detections], ensure_ascii=False, indent=2)
    replacements = {
        "{scene}": config.scene,
        "{labels}": "、".join(config.labels),
        "{candidate_path}": record.relative_path,
        "{prior_detections}": prior,
        "{kb_brief}": kb_brief,
        "{scene_ids}": "、".join(scene_ids) if scene_ids else "未确定",
    }
    text = template
    for key, value in replacements.items():
        text = text.replace(key, value)
    return text


def _consistency_check(
    objects: list[Any],
    report: HazardReport,
) -> list[str]:
    reasons: list[str] = []
    labels = {getattr(item, "label", "") for item in objects}
    reported_ids = {hazard.id for hazard in report.hazards}
    for hazard_id, expected_label in _PPE_CONFLICTS.items():
        if hazard_id in reported_ids and expected_label in labels:
            reasons.append(f"consistency_{expected_label}_conflict")
    return reasons


def _process_frame(
    config: ProjectConfig,
    compressor: KnowledgeCompressor,
    router: SceneRouter,
    annotator: Any,
    templates: dict[str, str],
    video_scene_ids: list[str],
    candidate: KeyframeRecord,
) -> DualChannelRecord:
    relative_path = Path(candidate.relative_path)
    label_path = config.review_labels_dir / relative_path.with_suffix(".txt")
    hazard_path = config.hazards_dir / relative_path.with_suffix(".json")
    raw_response_path = config.review_raw_responses_dir / relative_path.with_suffix(".json")

    scene_ids = video_scene_ids or router.route(candidate.detections)
    kb_brief = compressor.build_brief(scene_ids, config.knowledge.kb_max_chars)

    review_reasons: list[str] = []
    raw_payload: dict[str, Any] = {}
    objects: list[Any] = []
    hazard_dict: dict[str, Any] = {}

    try:
        if config.qwen_combined_call:
            prompt = _render_prompt(templates["combined"], config, candidate, kb_brief, scene_ids)
            objects, hazard_dict, raw_payload = annotator.annotate_combined(candidate, prompt)
        else:
            objects_prompt = _render_prompt(templates["objects"], config, candidate, kb_brief, scene_ids)
            objects, objects_payload = annotator.annotate(candidate, objects_prompt)
            hazard_prompt = _render_prompt(templates["hazard"], config, candidate, kb_brief, scene_ids)
            hazard_dict, hazard_payload = annotator.annotate_hazard(candidate, hazard_prompt)
            raw_payload = {
                "provider": config.qwen_provider,
                "combined_call": False,
                "objects_call": objects_payload,
                "hazard_call": hazard_payload,
            }
    except _AnnotationParseError as error:
        objects = []
        hazard_dict = {}
        raw_payload = error.raw_payload
        review_reasons.append("json_parse_failed")
    except Exception as error:  # noqa: BLE001 - we must never silently drop a frame
        objects = list(candidate.detections)
        hazard_dict = {}
        raw_payload = {"provider": config.qwen_provider, "error": str(error)}
        review_reasons.append("annotation_error")

    report = HazardReport.from_dict(hazard_dict)
    if not report.scene_type and scene_ids:
        report.scene_type = compressor.kb.scene_name(scene_ids[0])
    report.provider = config.qwen_provider
    compressor.backfill(report)

    invalid_ids = [hazard.id for hazard in report.hazards if hazard.invalid_id]
    if invalid_ids:
        review_reasons.append("invalid_id")

    diff = _compare_detections(candidate.detections, objects)
    if diff["added"] > 0:
        review_reasons.append("qwen_added_objects")
    if diff["removed"] > 0:
        review_reasons.append("qwen_removed_objects")
    if diff["adjusted"] > 0:
        review_reasons.append("bbox_adjusted")
    if any(item.confidence < config.review_confidence_threshold for item in objects):
        review_reasons.append("low_confidence")
    review_reasons.extend(_consistency_check(objects, report))

    label_lines = [
        item.to_yolo_line(config.label_to_index)
        for item in objects
        if item.label in config.label_to_index
    ]
    hazard_payload = report.to_dict()
    hazard_payload.update(
        {
            "relative_path": candidate.relative_path,
            "timestamp": round(float(candidate.timestamp), 3),
            "scene_ids": scene_ids,
            "keyframe_path": candidate.extracted_path,
            "provider": config.qwen_provider,
        }
    )

    ensure_dir(label_path.parent)
    ensure_dir(hazard_path.parent)
    ensure_dir(raw_response_path.parent)
    label_path.write_text("\n".join(label_lines), encoding="utf-8", newline="\n")
    write_json(hazard_path, hazard_payload)
    write_json(raw_response_path, raw_payload)

    review_required = bool(review_reasons)
    record = DualChannelRecord(
        keyframe=candidate,
        objects=list(objects),
        hazard_report=report,
        diff=diff,
        review_required=review_required,
        review_reasons=sorted(set(review_reasons)),
        label_path=str(label_path.resolve()),
        hazard_path=str(hazard_path.resolve()),
        raw_response_path=str(raw_response_path.resolve()),
        dataset_ready=(not review_required) or config.include_review_queue_in_training,
    )
    return record


def _completeness(records: list[DualChannelRecord]) -> float:
    total = 0
    complete = 0
    required = ("id", "category", "severity", "norm_reference", "action")
    for record in records:
        for hazard in record.hazard_report.hazards:
            total += 1
            payload = hazard.to_dict()
            if all(str(payload.get(field, "")).strip() for field in required):
                complete += 1
    if total == 0:
        return 1.0
    return round(complete / total, 4)


def run_annotation(config: ProjectConfig) -> dict[str, object]:
    candidates = [KeyframeRecord.from_dict(item) for item in read_jsonl(config.candidates_path)]

    kb = KnowledgeBase.load(config.kb_path)
    compressor = KnowledgeCompressor(
        kb,
        include_ten_no_lift=config.knowledge.include_ten_no_lift,
        include_crane_six_forbidden=config.knowledge.include_crane_six_forbidden,
    )
    router = SceneRouter(mode=config.knowledge.scene_routing)
    annotator = build_qwen_annotator(config)

    templates = {
        "combined": _resolve_prompt(config, "combined_zh.md", config.prompt_template_path),
        "objects": _resolve_prompt(config, "objects_only_zh.md"),
        "hazard": _resolve_prompt(config, "hazard_only_zh.md"),
    }

    ensure_dir(config.review_labels_dir)
    ensure_dir(config.hazards_dir)
    ensure_dir(config.review_raw_responses_dir)
    ensure_dir(config.status_dir)

    # Video-level routing (first N frames, fixed camera) only makes sense for a
    # single source; with multiple image-directory sources we route per frame.
    source_ids = {item.source_id for item in candidates}
    if len(source_ids) == 1 and candidates:
        video_scene_ids = router.route_by_video([item.detections for item in candidates[:5]])
    else:
        video_scene_ids = []

    existing = {
        record.keyframe.relative_path: record
        for record in (DualChannelRecord.from_dict(item) for item in read_jsonl(config.annotations_path))
    }

    pending: list[KeyframeRecord] = []
    resumed: list[DualChannelRecord] = []
    for candidate in candidates:
        previous = existing.get(candidate.relative_path)
        if (
            previous is not None
            and Path(previous.label_path).exists()
            and Path(previous.hazard_path).exists()
        ):
            resumed.append(previous)
        else:
            pending.append(candidate)

    def worker(candidate: KeyframeRecord) -> DualChannelRecord:
        return _process_frame(config, compressor, router, annotator, templates, video_scene_ids, candidate)

    concurrency = max(1, int(config.qwen_concurrency))
    if concurrency > 1 and len(pending) > 1:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            processed = list(executor.map(worker, pending))
    else:
        processed = [worker(candidate) for candidate in pending]

    order = {candidate.relative_path: index for index, candidate in enumerate(candidates)}
    records = resumed + processed
    records.sort(key=lambda record: order.get(record.keyframe.relative_path, 0))

    review_queue = [record for record in records if record.review_required]
    invalid_id_count = sum(
        1 for record in records for hazard in record.hazard_report.hazards if hazard.invalid_id
    )
    annotation_errors = sum(
        1
        for record in records
        if "annotation_error" in record.review_reasons or "json_parse_failed" in record.review_reasons
    )

    write_jsonl(config.annotations_path, [record.to_dict() for record in records])
    write_jsonl(config.review_queue_path, [record.to_dict() for record in review_queue])

    summary = {
        "project_name": config.name,
        "mode": config.run_mode,
        "qwen_provider": config.qwen_provider,
        "combined_call": config.qwen_combined_call,
        "scene_routing": config.knowledge.scene_routing,
        "video_scene_ids": video_scene_ids,
        "kb_total": kb.total,
        "annotated_frames": len(records),
        "resumed_frames": len(resumed),
        "auto_accepted": sum(1 for record in records if not record.review_required),
        "review_queue_count": len(review_queue),
        "annotation_errors": annotation_errors,
        "invalid_id_count": invalid_id_count,
        "required_field_completeness": _completeness(records),
        "overall_risk": _overall_risk(records),
        "annotations_path": str(config.annotations_path),
        "review_queue_path": str(config.review_queue_path),
    }
    write_json(config.annotation_summary_path, summary)
    write_json(config.status_dir / "annotate.json", summary)
    return summary


def _overall_risk(records: list[DualChannelRecord]) -> str:
    best = "none"
    for record in records:
        for hazard in record.hazard_report.hazards:
            if severity_rank(hazard.severity) > severity_rank(best):
                best = hazard.severity
    return best
