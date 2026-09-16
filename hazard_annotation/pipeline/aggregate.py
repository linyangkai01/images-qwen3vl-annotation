"""Stage 4a — aggregate per-frame hazards along the timeline.

Produces ``hazards/report.json`` with per-id aggregation, persistent hazards,
severity distribution, the overall risk level and a timeline.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from hazard_annotation.models import DualChannelRecord, severity_rank
from traffic_cv_closed_loop.config import ProjectConfig
from traffic_cv_closed_loop.io_utils import read_json, read_jsonl, write_json


def _empty_entry(hazard: Any, record: DualChannelRecord) -> dict[str, Any]:
    return {
        "id": hazard.id,
        "name": hazard.description or hazard.id,
        "category": hazard.category,
        "severity": hazard.severity,
        "norm_reference": hazard.norm_reference,
        "action": hazard.action,
        "count": 0,
        "first_frame": record.keyframe.relative_path,
        "last_frame": record.keyframe.relative_path,
        "first_timestamp": round(float(record.keyframe.timestamp), 3),
        "last_timestamp": round(float(record.keyframe.timestamp), 3),
        "frames": [],
        "evidence": [],
    }


def run_aggregate(config: ProjectConfig) -> dict[str, Any]:
    records = [DualChannelRecord.from_dict(item) for item in read_jsonl(config.annotations_path)]

    aggregated: dict[str, dict[str, Any]] = {}
    severity_counts: Counter[str] = Counter()
    scene_types: Counter[str] = Counter()
    timeline: list[dict[str, Any]] = []

    for record in records:
        hazard_ids: list[str] = []
        for hazard in record.hazard_report.hazards:
            hazard_ids.append(hazard.id)
            severity_counts[hazard.severity] += 1
            entry = aggregated.get(hazard.id)
            if entry is None:
                entry = _empty_entry(hazard, record)
                aggregated[hazard.id] = entry
            entry["count"] += 1
            entry["last_frame"] = record.keyframe.relative_path
            entry["last_timestamp"] = round(float(record.keyframe.timestamp), 3)
            if len(entry["frames"]) < 50:
                entry["frames"].append(record.keyframe.relative_path)
            if hazard.evidence and len(entry["evidence"]) < 5:
                entry["evidence"].append(hazard.evidence)
            if severity_rank(hazard.severity) > severity_rank(entry["severity"]):
                entry["severity"] = hazard.severity
        if record.hazard_report.scene_type:
            scene_types[record.hazard_report.scene_type] += 1
        timeline.append(
            {
                "relative_path": record.keyframe.relative_path,
                "timestamp": round(float(record.keyframe.timestamp), 3),
                "scene_type": record.hazard_report.scene_type,
                "overall_risk": record.hazard_report.overall_risk,
                "hazard_ids": hazard_ids,
                "keyframe_path": record.keyframe.extracted_path,
            }
        )

    overall_risk = "none"
    for entry in aggregated.values():
        if severity_rank(entry["severity"]) > severity_rank(overall_risk):
            overall_risk = entry["severity"]

    persistent = [
        entry for entry in aggregated.values() if entry["count"] >= 2
    ]

    extraction_summary = read_json(config.extraction_summary_path, default={}) or {}
    totals = extraction_summary.get("totals", {}) if isinstance(extraction_summary, dict) else {}
    total_frames = int(totals.get("total_frames", 0) or 0)
    keyframe_count = len(records)
    compression_ratio = round(total_frames / keyframe_count, 2) if keyframe_count and total_frames else 0.0

    report = {
        "project_name": config.name,
        "mode": config.run_mode,
        "provider": config.qwen_provider,
        "keyframe_count": keyframe_count,
        "overall_risk": overall_risk,
        "severity_counts": dict(severity_counts),
        "scene_types": dict(scene_types),
        "hazard_counts": {entry["id"]: entry["count"] for entry in aggregated.values()},
        "hazards": sorted(aggregated.values(), key=lambda item: (-item["count"], item["id"])),
        "persistent_hazards": sorted(
            [entry["id"] for entry in persistent], key=lambda value: value
        ),
        "timeline": timeline,
        "compression": {
            "total_frames": total_frames,
            "keyframe_count": keyframe_count,
            "compression_ratio": compression_ratio,
        },
    }

    write_json(config.hazard_report_path, report)
    write_json(config.status_dir / "aggregate.json", report)
    return report
