from __future__ import annotations

from pathlib import Path

from traffic_cv_closed_loop.adapters.qwen import build_qwen_annotator
from traffic_cv_closed_loop.config import ProjectConfig
from traffic_cv_closed_loop.io_utils import (
    copy_file,
    ensure_dir,
    read_jsonl,
    read_text,
    write_json,
    write_jsonl,
    write_text,
)
from traffic_cv_closed_loop.models import AnnotationRecord, CandidateFrame, Detection
from traffic_cv_closed_loop.prompting import render_annotation_prompt


def _compare_detections(yolo_detections: list[Detection], qwen_detections: list[Detection]) -> dict[str, int]:
    unmatched_yolo = set(range(len(yolo_detections)))
    unmatched_qwen = set(range(len(qwen_detections)))
    adjusted = 0

    for qwen_index, qwen_item in enumerate(qwen_detections):
        best_match_index = None
        best_iou = 0.0
        for yolo_index in list(unmatched_yolo):
            yolo_item = yolo_detections[yolo_index]
            if yolo_item.label != qwen_item.label:
                continue
            iou = qwen_item.iou(yolo_item)
            if iou > best_iou:
                best_iou = iou
                best_match_index = yolo_index
        if best_match_index is None or best_iou < 0.5:
            continue
        unmatched_yolo.discard(best_match_index)
        unmatched_qwen.discard(qwen_index)
        if best_iou < 0.9:
            adjusted += 1

    return {
        "added": len(unmatched_qwen),
        "removed": len(unmatched_yolo),
        "adjusted": adjusted,
    }


def run_annotation(config: ProjectConfig) -> dict[str, object]:
    candidates = [CandidateFrame.from_dict(item) for item in read_jsonl(config.candidates_path)]
    prompt_template = read_text(config.prompt_template_path)
    if not prompt_template.strip():
        raise FileNotFoundError(f"Prompt template is empty or missing: {config.prompt_template_path}")

    annotator = build_qwen_annotator(config)
    review_image_dir = config.review_dir / "images"
    prompt_dir = config.review_dir / "prompts"
    raw_response_dir = config.review_dir / "raw_responses"
    label_dir = config.review_dir / "labels"

    ensure_dir(review_image_dir)
    ensure_dir(prompt_dir)
    ensure_dir(raw_response_dir)
    ensure_dir(label_dir)

    records: list[AnnotationRecord] = []
    review_queue: list[AnnotationRecord] = []
    annotation_errors = 0

    for candidate in candidates:
        relative_path = Path(candidate.relative_path)
        prompt_path = prompt_dir / relative_path.with_suffix(".md")
        raw_response_path = raw_response_dir / relative_path.with_suffix(".json")
        label_path = label_dir / relative_path.with_suffix(".txt")
        review_image_path = review_image_dir / relative_path

        prompt_text = render_annotation_prompt(config, candidate, prompt_template)
        write_text(prompt_path, prompt_text)
        copy_file(candidate.extracted, review_image_path)

        review_required = False
        review_reasons: list[str] = []
        try:
            qwen_detections, raw_payload = annotator.annotate(candidate, prompt_text)
        except Exception as error:
            qwen_detections = list(candidate.detections)
            raw_payload = {
                "provider": config.qwen_provider,
                "error": str(error),
                "fallback": "reuse_yolo_detections",
            }
            review_required = True
            review_reasons.append("annotation_error")
            annotation_errors += 1

        diff = _compare_detections(candidate.detections, qwen_detections)
        if diff["added"] > 0:
            review_required = True
            review_reasons.append("qwen_added_objects")
        if diff["removed"] > 0:
            review_required = True
            review_reasons.append("qwen_removed_objects")
        if diff["adjusted"] > 0:
            review_required = True
            review_reasons.append("bbox_adjusted")
        if any(item.confidence < config.review_confidence_threshold for item in qwen_detections):
            review_required = True
            review_reasons.append("low_confidence")

        label_lines = [
            item.to_yolo_line(config.label_to_index)
            for item in qwen_detections
            if item.label in config.label_to_index
        ]
        write_text(label_path, "\n".join(label_lines))
        write_json(raw_response_path, raw_payload)

        record = AnnotationRecord(
            candidate=candidate,
            qwen_detections=qwen_detections,
            diff=diff,
            review_required=review_required,
            review_reasons=sorted(set(review_reasons)),
            prompt_path=str(prompt_path.resolve()),
            raw_response_path=str(raw_response_path.resolve()),
            label_path=str(label_path.resolve()),
            dataset_ready=(not review_required) or config.include_review_queue_in_training,
        )
        records.append(record)
        if review_required:
            review_queue.append(record)

    summary = {
        "project_name": config.name,
        "qwen_provider": config.qwen_provider,
        "annotated_candidates": len(records),
        "auto_accepted": sum(1 for item in records if not item.review_required),
        "review_queue_count": len(review_queue),
        "annotation_errors": annotation_errors,
        "annotations_path": str(config.annotations_path),
        "review_queue_path": str(config.review_queue_path),
    }
    write_jsonl(config.annotations_path, [item.to_dict() for item in records])
    write_jsonl(config.review_queue_path, [item.to_dict() for item in review_queue])
    write_json(config.annotation_summary_path, summary)
    write_json(config.status_dir / "annotate.json", summary)
    return summary
