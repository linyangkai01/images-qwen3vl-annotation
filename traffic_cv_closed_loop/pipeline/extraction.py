from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from traffic_cv_closed_loop.adapters.yolo import build_detector
from traffic_cv_closed_loop.config import ProjectConfig
from traffic_cv_closed_loop.io_utils import copy_file, ensure_dir, iter_image_files, write_json, write_jsonl
from traffic_cv_closed_loop.models import CandidateFrame


def run_extraction(config: ProjectConfig) -> dict[str, object]:
    ensure_dir(config.extracted_frames_dir)
    detector = build_detector(config)
    image_files = iter_image_files(config.input_frames_dir)

    grouped: dict[Path, list[Path]] = defaultdict(list)
    for image_path in image_files:
        grouped[image_path.parent].append(image_path)

    candidates: list[CandidateFrame] = []
    skipped_empty = 0
    sampled_total = 0

    for parent_dir in sorted(grouped):
        selected_images: list[tuple[int, Path]] = []
        sampled_count = 0
        for index, image_path in enumerate(sorted(grouped[parent_dir])):
            if index % max(1, config.frame_step) != 0:
                continue
            if sampled_count >= config.max_frames_per_source:
                break
            sampled_count += 1
            sampled_total += 1
            selected_images.append((index, image_path))

        detections_map = detector.detect_many([item[1] for item in selected_images])

        for index, image_path in selected_images:
            detections = detections_map.get(image_path.resolve(), [])
            has_valid_detection = any(
                item.confidence >= config.min_detection_confidence for item in detections
            )
            if not has_valid_detection and not config.include_empty_frames:
                skipped_empty += 1
                continue

            relative_path = image_path.relative_to(config.input_frames_dir)
            extracted_path = config.extracted_frames_dir / relative_path
            copy_file(image_path, extracted_path)

            candidate = CandidateFrame(
                source_id=parent_dir.name,
                relative_path=str(relative_path),
                original_path=str(image_path.resolve()),
                extracted_path=str(extracted_path.resolve()),
                detections=detections,
                scene=config.scene,
                metadata={
                    "source_directory": str(parent_dir.resolve()),
                    "sequence_index": index,
                    "detector_mode": config.yolo_detector,
                },
            )
            candidates.append(candidate)

    write_jsonl(config.candidates_path, [item.to_dict() for item in candidates])
    summary = {
        "project_name": config.name,
        "input_frames_dir": str(config.input_frames_dir),
        "detector_mode": config.yolo_detector,
        "total_images_found": len(image_files),
        "total_images_sampled": sampled_total,
        "selected_candidates": len(candidates),
        "skipped_empty": skipped_empty,
        "candidates_path": str(config.candidates_path),
    }
    write_json(config.extraction_summary_path, summary)
    write_json(config.status_dir / "extract.json", summary)
    return summary
