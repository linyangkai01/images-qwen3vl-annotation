from __future__ import annotations

import json

from traffic_cv_closed_loop.config import ProjectConfig
from traffic_cv_closed_loop.models import CandidateFrame


def render_annotation_prompt(
    config: ProjectConfig,
    candidate: CandidateFrame,
    template: str,
) -> str:
    prior_detections = json.dumps(
        [item.to_dict() for item in candidate.detections],
        ensure_ascii=False,
        indent=2,
    )
    return template.format(
        scene=config.scene,
        labels="、".join(config.labels),
        candidate_path=candidate.relative_path,
        prior_detections=prior_detections,
    )
