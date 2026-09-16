"""Domain data models for the construction-hazard annotation pipeline.

All classes provide ``to_dict``/``from_dict`` so they can be persisted as JSON
or JSONL. :class:`KeyframeRecord` deliberately keeps the same field names as the
traffic :class:`traffic_cv_closed_loop.models.CandidateFrame` (``source_id`` /
``relative_path`` / ``original_path`` / ``extracted_path`` / ``detections``) so
the existing detector / trainer helpers can consume hazard keyframes directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from traffic_cv_closed_loop.models import CandidateFrame, Detection

HAZARD_SEVERITIES = ("critical", "high", "medium", "low")
OVERALL_RISK_LEVELS = ("critical", "high", "medium", "low", "none")

#: Ordered from most to least severe; used for ``overall_risk`` roll-up.
SEVERITY_ORDER = {"critical": 4, "high": 3, "medium": 2, "low": 1, "none": 0}


def severity_rank(severity: str) -> int:
    return SEVERITY_ORDER.get(str(severity).strip().lower(), 0)


@dataclass(slots=True)
class HazardDef:
    """A single hazard definition normalised from the knowledge base."""

    id: str
    name: str
    category: str = ""
    severity: str = "medium"
    visual_cues: list[str] = field(default_factory=list)
    description: str = ""
    norm_ref: str = ""
    action: str = ""
    scene_id: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HazardDef":
        return cls(
            id=str(data.get("id", "")),
            name=str(data.get("name", "")),
            category=str(data.get("category", "")),
            severity=str(data.get("severity", "medium")),
            visual_cues=[str(item) for item in data.get("visual_cues", [])],
            description=str(data.get("description", "")),
            norm_ref=str(data.get("norm_ref", "")),
            action=str(data.get("action", "")),
            scene_id=str(data.get("scene_id", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "category": self.category,
            "severity": self.severity,
            "visual_cues": list(self.visual_cues),
            "description": self.description,
            "norm_ref": self.norm_ref,
            "action": self.action,
            "scene_id": self.scene_id,
        }


@dataclass(slots=True)
class HazardEvent:
    """A hazard instance detected on a single keyframe (channel B)."""

    id: str
    category: str = ""
    description: str = ""
    confidence: float = 0.0
    evidence: str = ""
    severity: str = "medium"
    norm_reference: str = ""
    action: str = ""
    scene_type: str = ""
    invalid_id: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HazardEvent":
        return cls(
            id=str(data.get("id", "")),
            category=str(data.get("category", "")),
            description=str(data.get("description", "")),
            confidence=float(data.get("confidence", 0.0)),
            evidence=str(data.get("evidence", "")),
            severity=str(data.get("severity", "medium")),
            norm_reference=str(data.get("norm_reference", data.get("norm_ref", ""))),
            action=str(data.get("action", "")),
            scene_type=str(data.get("scene_type", "")),
            invalid_id=bool(data.get("invalid_id", False)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "category": self.category,
            "description": self.description,
            "confidence": round(float(self.confidence), 4),
            "evidence": self.evidence,
            "severity": self.severity,
            "norm_reference": self.norm_reference,
            "action": self.action,
            "scene_type": self.scene_type,
            "invalid_id": self.invalid_id,
        }


@dataclass(slots=True)
class HazardReport:
    """The full hazard report produced for a single keyframe."""

    scene_type: str = ""
    hazards: list[HazardEvent] = field(default_factory=list)
    ten_no_lift_violations: list[str] = field(default_factory=list)
    crane_six_forbidden_violations: list[str] = field(default_factory=list)
    overall_risk: str = "none"
    recommendation: str = ""
    observations: list[str] = field(default_factory=list)
    provider: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HazardReport":
        return cls(
            scene_type=str(data.get("scene_type", "")),
            hazards=[HazardEvent.from_dict(item) for item in data.get("hazards", [])],
            ten_no_lift_violations=[str(item) for item in data.get("ten_no_lift_violations", [])],
            crane_six_forbidden_violations=[
                str(item) for item in data.get("crane_six_forbidden_violations", [])
            ],
            overall_risk=str(data.get("overall_risk", "none")),
            recommendation=str(data.get("recommendation", "")),
            observations=[str(item) for item in data.get("observations", [])],
            provider=str(data.get("provider", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "scene_type": self.scene_type,
            "hazards": [item.to_dict() for item in self.hazards],
            "ten_no_lift_violations": list(self.ten_no_lift_violations),
            "crane_six_forbidden_violations": list(self.crane_six_forbidden_violations),
            "overall_risk": self.overall_risk,
            "recommendation": self.recommendation,
            "observations": list(self.observations),
            "provider": self.provider,
        }

    @property
    def max_severity(self) -> str:
        best = "none"
        for hazard in self.hazards:
            if severity_rank(hazard.severity) > severity_rank(best):
                best = hazard.severity
        return best


@dataclass(slots=True)
class KeyframeRecord:
    """A selected keyframe plus its prior (channel A) detections."""

    source_id: str
    relative_path: str
    original_path: str
    extracted_path: str
    timestamp: float = 0.0
    reason: str = ""
    detections: list[Detection] = field(default_factory=list)
    scene: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "KeyframeRecord":
        return cls(
            source_id=str(data["source_id"]),
            relative_path=str(data["relative_path"]),
            original_path=str(data["original_path"]),
            extracted_path=str(data["extracted_path"]),
            timestamp=float(data.get("timestamp", 0.0)),
            reason=str(data.get("reason", "")),
            detections=[Detection.from_dict(item) for item in data.get("detections", [])],
            scene=str(data.get("scene", "")),
            metadata=dict(data.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "relative_path": self.relative_path,
            "original_path": self.original_path,
            "extracted_path": self.extracted_path,
            "timestamp": round(float(self.timestamp), 3),
            "reason": self.reason,
            "detections": [item.to_dict() for item in self.detections],
            "scene": self.scene,
            "metadata": self.metadata,
        }

    def to_candidate(self) -> CandidateFrame:
        """Return an equivalent traffic :class:`CandidateFrame`."""
        return CandidateFrame(
            source_id=self.source_id,
            relative_path=self.relative_path,
            original_path=self.original_path,
            extracted_path=self.extracted_path,
            detections=list(self.detections),
            scene=self.scene,
            metadata=dict(self.metadata),
        )

    @property
    def original(self) -> Path:
        return Path(self.original_path)

    @property
    def extracted(self) -> Path:
        return Path(self.extracted_path)


@dataclass(slots=True)
class DualChannelRecord:
    """The channel A + channel B result for one keyframe."""

    keyframe: KeyframeRecord
    objects: list[Detection] = field(default_factory=list)
    hazard_report: HazardReport = field(default_factory=HazardReport)
    diff: dict[str, int] = field(default_factory=dict)
    review_required: bool = False
    review_reasons: list[str] = field(default_factory=list)
    label_path: str = ""
    hazard_path: str = ""
    raw_response_path: str = ""
    dataset_ready: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DualChannelRecord":
        return cls(
            keyframe=KeyframeRecord.from_dict(data["keyframe"]),
            objects=[Detection.from_dict(item) for item in data.get("objects", [])],
            hazard_report=HazardReport.from_dict(data.get("hazard_report", {})),
            diff=dict(data.get("diff", {})),
            review_required=bool(data.get("review_required", False)),
            review_reasons=[str(item) for item in data.get("review_reasons", [])],
            label_path=str(data.get("label_path", "")),
            hazard_path=str(data.get("hazard_path", "")),
            raw_response_path=str(data.get("raw_response_path", "")),
            dataset_ready=bool(data.get("dataset_ready", False)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "keyframe": self.keyframe.to_dict(),
            "objects": [item.to_dict() for item in self.objects],
            "hazard_report": self.hazard_report.to_dict(),
            "diff": self.diff,
            "review_required": self.review_required,
            "review_reasons": list(self.review_reasons),
            "label_path": self.label_path,
            "hazard_path": self.hazard_path,
            "raw_response_path": self.raw_response_path,
            "dataset_ready": self.dataset_ready,
        }
