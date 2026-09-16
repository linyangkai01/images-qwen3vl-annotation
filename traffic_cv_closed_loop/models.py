from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, float(value)))


def normalize_bbox(bbox: list[float] | tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    if len(bbox) != 4:
        raise ValueError(f"bbox must contain 4 values, got {bbox}")
    x1, y1, x2, y2 = (_clamp(component) for component in bbox)
    if x2 <= x1:
        x2 = min(1.0, x1 + 0.001)
    if y2 <= y1:
        y2 = min(1.0, y1 + 0.001)
    return (x1, y1, x2, y2)


@dataclass(slots=True)
class Detection:
    label: str
    bbox: tuple[float, float, float, float]
    confidence: float = 1.0
    source: str = ""
    notes: str = ""

    def __post_init__(self) -> None:
        self.bbox = normalize_bbox(self.bbox)
        self.confidence = _clamp(self.confidence)

    @classmethod
    def from_dict(cls, data: dict[str, Any], default_source: str = "") -> "Detection":
        return cls(
            label=str(data["label"]),
            bbox=normalize_bbox(data["bbox"]),
            confidence=float(data.get("confidence", 1.0)),
            source=str(data.get("source", default_source)),
            notes=str(data.get("notes", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "bbox": list(self.bbox),
            "confidence": round(self.confidence, 4),
            "source": self.source,
            "notes": self.notes,
        }

    def to_yolo_line(self, label_to_index: dict[str, int]) -> str:
        x1, y1, x2, y2 = self.bbox
        x_center = (x1 + x2) / 2
        y_center = (y1 + y2) / 2
        width = x2 - x1
        height = y2 - y1
        return (
            f"{label_to_index[self.label]} "
            f"{x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
        )

    def iou(self, other: "Detection") -> float:
        ax1, ay1, ax2, ay2 = self.bbox
        bx1, by1, bx2, by2 = other.bbox
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        inter_width = max(0.0, inter_x2 - inter_x1)
        inter_height = max(0.0, inter_y2 - inter_y1)
        intersection = inter_width * inter_height
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        union = area_a + area_b - intersection
        if union <= 0:
            return 0.0
        return intersection / union


@dataclass(slots=True)
class CandidateFrame:
    source_id: str
    relative_path: str
    original_path: str
    extracted_path: str
    detections: list[Detection] = field(default_factory=list)
    scene: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CandidateFrame":
        return cls(
            source_id=str(data["source_id"]),
            relative_path=str(data["relative_path"]),
            original_path=str(data["original_path"]),
            extracted_path=str(data["extracted_path"]),
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
            "detections": [item.to_dict() for item in self.detections],
            "scene": self.scene,
            "metadata": self.metadata,
        }

    @property
    def original(self) -> Path:
        return Path(self.original_path)

    @property
    def extracted(self) -> Path:
        return Path(self.extracted_path)


@dataclass(slots=True)
class AnnotationRecord:
    candidate: CandidateFrame
    qwen_detections: list[Detection]
    diff: dict[str, int]
    review_required: bool
    review_reasons: list[str]
    prompt_path: str
    raw_response_path: str
    label_path: str
    dataset_ready: bool

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AnnotationRecord":
        return cls(
            candidate=CandidateFrame.from_dict(data["candidate"]),
            qwen_detections=[Detection.from_dict(item) for item in data.get("qwen_detections", [])],
            diff=dict(data.get("diff", {})),
            review_required=bool(data.get("review_required", False)),
            review_reasons=list(data.get("review_reasons", [])),
            prompt_path=str(data.get("prompt_path", "")),
            raw_response_path=str(data.get("raw_response_path", "")),
            label_path=str(data.get("label_path", "")),
            dataset_ready=bool(data.get("dataset_ready", False)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate": self.candidate.to_dict(),
            "qwen_detections": [item.to_dict() for item in self.qwen_detections],
            "diff": self.diff,
            "review_required": self.review_required,
            "review_reasons": self.review_reasons,
            "prompt_path": self.prompt_path,
            "raw_response_path": self.raw_response_path,
            "label_path": self.label_path,
            "dataset_ready": self.dataset_ready,
        }

