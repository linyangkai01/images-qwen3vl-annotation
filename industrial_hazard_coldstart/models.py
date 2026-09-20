"""冷启动链路的新增数据结构。

本模块**复用**母项目 ``traffic_cv_closed_loop.models`` 的
``Detection`` / ``CandidateFrame`` / ``AnnotationRecord``（直接 ``import``，
不重定义），并新增 9 个仅本链路使用的 ``dataclass``：

``MediaAsset`` / ``KeyframeRecord`` / ``TaxonomyClass`` / ``HazardCapability`` /
``HoldoutEntry`` / ``PriorityItem`` / ``EvalReport`` / ``RelationFinding`` /
``TemporalFinding``。

约定：
- 全部字段都有默认值（除少数主键），保证 ``from_dict`` 在缺字段时也能构造；
- 每个结构都提供 ``to_dict`` / ``from_dict``，且互为逆运算（round-trip 稳定）；
- 仅使用 Python 标准库 + 母项目，零新增第三方依赖。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from traffic_cv_closed_loop.models import (
    AnnotationRecord,
    CandidateFrame,
    Detection,
    normalize_bbox,
)

__all__ = [
    # 复用母项目的结构（原样再导出，便于本链路统一从 models 取用）
    "Detection",
    "CandidateFrame",
    "AnnotationRecord",
    "normalize_bbox",
    # 新增结构
    "MediaAsset",
    "KeyframeRecord",
    "TaxonomyClass",
    "HazardCapability",
    "HoldoutEntry",
    "PriorityItem",
    "EvalReport",
    "RelationFinding",
    "TemporalFinding",
]


def _as_dict_list(raw: Any) -> list[dict[str, Any]]:
    """把任意可迭代对象规整为 ``list[dict]``（非 dict 元素被丢弃）。"""
    if not raw:
        return []
    return [dict(item) for item in raw if isinstance(item, dict)]


def _as_str_list(raw: Any) -> list[str]:
    """把任意可迭代对象规整为 ``list[str]``。"""
    if not raw:
        return []
    return [str(item) for item in raw]


def _as_int_list(raw: Any) -> list[int]:
    """把任意可迭代对象规整为 ``list[int]``。"""
    if not raw:
        return []
    return [int(item) for item in raw]


@dataclass(slots=True)
class MediaAsset:
    """一条素材（视频或照片）的编目记录（对应功能点 F1）。

    ``media_type`` 取 ``image`` / ``video``；照片的 ``count`` 恒为 1，
    视频的 ``count`` 为可抽取帧数（未知时为 0）。
    """

    source_id: str
    path: str
    scene: str = "hoist"
    media_type: str = "image"
    duration_s: float = 0.0
    width: int = 0
    height: int = 0
    fps: float = 0.0
    count: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "path": self.path,
            "scene": self.scene,
            "media_type": self.media_type,
            "duration_s": self.duration_s,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "count": self.count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MediaAsset":
        return cls(
            source_id=str(data.get("source_id", "")),
            path=str(data.get("path", "")),
            scene=str(data.get("scene", "hoist")),
            media_type=str(data.get("media_type", "image")),
            duration_s=float(data.get("duration_s", 0.0)),
            width=int(data.get("width", 0)),
            height=int(data.get("height", 0)),
            fps=float(data.get("fps", 0.0)),
            count=int(data.get("count", 1)),
        )

    @property
    def resolved_path(self) -> Path:
        """素材原始路径（``Path`` 视图，中文路径安全）。"""
        return Path(self.path)


@dataclass(slots=True)
class KeyframeRecord:
    """一个被保留下来的关键帧（对应功能点 F2）。"""

    source_id: str
    frame_index: int
    timestamp_s: float = 0.0
    extracted_path: str = ""
    phash: int = 0
    motion_ratio: float = 0.0
    kept_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "frame_index": self.frame_index,
            "timestamp_s": self.timestamp_s,
            "extracted_path": self.extracted_path,
            "phash": self.phash,
            "motion_ratio": self.motion_ratio,
            "kept_reason": self.kept_reason,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "KeyframeRecord":
        return cls(
            source_id=str(data.get("source_id", "")),
            frame_index=int(data.get("frame_index", 0)),
            timestamp_s=float(data.get("timestamp_s", 0.0)),
            extracted_path=str(data.get("extracted_path", "")),
            phash=int(data.get("phash", 0)),
            motion_ratio=float(data.get("motion_ratio", 0.0)),
            kept_reason=str(data.get("kept_reason", "")),
        )


@dataclass(slots=True)
class TaxonomyClass:
    """类别体系中的一个检测类（共享契约，单一真源见 ``taxonomy.py``）。"""

    class_id: int
    name: str
    tier: str = ""
    served_hazards: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "class_id": self.class_id,
            "name": self.name,
            "tier": self.tier,
            "served_hazards": list(self.served_hazards),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaxonomyClass":
        return cls(
            class_id=int(data.get("class_id", 0)),
            name=str(data.get("name", "")),
            tier=str(data.get("tier", "")),
            served_hazards=_as_str_list(data.get("served_hazards")),
        )


@dataclass(slots=True)
class HazardCapability:
    """一条隐患的能力归属（目标检测层 / 几何关系层 / 时序层 / OCR / …）。"""

    hazard_id: str
    layer: str
    supported: bool = False
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "hazard_id": self.hazard_id,
            "layer": self.layer,
            "supported": self.supported,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HazardCapability":
        return cls(
            hazard_id=str(data.get("hazard_id", "")),
            layer=str(data.get("layer", "")),
            supported=bool(data.get("supported", False)),
            notes=str(data.get("notes", "")),
        )


@dataclass(slots=True)
class HoldoutEntry:
    """固定 holdout 中一帧的登记项（逐字节不变 + sha256 校验，需求 7.4）。"""

    dataset_version: str
    relative_path: str
    sha256: str
    registered_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_version": self.dataset_version,
            "relative_path": self.relative_path,
            "sha256": self.sha256,
            "registered_at": self.registered_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HoldoutEntry":
        return cls(
            dataset_version=str(data.get("dataset_version", "")),
            relative_path=str(data.get("relative_path", "")),
            sha256=str(data.get("sha256", "")),
            registered_at=str(data.get("registered_at", "")),
        )


@dataclass(slots=True)
class PriorityItem:
    """主动学习优先队列中的一项（对应功能点 F6）。"""

    relative_path: str
    score: float = 0.0
    signals: dict[str, Any] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "relative_path": self.relative_path,
            "score": self.score,
            "signals": dict(self.signals),
            "reasons": list(self.reasons),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PriorityItem":
        signals = data.get("signals")
        return cls(
            relative_path=str(data.get("relative_path", "")),
            score=float(data.get("score", 0.0)),
            signals=dict(signals) if isinstance(signals, dict) else {},
            reasons=_as_str_list(data.get("reasons")),
        )


@dataclass(slots=True)
class EvalReport:
    """固定 holdout 上的评估报告（逐类指标 + 三档判定 + 消费率）。"""

    dataset_version: str
    holdout_hash: str = ""
    per_class: dict[str, Any] = field(default_factory=dict)
    tier_verdict: dict[str, Any] = field(default_factory=dict)
    consumption_rate: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_version": self.dataset_version,
            "holdout_hash": self.holdout_hash,
            "per_class": dict(self.per_class),
            "tier_verdict": dict(self.tier_verdict),
            "consumption_rate": self.consumption_rate,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvalReport":
        per_class = data.get("per_class")
        tier_verdict = data.get("tier_verdict")
        return cls(
            dataset_version=str(data.get("dataset_version", "")),
            holdout_hash=str(data.get("holdout_hash", "")),
            per_class=dict(per_class) if isinstance(per_class, dict) else {},
            tier_verdict=dict(tier_verdict) if isinstance(tier_verdict, dict) else {},
            consumption_rate=float(data.get("consumption_rate", 0.0)),
        )


@dataclass(slots=True)
class RelationFinding:
    """几何关系层的结论（本期留位，不接入交付指标）。

    ``supporting`` 记录支撑该结论的检测框（``Detection.to_dict`` 形式）。
    """

    hazard_id: str
    relation: str
    confidence: float = 0.0
    supporting: list[dict[str, Any]] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "hazard_id": self.hazard_id,
            "relation": self.relation,
            "confidence": self.confidence,
            "supporting": [dict(item) for item in self.supporting],
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RelationFinding":
        return cls(
            hazard_id=str(data.get("hazard_id", "")),
            relation=str(data.get("relation", "")),
            confidence=float(data.get("confidence", 0.0)),
            supporting=_as_dict_list(data.get("supporting")),
            notes=str(data.get("notes", "")),
        )


@dataclass(slots=True)
class TemporalFinding:
    """时序层的结论（本期留位，不接入交付指标）。"""

    hazard_id: str
    source_id: str
    finding: str
    confidence: float = 0.0
    frame_indices: list[int] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "hazard_id": self.hazard_id,
            "source_id": self.source_id,
            "finding": self.finding,
            "confidence": self.confidence,
            "frame_indices": list(self.frame_indices),
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TemporalFinding":
        return cls(
            hazard_id=str(data.get("hazard_id", "")),
            source_id=str(data.get("source_id", "")),
            finding=str(data.get("finding", "")),
            confidence=float(data.get("confidence", 0.0)),
            frame_indices=_as_int_list(data.get("frame_indices")),
            notes=str(data.get("notes", "")),
        )
