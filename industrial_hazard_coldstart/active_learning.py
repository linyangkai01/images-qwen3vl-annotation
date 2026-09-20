"""主动学习 / 不确定性采样（功能点 F6）。

人工预算有限，把人的时间花在**模型最不确定**的样本上，单位人工收益最高——这是
「低人工成本」的关键杠杆（需求 §F6）。

本模块对每条标注记录计算一个**不确定性分数**，产出按分数降序的复核优先清单
``workspace/review/priority_queue.jsonl``。

**三类不确定信号**（需求 §F6 明确要求「至少三种」，本实现全部落实）：

1. **低置信度**（``low_confidence``）—— VLM 复核框里的最低置信度低于复核阈值越多，越可疑；
2. **通道间差异计数高**（``channel_disagreement``）—— 两段式标注里 YOLO 初检与 VLM 复核的
   新增 / 删除 / 纠偏总数越大，说明两通道分歧越大；
3. **复核原因命中**（``review_reason``）—— 母项目 ``run_annotation`` 写下的
   ``review_reasons``（如 ``qwen_added_objects`` / ``bbox_adjusted`` / ``annotation_error``）。

权重（见模块常量，**可通过覆盖模块常量调整**；函数本身没有权重入参）：
低置信度 ``0.4`` + 通道差异 ``0.4`` + 复核原因 ``0.2``。
各分量都归一到 ``[0, 1]``，故 ``score ∈ [0, 1]``；分数越大越优先送人工。

依赖：仅标准库 + 母项目（``AnnotationRecord``）+ 自身 ``models``，零新增第三方依赖。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from traffic_cv_closed_loop.io_utils import write_jsonl
from traffic_cv_closed_loop.models import AnnotationRecord

from industrial_hazard_coldstart.models import PriorityItem

__all__ = [
    "WEIGHT_LOW_CONFIDENCE",
    "WEIGHT_CHANNEL_DISAGREEMENT",
    "WEIGHT_REVIEW_REASON",
    "DEFAULT_REVIEW_CONFIDENCE_THRESHOLD",
    "DEFAULT_TOP_K",
    "score_uncertainty",
    "build_priority_queue",
    "write_priority_queue",
]

#: 低置信度信号的权重。
WEIGHT_LOW_CONFIDENCE: float = 0.4
#: 通道间差异信号的权重。
WEIGHT_CHANNEL_DISAGREEMENT: float = 0.4
#: 复核原因命中信号的权重。
WEIGHT_REVIEW_REASON: float = 0.2

#: 默认复核置信度阈值（与配置 ``pipeline.review_confidence_threshold`` 默认值一致）。
DEFAULT_REVIEW_CONFIDENCE_THRESHOLD: float = 0.8
#: 默认取前 K 条；``0`` / ``None`` 表示全部返回。
DEFAULT_TOP_K: int = 0


def _clamp_unit(value: float) -> float:
    """把数值夹到 ``[0, 1]``。"""
    return max(0.0, min(1.0, float(value)))


def _confidence_component(
    record: AnnotationRecord,
    threshold: float,
) -> tuple[float, float | None]:
    """低置信度分量：``max(0, 阈值 − 最低置信度) / 阈值``，归一化到 ``[0, 1]``。

    Returns:
        ``(component, min_confidence)``。无检测框时返回 ``(0.0, None)``——
        「没有框」不是「低置信度」，不在此信号计分（由复核原因信号兜底）。
    """
    confidences = [float(item.confidence) for item in record.qwen_detections]
    if not confidences:
        return 0.0, None
    min_conf = min(confidences)
    if threshold <= 0:
        return 0.0, min_conf
    return _clamp_unit((threshold - min_conf) / threshold), min_conf


def score_uncertainty(
    record: AnnotationRecord,
    *,
    confidence_threshold: float = DEFAULT_REVIEW_CONFIDENCE_THRESHOLD,
) -> PriorityItem:
    """对单条标注记录计算不确定性分数，产出 :class:`PriorityItem`。

    Args:
        record: 母项目 ``AnnotationRecord``（含 ``qwen_detections`` / ``diff`` /
            ``review_required`` / ``review_reasons`` / ``candidate``）。
        confidence_threshold: 低置信度判定的阈值（默认 ``0.8``）。

    Returns:
        含 ``relative_path`` / ``score`` / ``signals`` / ``reasons`` 的优先项。
        ``score`` 为三路信号的加权和，落在 ``[0, 1]``。
    """
    low_conf, min_conf = _confidence_component(record, confidence_threshold)

    diff = record.diff or {}
    added = int(diff.get("added", 0) or 0)
    removed = int(diff.get("removed", 0) or 0)
    adjusted = int(diff.get("adjusted", 0) or 0)
    disagreement_count = added + removed + adjusted
    denominator = max(1, len(record.candidate.detections) + len(record.qwen_detections))
    disagreement_ratio = _clamp_unit(disagreement_count / denominator)

    review_reasons = list(record.review_reasons or [])
    review_component = 1.0 if record.review_required else 0.0

    score = _clamp_unit(
        WEIGHT_LOW_CONFIDENCE * low_conf
        + WEIGHT_CHANNEL_DISAGREEMENT * disagreement_ratio
        + WEIGHT_REVIEW_REASON * review_component
    )

    signals: dict[str, Any] = {
        "low_confidence": round(low_conf, 6),
        "min_confidence": None if min_conf is None else round(min_conf, 6),
        "confidence_threshold": float(confidence_threshold),
        "channel_disagreement": round(disagreement_ratio, 6),
        "disagreement_count": disagreement_count,
        "disagreement_detail": {"added": added, "removed": removed, "adjusted": adjusted},
        "review_reason": round(review_component, 6),
        "review_reason_count": len(review_reasons),
        "weights": {
            "low_confidence": WEIGHT_LOW_CONFIDENCE,
            "channel_disagreement": WEIGHT_CHANNEL_DISAGREEMENT,
            "review_reason": WEIGHT_REVIEW_REASON,
        },
    }

    reasons: list[str] = []
    if low_conf > 0:
        reasons.append(
            f"低置信度：最低置信度 {min_conf:.3f} < 阈值 {confidence_threshold:.2f}"
        )
    if disagreement_count > 0:
        reasons.append(
            f"通道差异：新增 {added} / 删除 {removed} / 纠偏 {adjusted}（共 {disagreement_count}）"
        )
    if review_component > 0:
        detail = "、".join(review_reasons) if review_reasons else "review_required"
        reasons.append(f"复核原因命中：{detail}")

    return PriorityItem(
        relative_path=record.candidate.relative_path,
        score=round(score, 6),
        signals=signals,
        reasons=reasons,
    )


def build_priority_queue(
    records: Iterable[AnnotationRecord],
    top_k: int | None = DEFAULT_TOP_K,
    *,
    confidence_threshold: float = DEFAULT_REVIEW_CONFIDENCE_THRESHOLD,
) -> list[PriorityItem]:
    """把标注记录按不确定性**降序**排序，取前 ``top_k`` 条。

    排序稳定且确定：主键 ``score`` 降序，**并列时按 ``relative_path`` 升序**，
    保证同一批输入产出同一顺序（便于回归与复现）。

    Args:
        records: ``AnnotationRecord`` 可迭代对象。
        top_k: 取前 K 条；``0`` / ``None`` / 负数表示全部返回。
        confidence_threshold: 低置信度判定阈值。

    Returns:
        排序后的 :class:`PriorityItem` 列表。
    """
    items = [
        score_uncertainty(record, confidence_threshold=confidence_threshold)
        for record in records
    ]
    items.sort(key=lambda item: (-item.score, item.relative_path))
    if top_k is not None and top_k > 0:
        return items[:top_k]
    return items


def write_priority_queue(path: str | Path, items: Iterable[PriorityItem]) -> Path:
    """把优先队列写入 jsonl（中文安全）。"""
    target = Path(path)
    write_jsonl(target, [item.to_dict() for item in items])
    return target
