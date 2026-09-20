"""T05.3 几何关系层（**本期留位，不接入交付指标**）。

**性质说明**：本期交付物**只有目标检测权重**；几何关系层只**落代码 + 数据结构 +
单测**，证明结构正确，其结论**不得**混进 :class:`EvalReport` 的验收判定。
原因是这几条隐患**框本身做不了**——需要比较框间空间关系（包含 / 夹角 / 平衡），
框是**输入**不是结论：

- ``D-5-3``（人员站在吊物下方）：需判断 ``person`` 框是否落在 ``load`` 框的覆盖范围内；
- ``D-4-3``（歪拉斜吊）：需判断吊索/吊钩与吊物中心的连线**偏离竖直**的角度；
- （``D-2-1`` / ``D-2-4`` 同属本层，见 :data:`taxonomy.S1_GEOMETRY_HAZARD_IDS`。）

这些结论以 :class:`~industrial_hazard_coldstart.models.RelationFinding` 表达，
``supporting`` 记录支撑该结论的检测框（``Detection.to_dict`` 形式），便于人工复核。

**坐标系**：``Detection.bbox`` 是**归一化**坐标（0–1）。竖直夹角在归一化空间会因
画面长宽比而失真，故 :func:`vertical_angle_deg` 提供 ``aspect_ratio`` 参数
（= 宽/高）；调用方在已知图像尺寸时应传入真实长宽比。

仅使用 Python 标准库 + 母项目；零新增第三方依赖。
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

from industrial_hazard_coldstart.models import Detection, RelationFinding

__all__ = [
    "MIN_CONTAINMENT_RATIO",
    "OBLIQUE_ANGLE_THRESHOLD_DEG",
    "HAZARD_PERSON_UNDER_LOAD",
    "HAZARD_OBLIQUE_LIFT",
    "contains",
    "containment_ratio",
    "vertical_angle_deg",
    "find_person_under_load",
    "find_oblique_lift",
    "find_relations",
]

#: ``contains`` 判定的包含比阈值（内框面积被外框覆盖的比例）。
MIN_CONTAINMENT_RATIO: float = 0.85

#: 「歪拉斜吊」判定的偏离竖直角度阈值（度）。超过即视为歪拉斜吊。
OBLIQUE_ANGLE_THRESHOLD_DEG: float = 15.0

#: 本层服务的隐患 id（与 :mod:`taxonomy` 的分桶一致）。
HAZARD_PERSON_UNDER_LOAD: str = "D-5-3"
HAZARD_OBLIQUE_LIFT: str = "D-4-3"

#: 视为「吊索 / 吊钩」的检测类。
_HOOK_LABELS: tuple[str, ...] = ("hook", "rigging")


# --------------------------------------------------------------------------- #
# 基础几何
# --------------------------------------------------------------------------- #
def _area(bbox: Sequence[float]) -> float:
    """矩形面积（宽 × 高）。"""
    x1, y1, x2, y2 = bbox
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _center(bbox: Sequence[float]) -> tuple[float, float]:
    """矩形中心 ``(cx, cy)``。"""
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _intersection_area(a: Sequence[float], b: Sequence[float]) -> float:
    """两矩形交集面积。"""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    return inter_w * inter_h


def containment_ratio(outer: Detection, inner: Detection) -> float:
    """返回内框面积被外框覆盖的比例（``交集 / 内框面积``，范围 0–1）。"""
    inner_area = _area(inner.bbox)
    if inner_area <= 0.0:
        return 0.0
    return _intersection_area(outer.bbox, inner.bbox) / inner_area


def contains(outer: Detection, inner: Detection) -> bool:
    """判断 ``inner`` 是否（基本）被 ``outer`` 包含。

    判据：内框**中心**落在外框内，且内框面积被外框覆盖的比例 ≥
    :data:`MIN_CONTAINMENT_RATIO`（容忍检测框抖动，不要求逐像素包含）。

    Args:
        outer: 外框（如 ``load``）。
        inner: 内框（如 ``person``）。

    Returns:
        是否包含。
    """
    ox1, oy1, ox2, oy2 = outer.bbox
    cx, cy = _center(inner.bbox)
    if not (ox1 <= cx <= ox2 and oy1 <= cy <= oy2):
        return False
    return containment_ratio(outer, inner) >= MIN_CONTAINMENT_RATIO


def vertical_angle_deg(
    load: Detection,
    hook: Detection,
    aspect_ratio: float = 1.0,
) -> float:
    """返回「吊钩/吊索中心 → 吊物中心」连线偏离**竖直**方向的角度（度）。

    - ``0`` = 完全竖直（吊物正下方就是吊点）；数值越大越「歪拉斜吊」；
    - 返回 ``[0, 90]`` 的非负角度。

    Args:
        load: 吊物检测框。
        hook: 吊钩 / 吊索检测框。
        aspect_ratio: 画面宽高比（= 宽 / 高），用于把归一化 x 还原到等比例空间；
            未知时取 ``1.0``（此时角度是归一化空间的近似值）。

    Returns:
        偏离竖直的角度（度）。
    """
    load_cx, load_cy = _center(load.bbox)
    hook_cx, hook_cy = _center(hook.bbox)
    dx = (load_cx - hook_cx) * max(aspect_ratio, 1e-6)
    dy = load_cy - hook_cy  # 吊物应在吊点下方 → dy 为正。
    if dx == 0.0 and dy == 0.0:
        return 0.0
    return abs(math.degrees(math.atan2(abs(dx), abs(dy))))


# --------------------------------------------------------------------------- #
# 关系发现
# --------------------------------------------------------------------------- #
def _by_label(dets: Iterable[Detection], label: str) -> list[Detection]:
    """按类别名过滤检测框。"""
    return [item for item in dets if item.label == label]


def _finding(
    hazard_id: str,
    relation: str,
    supporting: Sequence[Detection],
    notes: str,
) -> RelationFinding:
    """构造一条关系结论（置信度取支撑框置信度之积，作保守估计）。"""
    confidence = 1.0
    for item in supporting:
        confidence *= float(item.confidence)
    return RelationFinding(
        hazard_id=hazard_id,
        relation=relation,
        confidence=round(confidence, 6),
        supporting=[item.to_dict() for item in supporting],
        notes=notes,
    )


def find_person_under_load(
    dets: Iterable[Detection],
    aspect_ratio: float = 1.0,
) -> list[RelationFinding]:
    """发现「人员站在吊物下方」（``D-5-3``）的关系。

    对每个 ``load`` 与 ``person`` 组合，若 :func:`contains` 成立（人落在吊物覆盖范围
    内）→ 产出一条 :class:`RelationFinding`。``supporting`` 依次为
    ``[load, person]``（便于复核）。

    Args:
        dets: 同一帧的检测框集合。
        aspect_ratio: 预留参数（本判据不依赖长宽比；保持签名一致）。

    Returns:
        关系结论列表（可能为空）。
    """
    _ = aspect_ratio
    detections = list(dets)
    findings: list[RelationFinding] = []
    for load in _by_label(detections, "load"):
        for person in _by_label(detections, "person"):
            if contains(load, person):
                findings.append(
                    _finding(
                        HAZARD_PERSON_UNDER_LOAD,
                        "person_under_load",
                        [load, person],
                        "person 框中心落在 load 框内且包含比≥阈值；本期留位，不接入交付指标。",
                    )
                )
    return findings


def find_oblique_lift(
    dets: Iterable[Detection],
    angle_threshold_deg: float = OBLIQUE_ANGLE_THRESHOLD_DEG,
    aspect_ratio: float = 1.0,
) -> list[RelationFinding]:
    """发现「歪拉斜吊」（``D-4-3``）的关系。

    对每个 ``load``，取其**上方最近**的 ``hook`` / ``rigging``，计算
    :func:`vertical_angle_deg`；超过 ``angle_threshold_deg`` → 产出结论。
    ``supporting`` 依次为 ``[load, hook]``。

    Args:
        dets: 同一帧的检测框集合。
        angle_threshold_deg: 偏离竖直的角度阈值（度）。
        aspect_ratio: 画面宽高比（见 :func:`vertical_angle_deg`）。

    Returns:
        关系结论列表（可能为空）。
    """
    detections = list(dets)
    hooks: list[Detection] = []
    for label in _HOOK_LABELS:
        hooks.extend(_by_label(detections, label))
    findings: list[RelationFinding] = []
    for load in _by_label(detections, "load"):
        _lx, load_cy = _center(load.bbox)
        above = [hook for hook in hooks if _center(hook.bbox)[1] < load_cy]
        if not above:
            continue
        nearest = min(
            above,
            key=lambda hook: abs(_center(hook.bbox)[0] - _lx)
            + abs(_center(hook.bbox)[1] - load_cy),
        )
        angle = vertical_angle_deg(load, nearest, aspect_ratio=aspect_ratio)
        if angle >= angle_threshold_deg:
            findings.append(
                _finding(
                    HAZARD_OBLIQUE_LIFT,
                    "oblique_lift",
                    [load, nearest],
                    f"吊点连线偏离竖直 {angle:.1f}°（阈值 {angle_threshold_deg:.1f}°）；"
                    "本期留位，不接入交付指标。",
                )
            )
    return findings


def find_relations(
    dets: Iterable[Detection],
    aspect_ratio: float = 1.0,
) -> list[RelationFinding]:
    """聚合本层全部关系结论（当前 = 人下吊物 ``D-5-3`` + 歪拉斜吊 ``D-4-3``）。"""
    detections = list(dets)
    return [
        *find_person_under_load(detections, aspect_ratio=aspect_ratio),
        *find_oblique_lift(detections, aspect_ratio=aspect_ratio),
    ]


# 显式标注本层「留位不接入指标」——供文档/自检读取。
LAYER_NOTE: str = "本期留位，不接入交付指标（检测权重为唯一交付物）。"
