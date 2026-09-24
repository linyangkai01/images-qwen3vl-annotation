"""T05.3 几何关系层（**本期留位，不接入交付指标**）。

**性质说明**：本期交付物**只有目标检测权重**；几何关系层只**落代码 + 数据结构 +
单测**，证明结构正确，其结论**不得**混进 :class:`EvalReport` 的验收判定。
原因是这几条隐患**框本身做不了**——需要比较框间空间关系（包含 / 夹角 / 平衡），
框是**输入**不是结论：

- ``D-5-3``（人员站在吊物下方）—— 见下「判据重写」；
- ``D-4-3``（歪拉斜吊）—— 需判断吊索/吊钩与吊物中心的连线**偏离竖直**的角度；
- （``D-2-1`` / ``D-2-4`` 同属本层，见 :data:`taxonomy.S1_GEOMETRY_HAZARD_IDS`。）

**判据重写（根治 v2 方案 §4.4，Singh & Seif 2026, arXiv:2607.16351）**：

旧判据 ``contains(load, person)`` 要求**人的框中心落在吊物框内**——这**只覆盖俯拍视角**；
**地面视角（人站在吊物下方的地面）永远判不出**（这正是真实素材里 ``D-5-3`` 报出 85 条
但几何层从未产出结论的「死代码根因」）。现改为**射影判据**（不需要深度 / 3D 重建）：

1. **工人脚点 = 框底中心**（``x = 框中心 x``，``y = 框底 y``）——**不是框中心**，关键修正；
2. **净空比** ``r_clr = (y_foot_worker − y_bottom_load) / h_worker``，阈值 **0.12**
   （用人像素高归一化，抗视角变化；该比值在归一化坐标与像素坐标下等价）；
3. **梯形坠落区**：半宽从载荷底部 ``w/2 + 30px`` 线性扩展到地面 ``w/2 + δ_lat``；
   ``δ_lat = v_eff · t_fall``，``t_fall = √(2h/g)``，漂移速度下限 ``0.4 m/s``；
4. **垂直门控**：脚点须在载荷底边下方 **≥40px**（抑制「载荷上方远处的人」误报）；
5. **像素-米代理**：``ρ = 中位工人像素高 / 1.75m``；
6. **俯拍回落**：仍保留 :func:`contains` 作为**俯拍视角**分支（人被吊物框包住 → 命中）。

**三级输出（方案阶段 2 第 7 项）**：每条结论带 ``verdict``（``confirmed_danger`` /
``suspected`` / ``needs_review``）与**决策余量** ``margin``（``|Δ|``），便于人工复核。

**时序（方案阶段 2 第 6 项）**：单帧命中不足以定论，需**连续多帧命中**
（:func:`find_person_under_load_sequence`）。论文原规则是「1 秒滑窗（15 帧）内 ≥10 帧」；
本项目抽帧为**秒级**（实测约 2~6 秒/帧），1 秒窗内不足 1 帧，故**按比例适配**为
「3 帧窗内 ≥2 帧命中」（保持论文的 2/3 命中比例，详见该函数 docstring）。

**坐标系**：``Detection.bbox`` 是**归一化**坐标（0–1）。净空比与长宽比无关；但
``30px`` / ``40px`` 等**像素阈值**需要图像尺寸——调用方应传 ``image_size=(宽, 高)``，
缺省用 :data:`REFERENCE_IMAGE_SIZE`（吊装视频原生分辨率 960×540，像素阈值在此标定）。

仅使用 Python 标准库 + 母项目；零新增第三方依赖。
"""

from __future__ import annotations

import math
import statistics
from typing import Any, Iterable, Sequence

from industrial_hazard_coldstart.models import (
    CandidateFrame,
    Detection,
    RelationFinding,
)

__all__ = [
    # 阈值常量
    "MIN_CONTAINMENT_RATIO",
    "OBLIQUE_ANGLE_THRESHOLD_DEG",
    "CLEARANCE_RATIO_THRESHOLD",
    "VERTICAL_GATE_PX",
    "FALL_ZONE_LATERAL_START_PX",
    "DRIFT_SPEED_MIN_MPS",
    "GRAVITY_MPS2",
    "WORKER_HEIGHT_M",
    "REFERENCE_IMAGE_SIZE",
    # 三级输出
    "VERDICT_CONFIRMED",
    "VERDICT_SUSPECTED",
    "VERDICT_REVIEW",
    "MARGIN_CONFIRMED",
    "MARGIN_SUSPECT",
    "verdict_for_margin",
    # 时序
    "DEFAULT_SEQUENCE_WINDOW",
    "DEFAULT_SEQUENCE_MIN_HITS",
    # 隐患 id
    "HAZARD_PERSON_UNDER_LOAD",
    "HAZARD_OBLIQUE_LIFT",
    # 基础几何
    "contains",
    "containment_ratio",
    "vertical_angle_deg",
    # 关系发现
    "find_person_under_load",
    "find_person_under_load_sequence",
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

# --------------------------------------------------------------------------- #
# Singh & Seif 2026（arXiv:2607.16351）射影判据参数（方案 §4.4，逐条对齐）
# --------------------------------------------------------------------------- #
#: 净空比阈值：脚点在载荷底边下方超过「0.12 × 工人像素高」才算「在下方」。
CLEARANCE_RATIO_THRESHOLD: float = 0.12

#: 垂直门控：脚点须在载荷底边下方至少该像素数（抑制载荷上方远处的人）。
VERTICAL_GATE_PX: float = 40.0

#: 梯形坠落区在载荷底部的起始半宽附加量（px）。
FALL_ZONE_LATERAL_START_PX: float = 30.0

#: 坠落物横向漂移速度下限（m/s）——用于 ``δ_lat = v_eff · t_fall``。
DRIFT_SPEED_MIN_MPS: float = 0.4

#: 重力加速度（m/s²），用于 ``t_fall = √(2h/g)``。
GRAVITY_MPS2: float = 9.81

#: 工人身高先验（m），用于像素-米代理 ``ρ = 中位工人像素高 / 1.75``。
WORKER_HEIGHT_M: float = 1.75

#: 像素阈值的标定参考分辨率（宽, 高）——吊装视频原生 960×540。
#: 调用方若已知真实图像尺寸，应显式传 ``image_size`` 覆盖。
REFERENCE_IMAGE_SIZE: tuple[int, int] = (960, 540)

# --------------------------------------------------------------------------- #
# 三级输出（方案阶段 2 第 7 项；分档阈值取自方案 §5 的不确定度带 0.10 / 0.25）
# --------------------------------------------------------------------------- #
#: 确定危险。
VERDICT_CONFIRMED: str = "confirmed_danger"
#: 疑似（在边界附近，倾向命中）。
VERDICT_SUSPECTED: str = "suspected"
#: 需人工看（紧贴边界，命中与否难分）。
VERDICT_REVIEW: str = "needs_review"

#: 决策余量 ≥ 该值 → 确定危险。
MARGIN_CONFIRMED: float = 0.25
#: 决策余量 ≥ 该值（且 < ``MARGIN_CONFIRMED``）→ 疑似。
MARGIN_SUSPECT: float = 0.10

# --------------------------------------------------------------------------- #
# 时序持久性（方案阶段 2 第 6 项，按实际帧率适配）
# --------------------------------------------------------------------------- #
#: 滑窗帧数（本项目秒级抽帧下的最小有意义窗）。
DEFAULT_SEQUENCE_WINDOW: int = 3
#: 窗内命中帧数下限（保持论文「10/15 = 2/3」的命中比例）。
DEFAULT_SEQUENCE_MIN_HITS: int = 2

#: 视为「吊索 / 吊钩」的检测类。
_HOOK_LABELS: tuple[str, ...] = ("hook", "rigging")

#: 视为「吊物」/「人员」的检测类。
_LOAD_LABELS: tuple[str, ...] = ("load",)
_PERSON_LABELS: tuple[str, ...] = ("person",)


def verdict_for_margin(margin: float) -> str:
    """把**决策余量** ``|Δ|`` 映射为三级判定。

    - ``margin ≥ MARGIN_CONFIRMED(0.25)`` → ``confirmed_danger``（确定危险）；
    - ``MARGIN_SUSPECT(0.10) ≤ margin < 0.25`` → ``suspected``（疑似）；
    - ``0 < margin < 0.10`` → ``needs_review``（需人工看，紧贴边界）；
    - ``margin ≤ 0`` → ``""``（未命中）。

    分档阈值取自根治方案 §5 的三级不确定度带（``|Δ|<0.10 疑似 / 0.10~0.25 危险 /
    ≥0.25 确定``）——同一套带用于「深度特征」与「几何余量」，保持全项目一致。
    """
    if margin >= MARGIN_CONFIRMED:
        return VERDICT_CONFIRMED
    if margin >= MARGIN_SUSPECT:
        return VERDICT_SUSPECTED
    if margin > 0.0:
        return VERDICT_REVIEW
    return ""


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
    """判断 ``inner`` 是否（基本）被 ``outer`` 包含（**俯拍视角**分支）。

    判据：内框**中心**落在外框内，且内框面积被外框覆盖的比例 ≥
    :data:`MIN_CONTAINMENT_RATIO`（容忍检测框抖动，不要求逐像素包含）。

    .. note::
        本函数只覆盖**俯拍视角**（人落在吊物框内）。**地面视角**（人站在吊物下方的
        地面）由 :func:`find_person_under_load` 的射影判据（脚点 / 净空比 / 坠落区）
        覆盖——这正是本次判据重写的核心。

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
# 射影判据（Singh & Seif 2026）内部实现
# --------------------------------------------------------------------------- #
def _to_pixels(
    bbox: Sequence[float],
    image_size: tuple[int, int],
) -> tuple[float, float, float, float]:
    """归一化框 → 像素框（``image_size`` = (宽, 高)）。"""
    width, height = float(image_size[0]), float(image_size[1])
    x1, y1, x2, y2 = bbox
    return (x1 * width, y1 * height, x2 * width, y2 * height)


def _worker_scale(
    detections: Sequence[Detection],
    image_size: tuple[int, int],
) -> float:
    """像素-米代理 ``ρ = 中位工人像素高 / 1.75m``（px per meter）。

    无 ``person`` 框时回落到 ``1.0``（避免除零；此时坠落区横向扩展近似退化，
    但净空比判据仍成立——净空比与 ρ 无关）。
    """
    height_px = float(image_size[1])
    heights = [
        (item.bbox[3] - item.bbox[1]) * height_px
        for item in detections
        if item.label in _PERSON_LABELS
    ]
    heights = [value for value in heights if value > 0.0]
    if not heights:
        return 1.0
    return max(statistics.median(heights), 1e-6) / WORKER_HEIGHT_M


def _fall_zone_half_width_px(
    load_px: Sequence[float],
    y_foot_px: float,
    rho_px_per_m: float,
    image_height: float,
) -> float:
    """梯形坠落区在 ``y_foot_px`` 处的半宽（px）。

    半宽从载荷底部的 ``w/2 + 30px`` 线性扩展到地面（画面底边）的 ``w/2 + δ_lat``；
    ``δ_lat = v_eff · t_fall · ρ``，``t_fall = √(2h/g)``，``h`` = 载荷底边到地面的米数
    （``(画面高 − 载荷底边)/ρ``）。为保证坠落区**单调扩张**，``δ_lat`` 取下限 30px。
    """
    lx1, _ly1, lx2, ly2 = load_px
    w_load = max(0.0, lx2 - lx1)
    base_half = w_load / 2.0 + FALL_ZONE_LATERAL_START_PX
    drop_to_ground = max(0.0, float(image_height) - ly2)
    if drop_to_ground <= 0.0:
        return base_half
    h_m = drop_to_ground / max(rho_px_per_m, 1e-6)
    t_fall = math.sqrt(2.0 * h_m / GRAVITY_MPS2)
    delta_lat_px = max(
        DRIFT_SPEED_MIN_MPS * t_fall * rho_px_per_m, FALL_ZONE_LATERAL_START_PX
    )
    ground_half = w_load / 2.0 + delta_lat_px
    fraction = min(1.0, max(0.0, (y_foot_px - ly2) / drop_to_ground))
    return base_half + (ground_half - base_half) * fraction


def _evaluate_pair(
    load: Detection,
    person: Detection,
    image_size: tuple[int, int],
    rho_px_per_m: float,
) -> tuple[bool, str, str, float, dict[str, Any]]:
    """对一对 ``(load, person)`` 给出判据结果。

    Returns:
        ``(hit, branch, verdict, margin, evidence)``：

        - ``hit``：是否命中（``margin > 0``）；
        - ``branch``：命中的分支（``"ground"`` 地面射影 / ``"overhead"`` 俯拍包含）；
        - ``verdict``：三级判定；
        - ``margin``：决策余量 ``|Δ|``；
        - ``evidence``：可复核的实测量。
    """
    width, height = image_size
    load_px = _to_pixels(load.bbox, image_size)
    person_px = _to_pixels(person.bbox, image_size)
    px1, py1, px2, py2 = person_px

    # --- 分支 A：地面射影（脚点 / 净空比 / 梯形坠落区 / 垂直门控）---
    x_foot = (px1 + px2) / 2.0
    y_foot = py2  # 脚点 = 框底中心（不是框中心）
    h_worker = max(0.0, py2 - py1)
    y_bottom_load = load_px[3]
    x_load_center = (load_px[0] + load_px[2]) / 2.0
    gap_px = y_foot - y_bottom_load
    r_clr = (gap_px / h_worker) if h_worker > 0.0 else float("-inf")
    dx = abs(x_foot - x_load_center)
    half_w = _fall_zone_half_width_px(load_px, y_foot, rho_px_per_m, height)
    gate_ok = gap_px >= VERTICAL_GATE_PX
    clearance_ok = r_clr >= CLEARANCE_RATIO_THRESHOLD
    within_zone = dx <= half_w
    ground_ok = gate_ok and clearance_ok and within_zone
    if ground_ok:
        # 净空比余量（相对阈值，封顶 1）+ 横向余量（到坠落区边界，封顶 1）。
        m_clear = min(1.0, (r_clr - CLEARANCE_RATIO_THRESHOLD) / CLEARANCE_RATIO_THRESHOLD)
        m_lateral = (half_w - dx) / half_w if half_w > 0.0 else 0.0
        m_ground = max(0.0, min(m_clear, m_lateral))
    else:
        m_ground = -1.0

    # --- 分支 B：俯拍包含（旧判据保留，覆盖俯拍视角）---
    ratio = containment_ratio(load, person)
    overhead_ok = contains(load, person)
    m_overhead = (
        (ratio - MIN_CONTAINMENT_RATIO) / (1.0 - MIN_CONTAINMENT_RATIO)
        if overhead_ok
        else -1.0
    )

    margin = max(m_ground, m_overhead)
    hit = margin > 0.0
    branch = "ground" if m_ground >= m_overhead else "overhead"
    verdict = verdict_for_margin(margin) if hit else ""
    evidence: dict[str, Any] = {
        "branch": branch,
        "clearance_ratio": round(r_clr, 6) if math.isfinite(r_clr) else None,
        "clearance_threshold": CLEARANCE_RATIO_THRESHOLD,
        "gap_px": round(gap_px, 3),
        "vertical_gate_px": VERTICAL_GATE_PX,
        "lateral_offset_px": round(dx, 3),
        "fall_zone_half_width_px": round(half_w, 3),
        "worker_height_px": round(h_worker, 3),
        "worker_scale_px_per_m": round(rho_px_per_m, 6),
        "containment_ratio": round(ratio, 6),
        "margin": round(margin, 6),
        "verdict": verdict,
        "image_size": [int(width), int(height)],
    }
    return hit, branch, verdict, margin, evidence


def _person_under_load_notes(
    verdict: str, margin: float, evidence: dict[str, Any]
) -> str:
    """构造可读的中文结论说明（含实测量与判据来源）。"""
    return (
        f"分支={evidence.get('branch')}；"
        f"净空比 r_clr={evidence.get('clearance_ratio')}"
        f"（阈值 {CLEARANCE_RATIO_THRESHOLD}）；"
        f"脚点距载荷底边 {evidence.get('gap_px')}px"
        f"（垂直门控 {VERTICAL_GATE_PX}px）；"
        f"水平距 {evidence.get('lateral_offset_px')}px"
        f" / 坠落区半宽 {evidence.get('fall_zone_half_width_px')}px；"
        f"包含比 {evidence.get('containment_ratio')}；"
        f"决策余量 |Δ|={margin:.3f} → {verdict}。"
        "判据：Singh & Seif 2026（脚点=框底中心 + 净空比 + 梯形坠落区 + 垂直门控）；"
        "本期留位，不接入交付指标。"
    )


# --------------------------------------------------------------------------- #
# 关系发现
# --------------------------------------------------------------------------- #
def _by_label(dets: Iterable[Detection], label: str) -> list[Detection]:
    """按类别名过滤检测框。"""
    return [item for item in dets if item.label == label]


def _by_labels(dets: Iterable[Detection], labels: Sequence[str]) -> list[Detection]:
    """按一组类别名过滤检测框。"""
    wanted = set(labels)
    return [item for item in dets if item.label in wanted]


def _finding(
    hazard_id: str,
    relation: str,
    supporting: Sequence[Detection],
    notes: str,
    *,
    verdict: str = "",
    margin: float = 0.0,
    evidence: dict[str, Any] | None = None,
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
        verdict=verdict,
        margin=round(float(margin), 6),
        evidence=dict(evidence or {}),
    )


def find_person_under_load(
    dets: Iterable[Detection],
    aspect_ratio: float = 1.0,
    image_size: tuple[int, int] | None = None,
) -> list[RelationFinding]:
    """发现「人员站在吊物下方」（``D-5-3``）的关系——**射影判据（方案 §4.4）**。

    对每个 ``load`` 与 ``person`` 组合，若命中下列**任一**分支即产出结论：

    - **地面射影**：脚点（框底中心）在载荷底边下方 ≥40px 且净空比 ≥0.12，
      且落在梯形坠落区内（覆盖**地面视角**）；
    - **俯拍包含**：人被吊物框（基本）包含（:func:`contains`，覆盖**俯拍视角**）。

    每条结论带三级判定（``verdict``）与决策余量（``margin``）。``supporting`` 依次为
    ``[load, person]``。

    Args:
        dets: 同一帧的检测框集合。
        aspect_ratio: 预留参数（本判据不依赖长宽比；保持签名一致）。
        image_size: 图像尺寸 ``(宽, 高)``（像素阈值 30px/40px 的标定基准）；
            缺省用 :data:`REFERENCE_IMAGE_SIZE`。

    Returns:
        关系结论列表（可能为空）。
    """
    _ = aspect_ratio
    size = (int(image_size[0]), int(image_size[1])) if image_size else REFERENCE_IMAGE_SIZE
    detections = list(dets)
    rho = _worker_scale(detections, size)
    loads = _by_labels(detections, _LOAD_LABELS)
    persons = _by_labels(detections, _PERSON_LABELS)
    findings: list[RelationFinding] = []
    for load in loads:
        for person in persons:
            hit, _branch, verdict, margin, evidence = _evaluate_pair(
                load, person, size, rho
            )
            if not hit:
                continue
            findings.append(
                _finding(
                    HAZARD_PERSON_UNDER_LOAD,
                    "person_under_load",
                    [load, person],
                    _person_under_load_notes(verdict, margin, evidence),
                    verdict=verdict,
                    margin=margin,
                    evidence=evidence,
                )
            )
    return findings


def _frame_index(frame: CandidateFrame, fallback: int) -> int:
    """取帧序号：优先 ``metadata['frame_index']``，否则用组内序号。"""
    value = (frame.metadata or {}).get("frame_index")
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def find_person_under_load_sequence(
    frames: Sequence[CandidateFrame],
    aspect_ratio: float = 1.0,
    image_size: tuple[int, int] | None = None,
    *,
    window_frames: int = DEFAULT_SEQUENCE_WINDOW,
    min_hits: int = DEFAULT_SEQUENCE_MIN_HITS,
) -> list[RelationFinding]:
    """对**序列**做「人下吊物」的**时序持久性**判定（方案阶段 2 第 6 项）。

    单帧命中不足以定论（检测抖动 / 单帧误报），需**连续多帧命中**。

    **帧率适配（换算依据）**：Singh & Seif 2026 的原规则是「**1 秒**滑窗（15 帧）内
    **≥10 帧**命中才判 Unsafe」——命中比例 10/15 ≈ **2/3**。本项目抽帧间隔是**秒级**
    （实测约 2~6 秒/帧），1 秒窗内**不足 1 帧**，机械照抄「15 帧窗 / ≥10 帧」无意义。
    故**按比例适配**：取最小有意义的窗 ``window_frames=3``，命中下限
    ``min_hits=2``（2/3 ≈ 论文比例）。稀疏帧下「连续 2 帧命中」= 至少跨约 2~6 秒持续
    处于危险区，是比 1 秒窗**更强**的悬停/滞留证据。

    实现：按 ``source_id`` 分组、按 ``frame_index`` 排序；逐帧取**最优**命中
    （``margin`` 最大的一对）；滑动窗内命中帧数 ≥ ``min_hits`` 的帧段视为**持续命中**，
    合并为连续段后每段产出一条结论（``frame_indices`` 记录命中帧、``evidence`` 记录
    窗参数与命中数）。

    Args:
        frames: 候选帧序列（可跨 ``source_id``，内部按 ``source_id`` 分组）。
        aspect_ratio: 预留参数（见 :func:`find_person_under_load`）。
        image_size: 图像尺寸 ``(宽, 高)``；缺省用 :data:`REFERENCE_IMAGE_SIZE`。
        window_frames: 滑窗帧数（默认 3）。
        min_hits: 窗内命中帧数下限（默认 2）。

    Returns:
        时序持久命中的关系结论列表（``verdict`` 取该段最优帧的三级判定）。
    """
    _ = aspect_ratio
    size = (int(image_size[0]), int(image_size[1])) if image_size else REFERENCE_IMAGE_SIZE
    window = max(1, int(window_frames))
    threshold = max(1, int(min_hits))

    grouped: dict[str, list[CandidateFrame]] = {}
    for frame in frames:
        grouped.setdefault(frame.source_id, []).append(frame)

    findings: list[RelationFinding] = []
    for source_id, group in grouped.items():
        ordered = sorted(enumerate(group), key=lambda pair: _frame_index(pair[1], pair[0]))
        ordered_frames = [frame for _index, frame in ordered]
        if not ordered_frames:
            continue

        per_frame: list[tuple[int, RelationFinding | None]] = []
        for position, frame in enumerate(ordered_frames):
            index = _frame_index(frame, position)
            hits = find_person_under_load(
                frame.detections or [], image_size=size
            )
            best = max(hits, key=lambda item: item.margin) if hits else None
            per_frame.append((index, best))

        total = len(per_frame)
        persistent = [False] * total
        for start in range(total):
            window_slice = per_frame[start : start + window]
            count = sum(1 for _index, finding in window_slice if finding is not None)
            if count >= threshold:
                for offset in range(start, min(total, start + window)):
                    if per_frame[offset][1] is not None:
                        persistent[offset] = True

        runs: list[list[int]] = []
        current: list[int] = []
        for position in range(total):
            if persistent[position]:
                current.append(position)
            elif current:
                runs.append(current)
                current = []
        if current:
            runs.append(current)

        for run in runs:
            hit_frames = [per_frame[position] for position in run if per_frame[position][1]]
            if len(hit_frames) < threshold:
                continue
            best_index, best = max(hit_frames, key=lambda pair: pair[1].margin)  # type: ignore[union-attr]
            frame_indices = [index for index, _finding_item in hit_frames]
            evidence = dict(best.evidence)
            evidence.update(
                {
                    "source_id": source_id,
                    "frame_indices": frame_indices,
                    "persistence": {
                        "window_frames": window,
                        "min_hits": threshold,
                        "hit_frames": len(hit_frames),
                    },
                }
            )
            findings.append(
                RelationFinding(
                    hazard_id=HAZARD_PERSON_UNDER_LOAD,
                    relation="person_under_load",
                    confidence=best.confidence,
                    supporting=list(best.supporting),
                    notes=(
                        f"时序持久命中：{len(hit_frames)}/{window} 帧命中"
                        f"（≥{threshold}）；代表帧 index={best_index}；"
                        + best.notes
                    ),
                    verdict=best.verdict,
                    margin=best.margin,
                    evidence=evidence,
                )
            )
    return findings


def find_oblique_lift(
    dets: Iterable[Detection],
    angle_threshold_deg: float = OBLIQUE_ANGLE_THRESHOLD_DEG,
    aspect_ratio: float = 1.0,
    image_size: tuple[int, int] | None = None,
) -> list[RelationFinding]:
    """发现「歪拉斜吊」（``D-4-3``）的关系。

    对每个 ``load``，取其**上方最近**的 ``hook`` / ``rigging``，计算
    :func:`vertical_angle_deg`；超过 ``angle_threshold_deg`` → 产出结论。
    ``supporting`` 依次为 ``[load, hook]``。

    Args:
        dets: 同一帧的检测框集合。
        angle_threshold_deg: 偏离竖直的角度阈值（度）。
        aspect_ratio: 画面宽高比（见 :func:`vertical_angle_deg`）。
        image_size: 图像尺寸（本判据在归一化空间计算，仅写入 evidence 供复核）。

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
            # 角度越大越确定（归一化到 [0,1] 的决策余量）。
            margin = min(1.0, (angle - angle_threshold_deg) / max(angle_threshold_deg, 1e-6))
            findings.append(
                _finding(
                    HAZARD_OBLIQUE_LIFT,
                    "oblique_lift",
                    [load, nearest],
                    f"吊点连线偏离竖直 {angle:.1f}°（阈值 {angle_threshold_deg:.1f}°）；"
                    "本期留位，不接入交付指标。",
                    verdict=verdict_for_margin(margin),
                    margin=margin,
                    evidence={
                        "angle_deg": round(angle, 6),
                        "angle_threshold_deg": angle_threshold_deg,
                        "aspect_ratio": aspect_ratio,
                        "image_size": list(image_size) if image_size else None,
                    },
                )
            )
    return findings


def find_relations(
    dets: Iterable[Detection],
    aspect_ratio: float = 1.0,
    image_size: tuple[int, int] | None = None,
) -> list[RelationFinding]:
    """聚合本层全部关系结论（当前 = 人下吊物 ``D-5-3`` + 歪拉斜吊 ``D-4-3``）。"""
    detections = list(dets)
    return [
        *find_person_under_load(detections, aspect_ratio=aspect_ratio, image_size=image_size),
        *find_oblique_lift(
            detections, aspect_ratio=aspect_ratio, image_size=image_size
        ),
    ]


# 显式标注本层「留位不接入指标」——供文档/自检读取。
LAYER_NOTE: str = "本期留位，不接入交付指标（检测权重为唯一交付物）。"
