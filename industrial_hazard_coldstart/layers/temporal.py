"""T05.4 时序层（**本期留位，不接入交付指标**）。

**性质说明**：本期交付物**只有目标检测权重**；时序层只**落代码 + 数据结构 + 单测**，
其结论**不得**混进 :class:`EvalReport` 的验收判定。原因是这类隐患**单帧做不了**——
必须比较**多帧**的运动 / 静止：

- ``D-3-6``（吊物悬停在空中）：需判断吊物在相邻帧间**静止**（悬停）；
- （``D-3-7`` 相邻龙门吊重物长时间悬停同属本层，见
  :data:`taxonomy.S1_TEMPORAL_HAZARD_IDS`。）

结论以 :class:`~industrial_hazard_coldstart.models.TemporalFinding` 表达，
``frame_indices`` 记录支撑该结论的帧序列，``finding`` 取 ``suspended`` / ``moving``。

**判定口径**：按 ``source_id`` 分组，对 ``load`` 检测框做跨帧 IoU 跟踪；某条轨迹持续
帧数 ≥ ``min_frames`` 且**最大中心位移** ≤ ``max_displacement`` → 判 ``suspended``，
否则判 ``moving``。``kf`` 为**可选**的像素级运动复核器（如母项目 ``ImageStats``）：
提供时对轨迹首末帧做像素运动比交叉核对，可把「框稳定但画面整体在动」降级为 ``moving``。

仅使用 Python 标准库 + 母项目；零新增第三方依赖。
"""

from __future__ import annotations

from typing import Any, Sequence

from industrial_hazard_coldstart.models import CandidateFrame, Detection, TemporalFinding

__all__ = [
    "HAZARD_SUSPENDED_LOAD",
    "DEFAULT_MIN_FRAMES",
    "DEFAULT_MAX_DISPLACEMENT",
    "DEFAULT_TRACK_IOU",
    "DEFAULT_MAX_MATCH_DISTANCE",
    "DEFAULT_PIXEL_MOTION_LIMIT",
    "detect_suspended_load",
    "track_loads",
]

#: 本层服务的隐患 id（与 :mod:`taxonomy` 的分桶一致）。
HAZARD_SUSPENDED_LOAD: str = "D-3-6"

#: 轨迹持续帧数下限（达到才判定）。
DEFAULT_MIN_FRAMES: int = 3

#: 判定「静止」的最大中心位移（归一化坐标）。
DEFAULT_MAX_DISPLACEMENT: float = 0.02

#: 跨帧跟踪的 IoU 阈值。
DEFAULT_TRACK_IOU: float = 0.3

#: 跨帧跟踪的中心距离上限（IoU 不达标时的回落匹配；保证「快速移动」也能成轨迹）。
DEFAULT_MAX_MATCH_DISTANCE: float = 0.25

#: 像素运动比上限（超过则即使框稳定也判「运动」）。
DEFAULT_PIXEL_MOTION_LIMIT: float = 0.15


def _center(bbox: Sequence[float]) -> tuple[float, float]:
    """矩形中心 ``(cx, cy)``。"""
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _iou(a: Sequence[float], b: Sequence[float]) -> float:
    """两框 IoU。"""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def _loads(frame: CandidateFrame) -> list[Detection]:
    """取一帧的 ``load`` 检测框。"""
    return [item for item in (frame.detections or []) if item.label == "load"]


def _frame_index(frame: CandidateFrame, fallback: int) -> int:
    """取帧序号：优先 ``metadata['frame_index']``，否则用组内序号。"""
    value = (frame.metadata or {}).get("frame_index")
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def track_loads(
    frames: Sequence[CandidateFrame],
    iou_threshold: float = DEFAULT_TRACK_IOU,
    max_match_distance: float = DEFAULT_MAX_MATCH_DISTANCE,
) -> list[list[tuple[int, Detection]]]:
    """对 ``load`` 检测框做跨帧贪心跟踪，返回若干条轨迹。

    匹配判据：优先 IoU ≥ ``iou_threshold``；否则若中心距离 ≤ ``max_match_distance``
    也可续接（保证「快速移动、相邻帧不重叠」的吊物仍能形成轨迹 → 才能判 ``moving``）。

    Args:
        frames: 同一 ``source_id`` 的候选帧（按时间顺序）。
        iou_threshold: 跨帧匹配的 IoU 阈值。
        max_match_distance: 中心距离回落匹配上限（归一化坐标）。

    Returns:
        轨迹列表；每条轨迹为 ``[(frame_index, Detection), ...]``。
    """
    tracks: list[list[tuple[int, Detection]]] = []
    active: list[dict[str, Any]] = []  # 每条轨迹的末尾状态：{"track": [...], "last": Detection}
    for position, frame in enumerate(frames):
        index = _frame_index(frame, position)
        loads = _loads(frame)

        # 贪心匹配：候选对按匹配分降序（IoU 命中优先于距离命中）。
        pairs: list[tuple[float, int, int]] = []
        for load_index, load in enumerate(loads):
            for slot, state in enumerate(active):
                overlap = _iou(load.bbox, state["last"].bbox)
                if overlap >= iou_threshold:
                    pairs.append((1.0 + overlap, load_index, slot))
                    continue
                load_cx, load_cy = _center(load.bbox)
                last_cx, last_cy = _center(state["last"].bbox)
                distance = max(abs(load_cx - last_cx), abs(load_cy - last_cy))
                if distance <= max_match_distance:
                    score = (max_match_distance - distance) / max_match_distance
                    pairs.append((score, load_index, slot))
        pairs.sort(reverse=True)
        matched_loads: set[int] = set()
        matched_slots: set[int] = set()
        for _score, load_index, slot in pairs:
            if load_index in matched_loads or slot in matched_slots:
                continue
            matched_loads.add(load_index)
            matched_slots.add(slot)
            active[slot]["track"].append((index, loads[load_index]))
            active[slot]["last"] = loads[load_index]

        # 本帧未续上的轨迹 → 归档（保证轨迹是**连续**帧段，符合「静止/运动」语义）。
        still_active: list[dict[str, Any]] = []
        for slot, state in enumerate(active):
            if slot in matched_slots:
                still_active.append(state)
            else:
                tracks.append(state["track"])
        active = still_active

        # 未匹配的 load → 开新轨迹。
        for load_index, load in enumerate(loads):
            if load_index not in matched_loads:
                active.append({"track": [(index, load)], "last": load})

    for state in active:
        tracks.append(state["track"])
    return tracks


def _pixel_motion(
    kf: Any,
    first: CandidateFrame,
    last: CandidateFrame,
) -> float | None:
    """用可选的 ``kf`` 运动复核器算首末帧像素运动比；不可用时返回 ``None``。"""
    if kf is None:
        return None
    motion_ratio = getattr(kf, "motion_ratio", None)
    if not callable(motion_ratio):
        return None
    try:
        from traffic_cv_closed_loop.imageio import load_image

        first_image = load_image(first.extracted)
        last_image = load_image(last.extracted)
        return float(motion_ratio(first_image, last_image))
    except (OSError, ValueError, AttributeError):
        return None


def detect_suspended_load(
    frames: Sequence[CandidateFrame],
    kf: Any = None,
    *,
    min_frames: int = DEFAULT_MIN_FRAMES,
    max_displacement: float = DEFAULT_MAX_DISPLACEMENT,
    iou_threshold: float = DEFAULT_TRACK_IOU,
    max_match_distance: float = DEFAULT_MAX_MATCH_DISTANCE,
    pixel_motion_limit: float = DEFAULT_PIXEL_MOTION_LIMIT,
) -> list[TemporalFinding]:
    """对相邻帧给出吊物**静止 / 运动**判定，服务 ``D-3-6``（吊物悬停在空中）。

    Args:
        frames: 候选帧序列（可跨 ``source_id``，内部按 ``source_id`` 分组）。
        kf: 可选的像素运动复核器（提供 ``motion_ratio(img1, img2)``，如母项目
            ``ImageStats``）；``None`` 时仅用检测框稳定性判定。
        min_frames: 轨迹持续帧数下限。
        max_displacement: 判定「静止」的最大中心位移（归一化坐标）。
        iou_threshold: 跨帧跟踪的 IoU 阈值。
        max_match_distance: 跨帧跟踪的中心距离回落匹配上限。
        pixel_motion_limit: 像素运动比上限（超过则降级为 ``moving``）。

    Returns:
        :class:`TemporalFinding` 列表；``finding`` 取 ``suspended`` / ``moving``，
        ``hazard_id`` 为 ``D-3-6``，``frame_indices`` 为轨迹帧序号。
    """
    grouped: dict[str, list[CandidateFrame]] = {}
    for frame in frames:
        grouped.setdefault(frame.source_id, []).append(frame)

    findings: list[TemporalFinding] = []
    for source_id, group in grouped.items():
        ordered = sorted(
            enumerate(group), key=lambda pair: _frame_index(pair[1], pair[0])
        )
        ordered_frames = [frame for _index, frame in ordered]
        for track in track_loads(
            ordered_frames,
            iou_threshold=iou_threshold,
            max_match_distance=max_match_distance,
        ):
            if len(track) < min_frames:
                continue
            centers = [_center(detection.bbox) for _index, detection in track]
            max_shift = 0.0
            for first_center in centers:
                for second_center in centers:
                    shift = max(
                        abs(first_center[0] - second_center[0]),
                        abs(first_center[1] - second_center[1]),
                    )
                    max_shift = max(max_shift, shift)

            frame_indices = [index for index, _detection in track]
            # 找到轨迹首末帧对象（按 frame_index 对齐）用于像素复核。
            by_index = {_frame_index(frame, i): frame for i, frame in enumerate(ordered_frames)}
            first_frame = by_index.get(frame_indices[0])
            last_frame = by_index.get(frame_indices[-1])
            motion = (
                _pixel_motion(kf, first_frame, last_frame)
                if first_frame is not None and last_frame is not None
                else None
            )

            stable = max_shift <= max_displacement
            if stable and motion is not None and motion > pixel_motion_limit:
                stable = False  # 框稳定但画面整体在动 → 不是「悬停」。
            finding = "suspended" if stable else "moving"

            stability = max(0.0, 1.0 - max_shift / max(max_displacement, 1e-6))
            span = min(1.0, len(track) / (min_frames * 2))
            confidence = round(min(1.0, 0.5 * stability + 0.5 * span), 6)
            notes = (
                f"轨迹持续 {len(track)} 帧，最大中心位移 {max_shift:.4f}"
                f"（阈值 {max_displacement}）"
                + (f"，像素运动比 {motion:.4f}" if motion is not None else "")
                + "；本期留位，不接入交付指标。"
            )
            findings.append(
                TemporalFinding(
                    hazard_id=HAZARD_SUSPENDED_LOAD,
                    source_id=source_id,
                    finding=finding,
                    confidence=confidence,
                    frame_indices=frame_indices,
                    notes=notes,
                )
            )
    return findings


# 显式标注本层「留位不接入指标」——供文档/自检读取。
LAYER_NOTE: str = "本期留位，不接入交付指标（检测权重为唯一交付物）。"
