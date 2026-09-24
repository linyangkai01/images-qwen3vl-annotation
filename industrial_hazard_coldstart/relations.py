"""``relations`` 命令（根治方案 v2 §8 阶段 3 第 9–10 项；**留位能力，不接入交付指标**）。

**做什么**：吃 ``workspace/review/annotations.jsonl`` 的**标注框**（``qwen_detections``，
复核后的框——**不是**模型对隐患的文本预测），调用 :mod:`industrial_hazard_coldstart.layers.geometry`
的射影判据（Singh & Seif 2026，方案 §4.4），产出**独立**的
``workspace/relations/relations.jsonl``。

**为什么独立、不接进 EvalReport**（方案 §8 阶段 3 第 9–10 项 + 需求决策）：

- 本期**唯一交付物是目标检测权重**；几何关系层的结论**未经检测验证**、也**无金标准**，
  混进验收判定会污染交付口径；
- 因此本命令**只落一个旁路产物**（``relations.jsonl``），并**保持隔离断言**：
  :mod:`industrial_hazard_coldstart.evaluate` **不得** ``import``
  ``industrial_hazard_coldstart.layers``（见 ``tests/test_layers.py``）。本模块是
  **唯一**把 ``layers`` 拉进运行链路的入口，且仅在 ``relations`` 阶段被调用。

**输出结构**：``relations.jsonl`` 每行 = 一条 :class:`RelationFinding` 的 ``to_dict()``
外加三项**定位上下文**：

- ``scope``：``"frame"``（单帧关系）或 ``"sequence"``（时序持久命中）；
- ``relative_path`` / ``source_id`` / ``frame_index``：该结论所属帧（时序结论取代表帧）。

每行仍带三级判定 ``verdict``（``confirmed_danger`` / ``suspected`` / ``needs_review``）
与决策余量 ``margin``（``|Δ|``），便于人工复核。

**坐标系**：``Detection.bbox`` 是**归一化**坐标；像素阈值（30px / 40px）的标定基准由
``image_size`` 决定，默认 :data:`~industrial_hazard_coldstart.layers.geometry.REFERENCE_IMAGE_SIZE`
（960×540）；若帧的 ``candidate.metadata`` 带 ``width`` / ``height`` 则优先采用。

仅使用 Python 标准库 + 母项目 + 本包 ``layers``；零新增第三方依赖。
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

from traffic_cv_closed_loop.config import ProjectConfig
from traffic_cv_closed_loop.io_utils import read_jsonl, write_json, write_jsonl
from traffic_cv_closed_loop.models import AnnotationRecord, CandidateFrame, Detection

from industrial_hazard_coldstart.layers import geometry
from industrial_hazard_coldstart.models import RelationFinding

__all__ = [
    "RELATIONS_DIRNAME",
    "RELATIONS_FILENAME",
    "RELATIONS_SUMMARY_FILENAME",
    "annotation_detections",
    "frame_image_size",
    "relations_for_records",
    "run_relations",
]

#: ``relations`` 产物目录名（相对 workspace）。
RELATIONS_DIRNAME: str = "relations"
#: 关系结论文件名。
RELATIONS_FILENAME: str = "relations.jsonl"
#: 关系结论摘要文件名。
RELATIONS_SUMMARY_FILENAME: str = "summary.json"


# --------------------------------------------------------------------------- #
# 标注框 / 图像尺寸
# --------------------------------------------------------------------------- #
def annotation_detections(record: AnnotationRecord) -> list[Detection]:
    """取一帧的**标注框**（不是模型预测）。

    优先 ``qwen_detections``（两段式复核后的标注框）；为空时回落到
    ``candidate.detections``（上游初检框）——**绝不**使用模型对隐患的**文本**预测
    （那属于 ``hazard_report``，由 ``annotate`` 阶段处理，不在本命令职责内）。
    """
    reviewed = list(record.qwen_detections or [])
    if reviewed:
        return reviewed
    candidate = getattr(record, "candidate", None)
    return list(getattr(candidate, "detections", None) or [])


def frame_image_size(
    record: AnnotationRecord,
    default: tuple[int, int] = geometry.REFERENCE_IMAGE_SIZE,
) -> tuple[int, int]:
    """取一帧的**像素尺寸** ``(宽, 高)``；缺失时用 ``default``（标定基准 960×540）。

    像素阈值（30px / 40px）依赖图像尺寸——真实帧若在 ``candidate.metadata`` 里记了
    ``width`` / ``height`` 就采用，否则回落到参考分辨率（吊装视频原生 960×540）。
    """
    candidate = getattr(record, "candidate", None)
    meta = getattr(candidate, "metadata", None) or {}
    try:
        width = int(meta.get("width"))
        height = int(meta.get("height"))
    except (TypeError, ValueError):
        return (int(default[0]), int(default[1]))
    if width > 0 and height > 0:
        return (width, height)
    return (int(default[0]), int(default[1]))


def _as_size(image_size: Sequence[int] | None) -> tuple[int, int]:
    """规整图像尺寸参数；``None`` → 参考分辨率。"""
    if image_size:
        return (int(image_size[0]), int(image_size[1]))
    return geometry.REFERENCE_IMAGE_SIZE


def _frame_index(frame: CandidateFrame, fallback: int) -> int:
    """取帧序号：优先 ``metadata['frame_index']``，否则用组内序号。"""
    value = (frame.metadata or {}).get("frame_index")
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _frame_with_annotations(record: AnnotationRecord) -> CandidateFrame:
    """用**标注框**替换 ``candidate.detections``，保留源信息（供时序分组 / 排序）。"""
    candidate = record.candidate
    return CandidateFrame(
        source_id=candidate.source_id,
        relative_path=candidate.relative_path,
        original_path=candidate.original_path,
        extracted_path=candidate.extracted_path,
        detections=annotation_detections(record),
        scene=candidate.scene,
        metadata=dict(candidate.metadata or {}),
    )


# --------------------------------------------------------------------------- #
# 关系结论构造
# --------------------------------------------------------------------------- #
def _row(
    finding: RelationFinding,
    *,
    scope: str,
    relative_path: str,
    source_id: str,
    frame_index: int,
) -> dict[str, Any]:
    """把一条 :class:`RelationFinding` 展平为一行（结论字段 + 定位上下文）。"""
    row = finding.to_dict()
    row.update(
        {
            "scope": scope,
            "relative_path": relative_path,
            "source_id": source_id,
            "frame_index": int(frame_index),
        }
    )
    return row


def relations_for_records(
    records: Iterable[AnnotationRecord],
    *,
    image_size: Sequence[int] | None = None,
    include_temporal: bool = True,
) -> list[dict[str, Any]]:
    """把标注记录转成 ``relations`` 行（**纯函数**，便于单测）。

    两类结论：

    1. **逐帧关系**（``scope="frame"``）：对每帧的标注框跑
       :func:`~industrial_hazard_coldstart.layers.geometry.find_relations`
       （人下吊物 ``D-5-3`` + 歪拉斜吊 ``D-4-3``）；
    2. **时序持久命中**（``scope="sequence"``）：按 ``source_id`` 分组、按
       ``frame_index`` 排序后跑
       :func:`~industrial_hazard_coldstart.layers.geometry.find_person_under_load_sequence`
       （单帧命中不足以定论，需连续多帧命中，见该函数 docstring 的帧率适配说明）。

    Args:
        records: 标注记录（``annotations.jsonl`` 反序列化而来）。
        image_size: 图像尺寸 ``(宽, 高)``；缺省逐帧从 ``metadata`` 取或回落参考分辨率。
        include_temporal: 是否附上时序持久结论（默认 ``True``）。

    Returns:
        关系结论行列表（每行 = ``RelationFinding.to_dict()`` + 定位上下文）。
    """
    record_list = list(records)
    size = _as_size(image_size)

    rows: list[dict[str, Any]] = []
    frames: list[CandidateFrame] = []
    #: ``(source_id, frame_index) → relative_path``，供时序结论回填代表帧路径。
    path_lookup: dict[tuple[str, int], str] = {}

    for index, record in enumerate(record_list):
        frame = _frame_with_annotations(record)
        frame_index = _frame_index(frame, index)
        frames.append(frame)
        path_lookup[(frame.source_id, frame_index)] = frame.relative_path

        per_frame_size = frame_image_size(record, size)
        findings = geometry.find_relations(frame.detections, image_size=per_frame_size)
        for finding in findings:
            rows.append(
                _row(
                    finding,
                    scope="frame",
                    relative_path=frame.relative_path,
                    source_id=frame.source_id,
                    frame_index=frame_index,
                )
            )

    if include_temporal:
        for finding in geometry.find_person_under_load_sequence(frames, image_size=size):
            evidence = finding.evidence or {}
            source_id = str(evidence.get("source_id", ""))
            indices = [int(item) for item in (evidence.get("frame_indices") or [])]
            representative = indices[0] if indices else -1
            rows.append(
                _row(
                    finding,
                    scope="sequence",
                    relative_path=path_lookup.get((source_id, representative), ""),
                    source_id=source_id,
                    frame_index=representative,
                )
            )

    return rows


# --------------------------------------------------------------------------- #
# 阶段入口
# --------------------------------------------------------------------------- #
def run_relations(
    config: ProjectConfig,
    *,
    image_size: Sequence[int] | None = None,
) -> dict[str, Any]:
    """执行 ``relations`` 阶段：标注框 → 几何关系结论（独立旁路产物）。

    流程：

    1. 读 ``workspace/review/annotations.jsonl``（**标注框，不是模型预测**）；
    2. :func:`relations_for_records` 逐帧 + 时序推理（射影判据，方案 §4.4）；
    3. 落盘 ``workspace/relations/relations.jsonl``（每行带 ``scope`` / 定位上下文 /
       三级判定 ``verdict`` / 决策余量 ``margin``）；
    4. 落盘 ``workspace/relations/summary.json`` + ``workspace/status/relations.json``。

    **留位性质**：本阶段结论**不接入** ``EvalReport`` 验收判定（见模块 docstring）。

    Args:
        config: 项目配置。
        image_size: 图像尺寸 ``(宽, 高)``；缺省逐帧从 ``metadata`` 取或回落 960×540。

    Returns:
        阶段摘要（含计数、按隐患 / 按 scope / 按三级判定的分布、产物路径）。

    Raises:
        ValueError: ``annotations.jsonl`` **缺失或为空**——无标注框可推理，
            **fail-loud**（与 ``annotate`` / ``rebuild-hazards`` 同一护栏，
            绝不产出空结果假装成功）。
    """
    if not config.annotations_path.exists():
        raise ValueError(
            f"未找到标注记录 {config.annotations_path}；"
            "relations 阶段需先有 `annotate` 产物（annotations.jsonl）。"
        )
    records = [
        AnnotationRecord.from_dict(item) for item in read_jsonl(config.annotations_path)
    ]
    if not records:
        raise ValueError(
            f"标注记录 {config.annotations_path} 为空；无标注框可推理，拒绝产出空结果。"
        )

    rows = relations_for_records(records, image_size=image_size)

    out_dir = config.workspace(RELATIONS_DIRNAME)
    out_path = out_dir / RELATIONS_FILENAME
    write_jsonl(out_path, rows)

    by_hazard: dict[str, int] = {}
    by_scope: dict[str, int] = {}
    by_verdict: dict[str, int] = {}
    for row in rows:
        hazard_id = str(row.get("hazard_id", ""))
        by_hazard[hazard_id] = by_hazard.get(hazard_id, 0) + 1
        scope = str(row.get("scope", ""))
        by_scope[scope] = by_scope.get(scope, 0) + 1
        verdict = str(row.get("verdict", ""))
        by_verdict[verdict] = by_verdict.get(verdict, 0) + 1

    summary: dict[str, Any] = {
        "stage": "relations",
        "annotations_used": len(records),
        "relation_count": len(rows),
        "by_hazard": by_hazard,
        "by_scope": by_scope,
        "by_verdict": by_verdict,
        "relations_path": str(out_path),
        "note": (
            "几何关系层结论（**留位能力，不接入交付指标**）："
            "吃 annotations.jsonl 的标注框，吐独立 relations.jsonl。"
        ),
    }
    write_json(out_dir / RELATIONS_SUMMARY_FILENAME, summary)
    write_json(config.status_dir / "relations.json", summary)
    return summary
