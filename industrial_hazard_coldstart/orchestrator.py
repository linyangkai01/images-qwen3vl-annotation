"""阶段编排：把素材层、母项目初检与两段式标注串起来。

实现的阶段：

- ``inventory`` —— 只读素材编目（:func:`inventory.run_inventory`）；
- ``extract-video`` —— 调 :func:`video_decode.run_video_extract`，把关键帧写入
  ``config.input_frames_dir/<source_id>/``；
- ``extract`` —— 调母项目 ``pipeline.extraction.run_extraction``，在关键帧上做
  通用 YOLO 零样本初检，产出 ``candidates.jsonl``；
- ``annotate``（T03.2 + T03.3）—— **复用母项目** ``pipeline.annotation.run_annotation``
  做两段式复核标注（产出 ``labels/`` + ``annotations.jsonl`` + ``review_queue.jsonl``），
  随后对模型返回的隐患做**受控词表校验 + 本地 KB 回填**，产出
  ``workspace/hazards/report.json``；
- ``rebuild-hazards`` —— **只重建** ``workspace/hazards/report.json``（从现有
  ``annotations.jsonl`` + ``raw_responses`` 重算），**不调用 VLM、不重新标注、不读
  candidates.jsonl**。用于给旧代码产出的报告补上新字段（如幻觉过滤标记），
  省下整轮标注耗时（:func:`rebuild_hazard_report`）；
- ``prioritize``（T03.4）—— 对标注记录做不确定性采样，产出
  ``workspace/review/priority_queue.jsonl``；
- ``curate``（T04.1）—— 质量过滤 + 版本化数据集构建（:func:`run_curate`）；
- ``freeze-holdout``（T04.2）—— 冻结 / 读回固定 holdout（:func:`run_freeze_holdout`）；
- ``train``（T04.3）—— 版本化数据集 + 固定 holdout 上的冷启动训练（:func:`run_train`）。

**串联逻辑**：``extract`` 阶段默认先确保关键帧已就位——若 ``input_frames_dir``
尚无任何图片且能发现视频，则自动先跑一遍 ``extract-video``，再调
``run_extraction``。该行为可用 ``auto_extract_video=False`` 关闭。

产出的 ``CandidateFrame`` / ``AnnotationRecord`` **直接复用母项目类**（不重定义），
结构与母项目完全兼容。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from traffic_cv_closed_loop.adapters.qwen import MockQwenAnnotator, build_qwen_annotator
from traffic_cv_closed_loop.adapters.yolo import build_trainer
from traffic_cv_closed_loop.config import ProjectConfig
from traffic_cv_closed_loop.io_utils import (
    ensure_dir,
    iter_image_files,
    read_json,
    read_jsonl,
    read_text,
    stem_sidecar,
    write_json,
    write_jsonl,
    write_text,
)
from traffic_cv_closed_loop.models import AnnotationRecord
from traffic_cv_closed_loop.pipeline.annotation import run_annotation
from traffic_cv_closed_loop.pipeline.extraction import run_extraction

from industrial_hazard_coldstart.active_learning import build_priority_queue
from industrial_hazard_coldstart.dataset import build_dataset, curate_records
from industrial_hazard_coldstart.evaluate import (
    DEFAULT_CONSUMPTION_TARGET,
    HARD_EXAMPLES_FILENAME,
    METRICS_FILENAME,
    NEXT_CYCLE_TASKS_FILENAME,
    REPORT_FILENAME,
    build_hard_examples,
    consumption_rate,
    evaluate,
    render_next_cycle_tasks,
)
from industrial_hazard_coldstart.holdout import (
    DEFAULT_HOLDOUT_RATIO,
    freeze_holdout,
    holdout_dir_of,
    holdout_hash_of,
    holdout_index_path,
    validate_holdout_ratio,
    verify_holdout,
)
from industrial_hazard_coldstart.inventory import is_ignorable_media_path, run_inventory
from industrial_hazard_coldstart.knowledge import (
    backfill_hazards,
    extract_model_hazards,
    load_knowledge_base,
    overall_risk_of,
    recommendation_of,
)
from industrial_hazard_coldstart.taxonomy import layer_of_hazard
from industrial_hazard_coldstart.video_decode import discover_videos, run_video_extract

__all__ = [
    "SUPPORTED_STAGES",
    "run_cycle",
    "run_annotate",
    "rebuild_hazard_report",
    "run_prioritize",
    "run_curate",
    "run_freeze_holdout",
    "run_train",
    "run_evaluate",
]

#: 默认数据集版本号（版本号**不用时间戳**，保证跨轮可比）。
DEFAULT_DATASET_VERSION: str = "coldstart-v1"

#: 本批实现的阶段名。
SUPPORTED_STAGES: tuple[str, ...] = (
    "inventory",
    "extract-video",
    "extract",
    "annotate",
    "rebuild-hazards",
    "prioritize",
    "curate",
    "freeze-holdout",
    "train",
    "evaluate",
)

#: 隐患被拒（KB 外 / 场景外）时写入复核队列的原因标记。
REVIEW_REASON_HAZARD_OUT_OF_VOCAB = "hazard_id_not_in_kb"

#: 模型「答了目标但取不到隐患报告」时写入复核队列的原因标记。
#:
#: 用于把「模型没答隐患题」（通道 B 静默为空）与「本帧确实没有隐患」区分开——
#: 二者若都不报错、都产出 ``hazards: []``，人工将无法察觉前者，是本项目的头号失败模式。
REVIEW_REASON_HAZARD_REPORT_MISSING = "hazard_report_missing"

#: 载荷里可能包裹模型输出的容器键（用于「多包一层」的容错）。
_PAYLOAD_CONTAINER_KEYS: tuple[str, ...] = (
    "parsed",
    "result",
    "data",
    "output",
    "message",
    "content",
    "response",
)

#: 解开嵌套容器的最大深度（防病态输入导致无限递归）。
_MAX_UNWRAP_DEPTH: int = 4

#: 判定「模型已作答」的输出键：出现任一即说明模型产出了结构化输出。
_MODEL_OUTPUT_KEYS: tuple[str, ...] = ("objects", "hazard_report", "hazards")


def _needs_video_extract(
    config: ProjectConfig,
    video_paths: list[Path] | None,
) -> bool:
    """判断 ``extract`` 前是否需要先抽帧：无图片、且有可发现的视频。"""
    if iter_image_files(config.input_frames_dir):
        return False
    return bool(discover_videos(config, video_paths))


def _count_sidecar_files(root: Path) -> int:
    """统计 ``root`` 下的 sidecar 检测文件数（``*.detections.json``，跳过隐藏/元数据）。"""
    if not root.exists():
        return 0
    return sum(
        1
        for path in root.rglob("*.detections.json")
        if path.is_file() and not is_ignorable_media_path(path, root)
    )


def _extract_warnings(config: ProjectConfig) -> list[str]:
    """``extract`` 阶段的可见告警（P2：sidecar 零命中不得静默）。

    真实场景误配 ``yolo.detector=sidecar``（默认值面向 mock）时，若**一帧 sidecar 都
    没读到**，全部帧会被判为空 → ``selected_candidates: 0`` 但退出码 0，调用方会误以为
    「这批素材真的没目标」。故此处显式告警，使其**可见**（不报错——sidecar 模式本身合法）。
    """
    warnings: list[str] = []
    if str(config.yolo_detector).lower() == "sidecar":
        if _count_sidecar_files(config.input_frames_dir) == 0:
            warnings.append(
                "sidecar_zero_hit：检测器为 `sidecar`，但在 input_frames_dir 下**未找到任何 "
                "`*.detections.json` sidecar 文件**，全部帧将因「无有效检测」被跳过。"
                "若本意是用真实模型初检，请把 `yolo.detector` 改为 `ultralytics`；"
                "若确实依赖 sidecar，请确认 sidecar 已随帧生成。"
            )
    return warnings


# --------------------------------------------------------------------------- #
# annotate 阶段（T03.2 两段式标注 + T03.3 语义层受控词表 / KB 回填）
# --------------------------------------------------------------------------- #
def _decode_json_string(value: Any) -> Any:
    """若 ``value`` 是 JSON 对象/数组字符串，解码**一次**；否则原样返回。

    真实模型（尤其 thinking 模型）偶尔把整个 JSON **二次编码**成字符串塞进某字段。
    这里只在字符串以 ``{`` / ``[`` 开头时才尝试解析，避免对普通文本做无谓解析；
    解析失败则原样返回（绝不因容错而抛错）。
    """
    if not isinstance(value, str):
        return value
    text = value.strip()
    if text[:1] not in ("{", "["):
        return value
    try:
        return json.loads(text)
    except (ValueError, TypeError):
        return value


def _locate_model_output(payload: Any, depth: int = 0) -> dict[str, Any] | None:
    """在原始载荷里定位「模型的输出对象」。

    模型输出对象是**含 ``objects`` / ``hazard_report`` / ``hazards`` 任一键的 dict**。
    为兼容真实 VLM 的多种吐法，逐层解开：

    - 二次编码的 JSON 字符串（:func:`_decode_json_string`）；
    - 列表包裹（取首个命中的元素）；
    - 一层或多层容器键（``parsed`` / ``result`` / ``data`` / ``output`` / ``response`` …），
      深度上限 :data:`_MAX_UNWRAP_DEPTH`。

    找不到返回 ``None``。
    """
    payload = _decode_json_string(payload)
    if isinstance(payload, list):
        for item in payload:
            found = _locate_model_output(item, depth + 1)
            if found is not None:
                return found
        return None
    if not isinstance(payload, dict):
        return None
    if any(key in payload for key in _MODEL_OUTPUT_KEYS):
        return payload
    if depth >= _MAX_UNWRAP_DEPTH:
        return None
    for key in _PAYLOAD_CONTAINER_KEYS:
        if key in payload:
            found = _locate_model_output(payload[key], depth + 1)
            if found is not None:
                return found
    return None


def _extract_hazard_from_raw(raw_payload: Any) -> dict[str, Any]:
    """从一帧的原始载荷里取出 ``hazard_report``。

    先定位模型输出对象（容忍二次编码 / 多包一层，见 :func:`_locate_model_output`），
    再取其 ``hazard_report``（同样容忍二次编码）；若定位到的对象**本身就是报告**
    （顶层直接是 ``{"hazards": [...]}``），则直接返回它。任何形态都取不到时返回 ``{}``。
    """
    output = _locate_model_output(raw_payload)
    if output is None:
        return {}
    report = _decode_json_string(output.get("hazard_report"))
    if isinstance(report, dict):
        return report
    if "hazards" in output:  # 定位到的对象本身就是隐患报告
        return output
    return {}


def _model_produced_output(raw_payload: Any) -> bool:
    """判断模型是否产出了可识别的输出（输出对象含 ``objects`` 键）。

    用于区分「**模型答了但没给 hazard_report**」与「**这帧本来就没隐患**」——
    只有前者才应被判为异常（否则会把「场景确实干净」误报成失败）。
    """
    output = _locate_model_output(raw_payload)
    return isinstance(output, dict) and "objects" in output


def _model_output_for(
    annotator: Any,
    record: AnnotationRecord,
) -> dict[str, Any]:
    """取一帧的**原始模型输出载荷**（供 :func:`_extract_hazard_from_raw` 解析）。

    - **mock 途径**：直接调 ``annotate_combined``（离线、零成本），它返回的合并载荷
      同时含 ``objects`` 与 ``hazard_report``（读 ``*.hazard.json`` 侧车或 ``*.qwen.json``
      里内嵌的 ``hazard_report``）；
    - **真实途径**：复用 ``run_annotation`` 已写下的原始响应，**不重复调用 VLM**
      （避免把昂贵的网络调用翻倍）。
    """
    if isinstance(annotator, MockQwenAnnotator):
        prompt_text = read_text(Path(record.prompt_path))
        _, _, combined = annotator.annotate_combined(record.candidate, prompt_text)
        return combined if isinstance(combined, dict) else {}
    raw = read_json(Path(record.raw_response_path), default={})
    return raw if isinstance(raw, dict) else {}


def _augment_review_queue(
    config: ProjectConfig,
    frames: set[str],
    reason: str,
) -> int:
    """把「隐患 id 越界」的帧补进复核队列（保持母项目 ``AnnotationRecord`` 格式）。

    只**追加原因并置 ``review_required=True``**，不新增自定义结构；已在队列里的帧就地
    补原因，不在队列里的帧从 ``annotations.jsonl`` 取回后加入。返回新增帧数。
    """
    if not frames:
        return 0
    queue = [AnnotationRecord.from_dict(item) for item in read_jsonl(config.review_queue_path)]
    annotations = [AnnotationRecord.from_dict(item) for item in read_jsonl(config.annotations_path)]
    by_path: dict[str, AnnotationRecord] = {
        record.candidate.relative_path: record for record in queue
    }
    added = 0
    for relative_path in sorted(frames):
        record = by_path.get(relative_path)
        if record is None:
            record = next(
                (item for item in annotations if item.candidate.relative_path == relative_path),
                None,
            )
            if record is None:
                continue
            by_path[relative_path] = record
            added += 1
        record.review_required = True
        record.review_reasons = sorted(set(record.review_reasons or []) | {reason})
        record.dataset_ready = config.include_review_queue_in_training

    ordered = [record for record in queue if record.candidate.relative_path in by_path]
    ordered_paths = {record.candidate.relative_path for record in ordered}
    ordered.extend(
        by_path[path] for path in by_path if path not in ordered_paths
    )
    write_jsonl(config.review_queue_path, [record.to_dict() for record in ordered])
    return added


def _build_hazard_report(
    config: ProjectConfig,
    records: list[AnnotationRecord],
    model_output_provider: Callable[[AnnotationRecord], dict[str, Any]],
) -> tuple[dict[str, Any], set[str], set[str]]:
    """逐帧取模型输出 → KB 回填 → 组装隐患报告（**不落盘、不增补复核队列**）。

    抽出来供两条路径共用，保证「正常标注」与「只重建报告」产出的 ``report.json``
    **结构完全一致**：

    - :func:`run_annotate`：``model_output_provider`` 走 annotator（真实 = 读原始响应，
      mock = 读侧车）；
    - :func:`rebuild_hazard_report`：``model_output_provider`` **只读磁盘**，
      **绝不构造或调用 annotator**（零 VLM 调用）。

    Args:
        config: 项目配置。
        records: 已存在的标注记录（``annotations.jsonl``）。
        model_output_provider: 取「单帧原始模型输出载荷」的可调用对象。

    Returns:
        ``(report, frames_needing_review, frames_without_report)``。
    """
    kb = load_knowledge_base(config.kb_path)
    allowed_ids = frozenset(kb.allowed_hazard_ids(config.scene))

    hazards: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    frames: list[dict[str, Any]] = []
    frames_needing_review: set[str] = set()
    frames_without_report: set[str] = set()
    model_overall = ""

    for record in records:
        relative_path = record.candidate.relative_path
        model_output = model_output_provider(record)
        hazard_report = _extract_hazard_from_raw(model_output)
        if hazard_report and not model_overall:
            model_overall = str(hazard_report.get("overall_risk", "") or "")

        # 护栏（QA-T03-01）：模型「答了目标却取不到隐患报告」→ 显式标记。
        # 只在「有 objects（说明模型答了）」却「无可用 hazard_report」时判定为异常，
        # 绝不把「本帧确实没有隐患」误报成失败。
        report_missing = not hazard_report and _model_produced_output(model_output)
        if report_missing:
            frames_without_report.add(relative_path)

        frame_resolved, frame_rejected = backfill_hazards(
            hazard_report, kb, allowed_ids, source_frame=relative_path
        )
        hazards.extend(frame_resolved)
        rejected.extend(frame_rejected)
        if frame_rejected:
            frames_needing_review.add(relative_path)
        frames.append(
            {
                "relative_path": relative_path,
                "model_hazard_count": len(extract_model_hazards(hazard_report)),
                "resolved_ids": [item["hazard_id"] for item in frame_resolved],
                "rejected_ids": [item["hazard_id"] for item in frame_rejected],
                "hazard_report_present": bool(hazard_report),
            }
        )

    # 幻觉过滤（需求 §8.3）：统计「能力范围外」隐患——几何/时序/OCR/超分辨/图像质量层，
    # 这些结论由 VLM 文本推断、未经检测验证，已在回填时逐条标记 needs_manual_verification。
    # 此处给出汇总计数，避免淹没在数十条隐患里被直接采信。**只统计、不过滤**。
    verification_by_layer: dict[str, int] = {}
    for item in hazards:
        if item.get("needs_manual_verification"):
            layer = layer_of_hazard(item["hazard_id"])
            verification_by_layer[layer] = verification_by_layer.get(layer, 0) + 1
    hazards_needing_verification = sum(verification_by_layer.values())

    report = {
        "scene_type": config.scene,
        "kb_scene_id": kb.scene_id_for(config.scene),
        "kb_source": kb.source_path,
        "allowed_hazard_count": len(allowed_ids),
        "hazards": hazards,
        "rejected": rejected,
        "overall_risk": overall_risk_of(hazards, model_overall),
        "recommendation": recommendation_of(hazards),
        "frames": frames,
        "verification": {
            "hazards_needing_verification": hazards_needing_verification,
            "by_layer": verification_by_layer,
            "note": (
                "needs_manual_verification=true 的隐患超出本期交付能力（非目标检测层），"
                "结论由 VLM 文本推断、未经检测验证，须人工确认；本报告只标记、不过滤。"
            ),
        },
        "counts": {
            "frames": len(records),
            "hazards_resolved": len(hazards),
            "hazards_rejected": len(rejected),
            "hazards_needing_verification": hazards_needing_verification,
            "frames_with_rejected_hazard": len(frames_needing_review),
            "frames_without_hazard_report": len(frames_without_report),
        },
    }
    return report, frames_needing_review, frames_without_report


def _rebuild_model_output_provider(record: AnnotationRecord) -> dict[str, Any]:
    """**重建专用**的模型输出来源：只读磁盘，**绝不构造或调用 annotator**（零 VLM 调用）。

    优先级：

    1. ``record.raw_response_path``（真实跑批产物：``parsed.hazard_report``）；
    2. 帧旁 ``*.hazard.json`` 侧车（mock 约定）；
    3. 仍无 → 返回原始响应，让「模型答了却无报告」护栏（QA-T03-01）照常生效。
    """
    raw: dict[str, Any] | None = None
    raw_path = str(record.raw_response_path or "").strip()
    if raw_path:
        loaded = read_json(Path(raw_path), default=None)
        raw = loaded if isinstance(loaded, dict) else None
    if raw is not None and _extract_hazard_from_raw(raw):
        return raw
    original = str(record.candidate.original_path or "").strip()
    if original:
        sidecar = read_json(stem_sidecar(Path(original), "hazard"), default=None)
        if isinstance(sidecar, dict) and sidecar:
            return {"hazard_report": sidecar}
    return raw or {}


def run_annotate(config: ProjectConfig) -> dict[str, Any]:
    """执行 ``annotate`` 阶段：两段式标注 + 隐患受控词表校验与本地回填。

    流程：

    1. **复用**母项目 :func:`traffic_cv_closed_loop.pipeline.annotation.run_annotation`
       完成两段式复核标注，产出 ``workspace/review/labels/`` + ``annotations.jsonl``
       + ``review_queue.jsonl``（**不重写标注逻辑**）；
    2. 加载本地知识库，取当前场景（``hoist`` → ``S1``）的**受控隐患 id 集合**；
    3. 逐帧取模型隐患报告，做 :func:`knowledge.backfill_hazards`：
       ``category / norm_reference / action`` **全部由 KB 回填**；KB 外 / 场景外的
       ``hazard_id`` **一律拒绝**并记入复核；
    4. **静默失败护栏**（QA-T03-01）：若某帧「模型答了（有 ``objects``）却取不到可用
       ``hazard_report``」，计入 ``counts.frames_without_hazard_report`` 并按
       ``hazard_report_missing`` 原因补进复核队列——把「模型没答隐患题」与
       「本帧确实没有隐患」区分开（二者否则完全不可区分）；
    5. **幻觉过滤**（需求 §8.3）：对**超出本期交付能力**的隐患（几何 / 时序 / OCR /
       超分辨 / 图像质量层，共 9 条）逐条加 ``needs_manual_verification`` +
       ``capability_note`` 标记，并在 ``counts.hazards_needing_verification`` /
       ``verification`` 给出汇总。**只标记、不过滤**（隐患可能确实存在，只是本期判不准）；
    6. 落盘 ``workspace/hazards/report.json`` 与 ``workspace/hazards/rejected.jsonl``。

    Returns:
        阶段摘要（含标注摘要与隐患回填统计）。

    Raises:
        ValueError: ``candidates.jsonl`` **缺失或为空**——上游 ``extract`` 未产出候选帧。
            此时**必须 fail-loud**：否则标注会「退出码 0、报告成功」但产物全空，
            流程显示绿灯而实际什么都没做（真实闭环暴露的 P1 静默缺陷）。
    """
    # P1 静默护栏：候选帧清单缺失或为空都必须 fail-loud，绝不「空跑成功」。
    if not config.candidates_path.exists():
        raise ValueError(
            f"未找到候选帧清单 {config.candidates_path}；"
            "请先运行 `extract` 阶段产出 candidates.jsonl。"
        )
    candidates = read_jsonl(config.candidates_path)
    if not candidates:
        raise ValueError(
            f"候选帧清单 {config.candidates_path} 为空；"
            "上游 `extract` 未产出任何候选帧（属异常），`annotate` 拒绝静默空跑。"
        )

    annotation_summary = run_annotation(config)

    records = [
        AnnotationRecord.from_dict(item) for item in read_jsonl(config.annotations_path)
    ]
    annotator = build_qwen_annotator(config)
    report, frames_needing_review, frames_without_report = _build_hazard_report(
        config, records, lambda record: _model_output_for(annotator, record)
    )

    # 两类复核原因分别补进队列（同一路径下原因会累加，不会互相覆盖）。
    added_to_queue = _augment_review_queue(
        config, frames_needing_review, REVIEW_REASON_HAZARD_OUT_OF_VOCAB
    )
    added_missing = _augment_review_queue(
        config, frames_without_report, REVIEW_REASON_HAZARD_REPORT_MISSING
    )

    write_json(config.hazard_report_path, report)
    write_jsonl(config.workspace("hazards", "rejected.jsonl"), report["rejected"])

    counts = report["counts"]
    summary = {
        "stage": "annotate",
        "annotation": annotation_summary,
        "hazard_report_path": str(config.hazard_report_path),
        "allowed_hazard_count": report["allowed_hazard_count"],
        "hazards_resolved": counts["hazards_resolved"],
        "hazards_rejected": counts["hazards_rejected"],
        "hazards_needing_verification": counts["hazards_needing_verification"],
        "frames_with_rejected_hazard": counts["frames_with_rejected_hazard"],
        "frames_without_hazard_report": counts["frames_without_hazard_report"],
        "review_queue_added": added_to_queue + added_missing,
        "overall_risk": report["overall_risk"],
    }
    write_json(config.workspace("status", "annotate.json"), summary)
    return summary


def rebuild_hazard_report(config: ProjectConfig) -> dict[str, Any]:
    """**只重建隐患报告**：从现有 ``annotations.jsonl``（+ ``raw_responses``）重算
    ``workspace/hazards/report.json``，**不调用 VLM、不重新标注、不读 candidates.jsonl**。

    用途：旧代码启动的长时间标注（如服务器上 12 小时的全量跑批）产出的报告缺少新增字段
    （如 ``needs_manual_verification`` / ``verification``）。标注的四类产物中只有
    ``report.json`` 受影响，而标记是在**回填阶段**加的、**不依赖 VLM 重新调用**——
    故无需重跑标注，从已有产物重建即可（省下整轮标注耗时）。

    **幂等**：同样的输入（``annotations.jsonl`` + 磁盘上的原始响应）重复跑结果一致。

    Args:
        config: 项目配置。

    Returns:
        重建摘要（``mode="rebuild"``，明确区别于 ``annotate``，避免误读为跑了 VLM）。

    Raises:
        ValueError: ``annotations.jsonl`` **缺失或为空**——无源可重建，**fail-loud**
            （与 :func:`run_annotate` 同一护栏，绝不产出空报告假装成功）。
    """
    if not config.annotations_path.exists():
        raise ValueError(
            f"未找到标注记录 {config.annotations_path}；"
            "重建隐患报告需先有 `annotate` 产物（annotations.jsonl）。"
        )
    records = [
        AnnotationRecord.from_dict(item) for item in read_jsonl(config.annotations_path)
    ]
    if not records:
        raise ValueError(
            f"标注记录 {config.annotations_path} 为空；无源可重建，拒绝产出空报告。"
        )

    report, _, _ = _build_hazard_report(config, records, _rebuild_model_output_provider)
    write_json(config.hazard_report_path, report)
    write_jsonl(config.workspace("hazards", "rejected.jsonl"), report["rejected"])

    counts = report["counts"]
    summary = {
        "stage": "rebuild-hazards",
        "mode": "rebuild",
        "note": (
            "由现有 annotations.jsonl + raw_responses 重建隐患报告；"
            "未调用 VLM、未重新标注、未读 candidates.jsonl。"
        ),
        "hazard_report_path": str(config.hazard_report_path),
        "source_annotations_path": str(config.annotations_path),
        "annotations_used": len(records),
        "allowed_hazard_count": report["allowed_hazard_count"],
        "hazards_resolved": counts["hazards_resolved"],
        "hazards_rejected": counts["hazards_rejected"],
        "hazards_needing_verification": counts["hazards_needing_verification"],
        "verification_by_layer": report["verification"]["by_layer"],
        "overall_risk": report["overall_risk"],
    }
    write_json(config.status_dir / "rebuild-hazards.json", summary)
    return summary


# --------------------------------------------------------------------------- #
# prioritize 阶段（T03.4 主动学习 / 不确定性采样）
# --------------------------------------------------------------------------- #
def run_prioritize(config: ProjectConfig, top_k: int | None = 0) -> dict[str, Any]:
    """执行 ``prioritize`` 阶段：按不确定性排序，产出 ``priority_queue.jsonl``。

    Args:
        config: 项目配置。
        top_k: 取前 K 条；``0`` / ``None`` 表示全部。

    Returns:
        阶段摘要（含队列条数与前若干条预览）。

    Raises:
        FileNotFoundError: 尚无可读的 ``annotations.jsonl``（应先跑 ``annotate``）。
    """
    if not config.annotations_path.exists():
        raise FileNotFoundError(
            f"未找到标注记录 {config.annotations_path}，请先运行 `annotate` 阶段。"
        )
    records = [
        AnnotationRecord.from_dict(item) for item in read_jsonl(config.annotations_path)
    ]
    items = build_priority_queue(
        records,
        top_k=top_k,
        confidence_threshold=config.review_confidence_threshold,
    )
    priority_path = config.workspace("review", "priority_queue.jsonl")
    write_jsonl(priority_path, [item.to_dict() for item in items])

    summary = {
        "stage": "prioritize",
        "records": len(records),
        "queued": len(items),
        "priority_queue_path": str(priority_path),
        "top": [item.to_dict() for item in items[:5]],
    }
    write_json(config.workspace("status", "prioritize.json"), summary)
    return summary


# --------------------------------------------------------------------------- #
# curate / freeze-holdout 阶段（T04.1 / T04.2）
# --------------------------------------------------------------------------- #
def run_curate(
    config: ProjectConfig,
    version: str | None = None,
    holdout_ratio: float = DEFAULT_HOLDOUT_RATIO,
    quality_thresholds: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """执行 ``curate`` 阶段：质量过滤 + 版本化数据集构建（T04.1）。

    复用 :func:`dataset.build_dataset`（其内部已包含固定 holdout 的冻结 / 读回），
    因此 ``curate`` 与 ``train`` 产出的数据集口径完全一致。版本号由调用方指定
    （**非时间戳**）→ 跨轮可比。

    Args:
        config: 项目配置。
        version: 数据集版本号；省略时用 :data:`DEFAULT_DATASET_VERSION`。
        holdout_ratio: 首次冻结 holdout 的占比。
        quality_thresholds: 覆盖默认质量过滤阈值。

    Returns:
        数据集摘要（含 ``stage: "curate"``）。
    """
    dataset_version = (
        str(version).strip() if version is not None else ""
    ) or DEFAULT_DATASET_VERSION
    dataset_summary = build_dataset(
        config, dataset_version, holdout_ratio, quality_thresholds
    )
    summary = {"stage": "curate", **dataset_summary}
    write_json(config.status_dir / "curate.json", summary)
    return summary


def run_freeze_holdout(
    config: ProjectConfig,
    version: str | None = None,
    holdout_ratio: float = DEFAULT_HOLDOUT_RATIO,
    quality_thresholds: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """执行 ``freeze-holdout`` 阶段：冻结 / 读回固定 holdout（T04.2）。

    - **首次**：从（质量过滤后的）记录里按 ``config.random_seed`` 切出 holdout，
      逐帧物化冻结副本并登记 sha256；
    - **之后**：``index.json`` 一旦存在即**读回**，绝不重新切分（holdout 铁律），
      因此新增记录只会进 train/val，跨轮指标可比。

    与 :func:`run_curate` 共用 :func:`dataset.curate_records` 的质量过滤口径，
    保证 holdout 候选池与数据集一致。

    Args:
        config: 项目配置。
        version: 首次冻结时写入清单的 ``dataset_version`` 标签（可追溯来源）。
        holdout_ratio: 首次冻结的占比（已冻结时以清单为准）。
        quality_thresholds: 覆盖默认质量过滤阈值。

    Returns:
        holdout 摘要（含计数、聚合哈希、是否首次冻结、逐字节校验结论）。
    """
    dataset_version = (
        str(version).strip() if version is not None else ""
    ) or DEFAULT_DATASET_VERSION
    holdout_ratio = validate_holdout_ratio(holdout_ratio)

    _, accepted, _, _ = curate_records(config, quality_thresholds)

    out_dir = holdout_dir_of(config)
    index_path = holdout_index_path(out_dir)
    already_frozen = index_path.exists()
    entries = freeze_holdout(
        accepted,
        holdout_ratio,
        config.random_seed,
        out_dir,
        version=dataset_version,
    )
    verification = verify_holdout(entries, out_dir)
    summary = {
        "stage": "freeze-holdout",
        "dataset_version": dataset_version,
        "holdout_count": len(entries),
        "holdout_hash": holdout_hash_of(entries),
        "holdout_index_path": str(index_path.resolve()),
        "frozen_dir": str(out_dir.resolve()),
        "newly_frozen": not already_frozen,
        "read_back": already_frozen,
        "hash_ok": verification["hash_ok"],
        "verified": verification["verified"],
        "tampered": verification["tampered"],
        "missing": verification["missing"],
    }
    write_json(config.status_dir / "freeze-holdout.json", summary)
    return summary


# --------------------------------------------------------------------------- #
# train 阶段（T04.3 训练编排）
# --------------------------------------------------------------------------- #
def run_train(
    config: ProjectConfig,
    version: str | None = None,
    holdout_ratio: float = DEFAULT_HOLDOUT_RATIO,
    execute: bool = False,
    quality_thresholds: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """执行 ``train`` 阶段：版本化数据集 + 固定 holdout + YOLO 冷启动训练。

    **关键裁决**：**不**调用母项目 ``run_training``（它每轮用时间戳作版本号并重新切分，
    会破坏固定 holdout）。改为**细粒度复用**：

    1. :func:`dataset.build_dataset` 自己构建**版本可指定**的数据集，并冻结 / 读回 holdout；
    2. :func:`traffic_cv_closed_loop.adapters.yolo.build_trainer` 取底层训练器
       （``MockTrainer`` / ``UltralyticsTrainer``）；
    3. ``trainer.train(dataset_yaml_path=<我们的 data.yaml>, run_dir=..., execute=...)``
       —— 只复用其**训练执行能力**，跳过其切分逻辑。

    本机无 ``ultralytics``，只能走 ``MockTrainer``（``execute=False`` 即可端到端跑通）；
    ``UltralyticsTrainer`` 在 ``execute=False`` 时**只准备命令**（``command.txt``），
    真实训练在 ``lab-server`` 上以 ``execute=True`` 执行。

    Args:
        config: 项目配置。
        version: 数据集 / 运行版本号；省略时用 :data:`DEFAULT_DATASET_VERSION`。
        holdout_ratio: 首次冻结 holdout 的占比。
        execute: 是否真正执行训练（``False`` = 只准备 / mock）。
        quality_thresholds: 覆盖默认质量过滤阈值。

    Returns:
        训练报告（含数据集摘要、holdout 计数 / 聚合哈希、训练器结果）。
    """
    dataset_version = (str(version).strip() if version is not None else "") or DEFAULT_DATASET_VERSION
    dataset_summary = build_dataset(
        config,
        dataset_version,
        holdout_ratio,
        quality_thresholds,
    )

    run_dir = config.runs_dir / dataset_version
    ensure_dir(run_dir)

    trainer = build_trainer(config)
    trainer_summary = {
        "dataset_version": dataset_version,
        "accepted_images": dataset_summary["accepted"],
        "split_counts": dataset_summary["split_counts"],
        "class_histogram": dataset_summary["class_histogram"],
        "review_queue_count": dataset_summary["review_queue_count"],
        "fallback_used": False,
    }
    trainer_result = trainer.train(
        config=config,
        dataset_yaml_path=Path(dataset_summary["data_yaml_path"]),
        run_dir=run_dir,
        dataset_summary=trainer_summary,
        execute=execute,
    )

    report = {
        "stage": "train",
        "project_name": config.name,
        "dataset_version": dataset_version,
        "dataset_dir": dataset_summary["dataset_dir"],
        "data_yaml_path": dataset_summary["data_yaml_path"],
        "manifest_path": dataset_summary["manifest_path"],
        "quality_report_path": dataset_summary["quality_report_path"],
        "run_dir": str(run_dir.resolve()),
        "detector_mode": config.yolo_detector,
        "trainer_mode": config.yolo_trainer,
        "accepted_images": dataset_summary["accepted"],
        "rejected_images": dataset_summary["rejected"],
        "split_counts": dataset_summary["split_counts"],
        "holdout_count": dataset_summary["holdout_count"],
        "holdout_hash": dataset_summary["holdout_hash"],
        "holdout_index_path": dataset_summary["holdout_index_path"],
        "review_queue_count": dataset_summary["review_queue_count"],
        "class_histogram": dataset_summary["class_histogram"],
        "trainer": trainer_result,
    }
    write_json(run_dir / "report.json", report)
    write_json(config.status_dir / "train.json", report)
    return report


# --------------------------------------------------------------------------- #
# evaluate 阶段（T05.1 逐类评估 + T05.2 困难样本回流与消费率）
# --------------------------------------------------------------------------- #
def run_evaluate(
    config: ProjectConfig,
    version: str | None = None,
    execute: bool = False,
) -> dict[str, Any]:
    """执行 ``evaluate`` 阶段：固定 holdout 逐类评估 + 困难样本回流。

    流程：

    1. 从 ``runs/<version>/weights/best.pt`` 取待评估权重（mock 路径可不存在）；
    2. 读上一轮 ``analysis/hard_examples.jsonl`` 与本轮标注，算**消费率**
       （判据 ≥ :data:`DEFAULT_CONSUMPTION_TARGET`，需求 §7.5.5「只写不读不算闭环」）；
    3. :func:`evaluate.evaluate` 在**固定 holdout**（哈希校验通过）上逐类评估并三档判定，
       产出 ``workspace/eval/<version>/report.json`` + ``metrics.json``；
    4. 汇总本轮困难样本 → ``workspace/analysis/hard_examples.jsonl`` +
       ``next_cycle_tasks.md``（供下一轮优先标注 / 消费）。

    Args:
        config: 项目配置。
        version: 数据集 / 评估版本号；省略时用 :data:`DEFAULT_DATASET_VERSION`。
        execute: 是否真正执行 ``yolo val`` / ``yolo predict``（默认否 = mock 路径）。

    Returns:
        评估摘要（含逐类指标、三档判定、消费率、困难样本计数与各产物路径）。
    """
    dataset_version = (
        str(version).strip() if version is not None else ""
    ) or DEFAULT_DATASET_VERSION

    holdout_dir = config.workspace("holdout")
    run_dir = config.runs_dir / dataset_version
    weights = run_dir / "weights" / "best.pt"
    eval_dir = config.workspace("eval", dataset_version)
    analysis_dir = config.workspace("analysis")

    # 消费率：上一轮困难样本 vs 本轮真正进入数据集的标注（dataset_ready）。
    prev_hard = read_jsonl(analysis_dir / HARD_EXAMPLES_FILENAME)
    annotations = [
        AnnotationRecord.from_dict(item) for item in read_jsonl(config.annotations_path)
    ]
    consumed = [
        record.candidate.relative_path for record in annotations if record.dataset_ready
    ]
    rate = consumption_rate(prev_hard, consumed)

    report = evaluate(
        weights,
        holdout_dir,
        config,
        version=dataset_version,
        eval_dir=eval_dir,
        execute=execute,
        consumption=rate,
    )

    # 困难样本回流：评估逐帧统计（误检/漏检）+ 标注信号（低置信 / 通道差异）。
    metrics_payload = read_json(eval_dir / METRICS_FILENAME, default={})
    frame_stats = (
        metrics_payload.get("per_frame", {}) if isinstance(metrics_payload, dict) else {}
    )
    hard_examples = build_hard_examples(config, report, annotations, frame_stats)
    write_jsonl(analysis_dir / HARD_EXAMPLES_FILENAME, hard_examples)
    write_text(
        analysis_dir / NEXT_CYCLE_TASKS_FILENAME,
        render_next_cycle_tasks(report, hard_examples, rate),
    )

    consumption_ok = rate >= DEFAULT_CONSUMPTION_TARGET
    summary: dict[str, Any] = {
        "stage": "evaluate",
        "dataset_version": dataset_version,
        "holdout_hash": report.holdout_hash,
        "report_path": str((eval_dir / REPORT_FILENAME).resolve()),
        "metrics_path": str((eval_dir / METRICS_FILENAME).resolve()),
        "per_class": report.per_class,
        "tier_verdict": report.tier_verdict,
        "false_detection_rate": (
            metrics_payload.get("false_detection_rate", 0.0)
            if isinstance(metrics_payload, dict)
            else 0.0
        ),
        "consumption_rate": rate,
        "consumption_target": DEFAULT_CONSUMPTION_TARGET,
        "consumption_ok": consumption_ok,
        "hard_example_count": len(hard_examples),
        "hard_examples_path": str((analysis_dir / HARD_EXAMPLES_FILENAME).resolve()),
        "next_cycle_tasks_path": str((analysis_dir / NEXT_CYCLE_TASKS_FILENAME).resolve()),
    }
    write_json(config.status_dir / "evaluate.json", summary)
    return summary


def run_cycle(
    config: ProjectConfig,
    stage: str,
    *,
    video_paths: list[Path] | None = None,
    media_root: Path | None = None,
    auto_extract_video: bool = True,
    version: str | None = None,
    holdout_ratio: float | None = None,
    execute: bool = False,
) -> dict[str, Any]:
    """执行一个链路阶段。

    Args:
        config: 项目配置。
        stage: 阶段名，见 :data:`SUPPORTED_STAGES`。
        video_paths: ``extract-video`` / ``extract`` 使用的显式视频列表。
        media_root: ``inventory`` 使用的素材根目录。
        auto_extract_video: ``extract`` 阶段是否在需要时自动先抽帧。
        version: ``curate`` / ``freeze-holdout`` / ``train`` / ``evaluate`` 使用的数据集
            版本号（省略用默认版本）。
        holdout_ratio: ``curate`` / ``freeze-holdout`` / ``train`` 首次冻结 holdout 的占比。
        execute: ``train`` / ``evaluate`` 阶段是否真正执行（默认否 = mock 路径）。

    Returns:
        阶段摘要字典（``extract`` 结果含母项目 ``run_extraction`` 的全部字段）。

    Raises:
        ValueError: 阶段名不受支持。
    """
    normalized = (stage or "").strip().lower()

    if normalized == "inventory":
        return {"stage": "inventory", **run_inventory(config, media_root=media_root)}

    if normalized == "extract-video":
        return {"stage": "extract-video", **run_video_extract(config, video_paths=video_paths)}

    if normalized == "extract":
        chained: dict[str, Any] | None = None
        if auto_extract_video and _needs_video_extract(config, video_paths):
            chained = run_video_extract(config, video_paths=video_paths)
        summary = run_extraction(config)
        result: dict[str, Any] = {"stage": "extract", "video_extract": chained, **summary}
        # P2：sidecar 零命中等「静默风险」以 warnings 形式暴露（不报错，但必须可见）。
        warnings = _extract_warnings(config)
        if warnings:
            result["warnings"] = warnings
        # 把内嵌 video_extract 的部分失败上浮到顶层，供调用方 / CLI 判定退出码。
        if isinstance(chained, dict) and chained.get("has_failures"):
            result["has_failures"] = True
            result["failed_videos"] = list(chained.get("failed_videos") or [])
        return result

    if normalized == "annotate":
        return run_annotate(config)

    if normalized == "rebuild-hazards":
        return rebuild_hazard_report(config)

    if normalized == "prioritize":
        return run_prioritize(config)

    if normalized == "curate":
        curate_kwargs: dict[str, Any] = {}
        if version is not None:
            curate_kwargs["version"] = version
        if holdout_ratio is not None:
            curate_kwargs["holdout_ratio"] = holdout_ratio
        return run_curate(config, **curate_kwargs)

    if normalized == "freeze-holdout":
        freeze_kwargs: dict[str, Any] = {}
        if version is not None:
            freeze_kwargs["version"] = version
        if holdout_ratio is not None:
            freeze_kwargs["holdout_ratio"] = holdout_ratio
        return run_freeze_holdout(config, **freeze_kwargs)

    if normalized == "train":
        train_kwargs: dict[str, Any] = {"execute": execute}
        if version is not None:
            train_kwargs["version"] = version
        if holdout_ratio is not None:
            train_kwargs["holdout_ratio"] = holdout_ratio
        return run_train(config, **train_kwargs)

    if normalized == "evaluate":
        evaluate_kwargs: dict[str, Any] = {"execute": execute}
        if version is not None:
            evaluate_kwargs["version"] = version
        return run_evaluate(config, **evaluate_kwargs)

    raise ValueError(
        f"未知阶段：{stage!r}（本批支持：{', '.join(SUPPORTED_STAGES)}）"
    )
