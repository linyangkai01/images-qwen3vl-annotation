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
- ``prioritize``（T03.4）—— 对标注记录做不确定性采样，产出
  ``workspace/review/priority_queue.jsonl``。

**串联逻辑**：``extract`` 阶段默认先确保关键帧已就位——若 ``input_frames_dir``
尚无任何图片且能发现视频，则自动先跑一遍 ``extract-video``，再调
``run_extraction``。该行为可用 ``auto_extract_video=False`` 关闭。

产出的 ``CandidateFrame`` / ``AnnotationRecord`` **直接复用母项目类**（不重定义），
结构与母项目完全兼容。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from traffic_cv_closed_loop.adapters.qwen import MockQwenAnnotator, build_qwen_annotator
from traffic_cv_closed_loop.config import ProjectConfig
from traffic_cv_closed_loop.io_utils import (
    iter_image_files,
    read_json,
    read_jsonl,
    read_text,
    write_json,
    write_jsonl,
)
from traffic_cv_closed_loop.models import AnnotationRecord
from traffic_cv_closed_loop.pipeline.annotation import run_annotation
from traffic_cv_closed_loop.pipeline.extraction import run_extraction

from industrial_hazard_coldstart.active_learning import build_priority_queue
from industrial_hazard_coldstart.inventory import run_inventory
from industrial_hazard_coldstart.knowledge import (
    backfill_hazards,
    extract_model_hazards,
    load_knowledge_base,
    overall_risk_of,
    recommendation_of,
)
from industrial_hazard_coldstart.video_decode import discover_videos, run_video_extract

__all__ = ["SUPPORTED_STAGES", "run_cycle", "run_annotate", "run_prioritize"]

#: 本批实现的阶段名。
SUPPORTED_STAGES: tuple[str, ...] = (
    "inventory",
    "extract-video",
    "extract",
    "annotate",
    "prioritize",
)

#: 隐患被拒（KB 外 / 场景外）时写入复核队列的原因标记。
REVIEW_REASON_HAZARD_OUT_OF_VOCAB = "hazard_id_not_in_kb"


def _needs_video_extract(
    config: ProjectConfig,
    video_paths: list[Path] | None,
) -> bool:
    """判断 ``extract`` 前是否需要先抽帧：无图片、且有可发现的视频。"""
    if iter_image_files(config.input_frames_dir):
        return False
    return bool(discover_videos(config, video_paths))


# --------------------------------------------------------------------------- #
# annotate 阶段（T03.2 两段式标注 + T03.3 语义层受控词表 / KB 回填）
# --------------------------------------------------------------------------- #
def _extract_hazard_from_raw(raw_payload: Any) -> dict[str, Any]:
    """从 ``raw_responses/*.json`` 里取出 ``hazard_report``。

    真实途径的 ``annotate`` 会把解析后的完整载荷存进 ``parsed``（含 ``hazard_report``），
    故优先看 ``parsed.hazard_report``；同时兼容顶层直接挂 ``hazard_report`` 的形态。
    """
    if not isinstance(raw_payload, dict):
        return {}
    top_level = raw_payload.get("hazard_report")
    if isinstance(top_level, dict):
        return top_level
    parsed = raw_payload.get("parsed")
    if isinstance(parsed, dict):
        nested = parsed.get("hazard_report")
        if isinstance(nested, dict):
            return nested
    return {}


def _hazard_report_for(
    annotator: Any,
    record: AnnotationRecord,
) -> dict[str, Any]:
    """取一帧的模型隐患报告。

    - **mock 途径**：直接调 ``annotate_combined``（离线、零成本），它会读取
      ``*.hazard.json`` 侧车或 ``*.qwen.json`` 里的内嵌 ``hazard_report``；
    - **真实途径**：复用 ``run_annotation`` 已写下的原始响应，**不重复调用 VLM**
      （避免把昂贵的网络调用翻倍）。
    """
    if isinstance(annotator, MockQwenAnnotator):
        prompt_text = read_text(Path(record.prompt_path))
        _, hazard_report, _ = annotator.annotate_combined(record.candidate, prompt_text)
        return hazard_report if isinstance(hazard_report, dict) else {}
    return _extract_hazard_from_raw(read_json(Path(record.raw_response_path), default={}))


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
    4. 落盘 ``workspace/hazards/report.json`` 与 ``workspace/hazards/rejected.jsonl``。

    Returns:
        阶段摘要（含标注摘要与隐患回填统计）。
    """
    annotation_summary = run_annotation(config)

    kb = load_knowledge_base(config.kb_path)
    allowed_ids = frozenset(kb.allowed_hazard_ids(config.scene))

    records = [
        AnnotationRecord.from_dict(item) for item in read_jsonl(config.annotations_path)
    ]
    annotator = build_qwen_annotator(config)

    hazards: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    frames: list[dict[str, Any]] = []
    frames_needing_review: set[str] = set()
    model_overall = ""

    for record in records:
        relative_path = record.candidate.relative_path
        hazard_report = _hazard_report_for(annotator, record)
        if isinstance(hazard_report, dict) and not model_overall:
            model_overall = str(hazard_report.get("overall_risk", "") or "")

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
            }
        )

    added_to_queue = _augment_review_queue(
        config, frames_needing_review, REVIEW_REASON_HAZARD_OUT_OF_VOCAB
    )

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
        "counts": {
            "frames": len(records),
            "hazards_resolved": len(hazards),
            "hazards_rejected": len(rejected),
            "frames_with_rejected_hazard": len(frames_needing_review),
        },
    }
    write_json(config.hazard_report_path, report)
    write_jsonl(config.workspace("hazards", "rejected.jsonl"), rejected)

    summary = {
        "stage": "annotate",
        "annotation": annotation_summary,
        "hazard_report_path": str(config.hazard_report_path),
        "allowed_hazard_count": len(allowed_ids),
        "hazards_resolved": len(hazards),
        "hazards_rejected": len(rejected),
        "frames_with_rejected_hazard": len(frames_needing_review),
        "review_queue_added": added_to_queue,
        "overall_risk": report["overall_risk"],
    }
    write_json(config.workspace("status", "annotate.json"), summary)
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


def run_cycle(
    config: ProjectConfig,
    stage: str,
    *,
    video_paths: list[Path] | None = None,
    media_root: Path | None = None,
    auto_extract_video: bool = True,
) -> dict[str, Any]:
    """执行一个链路阶段。

    Args:
        config: 项目配置。
        stage: 阶段名，见 :data:`SUPPORTED_STAGES`。
        video_paths: ``extract-video`` / ``extract`` 使用的显式视频列表。
        media_root: ``inventory`` 使用的素材根目录。
        auto_extract_video: ``extract`` 阶段是否在需要时自动先抽帧。

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
        # 把内嵌 video_extract 的部分失败上浮到顶层，供调用方 / CLI 判定退出码。
        if isinstance(chained, dict) and chained.get("has_failures"):
            result["has_failures"] = True
            result["failed_videos"] = list(chained.get("failed_videos") or [])
        return result

    if normalized == "annotate":
        return run_annotate(config)

    if normalized == "prioritize":
        return run_prioritize(config)

    raise ValueError(
        f"未知阶段：{stage!r}（本批支持：{', '.join(SUPPORTED_STAGES)}）"
    )
