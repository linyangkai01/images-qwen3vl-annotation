"""T04.1 质量过滤 + 版本化数据集（F8），并集成固定 holdout。

**为什么不能直接用母项目 ``run_training``**：它每轮都用 ``timestamp_token()`` 作版本号、
自己 ``random.shuffle`` 重新切分，导致 holdout 无法固定、跨轮指标不可比（需求 7 记录的
P1 问题）。本模块**细粒度复用**母项目的底层能力，自己做版本化与切分：

- 版本号由**调用方指定**（不用时间戳）→ 同名版本可跨轮对比；
- 切分 ``train / val / holdout``，其中 **holdout 由 :mod:`holdout` 冻结、不进 train/val**；
- 质量过滤**复用** ``scripts/curate_video_batch.py`` 的 ``frame_quality`` / ``keep_record``
  （该脚本不是包，故用 ``importlib`` 按文件路径加载，见 :func:`_curation_module`）；
- 图片物化**复用** ``dataset_media.materialize_dataset_image``（保证 Ultralytics 可读扩展名）。

产出 ``workspace/datasets/<version>/``：
``data.yaml`` / ``manifest.json`` / ``class_histogram.json`` / ``quality_report.json``
（外加 ``summary.json`` 便于编排读取）。

仅使用 Python 标准库 + 母项目；中文路径安全。
"""

from __future__ import annotations

import argparse
import importlib.util
import shutil
from functools import lru_cache
from pathlib import Path
from typing import Any

from traffic_cv_closed_loop.config import ProjectConfig
from traffic_cv_closed_loop.dataset_media import (
    dataset_image_relative_path,
    materialize_dataset_image,
)
from traffic_cv_closed_loop.io_utils import (
    copy_file,
    ensure_dir,
    read_json,
    read_jsonl,
    write_json,
    write_text,
)
from traffic_cv_closed_loop.models import AnnotationRecord

from industrial_hazard_coldstart.holdout import (
    DEFAULT_HOLDOUT_RATIO,
    freeze_holdout,
    holdout_dir_of,
    holdout_hash_of,
    validate_holdout_ratio,
)

__all__ = [
    "DEFAULT_QUALITY_THRESHOLDS",
    "DEFAULT_HOLDOUT_RATIO",
    "QUALITY_THRESHOLD_KEYS",
    "frame_quality",
    "keep_record",
    "split_counts",
    "build_dataset_yaml",
    "load_config_quality_thresholds",
    "resolve_quality_thresholds",
    "curate_records",
    "build_dataset",
]

#: 质量过滤默认阈值（与 ``scripts/curate_video_batch.py`` 的命令行默认值一致）。
#:
#: ⚠️ **该默认值假设「VLM 只纠偏、不新增」**（``max_added=0``）。真实场景下通用 YOLO
#: 初检漏检严重（实测：YOLO 55 个 vs VLM 216 个目标，36/37 帧都有 VLM 新增），若沿用
#: 该默认值会**把全部帧剔除**（``所有帧都被质量过滤剔除``）。真实素材跑批**必须放宽**
#: ——通过配置文件的 ``pipeline.quality_thresholds`` 覆盖，例如
#: ``{"max_added": 20, "max_adjusted": 10}``（见 :func:`load_config_quality_thresholds`）。
#: **本默认值刻意不改**：它与母项目脚本命令行默认值对齐，改了会破坏该对齐。
DEFAULT_QUALITY_THRESHOLDS: dict[str, Any] = {
    "min_confidence": 0.25,
    "max_added": 0,
    "max_removed": 10,
    "max_adjusted": 2,
    "min_labels": 1,
}

#: 合法的质量阈值键集合（未知键一律 fail-loud）。
QUALITY_THRESHOLD_KEYS: frozenset[str] = frozenset(DEFAULT_QUALITY_THRESHOLDS)

#: 配置文件里承载质量阈值的路径：``pipeline.quality_thresholds``。
_CONFIG_QUALITY_KEY: str = "quality_thresholds"

#: 素材筛选脚本（不是包，故按文件路径加载）。
_CURATION_SCRIPT: Path = (
    Path(__file__).resolve().parents[1] / "scripts" / "curate_video_batch.py"
)

#: 加载到内存的模块名（避免与真实模块名冲突）。
_CURATION_MODULE_NAME: str = "_industrial_curate_video_batch"


# --------------------------------------------------------------------------- #
# 复用 scripts/curate_video_batch.py（它不是包）
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def _curation_module() -> Any:
    """按**文件路径**加载 ``scripts/curate_video_batch.py`` 并缓存。

    该脚本不在任何包里（``scripts/`` 下无 ``__init__.py``），无法 ``import``，
    因此用 ``importlib.util.spec_from_file_location`` 从磁盘加载；其 ``__name__``
    不等于 ``__main__``，故模块级 ``main()`` 不会被执行。加载失败时给出明确错误。
    """
    if not _CURATION_SCRIPT.exists():
        raise ImportError(f"未找到素材筛选脚本：{_CURATION_SCRIPT}")
    spec = importlib.util.spec_from_file_location(_CURATION_MODULE_NAME, _CURATION_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法从文件加载素材筛选脚本：{_CURATION_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def frame_quality(record: AnnotationRecord) -> dict[str, Any]:
    """复用脚本的逐帧质量画像（返回 ``has_error/added/removed/adjusted/min_conf`` 等）。"""
    return _curation_module().frame_quality(record)


def keep_record(
    quality: dict[str, Any], args: argparse.Namespace
) -> tuple[bool, list[str]]:
    """复用脚本的保留判定；``args`` 为 ``argparse.Namespace``（阈值载体）。"""
    return _curation_module().keep_record(quality, args)


def split_counts(
    total: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> tuple[int, int, int]:
    """复用脚本的三路切分计数。"""
    return _curation_module().split_counts(total, train_ratio, val_ratio, test_ratio)


def build_dataset_yaml(
    config: ProjectConfig,
    dataset_dir: Path,
    split_count_map: dict[str, int],
) -> str:
    """复用脚本的 ``data.yaml`` 生成（纯函数）。"""
    return _curation_module().build_dataset_yaml(config, dataset_dir, split_count_map)


# --------------------------------------------------------------------------- #
# 内部工具
# --------------------------------------------------------------------------- #
def _validate_version(version: str) -> str:
    """校验数据集版本号：非空、且必须是**单一目录名**（拒绝路径穿越 / 绝对路径）。

    版本号会被拼进 ``workspace/datasets/<version>/``，并在同名重建时
    ``shutil.rmtree``。若放任 ``..`` / 路径分隔符 / 绝对路径，一条笔误（如
    ``"../../victim"``）就会把**工作区外**的既有目录连同文件一起删除——
    属数据销毁级风险，必须 fail-loud（本项目「绝不静默」）。

    Args:
        version: 调用方指定的版本号。

    Returns:
        ``strip`` 后的合法版本号（保持中文等字符原样）。

    Raises:
        ValueError: 空 / 含路径分隔符 / 含 ``..`` / 为绝对路径。
    """
    normalized = str(version).strip()
    if not normalized:
        raise ValueError("dataset version 不能为空（版本号由调用方指定，不用时间戳）。")
    if normalized in {".", ".."} or any(sep in normalized for sep in ("/", "\\")):
        raise ValueError(
            "dataset version 必须是单一目录名（不得含路径分隔符 / '..' / 为绝对路径）："
            f"{normalized!r}。"
        )
    candidate = Path(normalized)
    if candidate.is_absolute():
        raise ValueError(
            f"dataset version 不得是绝对路径：{normalized!r}。"
        )
    if ".." in candidate.parts:
        raise ValueError(f"dataset version 不得含 '..'（防路径穿越）：{normalized!r}。")
    return normalized


def _is_within(child: Path, parent: Path) -> bool:
    """判断 ``child`` 是否位于 ``parent`` 之内（用于路径越界的纵深防御）。"""
    try:
        return child.resolve().is_relative_to(parent.resolve())
    except (OSError, ValueError):
        return False


def load_config_quality_thresholds(config: ProjectConfig) -> dict[str, Any]:
    """从**配置文件**读取 ``pipeline.quality_thresholds``（可选）。

    **为什么在这里读**：母项目 ``ProjectConfig`` 硬约束不得改动（不加字段），故本包
    自行读取配置文件 JSON，取其 ``pipeline.quality_thresholds`` 子对象。这样真实素材
    跑批可**只改配置文件**（无需 API 绕过、无需新增 CLI 开关）。

    Args:
        config: 项目配置（用其 ``config_path`` 定位配置文件）。

    Returns:
        配置里的质量阈值字典（缺省 / 非法 / 非 dict 时返回 ``{}``）；值保留原样，
        ``None`` 表示「该键回落下一优先级」。
    """
    config_path = getattr(config, "config_path", None)
    if not config_path:
        return {}
    payload = read_json(Path(config_path), default=None)
    if not isinstance(payload, dict):
        return {}
    pipeline = payload.get("pipeline")
    if not isinstance(pipeline, dict):
        return {}
    raw = pipeline.get(_CONFIG_QUALITY_KEY)
    if not isinstance(raw, dict):
        return {}
    return {str(key): value for key, value in raw.items()}


def resolve_quality_thresholds(
    config: ProjectConfig,
    thresholds: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """按优先级合并质量阈值：**显式参数 > 配置文件 > 默认值**。

    Args:
        config: 项目配置。
        thresholds: 显式覆盖（函数参数）；``None`` 表示不覆盖。

    Returns:
        合并后的完整阈值字典。

    Raises:
        ValueError: 配置文件或显式参数里出现**未知键**（列出合法键）。
    """
    values = dict(DEFAULT_QUALITY_THRESHOLDS)
    layers: list[dict[str, Any]] = [load_config_quality_thresholds(config)]
    if thresholds:
        layers.append(dict(thresholds))
    for layer in layers:
        unknown = sorted(key for key in layer if key not in QUALITY_THRESHOLD_KEYS)
        if unknown:
            raise ValueError(
                f"未知的质量阈值键：{unknown}；合法键为 {sorted(QUALITY_THRESHOLD_KEYS)}。"
            )
        for key, value in layer.items():
            if value is not None:
                values[key] = value
    return values


def _quality_namespace(
    config: ProjectConfig,
    thresholds: dict[str, Any] | None,
) -> argparse.Namespace:
    """把阈值（显式 > 配置 > 默认）规整成 ``keep_record`` 需要的 ``argparse.Namespace``。

    - 优先级见 :func:`resolve_quality_thresholds`；
    - **未知键 → 立刻 ``ValueError``**（列出合法键）。拼错的键（如 ``min_conf``
      漏了 ``idence``）若被静默丢弃，使用者以为收紧了过滤、实际没生效，属典型静默
      失败；本项目一律 fail-loud。
    """
    return argparse.Namespace(**resolve_quality_thresholds(config, thresholds))


def _split_train_val(
    total: int, train_ratio: float, val_ratio: float
) -> tuple[int, int]:
    """把 ``total`` 帧切成 ``(train, val)`` 两份（至少给 train 留 1 帧）。"""
    if total <= 0:
        return (0, 0)
    if total == 1:
        return (1, 0)
    denominator = float(train_ratio) + float(val_ratio)
    weight = (float(train_ratio) / denominator) if denominator > 0 else 1.0
    train_count = int(round(total * weight))
    train_count = max(1, min(total - 1, train_count))
    return (train_count, total - train_count)


def _copy_label(label_path: str, target: Path) -> bool:
    """把标签拷进数据集；源缺失时写空标签（YOLO 视作背景）并返回 ``False``。

    返回 ``False`` 表示「源标签不存在，只能写空标签」。调用方据此累计
    ``missing_label_count``，把「丢标签被当成背景图」这件事**留痕**——否则这些帧会被
    YOLO 当作「无目标背景」参与训练，教模型「这里没东西」，直接制造漏检，而使用者
    从任何产物上都看不出来。故此处**不中断构建，但绝不静默**。
    """
    source = Path(label_path) if label_path else None
    if source is not None and source.exists():
        copy_file(source, target)
        return True
    ensure_dir(target.parent)
    target.write_text("", encoding="utf-8")
    return False


def _render_data_yaml(
    config: ProjectConfig,
    dataset_dir: Path,
    split_count_map: dict[str, int],
) -> str:
    """生成 ``data.yaml``，**修掉母项目「val 计数为 0 → val 指向 train」的回落坑**。

    母项目 ``scripts/curate_video_batch.py`` 在 ``val`` 计数为 0 时会把 ``val`` 指向
    ``images/train``（母项目冻结不改）。那会让 Ultralytics 在训练集上「验证」→ mAP
    虚高、早停与最优权重选择失真。这里**只复用其渲染逻辑**，但传入一个 ``val >= 1``
    的计数，强制输出 ``val: images/val``（目录可能为空 → 下游评估 fail-loud，
    绝不假装有验证集）。真实的 ``val`` 计数仍以 ``summary.split_counts`` 为准。
    """
    render_map = dict(split_count_map)
    if render_map.get("val", 0) <= 0:
        render_map["val"] = 1  # 仅为让母项目输出 images/val；实际目录可空。
    return build_dataset_yaml(config, dataset_dir, render_map)


def _manifest_entry(
    *,
    split: str,
    relative_path: str,
    image_path: str,
    label_path: str,
    review_required: bool = False,
    review_reasons: list[str] | None = None,
    diff: dict[str, Any] | None = None,
    sha256: str = "",
) -> dict[str, Any]:
    """构造 manifest 条目，**保证所有 split 的条目键集完全一致**（下游统一消费）。

    ``train`` / ``val`` 与 ``holdout`` 若键集不同构，下游（T05 逐类评估）按
    ``item["review_required"]`` 过滤就会 ``KeyError``，而这种缺键在构建期完全无感。
    故所有条目一律带同一组公共键：``split / relative_path / image_path / label_path /
    review_required / review_reasons / diff / sha256``（不适用者给空默认值）。
    """
    return {
        "split": str(split),
        "relative_path": str(relative_path),
        "image_path": str(image_path),
        "label_path": str(label_path),
        "review_required": bool(review_required),
        "review_reasons": [str(item) for item in (review_reasons or [])],
        "diff": dict(diff or {}),
        "sha256": str(sha256),
    }


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def curate_records(
    config: ProjectConfig,
    quality_thresholds: dict[str, Any] | None = None,
) -> tuple[
    list[AnnotationRecord],
    list[AnnotationRecord],
    list[AnnotationRecord],
    list[dict[str, Any]],
]:
    """读 ``annotations.jsonl`` 并逐帧质量过滤（``curate`` / ``freeze-holdout`` 共用）。

    抽出来是为了让独立的 ``freeze-holdout`` 阶段与 :func:`build_dataset` 使用**同一套**
    质量过滤口径——否则 holdout 的候选池会与数据集不一致。

    Args:
        config: 项目配置。
        quality_thresholds: 显式覆盖的质量阈值（**优先级最高**；未知键 → ``ValueError``）。
            未给出的键按 **配置文件 ``pipeline.quality_thresholds`` > 默认值** 回落。

    Returns:
        ``(all_records, accepted, rejected, quality_rows)``。

    Raises:
        FileNotFoundError: 尚无 ``annotations.jsonl``。
        RuntimeError: 所有帧都被质量过滤剔除。
    """
    records = [
        AnnotationRecord.from_dict(item) for item in read_jsonl(config.annotations_path)
    ]
    if not records:
        raise FileNotFoundError(
            f"未找到标注记录 {config.annotations_path}，请先运行 `annotate` 阶段。"
        )

    args = _quality_namespace(config, quality_thresholds)
    quality_rows: list[dict[str, Any]] = []
    accepted: list[AnnotationRecord] = []
    rejected: list[AnnotationRecord] = []
    for record in records:
        quality = frame_quality(record)
        keep, reasons = keep_record(quality, args)
        quality["accepted"] = keep
        quality["filter_reasons"] = reasons
        # D8 留痕：源标签是否存在（缺失会被写空标签当背景，必须在下游可见）。
        quality["label_path_exists"] = bool(record.label_path) and Path(
            record.label_path
        ).exists()
        quality_rows.append(quality)
        (accepted if keep else rejected).append(record)

    if not accepted:
        raise RuntimeError("所有帧都被质量过滤剔除；请放宽阈值或补充标注。")
    return records, accepted, rejected, quality_rows


def build_dataset(
    config: ProjectConfig,
    version: str,
    holdout_ratio: float = DEFAULT_HOLDOUT_RATIO,
    quality_thresholds: dict[str, Any] | None = None,
    *,
    seed: int | None = None,
) -> dict[str, Any]:
    """构建一个**版本化**数据集，并冻结 / 读回固定 holdout。

    步骤：

    1. 读 ``annotations.jsonl``（无记录 → ``FileNotFoundError``）；
    2. 逐帧质量过滤（复用脚本逻辑），被剔除的帧写入 ``quality_report.json``；
    3. 调 :func:`holdout.freeze_holdout` —— 首次按 ``seed`` 冻结，之后**读回**已冻结清单；
    4. 把「非 holdout」的合格帧切成 ``train / val``（holdout **不进** train/val）；
    5. 物化图片 + 标签，写 ``data.yaml / manifest.json / class_histogram.json /
       quality_report.json / summary.json``。

    Args:
        config: 项目配置。
        version: 数据集版本号（**由调用方指定**，不用时间戳；同名重建=覆盖该版本目录，
            holdout 因已冻结而保持不变）。
        holdout_ratio: 首次冻结时 holdout 占比。
        quality_thresholds: 覆盖默认质量阈值（``min_confidence/max_added/...``）。
        seed: 切分 / 冻结随机种子；省略时取 ``config.random_seed``。

    Returns:
        数据集摘要（版本、目录、各产物路径、切分计数、holdout 计数与聚合哈希等）。

    Raises:
        ValueError: 版本号为空 / 含路径分隔符或 ``..`` / 为绝对路径；``holdout_ratio`` 越界。
        FileNotFoundError: 尚无 ``annotations.jsonl``。
        RuntimeError: 所有帧都被质量过滤剔除。
    """
    # 先做参数校验（fail-loud），再动任何文件系统——避免「半路报错留下残料」。
    normalized_version = _validate_version(version)
    holdout_ratio = validate_holdout_ratio(holdout_ratio)

    # 读标注 + 逐帧质量过滤（与 freeze-holdout 阶段共用同一口径）。
    records, accepted, rejected, quality_rows = curate_records(
        config, quality_thresholds
    )

    # --- holdout：首次冻结 / 后续读回（绝不重新切分）---
    out_dir = holdout_dir_of(config)
    effective_seed = config.random_seed if seed is None else int(seed)
    entries = freeze_holdout(
        accepted,
        holdout_ratio,
        effective_seed,
        out_dir,
        version=normalized_version,
    )
    holdout_paths = {entry.relative_path for entry in entries}

    # --- train / val 切分（holdout 不进）---
    pool = sorted(
        (record for record in accepted if record.candidate.relative_path not in holdout_paths),
        key=lambda record: record.candidate.relative_path,
    )
    train_count, val_count = _split_train_val(
        len(pool), config.train_split, config.val_split
    )
    split_map: dict[str, list[AnnotationRecord]] = {
        "train": pool[:train_count],
        "val": pool[train_count : train_count + val_count],
    }

    dataset_dir = config.datasets_dir / normalized_version
    # 纵深防御：版本号已校验为单一目录名，这里再确认解析后仍落在 datasets 之内——
    # 任何情况下都不允许越界 rmtree（数据销毁级风险）。
    if not _is_within(dataset_dir, config.datasets_dir):
        raise ValueError(
            f"dataset version 解析后越出 datasets 目录：{normalized_version!r}。"
        )
    if dataset_dir.exists():
        # 同名版本重建：清空旧目录，避免上一轮残留文件污染本轮 manifest。
        shutil.rmtree(dataset_dir, ignore_errors=True)
    for split_name in split_map:
        ensure_dir(dataset_dir / "images" / split_name)
        ensure_dir(dataset_dir / "labels" / split_name)

    manifest: list[dict[str, Any]] = []
    class_histogram: dict[str, int] = {}
    missing_label_count = 0
    warnings: list[str] = []

    # --- train / val 物化 ---
    for split_name, split_records in split_map.items():
        images_split_dir = dataset_dir / "images" / split_name
        labels_split_dir = dataset_dir / "labels" / split_name
        for record in split_records:
            relative = Path(record.candidate.relative_path)
            produced = materialize_dataset_image(
                record.candidate.extracted, images_split_dir, relative
            )
            image_target = images_split_dir / produced
            label_target = labels_split_dir / produced.with_suffix(".txt")
            if not _copy_label(record.label_path, label_target):
                missing_label_count += 1
            detections = record.qwen_detections or record.candidate.detections
            for detection in detections:
                class_histogram[detection.label] = (
                    class_histogram.get(detection.label, 0) + 1
                )
            manifest.append(
                _manifest_entry(
                    split=split_name,
                    relative_path=record.candidate.relative_path,
                    image_path=str(image_target.resolve()),
                    label_path=str(label_target.resolve()),
                    review_required=record.review_required,
                    review_reasons=list(record.review_reasons),
                    diff=dict(record.diff),
                )
            )

    # --- holdout 物化（从**冻结副本**拷贝，保证逐字节不变）---
    ensure_dir(dataset_dir / "images" / "holdout")
    ensure_dir(dataset_dir / "labels" / "holdout")
    for entry in entries:
        produced = dataset_image_relative_path(Path(entry.relative_path))
        frozen_frame = out_dir / "frames" / produced
        frozen_label = out_dir / "labels" / produced.with_suffix(".txt")
        image_target = dataset_dir / "images" / "holdout" / produced
        label_target = dataset_dir / "labels" / "holdout" / produced.with_suffix(".txt")
        copy_file(frozen_frame, image_target)
        if frozen_label.exists():
            copy_file(frozen_label, label_target)
        else:
            ensure_dir(label_target.parent)
            label_target.write_text("", encoding="utf-8")
            missing_label_count += 1
        manifest.append(
            _manifest_entry(
                split="holdout",
                relative_path=entry.relative_path,
                image_path=str(image_target.resolve()),
                label_path=str(label_target.resolve()),
                review_required=False,
                review_reasons=[],
                diff={},
                sha256=entry.sha256,
            )
        )

    split_count_map = {
        "train": len(split_map["train"]),
        "val": len(split_map["val"]),
        "holdout": len(entries),
    }

    # --- data.yaml：修掉母项目「val=0 → val 指向 train」的回落坑（D2）---
    if split_count_map["val"] == 0:
        warnings.append(
            "val_empty：切分后验证集为空（val=0）；data.yaml 的 val 指向空的 images/val，"
            "绝不回落到训练集（否则 mAP 虚高）。下游评估会 fail-loud。"
        )
    if not entries and holdout_ratio > 0:
        warnings.append(
            "holdout_frozen_empty：index.json 已存在且冻结为空集合（首轮 ratio=0），"
            "本轮 holdout_ratio>0 也不会重新切分（holdout 铁律，跨轮可比优先）。"
        )

    data_yaml_path = dataset_dir / "data.yaml"
    write_text(
        data_yaml_path,
        _render_data_yaml(
            config,
            dataset_dir,
            {"train": len(split_map["train"]), "val": len(split_map["val"]), "test": 0},
        ),
    )
    manifest_path = dataset_dir / "manifest.json"
    histogram_path = dataset_dir / "class_histogram.json"
    quality_path = dataset_dir / "quality_report.json"
    write_json(manifest_path, manifest)
    write_json(histogram_path, class_histogram)
    write_json(quality_path, quality_rows)

    summary: dict[str, Any] = {
        "dataset_version": normalized_version,
        "dataset_dir": str(dataset_dir.resolve()),
        "data_yaml_path": str(data_yaml_path.resolve()),
        "manifest_path": str(manifest_path.resolve()),
        "class_histogram_path": str(histogram_path.resolve()),
        "quality_report_path": str(quality_path.resolve()),
        "split_counts": split_count_map,
        "class_histogram": class_histogram,
        "total_records": len(records),
        "accepted": len(accepted),
        "rejected": len(rejected),
        "holdout_count": len(entries),
        "holdout_hash": holdout_hash_of(entries),
        "holdout_index_path": str((out_dir / "index.json").resolve()),
        "review_queue_count": sum(1 for record in records if record.review_required),
        # D8：丢标签（被当背景图）的帧数——必须可见，绝不静默。
        "missing_label_count": missing_label_count,
        # D2：val 是否为空（data.yaml 的 val 指向空目录，不回落到 train）。
        "val_empty": split_count_map["val"] == 0,
        "warnings": warnings,
    }
    write_json(dataset_dir / "summary.json", summary)
    return summary
