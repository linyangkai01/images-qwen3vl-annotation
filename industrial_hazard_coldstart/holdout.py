"""T04.2 固定 holdout 冻结 + sha256 登记 + 逐字节校验（需求 7.4）。

**为什么必须单独做**：母项目 ``pipeline.training.run_training`` 每轮都用
``timestamp_token()`` 作版本号并**重新随机切分**，因此没有「跨轮逐字节不变」的评估集
——这正是需求第 7 节记录的 P1 问题「无固定评估集 → 跨轮不可比」。本模块提供
**可冻结、可校验、可读回**的 holdout：

1. **首次冻结**：按 ``seed`` 从（已通过质量过滤的）记录里切出 holdout，逐帧把图片
   物化到 ``<out_dir>/frames/``（保证 Ultralytics 可读扩展名）、标签物化到
   ``<out_dir>/labels/``，并登记每帧 ``sha256`` 与 ``registered_at``；
2. **后续轮次（本模块的关键判据）**：``index.json`` 一旦存在，:func:`freeze_holdout`
   **读回已冻结清单、绝不重新切分**——新的记录只能进 train/val，holdout 集合逐帧不变，
   因此跨轮指标可比；
3. **篡改 / 删除检出**：:func:`verify_holdout` 重算每帧 sha256，分别检出「内容被改」
   与「文件被删」两种情况。

存储布局::

    <out_dir>/
        index.json          # 冻结清单（entries: dataset_version/relative_path/sha256/registered_at）
        frames/<relative_path 经扩展名规范化>   # 冻结帧副本（逐字节不变）
        labels/<同上>.txt                        # 冻结标签副本

仅使用 Python 标准库 + 母项目；中文路径安全。
"""

from __future__ import annotations

import hashlib
import math
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from traffic_cv_closed_loop.config import ProjectConfig
from traffic_cv_closed_loop.dataset_media import (
    dataset_image_relative_path,
    materialize_dataset_image,
)
from traffic_cv_closed_loop.io_utils import copy_file, read_json, write_json

from industrial_hazard_coldstart.models import HoldoutEntry

__all__ = [
    "DEFAULT_HOLDOUT_RATIO",
    "holdout_dir_of",
    "holdout_index_path",
    "frozen_frame_path",
    "frozen_label_path",
    "holdout_count",
    "validate_holdout_ratio",
    "select_holdout_paths",
    "load_holdout_entries",
    "freeze_holdout",
    "verify_holdout",
    "holdout_hash_of",
]

#: 默认 holdout 占比。
DEFAULT_HOLDOUT_RATIO: float = 0.2

#: 冻结清单文件名。
INDEX_FILENAME: str = "index.json"

#: 冻结帧 / 标签子目录名。
FRAMES_DIRNAME: str = "frames"
LABELS_DIRNAME: str = "labels"


# --------------------------------------------------------------------------- #
# 路径与哈希工具
# --------------------------------------------------------------------------- #
def holdout_dir_of(config: ProjectConfig) -> Path:
    """返回项目配置对应的 holdout 根目录（``workspace/holdout``）。"""
    return config.workspace("holdout")


def holdout_index_path(out_dir: str | Path) -> Path:
    """返回冻结清单 ``<out_dir>/index.json`` 的路径。"""
    return Path(out_dir) / INDEX_FILENAME


def frozen_frame_path(out_dir: str | Path, relative_path: str) -> Path:
    """返回某帧冻结副本的路径（扩展名经 ``dataset_image_relative_path`` 规范化）。"""
    produced = dataset_image_relative_path(Path(relative_path))
    return Path(out_dir) / FRAMES_DIRNAME / produced


def frozen_label_path(out_dir: str | Path, relative_path: str) -> Path:
    """返回某帧冻结标签副本的路径。"""
    produced = dataset_image_relative_path(Path(relative_path)).with_suffix(".txt")
    return Path(out_dir) / LABELS_DIRNAME / produced


def _now_iso() -> str:
    """当前本地时间的 ISO 字符串（秒级精度，够用且稳定）。"""
    return datetime.now().isoformat(timespec="seconds")


def _sha256_file(path: Path) -> str:
    """流式计算文件 sha256（大文件也不吃内存）。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


# --------------------------------------------------------------------------- #
# 选择逻辑
# --------------------------------------------------------------------------- #
def validate_holdout_ratio(ratio: float) -> float:
    """校验 holdout 占比：必须是有限实数且落在 ``[0, 1]``，否则 ``ValueError``。

    本项目的既定规矩是「**绝不静默**」：一行配置笔误（``-1`` / ``25``（想写 25%）/
    ``nan``）若被静默钳制，会让固定评估集**悄悄失效或膨胀**且无任何提示——这正是
    需求 7.4 要消灭的隐患。故此处 fail-loud，并在消息里说清合法范围。

    Args:
        ratio: 待校验的占比。

    Returns:
        规整为 ``float`` 的合法占比。

    Raises:
        ValueError: 非数字 / NaN / Inf / 越界。
    """
    try:
        value = float(ratio)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"holdout_ratio 必须是 [0, 1] 之间的数字，收到 {ratio!r}。"
        ) from exc
    if math.isnan(value) or math.isinf(value):
        raise ValueError(f"holdout_ratio 不能是 NaN / Inf，收到 {ratio!r}。")
    if value < 0.0 or value > 1.0:
        raise ValueError(
            f"holdout_ratio 必须落在 [0, 1]，收到 {value}"
            "（若想表达百分比请写 0.25 而非 25）。"
        )
    return value


def holdout_count(total: int, ratio: float) -> int:
    """按占比算出 holdout 帧数（至少留 1 帧给 train）。

    - ``total <= 0`` → ``0``；
    - ``ratio`` 必须落在 ``[0, 1]``（NaN / 越界 → ``ValueError``，**绝不静默钳制**）；
    - ``ratio == 0`` → ``0``；
    - ``ratio > 0`` 且 ``total >= 2`` 时至少 1 帧（保证「冻结了东西」可被校验）；
    - 上限 ``total - 1``（绝不吃光全部帧，否则没有训练数据）。
    """
    if total <= 0:
        return 0
    value = validate_holdout_ratio(ratio)
    if value <= 0.0:
        return 0
    count = int(round(total * value))
    if count <= 0:
        count = 1
    if count >= total:
        count = max(0, total - 1)
    return count


def _select_holdout_records(records: Iterable[Any], ratio: float, seed: int) -> list[Any]:
    """确定性地选出 holdout 记录（先按 ``relative_path`` 排序再打乱）。"""
    ordered = sorted(records, key=lambda record: record.candidate.relative_path)
    shuffled = list(ordered)
    random.Random(int(seed)).shuffle(shuffled)
    return shuffled[: holdout_count(len(shuffled), ratio)]


def select_holdout_paths(
    records: Iterable[Any],
    ratio: float,
    seed: int,
) -> list[str]:
    """按 ``seed`` 确定性地选出 holdout 的 ``relative_path`` 列表。

    先按 ``relative_path`` 排序再打乱，因此结果只取决于「记录集合 + seed」，
    与输入顺序无关（可复现）。
    """
    return [
        record.candidate.relative_path
        for record in _select_holdout_records(records, ratio, seed)
    ]


# --------------------------------------------------------------------------- #
# 冻结清单读写
# --------------------------------------------------------------------------- #
def load_holdout_entries(out_dir: str | Path) -> list[HoldoutEntry]:
    """读回已冻结的 holdout 清单；不存在 / 损坏时返回空列表。"""
    payload = read_json(holdout_index_path(out_dir), default=None)
    if not isinstance(payload, dict):
        return []
    raw = payload.get("entries")
    if not isinstance(raw, list):
        return []
    entries = [HoldoutEntry.from_dict(item) for item in raw if isinstance(item, dict)]
    entries.sort(key=lambda entry: entry.relative_path)
    return entries


# --------------------------------------------------------------------------- #
# 冻结
# --------------------------------------------------------------------------- #
def freeze_holdout(
    records: Iterable[Any],
    ratio: float,
    seed: int,
    out_dir: str | Path,
    version: str = "",
) -> list[HoldoutEntry]:
    """冻结固定 holdout（幂等）。

    **核心行为**：若 ``<out_dir>/index.json`` **存在**，则**直接读回已冻结清单并返回**，
    绝不重新随机切分——这正是「holdout 逐字节不变、跨轮可比」的实现。只有首次
    （清单不存在）才按 ``seed`` 切分并登记。

    判据是「``index.json`` **是否存在**」而非「``entries`` 是否非空」：首轮
    ``ratio=0`` 会冻结出一个**空集合**（``entries: []``），它同样是一次已完成的冻结；
    若以「非空」为判据，第二轮改用 ``ratio>0`` 就会**静默重新切分**，跨轮可比性出现
    缺口（需求 7.4 要消灭的问题）。

    Args:
        records: 候选标注记录（``AnnotationRecord`` 列表，通常已通过质量过滤）。
        ratio: holdout 占比（仅**首次**冻结时生效；之后以清单为准）。
        seed: 随机种子（仅**首次**冻结时生效）。
        out_dir: holdout 根目录（如 ``workspace/holdout``）。
        version: 首次冻结时写入清单的 ``dataset_version`` 标签（可追溯来源）。

    Returns:
        冻结清单（``list[HoldoutEntry]``，按 ``relative_path`` 排序；可能为空）。

    Raises:
        ValueError: 首次冻结时 ``ratio`` 不在 ``[0, 1]``（或为 NaN）。
    """
    out_dir = Path(out_dir)
    index_path = holdout_index_path(out_dir)
    if index_path.exists():
        # 关键：index.json 一旦存在即视为「已冻结」——哪怕是空集合也绝不重新切分。
        return load_holdout_entries(out_dir)

    ordered = sorted(records, key=lambda record: record.candidate.relative_path)
    selected = _select_holdout_records(ordered, ratio, seed)

    frames_dir = out_dir / FRAMES_DIRNAME
    labels_dir = out_dir / LABELS_DIRNAME
    entries: list[HoldoutEntry] = []
    for record in selected:
        relative = Path(record.candidate.relative_path)
        produced = materialize_dataset_image(
            record.candidate.extracted,
            frames_dir,
            relative,
        )
        frame_path = frames_dir / produced
        label_source = Path(record.label_path) if record.label_path else None
        if label_source is not None and label_source.exists():
            copy_file(label_source, labels_dir / produced.with_suffix(".txt"))
        entries.append(
            HoldoutEntry(
                dataset_version=str(version),
                relative_path=record.candidate.relative_path,
                sha256=_sha256_file(frame_path),
                registered_at=_now_iso(),
            )
        )

    entries.sort(key=lambda entry: entry.relative_path)
    write_json(
        holdout_index_path(out_dir),
        {
            "version": str(version),
            "ratio": float(ratio),
            "seed": int(seed),
            "frames_dir": FRAMES_DIRNAME,
            "labels_dir": LABELS_DIRNAME,
            "entries": [entry.to_dict() for entry in entries],
        },
    )
    return entries


# --------------------------------------------------------------------------- #
# 校验
# --------------------------------------------------------------------------- #
def verify_holdout(
    entries: Iterable[HoldoutEntry],
    out_dir: str | Path | None = None,
) -> dict[str, Any]:
    """校验 holdout 是否逐字节未变。

    对每一帧重算 sha256：文件不存在 → 计入 ``missing``；哈希不一致 → 计入
    ``tampered``。``hash_ok`` 仅在两者皆空时为 ``True``。

    Args:
        entries: 冻结清单（``load_holdout_entries`` / ``freeze_holdout`` 的产物）。
        out_dir: holdout 根目录；省略时回退到 ``workspace/holdout``（相对当前工作目录）。

    Returns:
        ``{"hash_ok", "total", "verified", "tampered", "missing"}``。
    """
    base = Path(out_dir) if out_dir is not None else Path("workspace") / "holdout"
    entry_list = list(entries)
    tampered: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    verified = 0
    for entry in entry_list:
        frame_path = frozen_frame_path(base, entry.relative_path)
        if not frame_path.exists():
            missing.append(
                {
                    "relative_path": entry.relative_path,
                    "expected": entry.sha256,
                    "reason": "missing",
                }
            )
            continue
        actual = _sha256_file(frame_path)
        if actual != entry.sha256:
            tampered.append(
                {
                    "relative_path": entry.relative_path,
                    "expected": entry.sha256,
                    "actual": actual,
                    "reason": "hash_mismatch",
                }
            )
        else:
            verified += 1
    return {
        "hash_ok": not tampered and not missing,
        "total": len(entry_list),
        "verified": verified,
        "tampered": tampered,
        "missing": missing,
    }


def holdout_hash_of(entries: Iterable[HoldoutEntry]) -> str:
    """把冻结清单聚合成一个稳定摘要（供跨轮比较「holdout 是否完全一致」）。"""
    payload = "\n".join(
        f"{entry.relative_path}\t{entry.sha256}"
        for entry in sorted(entries, key=lambda item: item.relative_path)
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
