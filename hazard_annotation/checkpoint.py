"""抽帧阶段的断点续跑（checkpoint / resume）。

抽帧是小时级任务：一段 1 小时视频要解码 3600 帧分析帧再做六步筛选。
中途中断（Ctrl-C、机器重启、ffmpeg 崩）后如果只能从头再来，代价很高。

本模块提供**按数据源（source）粒度**的断点：

* 每个 source 抽完后，把它的关键帧记录落到 ``extracted/.parts/<source>.jsonl``，
  并在 ``extracted/.progress.json`` 里登记一个「指纹」。
* 重跑时，先比指纹；指纹一致且分片文件在 → 直接复用，跳过抽帧。
* 指纹不一致（换了视频 / 改了抽帧参数）→ **该 source 自动失效并重算**，
  不会拿旧结果冒充新配置的产物。

指纹分两层，缺一不可：

1. **配置指纹**（``config_fingerprint``）—— 覆盖所有会影响抽帧结果的配置项。
   改一个 ``motion_min_ratio`` 就必须重算，否则会得到"新旧参数混在一起"的脏数据集。
2. **数据源指纹**（``source_signature``）—— 覆盖输入本身，**基于文件内容哈希**。

关于数据源指纹为什么要哈希内容而不是看 mtime —— 这是踩过两次坑的结论：

* 第一次用 ``round(st_mtime, 3)``：同一毫秒内的两次写入检测不到。
* 改用 ``st_mtime_ns`` 后**在服务器上依然失败** —— 因为 ``/data/users/lyk/.tmp``
  挂在 **ext2/ext3** 上，**mtime 只有 1 秒粒度**，两次连续写入拿到完全相同的纳秒值。

**只要依赖 mtime，就无法在所有文件系统上正确。** 所以改为 sha256 内容哈希。
代价可接受：1 小时视频解码要几分钟，而哈希 300MB 只要 1 秒量级。
顺带一个语义好处：单纯 ``touch`` 不再触发重算（内容没变），这才是"输入是否变化"的正解。

写入一律走「临时文件 + 原子替换」，避免断电/中断留下半截 JSON 把后续重跑带偏。
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hazard_annotation.keyframes import KeyframeSource
from hazard_annotation.models import KeyframeRecord
from traffic_cv_closed_loop.config import ProjectConfig
from traffic_cv_closed_loop.io_utils import ensure_dir, read_jsonl, write_jsonl

#: 断点文件格式版本；结构变更时递增，旧版本会被整体作废。
CHECKPOINT_VERSION = 1

_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


def _sha256_of(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def config_fingerprint(config: ProjectConfig) -> str:
    """把所有会影响抽帧结果的配置压成一个指纹。

    只要这些值有一个变了，指纹就变，断点整体作废并重算。
    """
    relevant = {
        "labels": list(config.labels),
        "video": dataclasses.asdict(config.video),
        "keyframe": dataclasses.asdict(config.keyframe),
        "detector": config.yolo_detector,
        "detector_weights": config.yolo_detector_weights,
        "detector_confidence": config.yolo_detector_confidence,
        "detector_iou": config.yolo_detector_iou,
        "detector_image_size": config.yolo_detector_image_size,
    }
    # video_path 本身由 source_signature 负责，这里去掉避免重复失效
    relevant["video"].pop("video_path", None)
    return _sha256_of(json.dumps(relevant, ensure_ascii=False, sort_keys=True, default=str))


def _file_content_digest(path: Path, chunk_size: int = 1 << 20) -> str:
    """流式计算文件内容的 sha256，分块读避免把大文件整个塞进内存。"""
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            while True:
                chunk = stream.read(chunk_size)
                if not chunk:
                    break
                digest.update(chunk)
    except OSError:
        return "unreadable"
    return digest.hexdigest()


def _file_fingerprint(path: Path) -> dict[str, Any]:
    """文件指纹：路径 + 大小 + **内容哈希**。

    为什么不用 mtime —— 这是踩过坑的结论：

    最初用 ``round(st_mtime, 3)``，同一毫秒内的两次写入检测不到；改用
    ``st_mtime_ns`` 后**在服务器上依然失败**，因为 ``/data/users/lyk/.tmp``
    是 **ext2/ext3 文件系统，mtime 只有 1 秒粒度** —— 两次连续写入拿到的是
    完全相同的纳秒时间戳。

    结论：**只要依赖 mtime，就无法在所有文件系统上正确**。所以改为对内容做
    sha256。代价可接受：1 小时视频解码要几分钟，而哈希 300MB 只要 1 秒量级。

    顺带一个语义上的好处：单纯 ``touch`` 文件不再触发重算，因为内容没变 ——
    这才是"输入是否变化"的正确含义。
    """
    try:
        stat = path.stat()
    except OSError:
        return {"path": str(path), "missing": True}
    return {
        "path": str(path),
        "size": stat.st_size,
        "sha256": _file_content_digest(path),
    }


def source_signature(source: KeyframeSource) -> str:
    """给一个输入源算指纹；**内容**变了指纹就变。"""
    if source.video_path is not None:
        payload: dict[str, Any] = {
            "kind": "video",
            "source_id": source.source_id,
            "file": _file_fingerprint(Path(source.video_path)),
        }
    elif source.frames_dir is not None:
        frames_dir = Path(source.frames_dir)
        entries: list[dict[str, Any]] = []
        try:
            candidates = sorted(p for p in frames_dir.iterdir() if p.is_file())
        except OSError:
            candidates = []
        for path in candidates:
            entries.append({"name": path.name, **_file_fingerprint(path)})
        payload = {
            "kind": "frames_dir",
            "source_id": source.source_id,
            "dir": str(frames_dir),
            "file_count": len(entries),
            "files": entries,
        }
    else:
        payload = {"kind": "unknown", "source_id": source.source_id}

    return _sha256_of(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str))


def safe_part_name(source_id: str, max_length: int = 64) -> str:
    """把 source_id 转成安全的文件名片段（避免路径穿越与非法字符）。"""
    cleaned = _SAFE_NAME.sub("_", source_id).strip("._-") or "source"
    if len(cleaned) <= max_length:
        return cleaned
    digest = hashlib.sha256(source_id.encode("utf-8")).hexdigest()[:8]
    return f"{cleaned[: max_length - 9]}_{digest}"


def _atomic_write_text(path: Path, text: str) -> None:
    """先写同目录临时文件再原子替换，避免中断留下半截文件。"""
    ensure_dir(path.parent)
    handle, temp_name = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp-", suffix=".part")
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise


@dataclass
class SourceProgress:
    """单个数据源的完成记录。"""

    source_id: str
    signature: str
    record_count: int
    counts: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    finished_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "signature": self.signature,
            "record_count": self.record_count,
            "counts": dict(self.counts),
            "warnings": list(self.warnings),
            "finished_at": self.finished_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SourceProgress":
        return cls(
            source_id=str(data.get("source_id", "")),
            signature=str(data.get("signature", "")),
            record_count=int(data.get("record_count", 0)),
            counts={str(k): int(v) for k, v in (data.get("counts") or {}).items()},
            warnings=[str(item) for item in (data.get("warnings") or [])],
            finished_at=str(data.get("finished_at", "")),
        )


class ExtractionCheckpoint:
    """抽帧断点：管理 ``.progress.json`` 与 ``.parts/*.jsonl``。

    ``enabled=False`` 时所有写入与复用都被跳过，等价于原来的「每次全量重算」。
    """

    def __init__(self, config: ProjectConfig, enabled: bool = True) -> None:
        self.config = config
        self.enabled = enabled
        self.progress_path = Path(config.workspace_dir) / "extracted" / ".progress.json"
        self.parts_dir = Path(config.workspace_dir) / "extracted" / ".parts"
        self.fingerprint = config_fingerprint(config)
        self._sources: dict[str, SourceProgress] = {}
        self.reused_sources: list[str] = []
        self.recomputed_sources: list[str] = []
        self.invalidation_reason: str = ""

    # -- 载入与失效 -------------------------------------------------------- #
    def load(self) -> None:
        """读入断点文件；版本或配置指纹不符则整体作废。

        断点文件损坏（截断、非法 JSON）时**不抛异常**：断点只是加速手段，
        读不出来就当没有，退化成全量重算即可。
        """
        if not self.enabled:
            return
        if not self.progress_path.exists():
            return
        try:
            payload = json.loads(self.progress_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, UnicodeDecodeError):
            self.invalidation_reason = "checkpoint_unreadable"
            return
        if not isinstance(payload, dict):
            self.invalidation_reason = "checkpoint_malformed"
            return

        if int(payload.get("version", 0)) != CHECKPOINT_VERSION:
            self.invalidation_reason = "checkpoint_version_changed"
            return

        if str(payload.get("config_fingerprint", "")) != self.fingerprint:
            self.invalidation_reason = "config_changed"
            return

        for item in payload.get("sources", []):
            if isinstance(item, dict):
                progress = SourceProgress.from_dict(item)
                if progress.source_id:
                    self._sources[progress.source_id] = progress

    # -- 复用判定 ---------------------------------------------------------- #
    def is_reusable(self, source: KeyframeSource) -> bool:
        """该 source 能否直接复用已有结果。"""
        if not self.enabled:
            return False
        progress = self._sources.get(source.source_id)
        if progress is None:
            return False
        if progress.signature != source_signature(source):
            return False
        return self.part_path(source.source_id).exists()

    def read_records(self, source_id: str) -> list[KeyframeRecord]:
        """读回某个 source 已抽好的记录。"""
        rows = read_jsonl(self.part_path(source_id))
        return [KeyframeRecord.from_dict(row) for row in rows]

    def progress_of(self, source_id: str) -> SourceProgress | None:
        return self._sources.get(source_id)

    def part_path(self, source_id: str) -> Path:
        return self.parts_dir / f"{safe_part_name(source_id)}.jsonl"

    # -- 写入 -------------------------------------------------------------- #
    def mark_done(
        self,
        source: KeyframeSource,
        records: list[KeyframeRecord],
        counts: dict[str, int],
        warnings: list[str],
    ) -> None:
        """记录某个 source 已完成，并落盘它的分片。"""
        if not self.enabled:
            return
        write_jsonl(self.part_path(source.source_id), [item.to_dict() for item in records])
        self._sources[source.source_id] = SourceProgress(
            source_id=source.source_id,
            signature=source_signature(source),
            record_count=len(records),
            counts=dict(counts),
            warnings=list(warnings),
            finished_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
        self.flush()

    def flush(self) -> None:
        """把当前进度原子写盘。"""
        if not self.enabled:
            return
        payload = {
            "version": CHECKPOINT_VERSION,
            "config_fingerprint": self.fingerprint,
            "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "sources": [item.to_dict() for item in self._sources.values()],
        }
        _atomic_write_text(
            self.progress_path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
        )

    # -- 汇总 -------------------------------------------------------------- #
    def summary(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "fingerprint": self.fingerprint,
            "invalidation_reason": self.invalidation_reason,
            "reused_sources": list(self.reused_sources),
            "recomputed_sources": list(self.recomputed_sources),
            "reused_count": len(self.reused_sources),
            "recomputed_count": len(self.recomputed_sources),
        }
