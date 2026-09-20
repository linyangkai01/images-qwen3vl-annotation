"""T04.2/T04.1 单测：固定 holdout 冻结 / 逐字节不变 / 篡改检出 / 无泄漏。

覆盖点（对应任务书 T04.4 判据）：

1. **holdout 逐字节不变（核心判据）** —— 连续构建两次数据集（第二轮输入**更多**记录），
   断言 holdout 集合与首次**完全相同**，且新增记录**只进 train/val**；
2. **篡改检出** —— 改动一个 holdout 帧的字节 → ``verify_holdout`` 报 ``hash_ok=False``
   并列出该帧；
3. **删除检出** —— 删掉一个 holdout 帧 → 同样被检出（``missing``）；
4. **无泄漏** —— holdout 的 ``relative_path`` 不出现在 train/val 的 manifest 里，
   且 ``data.yaml`` 不含 holdout；
5. **版本号可指定（非时间戳）** + 同名版本重建行为明确；
6. **质量过滤** —— 被剔除的记录进 ``quality_report.json``，且不进 manifest；
7. **中文路径**。

约束：不修改任何生产代码；不联网、不调用真实 VLM；不读 ``industrial-data/``；
临时文件用 ``tempfile``。
"""

from __future__ import annotations

import hashlib
import json
import struct
import tempfile
import unittest
import zlib
from pathlib import Path

from traffic_cv_closed_loop.config import ProjectConfig
from traffic_cv_closed_loop.io_utils import read_json
from traffic_cv_closed_loop.models import AnnotationRecord, CandidateFrame, Detection

from industrial_hazard_coldstart.dataset import build_dataset
from industrial_hazard_coldstart.holdout import (
    freeze_holdout,
    frozen_frame_path,
    holdout_dir_of,
    holdout_hash_of,
    load_holdout_entries,
    verify_holdout,
)

ROOT = Path(__file__).resolve().parents[1]
HOIST_CONFIG = ROOT / "own" / "hoist_project" / "config" / "project.hoist.json"
KB_PATH = ROOT / "hazard-detection" / "knowledge_base.json"


# --------------------------------------------------------------------------- #
# 合成素材工具（绝不触碰真实素材）
# --------------------------------------------------------------------------- #
def _write_png(path: Path, width: int = 8, height: int = 8, color: tuple[int, int, int] = (120, 120, 120)) -> Path:
    """写一张**真实可解码**的最小 PNG（纯标准库 zlib + struct）。"""

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    row = b"\x00" + bytes(color) * width
    raw = row * height
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png)
    return path


def _make_config(tmp: Path) -> ProjectConfig:
    """在临时目录写一份 mock 吊装配置并加载（KB 显式指向仓库内知识库）。"""
    raw = json.loads(HOIST_CONFIG.read_text(encoding="utf-8"))
    raw["qwen"]["provider"] = "mock"
    raw["knowledge"]["kb_path"] = str(KB_PATH)
    raw["yolo"]["trainer"] = "mock"
    raw["yolo"]["detector"] = "sidecar"
    config_dir = tmp / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "project.json"
    config_path.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
    return ProjectConfig.load(config_path)


def _make_record(
    config: ProjectConfig,
    relative_path: str,
    *,
    label: str = "person",
    confidence: float = 0.9,
    det_count: int = 1,
    review_required: bool = False,
    review_reasons: list[str] | None = None,
) -> AnnotationRecord:
    """造一条**真实可物化**的标注记录（PNG + 标签文件都写到磁盘）。"""
    extracted = config.input_frames_dir / relative_path
    _write_png(extracted)
    label_path = config.workspace("review", "labels") / Path(relative_path).with_suffix(".txt")
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    detections = [
        Detection(label=label, bbox=(0.1, 0.1, 0.3, 0.5), confidence=confidence)
        for _ in range(det_count)
    ]
    candidate = CandidateFrame(
        source_id=Path(relative_path).parts[0],
        relative_path=relative_path,
        original_path=str(extracted),
        extracted_path=str(extracted),
        detections=detections,
        scene="hoist",
    )
    return AnnotationRecord(
        candidate=candidate,
        qwen_detections=detections,
        diff={"added": 0, "removed": 0, "adjusted": 0},
        review_required=review_required,
        review_reasons=review_reasons or [],
        prompt_path="",
        raw_response_path="",
        label_path=str(label_path),
        dataset_ready=True,
    )


def _write_annotations(config: ProjectConfig, records: list[AnnotationRecord]) -> None:
    config.annotations_path.parent.mkdir(parents=True, exist_ok=True)
    with config.annotations_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")


def _build_records(config: ProjectConfig, count: int, prefix: str = "clip") -> list[AnnotationRecord]:
    return [
        _make_record(config, f"{prefix}/frame_{index:06d}.png") for index in range(count)
    ]


def _manifest_by_split(manifest_path: str | Path) -> dict[str, set[str]]:
    by_split: dict[str, set[str]] = {}
    for item in read_json(Path(manifest_path)) or []:
        by_split.setdefault(item["split"], set()).add(item["relative_path"])
    return by_split


# --------------------------------------------------------------------------- #
# 1. 逐字节不变（核心判据）
# --------------------------------------------------------------------------- #
class HoldoutStabilityTest(unittest.TestCase):
    def test_holdout_identical_across_rebuilds_and_new_records_only_train_val(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = _make_config(Path(tmp))
            _write_annotations(config, _build_records(config, 10))

            first = build_dataset(config, "v1", holdout_ratio=0.3)
            first_entries = load_holdout_entries(holdout_dir_of(config))
            first_paths = {entry.relative_path for entry in first_entries}
            self.assertTrue(first_paths, "首次冻结应至少选出 1 帧 holdout")
            self.assertEqual(first["holdout_count"], len(first_paths))

            # 第二轮：输入**更多**记录（前 10 条不变 + 新增 6 条）。
            all_records = _build_records(config, 16)
            _write_annotations(config, all_records)
            second = build_dataset(config, "v2", holdout_ratio=0.3)

            second_entries = load_holdout_entries(holdout_dir_of(config))
            second_paths = {entry.relative_path for entry in second_entries}

            # 核心断言：holdout 集合与首次**完全相同**（读回冻结清单，未重新切分）。
            self.assertEqual(first_paths, second_paths)
            self.assertEqual(first["holdout_hash"], second["holdout_hash"])
            self.assertEqual(
                holdout_hash_of(first_entries), holdout_hash_of(second_entries)
            )

            # 新增记录只进 train/val，绝不进 holdout。
            new_paths = {
                record.candidate.relative_path
                for record in all_records
            } - {record.candidate.relative_path for record in _build_records(config, 10)}
            self.assertTrue(new_paths, "应存在新增记录")
            by_split = _manifest_by_split(second["manifest_path"])
            train_val = by_split.get("train", set()) | by_split.get("val", set())
            self.assertTrue(new_paths.issubset(train_val))
            self.assertFalse(new_paths & by_split.get("holdout", set()))

    def test_holdout_frames_byte_identical_across_versions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = _make_config(Path(tmp))
            _write_annotations(config, _build_records(config, 9))

            build_dataset(config, "v1", holdout_ratio=0.3)
            first = {e.relative_path: e.sha256 for e in load_holdout_entries(holdout_dir_of(config))}

            build_dataset(config, "v2", holdout_ratio=0.3)
            second = {e.relative_path: e.sha256 for e in load_holdout_entries(holdout_dir_of(config))}

            self.assertEqual(first, second)
            # 数据集内 holdout 副本的字节 == 冻结副本的字节。
            out_dir = holdout_dir_of(config)
            for relative_path, sha in second.items():
                frozen = frozen_frame_path(out_dir, relative_path)
                dataset_copy = (
                    config.datasets_dir
                    / "v2"
                    / "images"
                    / "holdout"
                    / frozen.relative_to(out_dir / "frames")
                )
                self.assertTrue(dataset_copy.exists())
                self.assertEqual(
                    hashlib.sha256(dataset_copy.read_bytes()).hexdigest(), sha
                )

    def test_freeze_is_idempotent_when_index_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = _make_config(Path(tmp))
            records = _build_records(config, 8)
            out_dir = holdout_dir_of(config)

            first = freeze_holdout(records, 0.5, 7, out_dir, version="v1")
            # 即便换 seed / ratio，已冻结清单也必须原样读回。
            second = freeze_holdout(records, 0.9, 123, out_dir, version="v2")
            self.assertEqual(
                [e.relative_path for e in first], [e.relative_path for e in second]
            )
            self.assertEqual(
                [e.dataset_version for e in first], [e.dataset_version for e in second]
            )


# --------------------------------------------------------------------------- #
# 2. 篡改 / 删除检出
# --------------------------------------------------------------------------- #
class HoldoutVerifyTest(unittest.TestCase):
    def _frozen(self) -> tuple[tempfile.TemporaryDirectory, ProjectConfig, list]:
        handle = tempfile.TemporaryDirectory()
        config = _make_config(Path(handle.name))
        _write_annotations(config, _build_records(config, 8))
        build_dataset(config, "v1", holdout_ratio=0.4)
        entries = load_holdout_entries(holdout_dir_of(config))
        return handle, config, entries

    def test_tampering_detected(self) -> None:
        handle, config, entries = self._frozen()
        try:
            out_dir = holdout_dir_of(config)
            target = entries[0]
            frame = frozen_frame_path(out_dir, target.relative_path)
            frame.write_bytes(frame.read_bytes() + b"\x00tampered")

            result = verify_holdout(entries, out_dir)
            self.assertFalse(result["hash_ok"])
            self.assertEqual(result["verified"], len(entries) - 1)
            self.assertIn(
                target.relative_path, {item["relative_path"] for item in result["tampered"]}
            )
        finally:
            handle.cleanup()

    def test_deletion_detected(self) -> None:
        handle, config, entries = self._frozen()
        try:
            out_dir = holdout_dir_of(config)
            target = entries[0]
            frozen_frame_path(out_dir, target.relative_path).unlink()

            result = verify_holdout(entries, out_dir)
            self.assertFalse(result["hash_ok"])
            self.assertIn(
                target.relative_path, {item["relative_path"] for item in result["missing"]}
            )
        finally:
            handle.cleanup()

    def test_untouched_holdout_passes(self) -> None:
        handle, config, entries = self._frozen()
        try:
            result = verify_holdout(entries, holdout_dir_of(config))
            self.assertTrue(result["hash_ok"])
            self.assertEqual(result["tampered"], [])
            self.assertEqual(result["missing"], [])
            self.assertEqual(result["verified"], len(entries))
        finally:
            handle.cleanup()


# --------------------------------------------------------------------------- #
# 3. 无泄漏
# --------------------------------------------------------------------------- #
class HoldoutLeakageTest(unittest.TestCase):
    def test_holdout_absent_from_train_val_manifest_and_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = _make_config(Path(tmp))
            _write_annotations(config, _build_records(config, 12))
            summary = build_dataset(config, "v1", holdout_ratio=0.3)

            by_split = _manifest_by_split(summary["manifest_path"])
            holdout = by_split.get("holdout", set())
            train = by_split.get("train", set())
            val = by_split.get("val", set())

            self.assertTrue(holdout)
            self.assertTrue(train)
            self.assertTrue(val)
            self.assertEqual(holdout & train, set())
            self.assertEqual(holdout & val, set())
            # 所有 manifest 记录恰好落在 train/val/holdout 之一，无重复归属。
            total = len(holdout) + len(train) + len(val)
            self.assertEqual(total, len(read_json(Path(summary["manifest_path"]))))

            yaml_text = Path(summary["data_yaml_path"]).read_text(encoding="utf-8")
            self.assertNotIn("holdout", yaml_text)


# --------------------------------------------------------------------------- #
# 4. 版本号 / 质量过滤 / 中文路径
# --------------------------------------------------------------------------- #
class DatasetBuildTest(unittest.TestCase):
    def test_version_is_caller_specified_not_timestamp(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = _make_config(Path(tmp))
            _write_annotations(config, _build_records(config, 6))
            summary = build_dataset(config, "alpha-1", holdout_ratio=0.3)
            self.assertEqual(summary["dataset_version"], "alpha-1")
            self.assertTrue((config.datasets_dir / "alpha-1" / "data.yaml").exists())
            self.assertTrue(
                (config.datasets_dir / "alpha-1" / "manifest.json").exists()
            )

    def test_rebuild_same_version_keeps_holdout_and_grows_train_val(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = _make_config(Path(tmp))
            _write_annotations(config, _build_records(config, 10))
            first = build_dataset(config, "same", holdout_ratio=0.3)
            first_by_split = _manifest_by_split(first["manifest_path"])
            first_holdout = first_by_split["holdout"]
            first_train_val = len(first_by_split["train"]) + len(first_by_split["val"])

            _write_annotations(config, _build_records(config, 16))
            second = build_dataset(config, "same", holdout_ratio=0.3)
            by_split = _manifest_by_split(second["manifest_path"])

            self.assertEqual(first_holdout, by_split["holdout"])
            self.assertGreater(
                len(by_split["train"]) + len(by_split["val"]),
                first_train_val,
            )

    def test_empty_version_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = _make_config(Path(tmp))
            _write_annotations(config, _build_records(config, 4))
            with self.assertRaises(ValueError):
                build_dataset(config, "   ")

    def test_missing_annotations_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = _make_config(Path(tmp))
            with self.assertRaises(FileNotFoundError):
                build_dataset(config, "v1")

    def test_quality_filter_records_rejections(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = _make_config(Path(tmp))
            good = _build_records(config, 6)
            error_frame = _make_record(
                config,
                "clip/frame_error.png",
                review_required=True,
                review_reasons=["annotation_error"],
            )
            low_conf = _make_record(config, "clip/frame_lowconf.png", confidence=0.05)
            _write_annotations(config, good + [error_frame, low_conf])

            summary = build_dataset(config, "v1", holdout_ratio=0.3)
            quality = read_json(Path(summary["quality_report_path"]))
            rejected = {
                row["frame"]: row["filter_reasons"]
                for row in quality
                if not row["accepted"]
            }
            self.assertIn("clip/frame_error.png", rejected)
            self.assertIn("annotation_error", rejected["clip/frame_error.png"])
            self.assertIn("clip/frame_lowconf.png", rejected)
            self.assertTrue(
                any(reason.startswith("min_conf_lt_") for reason in rejected["clip/frame_lowconf.png"])
            )

            manifest_paths = {
                item["relative_path"] for item in read_json(Path(summary["manifest_path"]))
            }
            self.assertNotIn("clip/frame_error.png", manifest_paths)
            self.assertNotIn("clip/frame_lowconf.png", manifest_paths)
            self.assertEqual(summary["rejected"], 2)

    def test_all_rejected_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = _make_config(Path(tmp))
            _write_annotations(
                config,
                [_make_record(config, f"clip/empty_{i}.png", det_count=0) for i in range(3)],
            )
            with self.assertRaises(RuntimeError):
                build_dataset(config, "v1")

    def test_chinese_paths_supported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = _make_config(Path(tmp))
            records = [
                _make_record(config, f"吊装场景/帧_{index:04d}.png") for index in range(8)
            ]
            _write_annotations(config, records)
            summary = build_dataset(config, "中文版本-v1", holdout_ratio=0.3)

            manifest_paths = {
                item["relative_path"] for item in read_json(Path(summary["manifest_path"]))
            }
            self.assertIn("吊装场景/帧_0000.png", manifest_paths)
            for item in read_json(Path(summary["manifest_path"])):
                self.assertTrue(Path(item["image_path"]).exists())

            entries = load_holdout_entries(holdout_dir_of(config))
            self.assertTrue(entries)
            result = verify_holdout(entries, holdout_dir_of(config))
            self.assertTrue(result["hash_ok"])


if __name__ == "__main__":
    unittest.main()
