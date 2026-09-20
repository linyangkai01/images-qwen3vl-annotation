"""T04 独立验证（QA 严过关）第 1 部分：holdout 铁律边界 + 无泄漏 + 极端切分 + data.yaml。

与工程师自测 ``tests/test_holdout.py`` / ``tests/test_dataset_train.py`` **独立**：本文件只攻
自测**没有覆盖**的边界，重点是「静默失败」与「边界情况下的错误行为」：

1. **holdout 铁律的更强边界**（换 seed / 换 ratio / **记录变少（含把 holdout 帧从输入里
   删掉、甚至删掉源图）** / 连调 / 输入顺序打乱）——铁律必须压过一切参数；
2. **文件系统级无泄漏** —— 不看 ``manifest.json`` 的 split 字段，直接扫
   ``datasets/<ver>/images|labels/{train,val}/``，按**文件名 + 内容哈希**双重比对
   （自测用的是**同色**合成帧，哈希全等会让「哈希比对」假绿，故本文件每帧颜色不同）；
3. **极端切分** —— ``ratio`` = 0 / 1 / 负 / >1 / NaN / 非法字符串，以及记录数 1/2/3；
4. **``data.yaml`` 正确性** —— ``names`` 下标即 class_id（不可重排）、``nc``、
   以及母项目 ``val==0 → val 指向 train`` 的坑。

已知缺陷（D1/D2/D3/D4）的用例**断言期望行为**（因此当前为红）；其余为通过用例或
「行为刻画 + 绊线」（母项目 ``scripts/curate_video_batch.py`` 的行为，按本项目 T01 既定
约定不改母项目，只刻画并在注释里写明绊线）。

约束：不修改任何生产代码；不联网、不调用真实 VLM、不跑真实训练；不读真实素材；
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
    holdout_count,
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
def _chunk(tag: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + tag
        + data
        + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    )


def _write_png(path: Path, color: tuple[int, int, int], width: int = 8, height: int = 8) -> Path:
    """写一张**真实可解码**的最小 PNG（纯标准库）。``color`` 逐帧不同 → 内容哈希唯一。"""
    row = b"\x00" + bytes(color) * width
    png = (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + _chunk(b"IDAT", zlib.compress(row * height))
        + _chunk(b"IEND", b"")
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png)
    return path


def _make_config(tmp: Path) -> ProjectConfig:
    """在 ``tmp`` 下写一份 mock 吊装配置并加载；``tmp`` 即项目根。"""
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
    color: tuple[int, int, int] = (120, 120, 120),
    confidence: float = 0.9,
    det_count: int = 1,
    review_reasons: list[str] | None = None,
) -> AnnotationRecord:
    """造一条真实可物化的标注记录（PNG + 标签都落盘）。"""
    extracted = config.input_frames_dir / relative_path
    _write_png(extracted, color)
    label_path = config.workspace("review", "labels") / Path(relative_path).with_suffix(".txt")
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    detections = [
        Detection(label="person", bbox=(0.1, 0.1, 0.3, 0.5), confidence=confidence)
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
        review_required=False,
        review_reasons=review_reasons or [],
        prompt_path="",
        raw_response_path="",
        label_path=str(label_path),
        dataset_ready=True,
    )


def _uniq_records(
    config: ProjectConfig, count: int, prefix: str = "clip", start: int = 0
) -> list[AnnotationRecord]:
    """``count`` 条记录，**每帧颜色不同** → 内容哈希两两不等（泄漏比对才有意义）。"""
    return [
        _make_record(
            config,
            f"{prefix}/frame_{index:06d}.png",
            color=((index * 7) % 256, (index * 13) % 256, (index * 29) % 256),
        )
        for index in range(start, start + count)
    ]


def _write_annotations(config: ProjectConfig, records: list[AnnotationRecord]) -> None:
    config.annotations_path.parent.mkdir(parents=True, exist_ok=True)
    with config.annotations_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _files_in(directory: Path) -> dict[str, str]:
    """返回 ``{相对路径: 内容 sha256}``（递归）。"""
    return {
        p.relative_to(directory).as_posix(): _sha256(p)
        for p in directory.rglob("*")
        if p.is_file()
    }


class _TempProject:
    """上下文管理器：一个干净的临时工程（配置 + 已写入的标注）。"""

    def __init__(self, record_count: int = 0, root: Path | None = None) -> None:
        self._outer = tempfile.TemporaryDirectory()
        self.root = Path(self._outer.name)
        self.project = root if root is not None else self.root / "proj"
        self.config = _make_config(self.project)
        self.records: list[AnnotationRecord] = []
        if record_count:
            self.records = _uniq_records(self.config, record_count)
            _write_annotations(self.config, self.records)

    def __enter__(self) -> "_TempProject":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self._outer.cleanup()


# --------------------------------------------------------------------------- #
# 1. holdout 铁律的更强边界
# --------------------------------------------------------------------------- #
class HoldoutIronLawBoundaryTest(unittest.TestCase):
    """铁律「一旦冻结，跨轮逐帧不变」必须压过 seed / ratio / 输入顺序 / 输入增减。"""

    def test_different_seed_does_not_refreeze(self) -> None:
        with _TempProject(24) as env:
            first = build_dataset(env.config, "v1", holdout_ratio=0.25, seed=7)
            # 第二轮换一个完全不同的 seed：铁律应压过 seed。
            second = build_dataset(env.config, "v2", holdout_ratio=0.25, seed=987654321)
            self.assertEqual(first["holdout_hash"], second["holdout_hash"])
            self.assertEqual(first["holdout_count"], second["holdout_count"])
            self.assertTrue(first["holdout_count"] > 0, "首轮必须真的冻结了帧")

    def test_different_ratio_does_not_refreeze(self) -> None:
        with _TempProject(24) as env:
            first = build_dataset(env.config, "v1", holdout_ratio=0.25, seed=7)
            # 第二轮 ratio 从 0.25 拉到 0.9：铁律应压过 ratio（否则 holdout 会膨胀）。
            second = build_dataset(env.config, "v2", holdout_ratio=0.9, seed=7)
            self.assertEqual(first["holdout_hash"], second["holdout_hash"])
            self.assertEqual(first["holdout_count"], second["holdout_count"])

    def test_input_order_shuffled_does_not_refreeze(self) -> None:
        with _TempProject(24) as env:
            first = build_dataset(env.config, "v1", holdout_ratio=0.25, seed=7)
            _write_annotations(env.config, list(reversed(env.records)))
            second = build_dataset(env.config, "v2", holdout_ratio=0.25, seed=7)
            self.assertEqual(first["holdout_hash"], second["holdout_hash"])

    def test_holdout_frames_removed_from_input_keeps_holdout(self) -> None:
        """**最担心的边界**：第一轮的 holdout 帧在第二轮输入里被删掉（如被质量过滤剔除）。

        期望：holdout 集合与哈希完全不变（用冻结副本），不报错、不静默丢帧；
        数据集里 holdout 分片的条目仍等于冻结集合。
        """
        with _TempProject(24) as env:
            first = build_dataset(env.config, "v1", holdout_ratio=0.25, seed=7)
            frozen = {e.relative_path for e in load_holdout_entries(holdout_dir_of(env.config))}
            self.assertTrue(frozen)

            remaining = [
                r for r in env.records if r.candidate.relative_path not in frozen
            ]
            self.assertEqual(len(remaining), 24 - len(frozen))
            _write_annotations(env.config, remaining)

            second = build_dataset(env.config, "v2", holdout_ratio=0.25, seed=7)
            # 铁律：不重新切分、不丢帧、不报错。
            self.assertEqual(first["holdout_hash"], second["holdout_hash"])
            self.assertEqual(second["holdout_count"], len(frozen))
            self.assertEqual(second["accepted"], len(remaining))

            manifest = read_json(Path(second["manifest_path"]))
            holdout_in_manifest = {
                m["relative_path"] for m in manifest if m["split"] == "holdout"
            }
            self.assertEqual(holdout_in_manifest, frozen)
            # 冻结副本仍在且逐字节未变。
            check = verify_holdout(
                load_holdout_entries(holdout_dir_of(env.config)), holdout_dir_of(env.config)
            )
            self.assertTrue(check["hash_ok"], f"冻结副本被破坏：{check}")

    def test_holdout_frames_removed_even_when_source_image_deleted(self) -> None:
        """再狠一层：连 holdout 帧的**源 PNG** 也从磁盘删掉，重建仍须成功且逐字节不变。"""
        with _TempProject(24) as env:
            first = build_dataset(env.config, "v1", holdout_ratio=0.25, seed=7)
            frozen = {
                e.relative_path: e.sha256
                for e in load_holdout_entries(holdout_dir_of(env.config))
            }

            remaining = [
                r for r in env.records if r.candidate.relative_path not in frozen
            ]
            _write_annotations(env.config, remaining)
            for rel in frozen:
                source = env.config.input_frames_dir / rel
                if source.exists():
                    source.unlink()

            second = build_dataset(env.config, "v2", holdout_ratio=0.25, seed=7)
            after = {
                e.relative_path: e.sha256
                for e in load_holdout_entries(holdout_dir_of(env.config))
            }
            self.assertEqual(frozen, after)
            self.assertEqual(first["holdout_hash"], second["holdout_hash"])

    def test_freeze_holdout_repeated_with_conflicting_params_is_stable(self) -> None:
        """直接连调 :func:`freeze_holdout` 三次，参数各不相同 → 返回的条目集合必须一致。"""
        with _TempProject(12) as env:
            out_dir = holdout_dir_of(env.config)
            first = freeze_holdout(env.records, 0.25, 7, out_dir, version="a")
            second = freeze_holdout(env.records, 0.9, 424242, out_dir, version="b")
            third = freeze_holdout(env.records, 0.5, 0, out_dir, version="c")
            self.assertTrue(first, "首轮应冻结至少 1 帧")
            self.assertEqual(len(first), len(second))
            self.assertEqual(len(first), len(third))
            self.assertEqual(
                {e.relative_path for e in first},
                {e.relative_path for e in second},
            )
            self.assertEqual(
                {e.relative_path for e in first},
                {e.relative_path for e in third},
            )
            self.assertEqual(holdout_hash_of(first), holdout_hash_of(third))
            # index.json 里登记的 ratio/seed 也不该被后续调用覆盖（可追溯首轮来源）。
            payload = read_json(out_dir / "index.json")
            self.assertEqual(payload["ratio"], 0.25)
            self.assertEqual(payload["seed"], 7)

    def test_verify_holdout_returns_structured_dicts(self) -> None:
        """``tampered`` / ``missing`` 的元素必须是结构化 dict（含键），不是裸字符串。"""
        with _TempProject(12) as env:
            out_dir = holdout_dir_of(env.config)
            entries = freeze_holdout(env.records, 0.25, 7, out_dir, version="v")
            self.assertTrue(entries)

            victim = out_dir / "frames" / entries[0].relative_path
            original = victim.read_bytes()
            victim.write_bytes(original + b"x")
            tampered_report = verify_holdout(entries, out_dir)
            self.assertFalse(tampered_report["hash_ok"])
            self.assertTrue(tampered_report["tampered"])
            item = tampered_report["tampered"][0]
            self.assertIsInstance(item, dict)
            for key in ("relative_path", "expected", "actual"):
                self.assertIn(key, item)

            victim.write_bytes(original)
            victim.unlink()
            missing_report = verify_holdout(entries, out_dir)
            self.assertFalse(missing_report["hash_ok"])
            self.assertTrue(missing_report["missing"])
            self.assertIsInstance(missing_report["missing"][0], dict)
            self.assertIn("reason", missing_report["missing"][0])


# --------------------------------------------------------------------------- #
# 2. 文件系统级无泄漏（不看 manifest，直接扫目录）
# --------------------------------------------------------------------------- #
class NoLeakFilesystemTest(unittest.TestCase):
    """按**文件名 + 内容哈希**双重比对真实落盘的文件。

    每帧颜色不同，因此「内容哈希比对」是真判据（工程师自测用同色帧，哈希全等会让
    该比对恒真）。
    """

    def test_no_holdout_frame_or_label_leaks_into_train_val(self) -> None:
        with _TempProject(24) as env:
            summary = build_dataset(env.config, "v1", holdout_ratio=0.25, seed=7)
            entries = load_holdout_entries(holdout_dir_of(env.config))
            holdout_names = {e.relative_path for e in entries}
            holdout_hashes = {e.sha256 for e in entries}
            self.assertTrue(holdout_names)

            dataset_dir = env.config.datasets_dir / "v1"
            for split in ("train", "val"):
                images = _files_in(dataset_dir / "images" / split)
                labels = _files_in(dataset_dir / "labels" / split)
                self.assertTrue(images, f"images/{split} 不应为空")
                self.assertEqual(
                    set(images) & holdout_names, set(), f"images/{split} 泄漏了 holdout 文件名"
                )
                self.assertEqual(
                    set(images.values()) & holdout_hashes, set(),
                    f"images/{split} 泄漏了 holdout 帧内容（同 sha256）",
                )
                leaked_labels = set(labels) & {
                    Path(name).with_suffix(".txt").as_posix() for name in holdout_names
                }
                self.assertEqual(leaked_labels, set(), f"labels/{split} 泄漏了 holdout 标签")

            # 反向：holdout 分片必须**逐字节**等于冻结副本。
            copied = _files_in(dataset_dir / "images" / "holdout")
            self.assertEqual(set(copied), holdout_names)
            for name, digest in copied.items():
                self.assertEqual(
                    digest,
                    next(e.sha256 for e in entries if e.relative_path == name),
                    f"holdout 副本与登记 sha256 不一致：{name}",
                )
            # 三个分片两两不重叠（按文件名）。
            train = set(_files_in(dataset_dir / "images" / "train"))
            val = set(_files_in(dataset_dir / "images" / "val"))
            hold = set(copied)
            self.assertEqual(train & val, set())
            self.assertEqual(train & hold, set())
            self.assertEqual(val & hold, set())
            self.assertEqual(
                len(train) + len(val) + len(hold), summary["accepted"]
            )


# --------------------------------------------------------------------------- #
# 3. 极端切分
# --------------------------------------------------------------------------- #
class HoldoutCountBoundaryTest(unittest.TestCase):
    """``holdout_count`` 的边界表（纯函数，便于一次看全）。"""

    #: 实测基线（``int(round(total * ratio))``，Python 的 banker's rounding：
    #: ``round(2.5) == 2``）。这里写死数字是为了**绊线**：哪天切分算法变了会立刻报警。
    TABLE: dict[tuple[int, float], int] = {
        (0, 0.25): 0,
        (1, 0.25): 0,   # 只有 1 帧 → 不能全进 holdout（否则没训练数据）
        (2, 0.25): 1,
        (3, 0.25): 1,
        (4, 0.25): 1,
        (8, 0.25): 2,
        (10, 0.0): 0,   # ratio=0 → 无 holdout
        (10, 0.25): 2,  # round(2.5) = 2
        (10, 0.5): 5,
        (24, 0.25): 6,
        (24, 1.0): 23,  # 绝不吃光最后一帧
    }

    def test_legal_ratios_produce_sane_counts(self) -> None:
        for (total, ratio), expected in self.TABLE.items():
            with self.subTest(total=total, ratio=ratio):
                self.assertEqual(holdout_count(total, ratio), expected)
                # 不变量：0 <= holdout <= total-1（至少留 1 帧给 train）。
                self.assertLessEqual(holdout_count(total, ratio), max(0, total - 1))
                self.assertGreaterEqual(holdout_count(total, ratio), 0)

    def test_ratio_one_never_eats_every_frame(self) -> None:
        for total in (1, 2, 3, 10, 50):
            with self.subTest(total=total):
                self.assertLess(holdout_count(total, 1.0), total)


class ExtremeRatioFailLoudTest(unittest.TestCase):
    """**已知缺陷 D1（P1）**：``holdout_ratio`` 越界被**静默钳制**，不 fail-loud。

    实测（12 帧）：
    - ``--holdout-ratio -1`` → 退出码 0，holdout=0（**静默禁用固定评估集**）；
    - ``--holdout-ratio 25``（用户想写 25%）→ 静默钳到 1.0 → 11/12 帧进 holdout，
      train 只剩 1 帧、val=0 → ``data.yaml`` 的 ``val`` 指向 ``images/train``；
    - ``float('nan')`` → 静默按 1.0 处理。

    这三种都是「一行配置笔误 → 数据集报废且无任何提示」，与本项目「绝不静默」的
    既定规矩冲突。期望：``ratio`` 不在 ``[0, 1]``（或为 NaN）时**显式抛 ValueError**。
    """

    def _assert_rejected(self, ratio: object) -> None:
        with _TempProject(12) as env:
            with self.assertRaises(ValueError) as ctx:
                build_dataset(
                    env.config, "v1", holdout_ratio=ratio, seed=7
                )  # type: ignore[arg-type]
            self.assertTrue(
                str(ctx.exception), "异常消息不得为空——必须说清哪个参数不合法"
            )

    def test_negative_ratio_is_rejected(self) -> None:
        self._assert_rejected(-0.5)

    def test_ratio_greater_than_one_is_rejected(self) -> None:
        self._assert_rejected(25.0)

    def test_nan_ratio_is_rejected(self) -> None:
        self._assert_rejected(float("nan"))

    def test_non_numeric_ratio_raises_value_error(self) -> None:
        """非法字符串**确实**报错（fail-loud ✅），但消息是 ``float()`` 的原始消息。

        本用例只要求「抛 ValueError」，不苛求消息（已知缺陷 D1 的附赠观察：消息为
        ``could not convert string to float: 'abc'``，对使用者不够友好）。
        """
        with _TempProject(12) as env:
            with self.assertRaises((ValueError, TypeError)):
                build_dataset(
                    env.config, "v1", holdout_ratio="abc", seed=7
                )  # type: ignore[arg-type]


class EmptyHoldoutRefreezeTest(unittest.TestCase):
    """**已知缺陷 D4（P2）**：「已冻结」的判据是 ``entries`` 非空，而非 ``index.json`` 存在。

    实测：``ratio=0`` 首轮写入 ``index.json`` 但 ``entries: []``；第二轮改用 ``ratio=0.25``
    时 ``freeze_holdout`` 认为「没冻结过」而**重新切分**，凭空多出 holdout ——
    跨轮可比性出现缺口（这正是需求 7.4 要消灭的问题）。

    期望（两种可接受修复之一）：要么 ``index.json`` 一旦存在即视为已冻结（哪怕为空），
    第二轮保持空；要么第二轮显式报错。唯一不可接受的是**静默重新切分**。
    """

    def test_empty_freeze_is_not_silently_re_split(self) -> None:
        with _TempProject(10) as env:
            first = build_dataset(env.config, "v1", holdout_ratio=0.0, seed=7)
            self.assertEqual(first["holdout_count"], 0)
            self.assertTrue((holdout_dir_of(env.config) / "index.json").exists())

            try:
                second = build_dataset(env.config, "v2", holdout_ratio=0.25, seed=7)
            except (ValueError, RuntimeError):
                return  # 显式 fail-loud 也是可接受的修复：本用例只禁止「静默重新切分」
            self.assertEqual(
                second["holdout_count"], 0,
                "首轮已冻结（空集合），第二轮不得静默重新切分出 holdout",
            )


class TinyRecordCountTest(unittest.TestCase):
    """记录数 1 / 2 / 3（母项目 ``split_counts`` 对 total<=0 / ==1 / ==2 有特殊分支）。"""

    def test_one_two_three_records_do_not_crash(self) -> None:
        observed: dict[int, dict[str, int]] = {}
        for total in (1, 2, 3):
            with _TempProject(total) as env:
                summary = build_dataset(env.config, "v1", holdout_ratio=0.25, seed=7)
                observed[total] = dict(summary["split_counts"])
                self.assertTrue(Path(summary["data_yaml_path"]).exists())
        # 不变量：train 至少 1 帧（只要有合格帧），且 holdout 不吃光全部。
        for total, counts in observed.items():
            with self.subTest(total=total):
                self.assertGreaterEqual(counts["train"], 1)
                self.assertLessEqual(counts["holdout"], max(0, total - 1))
                self.assertEqual(
                    counts["train"] + counts["val"] + counts["holdout"], total
                )


# --------------------------------------------------------------------------- #
# 4. data.yaml 正确性
# --------------------------------------------------------------------------- #
class DataYamlCorrectnessTest(unittest.TestCase):
    def _yaml_of(self, summary: dict) -> str:
        return Path(summary["data_yaml_path"]).read_text(encoding="utf-8")

    def test_names_index_is_class_id_and_matches_config_labels(self) -> None:
        """``names`` 的下标就是 class_id，必须与 ``config.labels`` **同序**，不可重排。"""
        with _TempProject(20) as env:
            summary = build_dataset(env.config, "v1", holdout_ratio=0.25, seed=7)
            text = self._yaml_of(summary)
            lines = text.splitlines()
            self.assertIn("names:", lines)
            start = lines.index("names:")
            mapping: dict[int, str] = {}
            for line in lines[start + 1:]:
                if not line.strip() or not line.startswith("  "):
                    break
                key, _, value = line.strip().partition(":")
                mapping[int(key)] = value.strip()
            self.assertEqual([mapping[i] for i in sorted(mapping)], list(env.config.labels))
            self.assertEqual(len(mapping), len(env.config.labels))
            self.assertEqual(len(mapping), 12, "吊装场景应为 12 类")

    def test_data_yaml_lacks_nc_field(self) -> None:
        """行为刻画 + 绊线（母项目 ``scripts/curate_video_batch.py:36`` 的行为）。

        当前 ``data.yaml`` **没有** ``nc:`` 字段，靠 Ultralytics 从 ``names`` 推断。
        这是母项目脚本的既有行为（本批不改母项目），故此处**刻画真实语义**并留绊线：
        若哪天 ``data.yaml`` 出现了 ``nc``，说明母项目已改，请同步更新本用例与文档。
        """
        with _TempProject(20) as env:
            summary = build_dataset(env.config, "v1", holdout_ratio=0.25, seed=7)
            text = self._yaml_of(summary)
            self.assertNotIn("nc:", text, "绊线：母项目已补 nc 字段，请更新本用例")
            self.assertIn("test: images/val", text, "绊线：test 回落逻辑已变，请更新本用例")

    def test_val_zero_must_not_point_at_train(self) -> None:
        """**已知缺陷 D2（P1）**：``val`` 计数为 0 时，``val`` 被指向 ``images/train``。

        母项目 ``scripts/curate_video_batch.py:39``：
        ``val_path = "images/val" if split_count_map.get("val", 0) > 0 else train_path``。
        实测触发路径：
        - 记录数 <= 2（如 total=2, ratio=0.25 → train=1, val=0, holdout=1）；
        - ``ratio >= 1``（如 total=10, ratio=1 → train=1, val=0, holdout=9）。

        后果：Ultralytics 训练期的验证集 == 训练集 → mAP 虚高、早停与最优权重选择失真；
        且 T05 若沿用 ``data.yaml`` 的 val 会得到「在训练集上评估」的结论。

        期望：``split_counts["val"] == 0`` 时，``data.yaml`` 的 ``val`` **不得**指向
        ``images/train``（应显式报错、或保持指向空的 ``images/val`` 让下游 fail-loud）。
        """
        cases = [("两条记录", 2, 0.25), ("ratio=1", 10, 1.0)]
        for label, total, ratio in cases:
            with self.subTest(case=label):
                with _TempProject(total) as env:
                    summary = build_dataset(env.config, "v1", holdout_ratio=ratio, seed=7)
                    if summary["split_counts"]["val"] == 0:
                        self.assertNotIn(
                            "val: images/train", self._yaml_of(summary),
                            "val 计数为 0 时 data.yaml 把 val 指向了训练集",
                        )

    def test_holdout_dir_is_not_declared_in_data_yaml(self) -> None:
        with _TempProject(20) as env:
            summary = build_dataset(env.config, "v1", holdout_ratio=0.25, seed=7)
            self.assertNotIn("holdout", self._yaml_of(summary))


if __name__ == "__main__":
    unittest.main()
