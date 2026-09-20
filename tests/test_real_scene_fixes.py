"""真实闭环暴露的「mock 默认值不适配真实场景」缺陷的回归单测。

真实闭环（抽帧→初检→VLM 标注→数据集→holdout→训练→评估）在服务器上跑通后暴露了
三个**合成测试发现不了**的静默/误配缺陷，本文件逐一回归：

1. **P1 annotate 静默成功** —— ``workspace/extracted/candidates.jsonl`` 缺失或为空时，
   ``annotate`` 旧行为是「退出码 0、产物全空」，流程绿灯而实际什么都没做。现必须
   **fail-loud**（``ValueError``；CLI 返回退出码 ``4``）。
2. **P1 CLI 无法传 ``quality_thresholds``** —— ``max_added=0`` 默认值面向 mock（假设
   「VLM 只纠偏、不新增」），真实素材下 VLM 大幅新增目标 → 全部帧被剔除。现支持从
   **配置文件 ``pipeline.quality_thresholds``** 读取，优先级
   **显式参数 > 配置文件 > 默认值**；默认值本身**不改**（与母项目脚本对齐）。
3. **P2 sidecar 零命中静默跳过** —— 真实场景误配 ``yolo.detector=sidecar`` 且一帧
   sidecar 都没读到时，全部帧被判空 → ``selected_candidates: 0`` 但退出码 0。现必须
   给出**可见告警**（``summary.warnings`` + CLI stderr），但**不报错**（sidecar 合法）。

约束：不联网、不调用真实 VLM / ultralytics、不读真实素材（只用合成 PNG）；
不修改 ``traffic_cv_closed_loop/``；临时文件用 ``tempfile``。
"""

from __future__ import annotations

import contextlib
import io
import json
import struct
import tempfile
import unittest
import zlib
from pathlib import Path

from traffic_cv_closed_loop.config import ProjectConfig
from traffic_cv_closed_loop.io_utils import read_json
from traffic_cv_closed_loop.models import AnnotationRecord, CandidateFrame, Detection

from industrial_hazard_coldstart import cli
from industrial_hazard_coldstart.dataset import (
    DEFAULT_QUALITY_THRESHOLDS,
    QUALITY_THRESHOLD_KEYS,
    curate_records,
    load_config_quality_thresholds,
    resolve_quality_thresholds,
)
from industrial_hazard_coldstart.orchestrator import (
    _count_sidecar_files,
    _extract_warnings,
    run_annotate,
    run_cycle,
)

ROOT = Path(__file__).resolve().parents[1]
HOIST_CONFIG = ROOT / "own" / "hoist_project" / "config" / "project.hoist.json"
HOIST_PROMPT = ROOT / "own" / "hoist_project" / "assets" / "prompts" / "hoist_zh.md"
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


def _make_config(
    tmp: Path,
    *,
    detector: str = "sidecar",
    include_empty_frames: bool = True,
    quality_thresholds: dict | None = None,
) -> ProjectConfig:
    """在临时目录写一份 mock 吊装配置并加载（可注入配置文件级质量阈值）。"""
    raw = json.loads(HOIST_CONFIG.read_text(encoding="utf-8"))
    raw["qwen"]["provider"] = "mock"
    raw["qwen"]["prompt_template"] = str(HOIST_PROMPT)
    raw["knowledge"]["kb_path"] = str(KB_PATH)
    raw["yolo"]["detector"] = detector
    raw["pipeline"]["include_empty_frames"] = include_empty_frames
    if quality_thresholds is not None:
        raw["pipeline"]["quality_thresholds"] = quality_thresholds
    config_dir = tmp / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "project.json"
    config_path.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
    return ProjectConfig.load(config_path)


def _make_record(
    config: ProjectConfig,
    relative_path: str,
    *,
    diff: dict | None = None,
    confidence: float = 0.9,
) -> AnnotationRecord:
    """造一条标注记录（真实 PNG + 标签），``diff`` 驱动质量过滤。"""
    extracted = config.input_frames_dir / relative_path
    _write_png(extracted)
    label_path = config.workspace("review", "labels") / Path(relative_path).with_suffix(".txt")
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    detections = [Detection(label="person", bbox=(0.1, 0.1, 0.3, 0.5), confidence=confidence)]
    return AnnotationRecord(
        candidate=CandidateFrame(
            source_id="clip",
            relative_path=relative_path,
            original_path=str(extracted),
            extracted_path=str(extracted),
            detections=detections,
            scene="hoist",
        ),
        qwen_detections=detections,
        diff=dict(diff or {"added": 0, "removed": 0, "adjusted": 0}),
        review_required=False,
        review_reasons=[],
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


# --------------------------------------------------------------------------- #
# P1-a：annotate 缺 / 空 candidates 必须 fail-loud
# --------------------------------------------------------------------------- #
class AnnotateFailLoudTest(unittest.TestCase):
    """annotate 在「上游未产出候选帧」时必须报错，绝不空跑成功。"""

    def test_missing_candidates_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = _make_config(Path(tmp))
            self.assertFalse(config.candidates_path.exists())
            with self.assertRaises(ValueError) as ctx:
                run_annotate(config)
            self.assertIn("候选帧", str(ctx.exception))

    def test_empty_candidates_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = _make_config(Path(tmp))
            config.candidates_path.parent.mkdir(parents=True, exist_ok=True)
            config.candidates_path.write_text("", encoding="utf-8")
            with self.assertRaises(ValueError) as ctx:
                run_annotate(config)
            self.assertIn("为空", str(ctx.exception))

    def test_guard_precedes_side_effects(self) -> None:
        """fail-loud 必须发生在**任何产物写入之前**（不留半成品残料）。"""
        with tempfile.TemporaryDirectory() as tmp:
            config = _make_config(Path(tmp))
            with self.assertRaises(ValueError):
                run_annotate(config)
            self.assertFalse(config.annotations_path.exists())
            self.assertFalse(config.hazard_report_path.exists())

    def test_cli_annotate_missing_candidates_returns_4(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = _make_config(Path(tmp))
            stderr = io.StringIO()
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(stderr):
                code = cli.main(["annotate", "--config", str(config.config_path)])
            self.assertEqual(code, cli.EXIT_STAGE_FAILED)
            self.assertIn("执行失败", stderr.getvalue())


# --------------------------------------------------------------------------- #
# P1-b：配置文件级 quality_thresholds 生效且优先级正确
# --------------------------------------------------------------------------- #
class QualityThresholdResolutionTest(unittest.TestCase):
    """``resolve_quality_thresholds`` 的优先级：显式参数 > 配置文件 > 默认值。"""

    def test_defaults_when_no_config_no_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = _make_config(Path(tmp))
            self.assertEqual(load_config_quality_thresholds(config), {})
            resolved = resolve_quality_thresholds(config, None)
            self.assertEqual(resolved, dict(DEFAULT_QUALITY_THRESHOLDS))

    def test_config_file_overrides_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = _make_config(Path(tmp), quality_thresholds={"max_added": 20})
            self.assertEqual(load_config_quality_thresholds(config), {"max_added": 20})
            resolved = resolve_quality_thresholds(config, None)
            self.assertEqual(resolved["max_added"], 20)
            # 未覆盖的键仍取默认值。
            self.assertEqual(resolved["max_removed"], DEFAULT_QUALITY_THRESHOLDS["max_removed"])

    def test_explicit_param_overrides_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = _make_config(Path(tmp), quality_thresholds={"max_added": 20})
            resolved = resolve_quality_thresholds(config, {"max_added": 5})
            self.assertEqual(resolved["max_added"], 5)

    def test_none_value_falls_back(self) -> None:
        """显式传 ``None`` 表示「该键不覆盖」，应回落到配置文件值。"""
        with tempfile.TemporaryDirectory() as tmp:
            config = _make_config(Path(tmp), quality_thresholds={"max_added": 20})
            resolved = resolve_quality_thresholds(config, {"max_added": None})
            self.assertEqual(resolved["max_added"], 20)

    def test_unknown_key_in_config_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = _make_config(Path(tmp), quality_thresholds={"min_conf": 0.5})
            with self.assertRaises(ValueError) as ctx:
                resolve_quality_thresholds(config, None)
            self.assertIn("未知的质量阈值键", str(ctx.exception))

    def test_unknown_key_in_override_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = _make_config(Path(tmp))
            with self.assertRaises(ValueError):
                resolve_quality_thresholds(config, {"bogus": 1})

    def test_default_max_added_unchanged(self) -> None:
        """默认值刻意不改（与母项目脚本对齐）——回归护栏。"""
        self.assertEqual(DEFAULT_QUALITY_THRESHOLDS["max_added"], 0)
        self.assertEqual(QUALITY_THRESHOLD_KEYS, frozenset(DEFAULT_QUALITY_THRESHOLDS))

    def test_config_thresholds_make_real_scene_pass(self) -> None:
        """真实场景复现：VLM 大幅新增目标（added>0）时默认值剔除全部帧。

        默认 ``max_added=0`` → 全部帧被过滤 → ``RuntimeError``；
        配置文件放宽 ``max_added`` → 全部帧被接受。
        """
        with tempfile.TemporaryDirectory() as tmp:
            config = _make_config(Path(tmp))
            records = [
                _make_record(config, f"clip/frame_{i:06d}.png", diff={"added": 5, "removed": 0, "adjusted": 0})
                for i in range(6)
            ]
            _write_annotations(config, records)

            # 默认阈值：added=5 > max_added=0 → 全部剔除。
            with self.assertRaises(RuntimeError):
                curate_records(config)

            # 配置文件放宽后：显式传入的阈值同理生效（走同一解析路径）。
            _, accepted, rejected, _ = curate_records(config, {"max_added": 20})
            self.assertEqual(len(accepted), 6)
            self.assertEqual(len(rejected), 0)

    def test_config_file_thresholds_used_by_curate_records(self) -> None:
        """不传显式参数时，``curate_records`` 也读**配置文件**里的阈值。"""
        with tempfile.TemporaryDirectory() as tmp:
            config = _make_config(Path(tmp), quality_thresholds={"max_added": 20})
            records = [
                _make_record(config, f"clip/frame_{i:06d}.png", diff={"added": 5, "removed": 0, "adjusted": 0})
                for i in range(4)
            ]
            _write_annotations(config, records)
            _, accepted, rejected, _ = curate_records(config)
            self.assertEqual(len(accepted), 4)
            self.assertEqual(len(rejected), 0)


# --------------------------------------------------------------------------- #
# P2：sidecar 零命中必须给出可见告警（不报错）
# --------------------------------------------------------------------------- #
class SidecarZeroHitWarningTest(unittest.TestCase):
    """``yolo.detector=sidecar`` 且零命中时，告警必须可见但不改变退出码。"""

    def test_count_sidecar_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = _make_config(Path(tmp))
            root = config.input_frames_dir
            (root / "clip").mkdir(parents=True, exist_ok=True)
            _write_png(root / "clip" / "frame_0.png")
            self.assertEqual(_count_sidecar_files(root), 0)
            (root / "clip" / "frame_0.detections.json").write_text("{}", encoding="utf-8")
            self.assertEqual(_count_sidecar_files(root), 1)

    def test_extract_warnings_flags_sidecar_zero_hit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = _make_config(Path(tmp), detector="sidecar")
            warnings = _extract_warnings(config)
            self.assertEqual(len(warnings), 1)
            self.assertIn("sidecar_zero_hit", warnings[0])

    def test_extract_warnings_silent_when_sidecar_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = _make_config(Path(tmp), detector="sidecar")
            root = config.input_frames_dir
            (root / "clip").mkdir(parents=True, exist_ok=True)
            (root / "clip" / "frame_0.detections.json").write_text("{}", encoding="utf-8")
            self.assertEqual(_extract_warnings(config), [])

    def test_extract_warnings_silent_for_ultralytics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = _make_config(Path(tmp), detector="ultralytics")
            self.assertEqual(_extract_warnings(config), [])

    def test_extract_summary_carries_warning(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = _make_config(Path(tmp), detector="sidecar", include_empty_frames=False)
            root = config.input_frames_dir
            (root / "clip").mkdir(parents=True, exist_ok=True)
            _write_png(root / "clip" / "frame_0.png")
            result = run_cycle(config, "extract", auto_extract_video=False)
            self.assertEqual(result["stage"], "extract")
            # 帧存在但无 sidecar → 被判空跳过；告警必须出现在摘要里。
            self.assertEqual(result["selected_candidates"], 0)
            self.assertIn("warnings", result)
            self.assertTrue(any("sidecar_zero_hit" in item for item in result["warnings"]))

    def test_cli_extract_prints_warning_but_exit_zero(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = _make_config(Path(tmp), detector="sidecar", include_empty_frames=False)
            root = config.input_frames_dir
            (root / "clip").mkdir(parents=True, exist_ok=True)
            _write_png(root / "clip" / "frame_0.png")
            stderr = io.StringIO()
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(stderr):
                code = cli.main(["extract", "--config", str(config.config_path)])
            # 告警不改变退出码（sidecar 合法）。
            self.assertEqual(code, cli.EXIT_READY)
            self.assertIn("告警", stderr.getvalue())
            self.assertIn("sidecar_zero_hit", stderr.getvalue())

    def test_no_warning_when_sidecar_hits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = _make_config(Path(tmp), detector="sidecar", include_empty_frames=False)
            root = config.input_frames_dir
            (root / "clip").mkdir(parents=True, exist_ok=True)
            _write_png(root / "clip" / "frame_0.png")
            (root / "clip" / "frame_0.detections.json").write_text(
                json.dumps(
                    {"objects": [{"label": "person", "bbox": [0.1, 0.1, 0.4, 0.5], "confidence": 0.9}]}
                ),
                encoding="utf-8",
            )
            result = run_cycle(config, "extract", auto_extract_video=False)
            self.assertGreaterEqual(result["selected_candidates"], 1)
            self.assertNotIn("warnings", result)


if __name__ == "__main__":
    unittest.main()
