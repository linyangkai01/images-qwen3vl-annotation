"""「只重建隐患报告」入口单测（省下整轮 VLM 标注耗时）。

背景：服务器上一轮全量标注（323 帧 / ~12 小时）是用**旧代码**启动的，产出的
``hazards/report.json`` 不含新增的幻觉过滤标记（``needs_manual_verification`` /
``verification``）。标注四类产物里只有 ``report.json`` 受影响，而标记是在**回填阶段**
加的、**不依赖 VLM 重新调用**——故无需重跑标注，从现有 ``annotations.jsonl`` +
``raw_responses`` 重建即可（:func:`orchestrator.rebuild_hazard_report`）。

覆盖点（对应任务书测试要求）：
1. 造含 ``annotations.jsonl``（检测层 + 非检测层隐患）的临时工程 → 重建 → 断言报告
   **结构与正常 ``annotate`` 一致**、标记正确、``by_layer`` 正确；
2. **断言重建未调用 VLM**（``build_qwen_annotator`` / ``run_annotation`` 零调用）；
3. **幂等**：连跑两次结果相同；
4. ``annotations.jsonl`` 缺失 / 为空 → **fail-loud**（与 ``annotate`` 一致护栏）。

约束：不联网、不调用真实 VLM、不读真实素材；只用合成 PNG + ``tempfile``；
不修改 ``traffic_cv_closed_loop/``。
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
from unittest import mock

from traffic_cv_closed_loop.config import ProjectConfig
from traffic_cv_closed_loop.io_utils import read_json
from traffic_cv_closed_loop.models import AnnotationRecord, CandidateFrame

from industrial_hazard_coldstart import cli
from industrial_hazard_coldstart.cli import IMPLEMENTED_COMMANDS
from industrial_hazard_coldstart.orchestrator import (
    rebuild_hazard_report,
    run_cycle,
)

ROOT = Path(__file__).resolve().parents[1]
HOIST_CONFIG = ROOT / "own" / "hoist_project" / "config" / "project.hoist.json"
HOIST_PROMPT = ROOT / "own" / "hoist_project" / "assets" / "prompts" / "hoist_zh.md"
KB_PATH = ROOT / "hazard-detection" / "knowledge_base.json"

#: 每帧的隐患（id + evidence）：含检测层（D-5-1/D-5-2）与非检测层（D-1-6 OCR / D-3-6 时序）。
FRAME_HAZARDS: dict[str, list[dict[str, str]]] = {
    "clip/frame_000001.png": [
        {"id": "D-5-1", "evidence": "人员头部未见安全帽"},
        {"id": "D-1-6", "evidence": "吊具铭牌未见承重标识"},
    ],
    "clip/frame_000002.png": [
        {"id": "D-5-2", "evidence": "人员未穿反光背心"},
        {"id": "D-3-6", "evidence": "吊物长时间悬停"},
    ],
}


# --------------------------------------------------------------------------- #
# 合成素材工具（绝不触碰真实素材）
# --------------------------------------------------------------------------- #
def _write_png(path: Path, width: int = 8, height: int = 8, color: tuple[int, int, int] = (120, 120, 120)) -> Path:
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
    raw = json.loads(HOIST_CONFIG.read_text(encoding="utf-8"))
    raw["qwen"]["provider"] = "mock"
    raw["qwen"]["prompt_template"] = str(HOIST_PROMPT)
    raw["yolo"]["detector"] = "sidecar"
    raw["pipeline"]["include_empty_frames"] = True
    raw["knowledge"]["kb_path"] = str(KB_PATH)
    config_dir = tmp / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "project.json"
    config_path.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
    return ProjectConfig.load(config_path)


def _write_annotations(config: ProjectConfig, records: list[AnnotationRecord]) -> None:
    config.annotations_path.parent.mkdir(parents=True, exist_ok=True)
    with config.annotations_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")


def _seed_annotations_only(config: ProjectConfig, frame_hazards: dict[str, list[dict[str, str]]]) -> None:
    """模拟「真实跑批已完成」的产物：annotations.jsonl + raw_responses（真实形态）。

    真实形态的原始响应 = ``{"parsed": {"objects": [...], "hazard_report": {...}}}``，
    与母项目 ``OpenAICompatibleQwenAnnotator.annotate`` 写下的结构一致。
    **不写 candidates.jsonl**——重建不该依赖它。
    """
    records: list[AnnotationRecord] = []
    for relative_path, hazards in frame_hazards.items():
        original = config.input_frames_dir.parent / "media" / relative_path
        extracted = config.input_frames_dir / relative_path
        _write_png(original)
        _write_png(extracted)
        raw_path = config.review_dir / "raw_responses" / Path(relative_path).with_suffix(".json")
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_text(
            json.dumps(
                {
                    "provider": "qwen-api",
                    "request": {"preview": "..."},
                    "response": {"raw": "..."},
                    "parsed": {
                        "objects": [],
                        "hazard_report": {"hazards": hazards, "overall_risk": "high"},
                    },
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        records.append(
            AnnotationRecord(
                candidate=CandidateFrame(
                    source_id="clip",
                    relative_path=relative_path,
                    original_path=str(original),
                    extracted_path=str(extracted),
                    detections=[],
                    scene="hoist",
                ),
                qwen_detections=[],
                diff={"added": 0, "removed": 0, "adjusted": 0},
                review_required=False,
                review_reasons=[],
                prompt_path="",
                raw_response_path=str(raw_path),
                label_path="",
                dataset_ready=True,
            )
        )
    _write_annotations(config, records)


def _seed_full_annotate(config: ProjectConfig, frame_hazards: dict[str, list[dict[str, str]]]) -> None:
    """走**正常 mock annotate**（candidates + 侧车 + run_annotation）作为对照基准。"""
    candidates = []
    for relative_path, hazards in frame_hazards.items():
        original = config.input_frames_dir.parent / "media" / relative_path
        extracted = config.input_frames_dir / relative_path
        _write_png(original)
        _write_png(extracted)
        (original.with_suffix(".detections.json")).write_text(
            json.dumps({"objects": []}, ensure_ascii=False), encoding="utf-8"
        )
        (original.with_suffix(".qwen.json")).write_text(
            json.dumps({"objects": []}, ensure_ascii=False), encoding="utf-8"
        )
        (original.with_suffix(".hazard.json")).write_text(
            json.dumps({"hazards": hazards, "overall_risk": "high"}, ensure_ascii=False),
            encoding="utf-8",
        )
        candidates.append(
            {
                "source_id": "clip",
                "relative_path": relative_path,
                "original_path": str(original),
                "extracted_path": str(extracted),
                "detections": [],
                "scene": "hoist",
                "metadata": {},
            }
        )
    config.candidates_path.parent.mkdir(parents=True, exist_ok=True)
    with config.candidates_path.open("w", encoding="utf-8", newline="\n") as handle:
        for item in candidates:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    run_cycle(config, "annotate")


# --------------------------------------------------------------------------- #
# 1. 结构与内容一致 + 标记正确
# --------------------------------------------------------------------------- #
class RebuildStructureTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_structure_matches_normal_annotate(self) -> None:
        # 对照：正常 mock annotate。
        annotate_dir = self.tmp / "annotate"
        config_a = _make_config(annotate_dir)
        _seed_full_annotate(config_a, FRAME_HAZARDS)
        report_a = read_json(config_a.hazard_report_path)

        # 重建：只喂 annotations.jsonl + raw_responses。
        rebuild_dir = self.tmp / "rebuild"
        config_b = _make_config(rebuild_dir)
        _seed_annotations_only(config_b, FRAME_HAZARDS)
        rebuild_hazard_report(config_b)
        report_b = read_json(config_b.hazard_report_path)

        # 顶层键集完全一致。
        self.assertEqual(set(report_a), set(report_b))
        self.assertEqual(set(report_a["counts"]), set(report_b["counts"]))
        self.assertEqual(set(report_a["verification"]), set(report_b["verification"]))
        # hazards 逐条一致（含标记）。
        self.assertEqual(report_a["hazards"], report_b["hazards"])
        self.assertEqual(report_a["counts"], report_b["counts"])
        self.assertEqual(report_a["verification"], report_b["verification"])

    def test_markings_and_by_layer_correct(self) -> None:
        config = _make_config(self.tmp)
        _seed_annotations_only(config, FRAME_HAZARDS)
        rebuild_hazard_report(config)
        report = read_json(config.hazard_report_path)

        by_id = {item["hazard_id"]: item for item in report["hazards"]}
        self.assertEqual(set(by_id), {"D-5-1", "D-5-2", "D-1-6", "D-3-6"})

        # 检测层无标记。
        self.assertNotIn("needs_manual_verification", by_id["D-5-1"])
        self.assertNotIn("needs_manual_verification", by_id["D-5-2"])
        # 非检测层带标记 + 说明正确。
        self.assertTrue(by_id["D-1-6"]["needs_manual_verification"])
        self.assertIn("OCR 文字识别", by_id["D-1-6"]["capability_note"])
        self.assertTrue(by_id["D-3-6"]["needs_manual_verification"])
        self.assertIn("时序分析", by_id["D-3-6"]["capability_note"])

        # 汇总计数 + by_layer。
        self.assertEqual(report["counts"]["hazards_needing_verification"], 2)
        self.assertEqual(report["counts"]["hazards_resolved"], 4)
        self.assertEqual(report["verification"]["by_layer"], {"ocr": 1, "temporal": 1})
        # KB 回填不受影响。
        self.assertTrue(by_id["D-5-1"]["norm_reference"])

    def test_summary_declares_rebuild_mode(self) -> None:
        config = _make_config(self.tmp)
        _seed_annotations_only(config, FRAME_HAZARDS)
        summary = rebuild_hazard_report(config)
        self.assertEqual(summary["stage"], "rebuild-hazards")
        self.assertEqual(summary["mode"], "rebuild")
        self.assertIn("未调用 VLM", summary["note"])
        self.assertEqual(summary["hazards_needing_verification"], 2)
        self.assertTrue((config.status_dir / "rebuild-hazards.json").exists())


# --------------------------------------------------------------------------- #
# 2. 不调用 VLM
# --------------------------------------------------------------------------- #
class RebuildNoVlmTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_rebuild_never_builds_annotator_or_annotates(self) -> None:
        config = _make_config(self.tmp)
        _seed_annotations_only(config, FRAME_HAZARDS)
        with mock.patch(
            "industrial_hazard_coldstart.orchestrator.build_qwen_annotator"
        ) as spy_build, mock.patch(
            "industrial_hazard_coldstart.orchestrator.run_annotation"
        ) as spy_annotate:
            rebuild_hazard_report(config)
            spy_build.assert_not_called()
            spy_annotate.assert_not_called()

    def test_rebuild_without_candidates_jsonl(self) -> None:
        """重建不依赖 candidates.jsonl——删掉它仍应成功。"""
        config = _make_config(self.tmp)
        _seed_annotations_only(config, FRAME_HAZARDS)
        self.assertFalse(config.candidates_path.exists())
        rebuild_hazard_report(config)
        self.assertTrue(config.hazard_report_path.exists())


# --------------------------------------------------------------------------- #
# 3. 幂等
# --------------------------------------------------------------------------- #
class RebuildIdempotencyTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_double_run_is_byte_identical(self) -> None:
        config = _make_config(self.tmp)
        _seed_annotations_only(config, FRAME_HAZARDS)
        rebuild_hazard_report(config)
        first = config.hazard_report_path.read_text(encoding="utf-8")
        rebuild_hazard_report(config)
        second = config.hazard_report_path.read_text(encoding="utf-8")
        self.assertEqual(first, second)


# --------------------------------------------------------------------------- #
# 4. fail-loud 护栏
# --------------------------------------------------------------------------- #
class RebuildFailLoudTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_missing_annotations_raises(self) -> None:
        config = _make_config(self.tmp)
        self.assertFalse(config.annotations_path.exists())
        with self.assertRaises(ValueError) as ctx:
            rebuild_hazard_report(config)
        self.assertIn("标注记录", str(ctx.exception))

    def test_empty_annotations_raises(self) -> None:
        config = _make_config(self.tmp)
        config.annotations_path.parent.mkdir(parents=True, exist_ok=True)
        config.annotations_path.write_text("", encoding="utf-8")
        with self.assertRaises(ValueError) as ctx:
            rebuild_hazard_report(config)
        self.assertIn("为空", str(ctx.exception))


# --------------------------------------------------------------------------- #
# 5. mock 侧车回退（无 raw_response 时）
# --------------------------------------------------------------------------- #
class RebuildSidecarFallbackTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_falls_back_to_hazard_sidecar(self) -> None:
        config = _make_config(self.tmp)
        # 造 annotations.jsonl，但 raw_response_path 指向不存在的文件；
        # 隐患只在帧旁 *.hazard.json 侧车里（mock 约定）。
        records = []
        for relative_path, hazards in FRAME_HAZARDS.items():
            original = config.input_frames_dir.parent / "media" / relative_path
            extracted = config.input_frames_dir / relative_path
            _write_png(original)
            _write_png(extracted)
            (original.with_suffix(".hazard.json")).write_text(
                json.dumps({"hazards": hazards, "overall_risk": "high"}, ensure_ascii=False),
                encoding="utf-8",
            )
            records.append(
                AnnotationRecord(
                    candidate=CandidateFrame(
                        source_id="clip",
                        relative_path=relative_path,
                        original_path=str(original),
                        extracted_path=str(extracted),
                        detections=[],
                        scene="hoist",
                    ),
                    qwen_detections=[],
                    diff={"added": 0, "removed": 0, "adjusted": 0},
                    review_required=False,
                    review_reasons=[],
                    prompt_path="",
                    raw_response_path="",
                    label_path="",
                    dataset_ready=True,
                )
            )
        _write_annotations(config, records)
        rebuild_hazard_report(config)
        report = read_json(config.hazard_report_path)
        self.assertEqual(report["counts"]["hazards_resolved"], 4)
        self.assertEqual(report["counts"]["hazards_needing_verification"], 2)


# --------------------------------------------------------------------------- #
# 6. CLI / run_cycle 接线
# --------------------------------------------------------------------------- #
class RebuildCliTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_command_is_implemented(self) -> None:
        self.assertIn("rebuild-hazards", IMPLEMENTED_COMMANDS)

    def test_run_cycle_branch(self) -> None:
        config = _make_config(self.tmp)
        _seed_annotations_only(config, FRAME_HAZARDS)
        summary = run_cycle(config, "rebuild-hazards")
        self.assertEqual(summary["stage"], "rebuild-hazards")
        self.assertEqual(summary["mode"], "rebuild")

    def test_cli_end_to_end(self) -> None:
        config = _make_config(self.tmp)
        _seed_annotations_only(config, FRAME_HAZARDS)
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = cli.main(["rebuild-hazards", "--config", str(config.config_path)])
        self.assertEqual(code, cli.EXIT_READY, err.getvalue())
        report = read_json(config.hazard_report_path)
        self.assertEqual(report["counts"]["hazards_needing_verification"], 2)

    def test_cli_missing_annotations_returns_4(self) -> None:
        config = _make_config(self.tmp)
        stderr = io.StringIO()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(stderr):
            code = cli.main(["rebuild-hazards", "--config", str(config.config_path)])
        self.assertEqual(code, cli.EXIT_STAGE_FAILED)
        self.assertIn("执行失败", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
