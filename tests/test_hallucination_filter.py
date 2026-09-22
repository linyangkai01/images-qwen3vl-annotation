"""幻觉过滤单测：把「能力范围外」的隐患显式标记为需人工确认（需求 §8.3）。

真实跑批（37 帧）暴露：VLM 对**能力范围外**的隐患也给了结论——例如 `D-1-6`
（吊具/索具未标识承重重量）**需 OCR 读铭牌**，属本期「留位不实现」的能力层，却出现在
隐患报告里且与检测层隐患混在一起。若不区分，甲方会误以为「系统能自动判定铭牌标识」。

修复：以 :mod:`industrial_hazard_coldstart.taxonomy` 的能力分层表为判据，对
``layer != "detection"`` 的隐患（几何 / 时序 / OCR / 超分辨 / 图像质量层，共 9 条）在
报告条目上加 ``needs_manual_verification=True`` + ``capability_note``，并给出汇总计数
``counts.hazards_needing_verification``。**只标记、不过滤。**

覆盖点（对应任务书测试要求）：
1. 检测层隐患（如 ``D-5-1``）**不带** ``needs_manual_verification`` 标记；
2. 非检测层隐患（``D-1-6`` OCR / ``D-5-3`` 几何 / ``D-3-6`` 时序 / ``D-1-4`` 超分辨 /
   ``D-4-1`` 图像质量）**带**标记且 ``capability_note`` 说明正确；
3. 汇总计数 ``hazards_needing_verification`` 正确；
4. 标记**不影响**隐患被正常回填（``category``/``norm_reference``/``action`` 仍来自 KB）。

约束：不联网、不调用真实 VLM、不读真实素材；只用合成 PNG + ``tempfile``；
不修改 ``traffic_cv_closed_loop/``。
"""

from __future__ import annotations

import json
import struct
import tempfile
import unittest
import zlib
from pathlib import Path

from traffic_cv_closed_loop.config import ProjectConfig
from traffic_cv_closed_loop.io_utils import read_json

from industrial_hazard_coldstart.knowledge import backfill_hazards, load_knowledge_base
from industrial_hazard_coldstart.orchestrator import run_cycle
from industrial_hazard_coldstart.taxonomy import (
    S1_DETECTION_HAZARD_IDS,
    S1_HAZARD_IDS,
    capability_note_for_hazard,
    hazard_needs_manual_verification,
    layer_of_hazard,
)

ROOT = Path(__file__).resolve().parents[1]
HOIST_CONFIG = ROOT / "own" / "hoist_project" / "config" / "project.hoist.json"
HOIST_PROMPT = ROOT / "own" / "hoist_project" / "assets" / "prompts" / "hoist_zh.md"
KB_PATH = ROOT / "hazard-detection" / "knowledge_base.json"

#: 5 个「能力范围外」能力层 → 该层应有的一条代表隐患 id 与能力短语。
_LAYER_SAMPLES: dict[str, tuple[str, str]] = {
    "ocr": ("D-1-6", "OCR 文字识别"),
    "geometry": ("D-5-3", "几何关系推理"),
    "temporal": ("D-3-6", "时序分析"),
    "super_resolution": ("D-1-4", "超分辨 / 细节识别"),
    "image_quality": ("D-4-1", "图像质量评估"),
}


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
    """在临时目录写一份 mock 吊装配置并加载（KB 指向仓库内知识库）。"""
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


def _make_candidate(config: ProjectConfig, relative_path: str, hazard_report: dict) -> dict:
    """造一条候选帧 + 侧车（detections / qwen / hazard），返回候选 dict。"""
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
        json.dumps(hazard_report, ensure_ascii=False), encoding="utf-8"
    )
    return {
        "source_id": Path(relative_path).parts[0],
        "relative_path": relative_path,
        "original_path": str(original),
        "extracted_path": str(extracted),
        "detections": [],
        "scene": "hoist",
        "metadata": {},
    }


def _write_candidates(config: ProjectConfig, candidates: list[dict]) -> None:
    config.candidates_path.parent.mkdir(parents=True, exist_ok=True)
    with config.candidates_path.open("w", encoding="utf-8", newline="\n") as handle:
        for item in candidates:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------- #
# 1. taxonomy 判据单测
# --------------------------------------------------------------------------- #
class CapabilityJudgementTest(unittest.TestCase):
    """``hazard_needs_manual_verification`` / ``capability_note_for_hazard`` 的判据。"""

    def test_detection_layer_not_marked(self) -> None:
        for hazard_id in S1_DETECTION_HAZARD_IDS:
            with self.subTest(hazard_id=hazard_id):
                self.assertFalse(hazard_needs_manual_verification(hazard_id))
                self.assertEqual(capability_note_for_hazard(hazard_id), "")

    def test_all_nine_non_detection_marked(self) -> None:
        marked = [h for h in S1_HAZARD_IDS if hazard_needs_manual_verification(h)]
        self.assertEqual(len(marked), 9)
        # 与「检测层 18 条」互补，恰好覆盖 27 条。
        self.assertEqual(len(S1_DETECTION_HAZARD_IDS), 18)
        self.assertEqual(set(marked) & set(S1_DETECTION_HAZARD_IDS), set())

    def test_each_layer_marked_with_correct_note(self) -> None:
        for layer, (hazard_id, capability) in _LAYER_SAMPLES.items():
            with self.subTest(hazard_id=hazard_id):
                self.assertEqual(layer_of_hazard(hazard_id), layer)
                self.assertTrue(hazard_needs_manual_verification(hazard_id))
                note = capability_note_for_hazard(hazard_id)
                self.assertIn(capability, note)
                self.assertIn("未经检测验证", note)
                self.assertIn("VLM 文本推断", note)

    def test_unsupported_layers_say_not_implemented(self) -> None:
        """几何/时序/OCR/超分辨层：如实标注「本期留位未实现」。"""
        for layer in ("geometry", "temporal", "ocr", "super_resolution"):
            hazard_id, _ = _LAYER_SAMPLES[layer]
            with self.subTest(hazard_id=hazard_id):
                self.assertIn("本期留位未实现", capability_note_for_hazard(hazard_id))

    def test_partial_layer_says_partial_not_absent(self) -> None:
        """图像质量层本期「部分实现」——不得笼统说「未实现」（措辞精度）。"""
        note = capability_note_for_hazard("D-4-1")
        self.assertIn("部分实现", note)
        self.assertNotIn("留位未实现", note)

    def test_unknown_and_generic_not_marked(self) -> None:
        """非 S1 隐患（通用 T-1 / 高处 G-1-1）与未知 id：保守不误标。"""
        for hazard_id in ("T-1", "G-1-1", "X-9-9", "", "  "):
            with self.subTest(hazard_id=hazard_id):
                self.assertFalse(hazard_needs_manual_verification(hazard_id))
                self.assertEqual(capability_note_for_hazard(hazard_id), "")


# --------------------------------------------------------------------------- #
# 2. 回填层单测：标记不破坏 KB 回填
# --------------------------------------------------------------------------- #
class BackfillMarkingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.kb = load_knowledge_base(KB_PATH)
        self.allowed = frozenset(self.kb.allowed_hazard_ids("hoist"))

    def _resolve_one(self, hazard_id: str, evidence: str = "证据") -> dict:
        resolved, rejected = backfill_hazards(
            {"hazards": [{"id": hazard_id, "evidence": evidence}]},
            self.kb,
            self.allowed,
            source_frame="clip/frame_000001.png",
        )
        self.assertEqual(rejected, [])
        self.assertEqual(len(resolved), 1)
        return resolved[0]

    def test_detection_hazard_has_no_marking(self) -> None:
        item = self._resolve_one("D-5-1", "人员头部未见安全帽")
        self.assertNotIn("needs_manual_verification", item)
        self.assertNotIn("capability_note", item)

    def test_ocr_hazard_marked_and_kb_fields_intact(self) -> None:
        item = self._resolve_one("D-1-6", "吊具铭牌未见承重标识")
        self.assertTrue(item["needs_manual_verification"])
        self.assertIn("OCR 文字识别", item["capability_note"])
        # 标记不影响 KB 回填：三个字段仍逐字等于 KB。
        entry = self.kb.get("D-1-6")
        self.assertEqual(item["category"], entry.category_name)
        self.assertEqual(item["norm_reference"], entry.norm_reference)
        self.assertEqual(item["action"], entry.action)

    def test_geometry_and_temporal_marked(self) -> None:
        for hazard_id, capability in (("D-5-3", "几何关系推理"), ("D-3-6", "时序分析")):
            with self.subTest(hazard_id=hazard_id):
                item = self._resolve_one(hazard_id)
                self.assertTrue(item["needs_manual_verification"])
                self.assertIn(capability, item["capability_note"])
                entry = self.kb.get(hazard_id)
                self.assertEqual(item["norm_reference"], entry.norm_reference)


# --------------------------------------------------------------------------- #
# 3. annotate 端到端：报告标记 + 汇总计数
# --------------------------------------------------------------------------- #
class AnnotateReportMarkingTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.config = _make_config(self.tmp)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_report_marks_out_of_capability_and_counts(self) -> None:
        candidates = [
            _make_candidate(
                self.config,
                "clip/frame_000001.png",
                hazard_report={
                    "hazards": [
                        {"id": "D-5-1", "evidence": "人员头部未见安全帽"},   # 检测层
                        {"id": "D-1-6", "evidence": "吊具铭牌未见承重标识"},  # OCR 层
                        {"id": "D-3-6", "evidence": "吊物长时间悬停"},       # 时序层
                    ]
                },
            )
        ]
        _write_candidates(self.config, candidates)

        summary = run_cycle(self.config, "annotate")
        report = read_json(self.config.hazard_report_path)
        by_id = {item["hazard_id"]: item for item in report["hazards"]}

        # 三条都保留（只标记、不过滤）。
        self.assertEqual(report["counts"]["hazards_resolved"], 3)
        self.assertEqual(set(by_id), {"D-5-1", "D-1-6", "D-3-6"})

        # 检测层：不带标记。
        self.assertNotIn("needs_manual_verification", by_id["D-5-1"])

        # 非检测层：带标记 + 说明正确。
        self.assertTrue(by_id["D-1-6"]["needs_manual_verification"])
        self.assertIn("OCR 文字识别", by_id["D-1-6"]["capability_note"])
        self.assertTrue(by_id["D-3-6"]["needs_manual_verification"])
        self.assertIn("时序分析", by_id["D-3-6"]["capability_note"])

        # 汇总计数。
        self.assertEqual(report["counts"]["hazards_needing_verification"], 2)
        self.assertEqual(summary["hazards_needing_verification"], 2)
        self.assertEqual(
            report["verification"]["by_layer"], {"ocr": 1, "temporal": 1}
        )

    def test_all_detection_hazards_report_zero_verification(self) -> None:
        candidates = [
            _make_candidate(
                self.config,
                "clip/frame_000001.png",
                hazard_report={
                    "hazards": [
                        {"id": "D-5-1", "evidence": "未戴安全帽"},
                        {"id": "D-5-2", "evidence": "无反光背心"},
                    ]
                },
            )
        ]
        _write_candidates(self.config, candidates)
        summary = run_cycle(self.config, "annotate")
        report = read_json(self.config.hazard_report_path)
        self.assertEqual(report["counts"]["hazards_needing_verification"], 0)
        self.assertEqual(report["verification"]["by_layer"], {})
        self.assertEqual(summary["hazards_needing_verification"], 0)
        for item in report["hazards"]:
            self.assertNotIn("needs_manual_verification", item)

    def test_real_scene_d_1_6_is_marked_not_dropped(self) -> None:
        """真实跑批复现：D-1-6 仍出现（不被过滤），但被标记为需人工确认。"""
        candidates = [
            _make_candidate(
                self.config,
                "clip/frame_000001.png",
                hazard_report={
                    "hazards": [{"id": "D-1-6", "evidence": "索具未见承重重量标识"}]
                },
            )
        ]
        _write_candidates(self.config, candidates)
        summary = run_cycle(self.config, "annotate")
        report = read_json(self.config.hazard_report_path)
        self.assertEqual(report["counts"]["hazards_resolved"], 1)
        item = report["hazards"][0]
        self.assertEqual(item["hazard_id"], "D-1-6")
        self.assertTrue(item["needs_manual_verification"])
        self.assertEqual(report["counts"]["hazards_needing_verification"], 1)
        self.assertEqual(summary["hazards_needing_verification"], 1)


if __name__ == "__main__":
    unittest.main()
