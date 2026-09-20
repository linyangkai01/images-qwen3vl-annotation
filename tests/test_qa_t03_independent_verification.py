"""T03 独立验证（QA 严过关）：两段式标注 + 主动学习 + 语义约束。

与 ``tests/test_annotation_flow.py``（工程师自测）**独立**：本文件聚焦工程师**自认未验证**
的 **非 mock 路径**，以及需求书明确要求、但自测未覆盖的边界。

核心手法（不联网、不调真实 VLM）：
- **半静态复现**：直接实例化 ``PrivateVlmAnnotator``，把 ``_post_json`` 替换为返回**手工构造的
  真实 VLM 可能响应**，从而完整走通真实 provider 的 ``annotate()`` 与 ``run_annotate()`` 数据流；
- 判定哪些响应形态会 **静默拿不到隐患**（通道 B 全空但流程显示成功）——本项目的头号失败模式。

约束：不修改任何生产代码；不联网；不调用真实 VLM；不读真实素材；临时文件用 ``tempfile``。
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import struct
import tempfile
import unittest
import zlib
from pathlib import Path

from traffic_cv_closed_loop.adapters.qwen import PrivateVlmAnnotator
from traffic_cv_closed_loop.config import ProjectConfig
from traffic_cv_closed_loop.io_utils import read_json, read_jsonl, read_text
from traffic_cv_closed_loop.models import AnnotationRecord, CandidateFrame, Detection
from traffic_cv_closed_loop.prompting import render_annotation_prompt

import industrial_hazard_coldstart.active_learning as active_learning
from industrial_hazard_coldstart import cli, taxonomy
from industrial_hazard_coldstart.active_learning import (
    build_priority_queue,
    score_uncertainty,
)
from industrial_hazard_coldstart.knowledge import (
    backfill_hazards,
    extract_model_hazards,
    load_knowledge_base,
    overall_risk_of,
    resolve_hazard,
    severity_rank,
)
from industrial_hazard_coldstart.orchestrator import (
    _extract_hazard_from_raw,
    run_cycle,
)

ROOT = Path(__file__).resolve().parents[1]
HOIST_CONFIG = ROOT / "own" / "hoist_project" / "config" / "project.hoist.json"
HOIST_PROMPT = ROOT / "own" / "hoist_project" / "assets" / "prompts" / "hoist_zh.md"
KB_PATH = ROOT / "hazard-detection" / "knowledge_base.json"


# --------------------------------------------------------------------------- #
# 合成素材 / 配置工具（绝不触碰真实素材）
# --------------------------------------------------------------------------- #
def _write_png(path: Path, width: int = 8, height: int = 8) -> Path:
    """写一张真实可解码的最小 PNG（纯标准库），保证图片魔数校验通过。"""

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    row = b"\x00" + bytes((120, 120, 120)) * width
    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(row * height))
        + chunk(b"IEND", b"")
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png)
    return path


def _build_config(tmp: Path, *, provider: str = "mock") -> ProjectConfig:
    """在临时目录写一份配置并加载（KB / 提示词指向仓库内真实文件）。"""
    raw = json.loads(HOIST_CONFIG.read_text(encoding="utf-8"))
    raw["qwen"]["provider"] = provider
    raw["qwen"]["prompt_template"] = str(HOIST_PROMPT)
    raw["yolo"]["detector"] = "sidecar"
    raw["pipeline"]["include_empty_frames"] = True
    raw["knowledge"]["kb_path"] = str(KB_PATH)
    config_dir = tmp / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "project.json"
    config_path.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
    return ProjectConfig.load(config_path)


def _write_candidate(
    config: ProjectConfig,
    relative_path: str,
    *,
    yolo_detections: list[dict] | None = None,
) -> None:
    """写一条候选帧 + 真实 PNG，并落 ``candidates.jsonl``。"""
    original = config.input_frames_dir.parent / "media" / relative_path
    extracted = config.input_frames_dir / relative_path
    _write_png(original)
    _write_png(extracted)
    (original.with_suffix(".detections.json")).write_text(
        json.dumps({"objects": yolo_detections or []}, ensure_ascii=False), encoding="utf-8"
    )
    config.candidates_path.parent.mkdir(parents=True, exist_ok=True)
    candidate = {
        "source_id": Path(relative_path).parts[0],
        "relative_path": relative_path,
        "original_path": str(original),
        "extracted_path": str(extracted),
        "detections": yolo_detections or [],
        "scene": "hoist",
        "metadata": {},
    }
    config.candidates_path.write_text(
        json.dumps(candidate, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _make_record(
    relative_path: str,
    *,
    confidences: list[float],
    diff: dict | None = None,
    review_required: bool = False,
    review_reasons: list[str] | None = None,
    yolo_count: int = 1,
) -> AnnotationRecord:
    candidate = CandidateFrame(
        source_id="clip",
        relative_path=relative_path,
        original_path="",
        extracted_path="",
        detections=[
            Detection(label="person", bbox=(0.1, 0.1, 0.2, 0.2)) for _ in range(yolo_count)
        ],
    )
    return AnnotationRecord(
        candidate=candidate,
        qwen_detections=[
            Detection(label="person", bbox=(0.1, 0.1, 0.2, 0.2), confidence=value)
            for value in confidences
        ],
        diff=diff or {"added": 0, "removed": 0, "adjusted": 0},
        review_required=review_required,
        review_reasons=review_reasons or [],
        prompt_path="",
        raw_response_path="",
        label_path="",
        dataset_ready=True,
    )


# --------------------------------------------------------------------------- #
# 1. 非 mock 路径容错性（工程师自认未验证 —— 本文件最重要的一节）
# --------------------------------------------------------------------------- #
class NonMockPathTest(unittest.TestCase):
    """判定真实 VLM 响应各形态是否会被 ``_extract_hazard_from_raw`` 命中。

    结论先行（证据见各用例）：

    * **能命中**：``parsed.hazard_report`` 为 dict（含 thinking / 围栏 / 前后散文包裹的响应，
      因为 ``ThinkingJsonParser`` 已在写盘前把响应解析成 dict）；
    * **修复后可恢复（QA-T03-01/03）**：``hazard_report`` 被二次编码成字符串、被多包一层，
      以及 ``parsed`` 本身是字符串/列表、Ollama 原生 ``response`` 字符串 —— 现均能被
      ``_locate_model_output`` / ``_decode_json_string`` 解开并命中；
    * **仍取不到（返回 ``{}``）但不再静默**：``hazard_report`` **缺失 / 为 null**。
      此时若模型确实答了目标（输出含 ``objects``），``run_annotate`` 会把它计入
      ``counts.frames_without_hazard_report`` 并按 ``hazard_report_missing`` 补进复核队列，
      从而与「本帧确实没有隐患」区分开（见 ``test_real_annotate_missing_report_is_flagged_and_queued``）。
    """

    _GOOD_REPORT = {"hazards": [{"id": "D-5-1", "evidence": "未戴安全帽"}], "overall_risk": "high"}

    def setUp(self) -> None:
        self._env_backup = os.environ.get("LAB_VLM_TOKEN")
        os.environ["LAB_VLM_TOKEN"] = "qa-dummy-token"
        self._orig_post_json = PrivateVlmAnnotator._post_json
        self.addCleanup(self._restore)

    def _restore(self) -> None:
        PrivateVlmAnnotator._post_json = self._orig_post_json
        if self._env_backup is None:
            os.environ.pop("LAB_VLM_TOKEN", None)
        else:
            os.environ["LAB_VLM_TOKEN"] = self._env_backup

    def _patch_post_json(self, response_text: str) -> None:
        def fake(self, payload, headers):  # noqa: ANN001 - 对齐被测签名
            return {"response": response_text, "done": True, "done_reason": "stop"}

        PrivateVlmAnnotator._post_json = fake

    # -- 单元级：_extract_hazard_from_raw 直接吃 raw payload ------------------ #
    def test_valid_parsed_hazard_report_is_extracted(self) -> None:
        payload = {"provider": "private_vlm", "parsed": {"objects": [], "hazard_report": self._GOOD_REPORT}}
        self.assertEqual(_extract_hazard_from_raw(payload), self._GOOD_REPORT)

    def test_top_level_hazard_report_is_extracted(self) -> None:
        payload = {"hazard_report": self._GOOD_REPORT}
        self.assertEqual(_extract_hazard_from_raw(payload), self._GOOD_REPORT)

    def test_still_unrecoverable_shapes_return_empty(self) -> None:
        """仍取不到的两种形态：``hazard_report`` 缺失 / 为 null → ``{}``。

        这两种并非"容错失败"：前者模型压根没答隐患题（会由 ``frames_without_hazard_report``
        显式标记），后者语义上就是"无隐患"。
        """
        shapes = {
            "no hazard_report key": {"parsed": {"objects": [{"label": "person"}]}},
            "hazard_report=null": {"parsed": {"objects": [], "hazard_report": None}},
        }
        for name, payload in shapes.items():
            with self.subTest(shape=name):
                self.assertEqual(_extract_hazard_from_raw(payload), {}, f"{name} 应为空")

    def test_recoverable_shapes_are_now_extracted(self) -> None:
        """修复后（QA-T03-01/03）：这 5 种形态**均能恢复**（独立复现，非仅打标）。

        工程师声称「7 种里 5 种现在能真正恢复」——本用例独立复核，确认属实。
        """
        shapes = {
            "hazard_report=str(double-encoded)": {
                "parsed": {"objects": [], "hazard_report": json.dumps(self._GOOD_REPORT)}
            },
            "wrapped one level deeper": {"parsed": {"result": {"objects": [], "hazard_report": self._GOOD_REPORT}}},
            "parsed is a str": {"parsed": json.dumps(self._GOOD_REPORT)},
            "parsed is a list": {"parsed": [{"hazard_report": self._GOOD_REPORT}]},
            "ollama response str, no parsed": {
                "provider": "private_vlm",
                "response": json.dumps({"objects": [], "hazard_report": self._GOOD_REPORT}),
            },
        }
        for name, payload in shapes.items():
            with self.subTest(shape=name):
                self.assertEqual(_extract_hazard_from_raw(payload), self._GOOD_REPORT, f"{name} 应已恢复")

    # -- 半静态：走真实 provider 的 annotate()（含 parser），只替换网络层 ----- #
    def test_real_annotate_handles_thinking_and_fences(self) -> None:
        """thinking 块 / 结束标记 / markdown 围栏 / 前后散文 —— 均应命中 hazard_report。"""
        tmp = Path(tempfile.mkdtemp())
        config = _build_config(tmp, provider="private_vlm")
        image = _write_png(tmp / "f.png")
        candidate = CandidateFrame(
            source_id="clip",
            relative_path="clip/f.png",
            original_path=str(image),
            extracted_path=str(image),
            detections=[Detection(label="person", bbox=(0.1, 0.1, 0.3, 0.5), confidence=0.7)],
        )
        final = json.dumps(
            {"objects": [{"label": "person", "bbox": [0.1, 0.1, 0.3, 0.5], "confidence": 0.9}],
             "hazard_report": self._GOOD_REPORT},
            ensure_ascii=False,
        )
        responses = {
            "plain": final,
            "thinking_end_marker": "推理过程……<|end▁of▁thinking｜>" + final,
            "think_xml": " thinkingframe has a worker, no json here<｜end▁of▁thinking｜>" + final,
            "markdown_fence": "```json\n" + final + "\n```",
            "prose_then_fence": "结果如下：\n```json\n" + final + "\n```\n完成。",
        }
        for name, response_text in responses.items():
            with self.subTest(shape=name):
                self._patch_post_json(response_text)
                _, payload = PrivateVlmAnnotator(config).annotate(candidate, "prompt")
                self.assertEqual(_extract_hazard_from_raw(payload), self._GOOD_REPORT)

    def test_real_annotate_missing_report_is_flagged_and_queued(self) -> None:
        """端到端：模型只回 objects（未答隐患题）→ **显式标记 + 入复核队列**（修复 QA-T03-01）。

        隐患通道仍为空（``hazards=[]``），但不再是"静默"：报告与摘要都给出
        ``frames_without_hazard_report=1``，且该帧按 ``hazard_report_missing`` 入队。
        """
        tmp = Path(tempfile.mkdtemp())
        config = _build_config(tmp, provider="private_vlm")
        _write_candidate(
            config,
            "clip/frame_000001.png",
            yolo_detections=[{"label": "person", "bbox": [0.1, 0.1, 0.3, 0.5], "confidence": 0.7}],
        )
        self._patch_post_json(
            json.dumps(
                {"objects": [{"label": "person", "bbox": [0.1, 0.1, 0.3, 0.5], "confidence": 0.9}]},
                ensure_ascii=False,
            )
        )
        summary = run_cycle(config, "annotate")
        report = read_json(config.hazard_report_path)

        # 隐患通道仍为空 —— 这部分行为不变。
        self.assertEqual(summary["hazards_resolved"], 0)
        self.assertEqual(summary["hazards_rejected"], 0)
        self.assertEqual(summary["frames_with_rejected_hazard"], 0)
        self.assertEqual(report["overall_risk"], "none")
        self.assertEqual(report["hazards"], [])
        self.assertEqual(report["frames"][0]["model_hazard_count"], 0)

        # 关键变化：显式信号出现，不再是静默失败。
        self.assertEqual(report["counts"]["frames_without_hazard_report"], 1)
        self.assertEqual(summary["frames_without_hazard_report"], 1)
        self.assertFalse(report["frames"][0]["hazard_report_present"])
        queue = read_jsonl(config.review_queue_path)
        queued = [item for item in queue
                  if item["candidate"]["relative_path"] == "clip/frame_000001.png"]
        self.assertTrue(queued)
        self.assertIn("hazard_report_missing", queued[0]["review_reasons"])
        self.assertTrue(queued[0]["review_required"])

    def test_real_annotate_explicit_empty_report_is_not_flagged(self) -> None:
        """反向边界：模型明确答"无隐患"（``hazard_report={"hazards":[]}``）**不得**误报。

        护栏只在「答了目标却取不到 hazard_report」时触发；这条守住不误报的下界。
        """
        tmp = Path(tempfile.mkdtemp())
        config = _build_config(tmp, provider="private_vlm")
        _write_candidate(
            config,
            "clip/frame_000001.png",
            yolo_detections=[{"label": "person", "bbox": [0.1, 0.1, 0.3, 0.5], "confidence": 0.7}],
        )
        self._patch_post_json(
            json.dumps(
                {"objects": [{"label": "person", "bbox": [0.1, 0.1, 0.3, 0.5], "confidence": 0.9}],
                 "hazard_report": {"hazards": []}},
                ensure_ascii=False,
            )
        )
        summary = run_cycle(config, "annotate")
        report = read_json(config.hazard_report_path)

        self.assertEqual(report["counts"]["frames_without_hazard_report"], 0)
        self.assertEqual(summary["frames_without_hazard_report"], 0)
        self.assertTrue(report["frames"][0]["hazard_report_present"])
        # 该帧 qwen 与 yolo 一致、置信度 0.9 → 本无需复核；不得因"无隐患"被塞进队列。
        queue = read_jsonl(config.review_queue_path)
        queued = [item for item in queue
                  if item["candidate"]["relative_path"] == "clip/frame_000001.png"]
        self.assertFalse(queued)

    def test_real_annotate_valid_report_end_to_end(self) -> None:
        """对照组：模型正常答隐患题 → 端到端命中并回填。"""
        tmp = Path(tempfile.mkdtemp())
        config = _build_config(tmp, provider="private_vlm")
        _write_candidate(
            config,
            "clip/frame_000001.png",
            yolo_detections=[{"label": "person", "bbox": [0.1, 0.1, 0.3, 0.5], "confidence": 0.7}],
        )
        self._patch_post_json(
            json.dumps({"objects": [], "hazard_report": self._GOOD_REPORT}, ensure_ascii=False)
        )
        summary = run_cycle(config, "annotate")
        report = read_json(config.hazard_report_path)
        self.assertEqual(summary["hazards_resolved"], 1)
        self.assertEqual(report["hazards"][0]["hazard_id"], "D-5-1")
        self.assertEqual(report["overall_risk"], "high")


# --------------------------------------------------------------------------- #
# 2. KB 三级校验边界
# --------------------------------------------------------------------------- #
class KbValidationEdgeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.kb = load_knowledge_base(KB_PATH)
        self.allowed = frozenset(self.kb.allowed_hazard_ids("hoist"))

    def test_empty_whitespace_none_id_are_missing(self) -> None:
        for hazard_id in ("", "   ", None, "\t"):
            with self.subTest(hazard_id=hazard_id):
                good, bad = resolve_hazard(hazard_id, "e", self.kb, self.allowed)
                self.assertIsNone(good)
                self.assertEqual(bad["reason"], "missing_hazard_id")

    def test_surrounding_whitespace_is_normalized(self) -> None:
        for hazard_id in (" D-1-1 ", "D-1-1\t", "\nD-1-1"):
            with self.subTest(hazard_id=hazard_id):
                good, bad = resolve_hazard(hazard_id, "e", self.kb, self.allowed)
                self.assertIsNotNone(good)
                self.assertEqual(good["hazard_id"], "D-1-1")
                self.assertIsNone(bad)

    def test_lowercase_id_is_rejected(self) -> None:
        """KB 键区分大小写：``d-1-1`` 不在 KB → not_in_knowledge_base（不放行）。"""
        good, bad = resolve_hazard("d-1-1", "e", self.kb, self.allowed)
        self.assertIsNone(good)
        self.assertEqual(bad["reason"], "not_in_knowledge_base")

    def test_empty_allowed_set_rejects_everything(self) -> None:
        """受控词表为空 → **全拒**（宁缺毋滥），绝不放行。"""
        good, bad = resolve_hazard("D-5-1", "e", self.kb, frozenset())
        self.assertIsNone(good)
        self.assertEqual(bad["reason"], "out_of_scene_vocabulary")

    def test_unknown_scene_yields_empty_vocabulary(self) -> None:
        self.assertEqual(self.kb.allowed_hazard_ids("bogus"), [])
        self.assertEqual(self.kb.scene_id_for("bogus"), "")

    def test_s3_generic_hazard_included_in_hoist_scene(self) -> None:
        """S3 通用（T-1）**并入吊装受控词表** → 可被引用并回填（修复 QA-T03-02）。

        与 ARCH §2.2 / REQ §4.2.7「吊装受控词表 = S1 27 条 + S3 通用 1 条 = 28 条」一致。
        """
        self.assertIsNotNone(self.kb.get("T-1"))  # 确实在 KB 内
        self.assertIn("T-1", self.allowed)
        self.assertEqual(len(self.allowed), 28)
        good, bad = resolve_hazard("T-1", "地面大片积水", self.kb, self.allowed)
        self.assertIsNone(bad)
        self.assertIsNotNone(good)
        self.assertEqual(good["hazard_id"], "T-1")
        # 回填值来自 KB（T-1 属 S3，无 hazard_categories → category 为空，属忠实回填）。
        self.assertEqual(good["norm_reference"], self.kb.get("T-1").norm_reference)
        self.assertTrue(good["action"])

    def test_s3_scene_does_not_duplicate_generic_ids(self) -> None:
        """``scene='S3'`` 时不应把 S3 再并入一次（结果与自身一致，不重复）。"""
        self.assertEqual(self.kb.allowed_hazard_ids("S3"), ["T-1"])

    def test_severity_and_overall_risk_helpers(self) -> None:
        self.assertEqual(severity_rank("HIGH"), 2)  # 大小写不敏感
        self.assertEqual(severity_rank("weird"), -1)
        self.assertEqual(severity_rank(None), -1)
        self.assertEqual(overall_risk_of([], "critical"), "critical")  # 合法枚举透传
        self.assertEqual(overall_risk_of([], "bogus"), "none")  # 非法值 → none
        self.assertEqual(overall_risk_of([], None), "none")
        # 有隐患时以 KB 严重度为准，忽略模型自报值。
        self.assertEqual(overall_risk_of([{"severity": "high"}], "low"), "high")


# --------------------------------------------------------------------------- #
# 3. 提示词渲染边界
# --------------------------------------------------------------------------- #
class PromptRenderingEdgeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = ProjectConfig.load(HOIST_CONFIG)
        self.candidate = CandidateFrame(
            source_id="clip",
            relative_path="clip/frame_000001.png",
            original_path="",
            extracted_path="",
            detections=[Detection(label="person", bbox=(0.1, 0.1, 0.3, 0.5), confidence=0.7)],
        )

    def test_hoist_template_renders_and_lists_controlled_vocab(self) -> None:
        rendered = render_annotation_prompt(
            self.config, self.candidate, HOIST_PROMPT.read_text(encoding="utf-8")
        )
        for placeholder in ("{scene}", "{labels}", "{candidate_path}", "{prior_detections}"):
            self.assertNotIn(placeholder, rendered)
        for hazard_id in taxonomy.S1_HAZARD_IDS:
            self.assertIn(hazard_id, rendered)
        # 受控词表现为 S1 27 条 + S3 通用 1 条（T-1），提示词同步补齐（修复 QA-T03-02）。
        self.assertIn("T-1", rendered)

    def test_template_with_subset_of_placeholders_is_ok(self) -> None:
        """模板只写 ``{scene}`` 仍可渲染（缺失的是参数才会报错）。"""
        self.assertEqual(render_annotation_prompt(self.config, self.candidate, "S={scene}"), "S=hoist")

    def test_raw_braces_in_template_raise_key_error(self) -> None:
        """未转义的 JSON 示例 ``{"a":1}`` 会 KeyError —— 模板必须转义花括号。"""
        with self.assertRaises(KeyError):
            render_annotation_prompt(self.config, self.candidate, '{"a":1} scene={scene}')

    def test_unbalanced_brace_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            render_annotation_prompt(self.config, self.candidate, "{scene")

    def test_empty_detections_render_as_empty_list(self) -> None:
        empty = CandidateFrame(
            source_id="clip",
            relative_path="clip/frame_000001.png",
            original_path="",
            extracted_path="",
            detections=[],
        )
        self.assertEqual(
            render_annotation_prompt(self.config, empty, "prior={prior_detections}"), "prior=[]"
        )

    def test_empty_template_fails_loud_in_run_annotation(self) -> None:
        """空模板：``render`` 本身返回空串，但母项目 ``run_annotation`` 会 fail-loud。"""
        from traffic_cv_closed_loop.pipeline.annotation import run_annotation

        tmp = Path(tempfile.mkdtemp())
        config = _build_config(tmp, provider="mock")
        empty_template = tmp / "empty.md"
        empty_template.write_text("", encoding="utf-8")
        raw = json.loads(config.config_path.read_text(encoding="utf-8"))
        raw["qwen"]["prompt_template"] = str(empty_template)
        config.config_path.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
        config = ProjectConfig.load(config.config_path)
        config.candidates_path.parent.mkdir(parents=True, exist_ok=True)
        config.candidates_path.write_text("", encoding="utf-8")
        with self.assertRaises(FileNotFoundError):
            run_annotation(config)


# --------------------------------------------------------------------------- #
# 4. 主动学习边界
# --------------------------------------------------------------------------- #
class ActiveLearningEdgeTest(unittest.TestCase):
    def test_empty_input_returns_empty_queue(self) -> None:
        self.assertEqual(build_priority_queue([]), [])

    def test_record_without_detections_does_not_divide_by_zero(self) -> None:
        item = score_uncertainty(_make_record("clip/a.png", confidences=[]))
        self.assertEqual(item.score, 0.0)
        self.assertEqual(item.signals["channel_disagreement"], 0.0)
        self.assertIsNone(item.signals["min_confidence"])

    def test_tie_is_broken_by_relative_path_ascending(self) -> None:
        records = [
            _make_record("clip/b.png", confidences=[0.4], review_required=True),
            _make_record("clip/a.png", confidences=[0.4], review_required=True),
            _make_record("clip/c.png", confidences=[0.4], review_required=True),
        ]
        queue = build_priority_queue(records)
        self.assertEqual([item.relative_path for item in queue], ["clip/a.png", "clip/b.png", "clip/c.png"])
        # 同一输入多次调用顺序一致（确定性）。
        self.assertEqual(
            [i.relative_path for i in build_priority_queue(records)],
            [i.relative_path for i in build_priority_queue(records)],
        )

    def test_top_k_variants(self) -> None:
        records = [_make_record(f"clip/f{i}.png", confidences=[0.3]) for i in range(3)]
        self.assertEqual(len(build_priority_queue(records, top_k=2)), 2)
        self.assertEqual(len(build_priority_queue(records, top_k=0)), 3)
        self.assertEqual(len(build_priority_queue(records, top_k=None)), 3)
        self.assertEqual(len(build_priority_queue(records, top_k=-1)), 3)

    def test_weights_are_overridable_at_module_scope(self) -> None:
        """权重无函数入参，只能改模块常量 —— 改后立即生效，且 signals 如实反映。"""
        record = _make_record("clip/a.png", confidences=[0.3], review_required=True, review_reasons=["low_confidence"])
        self.assertAlmostEqual(score_uncertainty(record).score, 0.45, places=6)

        backup = (
            active_learning.WEIGHT_LOW_CONFIDENCE,
            active_learning.WEIGHT_CHANNEL_DISAGREEMENT,
            active_learning.WEIGHT_REVIEW_REASON,
        )
        active_learning.WEIGHT_LOW_CONFIDENCE = 0.0
        active_learning.WEIGHT_CHANNEL_DISAGREEMENT = 0.0
        active_learning.WEIGHT_REVIEW_REASON = 0.0
        try:
            zeroed = score_uncertainty(record)
            self.assertEqual(zeroed.score, 0.0)
            self.assertEqual(zeroed.signals["weights"]["low_confidence"], 0.0)
            # 全 0 权重下仍按 path 升序，确定性不受影响。
            queue = build_priority_queue(
                [_make_record("clip/b.png", confidences=[0.3], review_required=True),
                 _make_record("clip/a.png", confidences=[0.3], review_required=True)]
            )
            self.assertEqual([i.relative_path for i in queue], ["clip/a.png", "clip/b.png"])
        finally:
            (
                active_learning.WEIGHT_LOW_CONFIDENCE,
                active_learning.WEIGHT_CHANNEL_DISAGREEMENT,
                active_learning.WEIGHT_REVIEW_REASON,
            ) = backup

    def test_score_is_clamped_to_unit_interval(self) -> None:
        record = _make_record(
            "clip/a.png",
            confidences=[0.0],
            diff={"added": 9, "removed": 9, "adjusted": 9},
            review_required=True,
            review_reasons=["annotation_error"],
        )
        self.assertLessEqual(score_uncertainty(record).score, 1.0)
        self.assertGreaterEqual(score_uncertainty(record).score, 0.0)


# --------------------------------------------------------------------------- #
# 5. KB 回填「只来自 KB」
# --------------------------------------------------------------------------- #
class BackfillOnlyFromKbTest(unittest.TestCase):
    def setUp(self) -> None:
        self.kb = load_knowledge_base(KB_PATH)
        self.allowed = frozenset(self.kb.allowed_hazard_ids("hoist"))

    def test_forged_fields_are_discarded_and_kb_wins(self) -> None:
        entry = self.kb.get("D-1-1")
        resolved, rejected = backfill_hazards(
            {
                "hazards": [
                    {
                        "id": "D-1-1",
                        "evidence": "绳索呈 X 形交叉",
                        # 伪造值**刻意不同于** KB 真值，避免"碰巧相同"的假象。
                        "category": "伪造类别",
                        "norm_reference": "GB 9999-2099-XX",
                        "action": "伪造整改措施",
                        "severity": "critical",
                        "name": "伪造名称",
                        "scene_id": "S9",
                        "extra": "junk",
                    }
                ]
            },
            self.kb,
            self.allowed,
        )
        self.assertEqual(rejected, [])
        item = resolved[0]
        self.assertEqual(item["norm_reference"], entry.norm_reference)
        self.assertNotEqual(item["norm_reference"], "GB 9999-2099-XX")
        self.assertEqual(item["category"], entry.category_name)
        self.assertEqual(item["action"], entry.action)
        self.assertEqual(item["severity"], entry.severity)  # 伪造 critical 被丢弃
        self.assertEqual(item["name"], entry.name)
        blob = json.dumps(item, ensure_ascii=False)
        self.assertNotIn("GB 9999-2099-XX", blob)
        self.assertNotIn("伪造", blob)
        self.assertNotIn("extra", item)

    def test_forged_value_equal_to_kb_still_sourced_from_kb(self) -> None:
        """伪造值恰好等于 KB 真值时，输出仍应等于 KB（无法从输出区分，故用字段整体一致来断言）。"""
        entry = self.kb.get("D-1-1")
        resolved, _ = backfill_hazards(
            {"hazards": [{"id": "D-1-1", "evidence": "x", "norm_reference": entry.norm_reference,
                          "category": entry.category_name, "action": entry.action}]},
            self.kb,
            self.allowed,
        )
        self.assertEqual(resolved[0]["norm_reference"], entry.norm_reference)
        self.assertEqual(resolved[0]["category"], entry.category_name)
        self.assertEqual(resolved[0]["action"], entry.action)

    def test_extract_model_hazards_shape_tolerance(self) -> None:
        self.assertEqual(extract_model_hazards({"hazards": [{"id": "D-5-1", "evidence": "e"}]}),
                         [{"hazard_id": "D-5-1", "evidence": "e"}])
        self.assertEqual(extract_model_hazards([{"hazard_id": "D-5-1", "evidence": "e"}]),
                         [{"hazard_id": "D-5-1", "evidence": "e"}])
        for bad in (None, "x", 123, {"hazards": "not-a-list"}, {"hazards": None}):
            with self.subTest(bad=bad):
                self.assertEqual(extract_model_hazards(bad), [])
        # 非 dict 元素被丢弃；缺 id → 空串（随后被 missing_hazard_id 拒绝）。
        self.assertEqual(extract_model_hazards({"hazards": ["x", {"id": "D-5-1"}]}),
                         [{"hazard_id": "D-5-1", "evidence": ""}])
        self.assertEqual(extract_model_hazards({"hazards": [{"evidence": "e"}]}),
                         [{"hazard_id": "", "evidence": "e"}])


# --------------------------------------------------------------------------- #
# 6. mock 端到端 + 复核队列兼容性
# --------------------------------------------------------------------------- #
class MockEndToEndTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.config = _build_config(self.tmp, provider="mock")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _seed(self) -> None:
        relative_path = "clip/frame_000001.png"
        _write_candidate(
            self.config,
            relative_path,
            yolo_detections=[{"label": "person", "bbox": [0.1, 0.1, 0.3, 0.5], "confidence": 0.9}],
        )
        original = self.config.input_frames_dir.parent / "media" / relative_path
        (original.with_suffix(".qwen.json")).write_text(
            json.dumps({"objects": [{"label": "person", "bbox": [0.1, 0.1, 0.3, 0.5], "confidence": 0.4}]}),
            encoding="utf-8",
        )
        (original.with_suffix(".hazard.json")).write_text(
            json.dumps({"hazards": [{"id": "D-5-1", "evidence": "未戴帽"},
                                    {"id": "D-9-9", "evidence": "伪造"}],
                        "overall_risk": "high"}, ensure_ascii=False),
            encoding="utf-8",
        )

    def test_annotate_produces_all_artifacts(self) -> None:
        self._seed()
        run_cycle(self.config, "annotate")
        for path in (
            self.config.review_labels_dir / "clip" / "frame_000001.txt",
            self.config.annotations_path,
            self.config.review_queue_path,
            self.config.hazard_report_path,
            self.config.workspace("hazards", "rejected.jsonl"),
        ):
            with self.subTest(path=str(path)):
                self.assertTrue(path.exists(), f"缺少产物 {path}")

    def test_rejected_hazard_is_queued_in_mother_project_format(self) -> None:
        self._seed()
        run_cycle(self.config, "annotate")
        report = read_json(self.config.hazard_report_path)
        self.assertEqual([h["hazard_id"] for h in report["hazards"]], ["D-5-1"])
        self.assertEqual([r["hazard_id"] for r in report["rejected"]], ["D-9-9"])
        self.assertEqual(report["rejected"][0]["reason"], "not_in_knowledge_base")

        queue = read_jsonl(self.config.review_queue_path)
        queued = [item for item in queue
                  if item["candidate"]["relative_path"] == "clip/frame_000001.png"]
        self.assertTrue(queued)
        self.assertIn("hazard_id_not_in_kb", queued[0]["review_reasons"])
        self.assertTrue(queued[0]["review_required"])
        # 与母项目 AnnotationRecord **双向兼容**：能无损 round-trip。
        record = AnnotationRecord.from_dict(queued[0])
        self.assertEqual(record.to_dict(), queued[0])

    def test_annotate_is_idempotent(self) -> None:
        self._seed()
        run_cycle(self.config, "annotate")
        first = read_json(self.config.hazard_report_path)
        run_cycle(self.config, "annotate")
        second = read_json(self.config.hazard_report_path)
        self.assertEqual(first, second)

    def test_prioritize_produces_queue(self) -> None:
        self._seed()
        run_cycle(self.config, "annotate")
        summary = run_cycle(self.config, "prioritize")
        queue_path = self.config.workspace("review", "priority_queue.jsonl")
        self.assertTrue(queue_path.exists())
        self.assertEqual(summary["queued"], 1)
        items = read_jsonl(queue_path)
        self.assertIn("score", items[0])
        self.assertIn("signals", items[0])


# --------------------------------------------------------------------------- #
# 7. CLI 退出码
# --------------------------------------------------------------------------- #
class CliExitCodeTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.config = _build_config(self.tmp, provider="mock")
        self.config_path = self.config.config_path

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _run(self, argv: list[str]) -> int:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return cli.main(argv)

    def test_annotate_and_prioritize_exit_zero(self) -> None:
        _write_candidate(
            self.config,
            "clip/frame_000001.png",
            yolo_detections=[{"label": "person", "bbox": [0.1, 0.1, 0.3, 0.5], "confidence": 0.9}],
        )
        original = self.config.input_frames_dir.parent / "media" / "clip" / "frame_000001.png"
        (original.with_suffix(".hazard.json")).write_text(
            json.dumps({"hazards": [{"id": "D-5-1", "evidence": "x"}]}), encoding="utf-8"
        )
        self.assertEqual(self._run(["annotate", "--config", str(self.config_path)]), cli.EXIT_READY)
        self.assertEqual(self._run(["prioritize", "--config", str(self.config_path)]), cli.EXIT_READY)

    def test_unimplemented_stage_exits_two(self) -> None:
        for command in ("train", "evaluate", "curate", "freeze-holdout"):
            with self.subTest(command=command):
                self.assertEqual(self._run([command, "--config", str(self.config_path)]),
                                 cli.EXIT_NOT_IMPLEMENTED)

    def test_prioritize_without_annotate_is_readable_failure(self) -> None:
        stderr = io.StringIO()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(stderr):
            code = cli.main(["prioritize", "--config", str(self.config_path)])
        self.assertEqual(code, cli.EXIT_STAGE_FAILED)
        message = stderr.getvalue()
        self.assertIn("annotate", message)  # 提示先跑 annotate
        self.assertIn("执行失败", message)

    def test_unknown_command_exits_two(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as ctx:
                cli.main(["bogus-command"])
        self.assertEqual(ctx.exception.code, cli.EXIT_NOT_IMPLEMENTED)


if __name__ == "__main__":
    unittest.main()
