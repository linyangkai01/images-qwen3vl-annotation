"""T03 单测：两段式标注（mock 端到端）+ 语义层受控词表 / KB 回填 + 主动学习。

覆盖点（对应任务书 T03.5 判据）：

1. **提示词渲染** —— 母项目 ``render_annotation_prompt`` 能成功渲染吊装提示词；
   提示词含受控词表要求，且其 S1 隐患清单与知识库逐条一致（27 条）；
2. **KB 本地回填** —— ``category / norm_reference / action`` 全部来自本地知识库，
   **抽样比对零差异**；
3. **KB 外 / 场景外的 ``hazard_id`` 被拒** —— 不得凭空出现在报告里，且被记入复核队列；
4. **主动学习排序** —— 三路不确定信号齐备，队列按分数降序；
5. **mock 端到端** —— ``annotate`` 产出 ``review/labels/`` + ``annotations.jsonl`` +
   ``review_queue.jsonl``；``prioritize`` 产出 ``priority_queue.jsonl``。

约束：不修改任何生产代码；只走 mock 路径；不联网、不调用真实 VLM；不读真实素材；
临时文件用 ``tempfile``。
"""

from __future__ import annotations

import json
import struct
import tempfile
import unittest
import zlib
from pathlib import Path

from traffic_cv_closed_loop.config import ProjectConfig
from traffic_cv_closed_loop.io_utils import read_json, read_jsonl
from traffic_cv_closed_loop.models import AnnotationRecord, CandidateFrame, Detection
from traffic_cv_closed_loop.prompting import render_annotation_prompt

from industrial_hazard_coldstart import taxonomy
from industrial_hazard_coldstart.active_learning import (
    build_priority_queue,
    score_uncertainty,
)
from industrial_hazard_coldstart.cli import IMPLEMENTED_COMMANDS
from industrial_hazard_coldstart.knowledge import (
    backfill_hazards,
    load_knowledge_base,
    resolve_hazard,
)
from industrial_hazard_coldstart.orchestrator import run_cycle

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


def _make_config(tmp: Path) -> ProjectConfig:
    """在临时目录写一份 mock 吊装配置并加载（KB 显式指向仓库内知识库）。"""
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


def _make_candidate(
    config: ProjectConfig,
    relative_path: str,
    *,
    yolo_detections: list[dict] | None = None,
    qwen_objects: list[dict] | None = None,
    hazard_report: dict | None = None,
) -> dict:
    """造一条候选帧 + 三个侧车（detections / qwen / hazard），返回候选 dict。

    - 侧车放在 ``original_path`` 旁边（``MockQwenAnnotator`` 的读取约定）；
    - ``extracted_path`` 指向真实存在的 PNG（``run_annotation`` 会拷贝它）。
    """
    original = config.input_frames_dir.parent / "media" / relative_path
    extracted = config.input_frames_dir / relative_path
    _write_png(original)
    _write_png(extracted)
    (original.with_suffix(".detections.json")).write_text(
        json.dumps({"objects": yolo_detections or []}, ensure_ascii=False), encoding="utf-8"
    )
    (original.with_suffix(".qwen.json")).write_text(
        json.dumps({"objects": qwen_objects or []}, ensure_ascii=False), encoding="utf-8"
    )
    if hazard_report is not None:
        (original.with_suffix(".hazard.json")).write_text(
            json.dumps(hazard_report, ensure_ascii=False), encoding="utf-8"
        )
    return {
        "source_id": Path(relative_path).parts[0],
        "relative_path": relative_path,
        "original_path": str(original),
        "extracted_path": str(extracted),
        "detections": yolo_detections or [],
        "scene": "hoist",
        "metadata": {},
    }


def _write_candidates(config: ProjectConfig, candidates: list[dict]) -> None:
    config.candidates_path.parent.mkdir(parents=True, exist_ok=True)
    with config.candidates_path.open("w", encoding="utf-8", newline="\n") as handle:
        for item in candidates:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------- #
# 1. 提示词渲染（T03.1）
# --------------------------------------------------------------------------- #
class PromptRenderingTest(unittest.TestCase):
    def test_prompt_renders_with_mother_project(self) -> None:
        config = ProjectConfig.load(HOIST_CONFIG)
        template = HOIST_PROMPT.read_text(encoding="utf-8")
        candidate = CandidateFrame(
            source_id="clip",
            relative_path="clip/frame_000001.png",
            original_path="",
            extracted_path="",
            detections=[Detection(label="person", bbox=(0.1, 0.1, 0.3, 0.5), confidence=0.7)],
            scene="hoist",
        )
        rendered = render_annotation_prompt(config, candidate, template)
        # 四个占位符全部被替换。
        for placeholder in ("{scene}", "{labels}", "{candidate_path}", "{prior_detections}"):
            self.assertNotIn(placeholder, rendered)
        self.assertIn("clip/frame_000001.png", rendered)
        self.assertIn('"objects"', rendered)  # 花括号已转义，format 后还原为 JSON 示意
        # 受控词表要求与「禁止输出规范号」的硬约束。
        self.assertIn("受控词表", rendered)
        self.assertIn("规范号", rendered)

    def test_prompt_controlled_vocab_matches_kb_s1(self) -> None:
        """提示词里的 S1 隐患 id 必须与知识库 S1 逐条一致（27 条，顺序一致）。"""
        kb = load_knowledge_base(KB_PATH)
        kb_ids = kb.scene_hazard_ids("S1")
        self.assertEqual(len(kb_ids), 27)
        template = HOIST_PROMPT.read_text(encoding="utf-8")
        for hazard_id in kb_ids:
            self.assertIn(hazard_id, template, f"提示词缺少受控 id {hazard_id}")
        self.assertEqual(tuple(kb_ids), taxonomy.S1_HAZARD_IDS)


# --------------------------------------------------------------------------- #
# 2. 语义层：受控词表 + KB 本地回填（T03.3）
# --------------------------------------------------------------------------- #
class KnowledgeBackfillTest(unittest.TestCase):
    def setUp(self) -> None:
        self.kb = load_knowledge_base(KB_PATH)
        self.allowed = frozenset(self.kb.allowed_hazard_ids("hoist"))

    def test_known_hazard_backfilled_from_kb(self) -> None:
        resolved, rejected = backfill_hazards(
            {"hazards": [{"id": "D-5-1", "evidence": "画面左侧人员头部未见安全帽"}]},
            self.kb,
            self.allowed,
            source_frame="clip/frame_000001.png",
        )
        self.assertEqual(rejected, [])
        self.assertEqual(len(resolved), 1)
        item = resolved[0]
        entry = self.kb.get("D-5-1")
        self.assertEqual(item["hazard_id"], "D-5-1")
        self.assertEqual(item["evidence"], "画面左侧人员头部未见安全帽")
        # 回填字段与 KB 零差异。
        self.assertEqual(item["category"], entry.category_name)
        self.assertEqual(item["norm_reference"], entry.norm_reference)
        self.assertEqual(item["action"], entry.action)
        self.assertEqual(item["severity"], entry.severity)

    def test_unknown_hazard_id_is_rejected(self) -> None:
        resolved, rejected = backfill_hazards(
            {"hazards": [{"id": "X-9-9", "evidence": "凭空捏造"}]},
            self.kb,
            self.allowed,
            source_frame="clip/frame_000002.png",
        )
        self.assertEqual(resolved, [])
        self.assertEqual(len(rejected), 1)
        self.assertEqual(rejected[0]["reason"], "not_in_knowledge_base")
        self.assertEqual(rejected[0]["hazard_id"], "X-9-9")

    def test_out_of_scene_hazard_is_rejected(self) -> None:
        """G-1-1 在 KB（高处作业）里，但不属吊装场景受控词表 → 同样拒绝。"""
        self.assertIsNotNone(self.kb.get("G-1-1"))  # 确认它确实在 KB 内
        resolved, rejected = backfill_hazards(
            {"hazards": [{"id": "G-1-1", "evidence": "高处作业隐患混入"}]},
            self.kb,
            self.allowed,
        )
        self.assertEqual(resolved, [])
        self.assertEqual(rejected[0]["reason"], "out_of_scene_vocabulary")

    def test_model_cannot_inject_norm_reference(self) -> None:
        """模型即便自报 category/norm_reference/action，也必须被丢弃、只认 KB。"""
        entry = self.kb.get("D-1-1")
        resolved, _ = backfill_hazards(
            {
                "hazards": [
                    {
                        "id": "D-1-1",
                        "evidence": "绳索呈 X 形交叉",
                        "category": "伪造类别",
                        "norm_reference": "GB 9999-9999",
                        "action": "伪造措施",
                    }
                ]
            },
            self.kb,
            self.allowed,
        )
        self.assertEqual(resolved[0]["category"], entry.category_name)
        self.assertEqual(resolved[0]["norm_reference"], entry.norm_reference)
        self.assertEqual(resolved[0]["action"], entry.action)
        self.assertNotIn("GB 9999-9999", json.dumps(resolved, ensure_ascii=False))

    def test_resolve_missing_id(self) -> None:
        good, bad = resolve_hazard("", "无 id", self.kb, self.allowed)
        self.assertIsNone(good)
        self.assertEqual(bad["reason"], "missing_hazard_id")

    def test_empty_allowed_set_rejects_everything(self) -> None:
        resolved, rejected = backfill_hazards(
            {"hazards": [{"id": "D-5-1", "evidence": "x"}]}, self.kb, frozenset()
        )
        self.assertEqual(resolved, [])
        self.assertEqual(rejected[0]["reason"], "out_of_scene_vocabulary")


# --------------------------------------------------------------------------- #
# 3. mock 端到端（T03.2 + T03.3）
# --------------------------------------------------------------------------- #
class AnnotateStageTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.config = _make_config(self.tmp)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_mock_end_to_end_produces_artifacts(self) -> None:
        candidates = [
            _make_candidate(
                self.config,
                "clip/frame_000001.png",
                yolo_detections=[
                    {"label": "person", "bbox": [0.1, 0.1, 0.3, 0.5], "confidence": 0.7}
                ],
                qwen_objects=[
                    {"label": "person", "bbox": [0.1, 0.1, 0.3, 0.5], "confidence": 0.9}
                ],
                hazard_report={
                    "hazards": [{"id": "D-5-1", "evidence": "人员头部未见安全帽"}],
                    "overall_risk": "high",
                },
            )
        ]
        _write_candidates(self.config, candidates)

        summary = run_cycle(self.config, "annotate")

        # T03.2 判据：labels/ + annotations.jsonl + review_queue.jsonl。
        self.assertTrue(self.config.review_labels_dir.exists())
        self.assertTrue((self.config.review_labels_dir / "clip" / "frame_000001.txt").exists())
        self.assertTrue(self.config.annotations_path.exists())
        self.assertTrue(self.config.review_queue_path.exists())
        self.assertEqual(summary["annotation"]["annotated_candidates"], 1)

        # T03.3 判据：hazard 报告含 hazard_id + evidence，回填字段来自 KB。
        report = read_json(self.config.hazard_report_path)
        self.assertEqual(report["scene_type"], "hoist")
        self.assertEqual(report["kb_scene_id"], "S1")
        self.assertEqual(report["counts"]["hazards_resolved"], 1)
        hazard = report["hazards"][0]
        self.assertEqual(hazard["hazard_id"], "D-5-1")
        self.assertEqual(hazard["evidence"], "人员头部未见安全帽")

        kb = load_knowledge_base(KB_PATH)
        entry = kb.get("D-5-1")
        self.assertEqual(hazard["category"], entry.category_name)
        self.assertEqual(hazard["norm_reference"], entry.norm_reference)
        self.assertEqual(hazard["action"], entry.action)
        self.assertEqual(report["overall_risk"], "high")

    def test_backfill_matches_kb_with_zero_diff_sampling(self) -> None:
        candidates = [
            _make_candidate(
                self.config,
                "clip/frame_000001.png",
                yolo_detections=[
                    {"label": "rigging", "bbox": [0.2, 0.2, 0.5, 0.6], "confidence": 0.6}
                ],
                qwen_objects=[
                    {"label": "rigging", "bbox": [0.2, 0.2, 0.5, 0.6], "confidence": 0.9}
                ],
                hazard_report={
                    "hazards": [
                        {"id": "D-1-1", "evidence": "绳索交叉"},
                        {"id": "D-2-3", "evidence": "托盘直接吊运"},
                        {"id": "D-5-2", "evidence": "无反光背心"},
                    ]
                },
            )
        ]
        _write_candidates(self.config, candidates)
        run_cycle(self.config, "annotate")

        report = read_json(self.config.hazard_report_path)
        kb = load_knowledge_base(KB_PATH)
        self.assertEqual(len(report["hazards"]), 3)
        for hazard in report["hazards"]:
            entry = kb.get(hazard["hazard_id"])
            self.assertIsNotNone(entry)
            # 抽样比对零差异：三个回填字段逐一等于 KB。
            self.assertEqual(hazard["category"], entry.category_name)
            self.assertEqual(hazard["norm_reference"], entry.norm_reference)
            self.assertEqual(hazard["action"], entry.action)
            # 且必须逐字出现在 KB 中（防伪造）。
            self.assertIn(hazard["norm_reference"], {entry.norm_reference})

    def test_out_of_kb_hazard_rejected_and_queued(self) -> None:
        candidates = [
            _make_candidate(
                self.config,
                "clip/frame_000001.png",
                yolo_detections=[
                    {"label": "person", "bbox": [0.1, 0.1, 0.3, 0.5], "confidence": 0.9}
                ],
                qwen_objects=[
                    {"label": "person", "bbox": [0.1, 0.1, 0.3, 0.5], "confidence": 0.9}
                ],
                hazard_report={
                    "hazards": [
                        {"id": "D-5-1", "evidence": "有效隐患"},
                        {"id": "D-9-9", "evidence": "KB 外的伪造隐患"},
                    ]
                },
            )
        ]
        _write_candidates(self.config, candidates)
        summary = run_cycle(self.config, "annotate")

        report = read_json(self.config.hazard_report_path)
        # 有效隐患保留，伪造隐患不得出现在 hazards 里。
        self.assertEqual([item["hazard_id"] for item in report["hazards"]], ["D-5-1"])
        rejected_ids = [item["hazard_id"] for item in report["rejected"]]
        self.assertEqual(rejected_ids, ["D-9-9"])
        self.assertEqual(report["rejected"][0]["reason"], "not_in_knowledge_base")
        # 不得凭空出现 KB 之外的规范号。
        self.assertNotIn("D-9-9", json.dumps(report["hazards"], ensure_ascii=False))

        # 被拒帧必须记入复核队列。
        self.assertEqual(summary["frames_with_rejected_hazard"], 1)
        queue = read_jsonl(self.config.review_queue_path)
        queued = [item for item in queue if item["candidate"]["relative_path"] == "clip/frame_000001.png"]
        self.assertTrue(queued)
        self.assertIn("hazard_id_not_in_kb", queued[0]["review_reasons"])
        self.assertTrue(queued[0]["review_required"])
        # 独立的拒绝清单也落盘。
        self.assertTrue((self.config.workspace("hazards", "rejected.jsonl")).exists())


# --------------------------------------------------------------------------- #
# 4. 主动学习（T03.4）
# --------------------------------------------------------------------------- #
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


class ActiveLearningTest(unittest.TestCase):
    def test_three_signals_are_reported(self) -> None:
        record = _make_record(
            "clip/a.png",
            confidences=[0.3],
            diff={"added": 1, "removed": 1, "adjusted": 1},
            review_required=True,
            review_reasons=["qwen_added_objects"],
        )
        item = score_uncertainty(record)
        self.assertIn("low_confidence", item.signals)
        self.assertIn("channel_disagreement", item.signals)
        self.assertIn("review_reason", item.signals)
        self.assertGreater(item.score, 0.0)
        self.assertLessEqual(item.score, 1.0)
        # 三路信号各自都应为正。
        self.assertGreater(item.signals["low_confidence"], 0.0)
        self.assertGreater(item.signals["channel_disagreement"], 0.0)
        self.assertEqual(item.signals["review_reason"], 1.0)
        self.assertTrue(item.reasons)

    def test_clean_record_scores_zero(self) -> None:
        record = _make_record("clip/clean.png", confidences=[0.95])
        item = score_uncertainty(record)
        self.assertEqual(item.score, 0.0)
        self.assertEqual(item.reasons, [])

    def test_priority_queue_sorted_desc(self) -> None:
        low = _make_record("clip/low.png", confidences=[0.3], review_required=True,
                           review_reasons=["low_confidence"])
        mid = _make_record("clip/mid.png", confidences=[0.9],
                           diff={"added": 1, "removed": 0, "adjusted": 0},
                           review_required=True, review_reasons=["qwen_added_objects"])
        clean = _make_record("clip/clean.png", confidences=[0.95])
        queue = build_priority_queue([clean, low, mid])
        self.assertEqual([item.relative_path for item in queue],
                         ["clip/low.png", "clip/mid.png", "clip/clean.png"])
        scores = [item.score for item in queue]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_top_k_truncates(self) -> None:
        records = [
            _make_record(f"clip/f{index}.png", confidences=[0.3 + 0.1 * index])
            for index in range(4)
        ]
        queue = build_priority_queue(records, top_k=2)
        self.assertEqual(len(queue), 2)

    def test_deterministic_tie_break_by_path(self) -> None:
        first = _make_record("clip/b.png", confidences=[0.4], review_required=True)
        second = _make_record("clip/a.png", confidences=[0.4], review_required=True)
        queue = build_priority_queue([first, second])
        self.assertEqual([item.relative_path for item in queue], ["clip/a.png", "clip/b.png"])

    def test_prioritize_stage_writes_queue(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = _make_config(Path(tmp))
            candidates = [
                _make_candidate(
                    config,
                    "clip/frame_000001.png",
                    yolo_detections=[
                        {"label": "person", "bbox": [0.1, 0.1, 0.3, 0.5], "confidence": 0.9}
                    ],
                    qwen_objects=[
                        {"label": "person", "bbox": [0.1, 0.1, 0.3, 0.5], "confidence": 0.4}
                    ],
                    hazard_report={"hazards": [{"id": "D-5-1", "evidence": "x"}]},
                ),
                _make_candidate(
                    config,
                    "clip/frame_000002.png",
                    yolo_detections=[
                        {"label": "person", "bbox": [0.1, 0.1, 0.3, 0.5], "confidence": 0.9}
                    ],
                    qwen_objects=[
                        {"label": "person", "bbox": [0.1, 0.1, 0.3, 0.5], "confidence": 0.95}
                    ],
                    hazard_report={"hazards": []},
                ),
            ]
            _write_candidates(config, candidates)
            run_cycle(config, "annotate")
            summary = run_cycle(config, "prioritize")

            queue_path = config.workspace("review", "priority_queue.jsonl")
            self.assertTrue(queue_path.exists())
            self.assertEqual(summary["queued"], 2)
            items = read_jsonl(queue_path)
            self.assertEqual(items[0]["relative_path"], "clip/frame_000001.png")  # 低置信度优先
            self.assertGreater(items[0]["score"], items[1]["score"])
            self.assertIn("signals", items[0])
            self.assertIn("score", items[0])


# --------------------------------------------------------------------------- #
# 5. CLI 接线
# --------------------------------------------------------------------------- #
class CliWiringTest(unittest.TestCase):
    def test_annotate_and_prioritize_are_implemented(self) -> None:
        self.assertIn("annotate", IMPLEMENTED_COMMANDS)
        self.assertIn("prioritize", IMPLEMENTED_COMMANDS)


if __name__ == "__main__":
    unittest.main()
