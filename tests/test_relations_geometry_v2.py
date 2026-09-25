"""根治方案 v2 §4.4 / §8 阶段 2–3 单测：**射影判据重写** + ``relations`` 命令。

覆盖点（对应任务书「必须补的测试」）：

1. **地面视角**（脚点在载荷框下方）→ 命中（旧 ``contains`` 判据判不出的「死代码根因」）；
2. **俯拍视角**（人被吊物框包住）→ 命中（旧分支保留）；
3. **安全场景**（水平距离超出梯形坠落区）→ 不命中；
4. **垂直门控**（脚点在载荷底边**上方**）→ 不命中；
5. **净空比阈值 0.12**（脚点略低于 / 略高于阈值 → 不命中 / 命中）；
6. **三级输出**（|Δ| 近边界 → 疑似 / 需人工看，非二值）；
7. **梯形坠落区**随脚点下移**单调扩张**；
8. **时序持久**（连续 ≥2 帧命中才出结论，单帧不出）；
9. **``relations`` 命令**：吃标注框（非模型预测）→ 吐独立 ``relations.jsonl``，
   fail-loud、CLI / run_cycle 接线、**保持 evaluate 隔离断言**。

约束：不联网、不调用真实 VLM / ultralytics、不读真实素材；临时文件用 ``tempfile``；
不修改 ``traffic_cv_closed_loop/`` 与 QA 的 ``test_qa_*.py``。
"""

from __future__ import annotations

import contextlib
import io
import json
import math
import tempfile
import unittest
from pathlib import Path

from traffic_cv_closed_loop.config import ProjectConfig
from traffic_cv_closed_loop.io_utils import read_jsonl
from traffic_cv_closed_loop.models import AnnotationRecord, CandidateFrame, Detection

from industrial_hazard_coldstart import cli
from industrial_hazard_coldstart.cli import IMPLEMENTED_COMMANDS
from industrial_hazard_coldstart.layers import geometry
from industrial_hazard_coldstart.orchestrator import run_cycle
from industrial_hazard_coldstart.relations import (
    RELATIONS_FILENAME,
    relations_for_records,
    run_relations,
)

ROOT = Path(__file__).resolve().parents[1]
HOIST_CONFIG = ROOT / "own" / "hoist_project" / "config" / "project.hoist.json"
HOIST_PROMPT = ROOT / "own" / "hoist_project" / "assets" / "prompts" / "hoist_zh.md"
KB_PATH = ROOT / "hazard-detection" / "knowledge_base.json"

#: 图像尺寸标定基准（吊装视频原生分辨率）——与 geometry.REFERENCE_IMAGE_SIZE 一致。
IMAGE_W, IMAGE_H = 960, 540


# --------------------------------------------------------------------------- #
# 夹具
# --------------------------------------------------------------------------- #
def _det(label: str, bbox: tuple[float, float, float, float], conf: float = 0.9) -> Detection:
    return Detection(label=label, bbox=bbox, confidence=conf)


def _ground_view_dets() -> list[Detection]:
    """地面视角：人站在吊物下方的地面（脚点 = 框底中心，落在载荷底边之下）。"""
    return [
        _det("load", (0.4, 0.2, 0.6, 0.3)),
        _det("person", (0.45, 0.75, 0.55, 0.95)),
    ]


def _safe_dets() -> list[Detection]:
    """安全场景：人远离吊物正下方（水平距远超梯形坠落区半宽）。"""
    return [
        _det("load", (0.4, 0.2, 0.6, 0.3)),
        _det("person", (0.02, 0.75, 0.08, 0.95)),
    ]


def _frame(source_id: str, index: int, dets: list[Detection]) -> CandidateFrame:
    return CandidateFrame(
        source_id=source_id,
        relative_path=f"{source_id}/frame_{index:06d}.png",
        original_path="",
        extracted_path="",
        detections=dets,
        scene="hoist",
        metadata={"frame_index": index},
    )


def _make_config(tmp: Path) -> ProjectConfig:
    """造一个 mock 临时工程（绝不触碰真端点 / 真素材）。"""
    raw = json.loads(HOIST_CONFIG.read_text(encoding="utf-8"))
    raw["qwen"]["provider"] = "mock"
    raw["qwen"]["prompt_template"] = str(HOIST_PROMPT)
    raw["yolo"]["detector"] = "sidecar"
    raw["knowledge"]["kb_path"] = str(KB_PATH)
    config_dir = tmp / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "project.json"
    config_path.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
    return ProjectConfig.load(config_path)


def _record(
    source_id: str,
    index: int,
    *,
    annotations: list[Detection],
    predictions: list[Detection] | None = None,
) -> AnnotationRecord:
    """构造一条标注记录：``qwen_detections`` = 标注框，``candidate.detections`` = 模型预测。"""
    return AnnotationRecord(
        candidate=CandidateFrame(
            source_id=source_id,
            relative_path=f"{source_id}/frame_{index:06d}.png",
            original_path="",
            extracted_path="",
            detections=list(predictions or []),
            scene="hoist",
            metadata={"frame_index": index},
        ),
        qwen_detections=list(annotations),
        diff={"added": 0, "removed": 0, "adjusted": 0},
        review_required=False,
        review_reasons=[],
        prompt_path="",
        raw_response_path="",
        label_path="",
        dataset_ready=True,
    )


def _write_annotations(config: ProjectConfig, records: list[AnnotationRecord]) -> None:
    config.annotations_path.parent.mkdir(parents=True, exist_ok=True)
    with config.annotations_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------- #
# 1–4. 射影判据核心（地面 / 俯拍 / 安全 / 垂直门控）
# --------------------------------------------------------------------------- #
class ProjectiveCriterionTest(unittest.TestCase):
    def test_ground_view_detected(self) -> None:
        """地面视角：脚点（框底中心）落在载荷底边之下 → 命中（branch=ground）。"""
        findings = geometry.find_person_under_load(_ground_view_dets())
        self.assertEqual(len(findings), 1)
        finding = findings[0]
        self.assertEqual(finding.hazard_id, "D-5-3")
        self.assertEqual(finding.evidence["branch"], "ground")
        self.assertGreater(finding.margin, 0.0)
        self.assertGreater(finding.evidence["clearance_ratio"], geometry.CLEARANCE_RATIO_THRESHOLD)
        self.assertEqual(finding.verdict, geometry.VERDICT_CONFIRMED)

    def test_ground_view_is_dead_code_root_cause_fix(self) -> None:
        """旧 ``contains`` 判据（俯拍专属）对地面视角判不出；新判据判得出。"""
        dets = _ground_view_dets()
        load, person = dets[0], dets[1]
        self.assertFalse(geometry.contains(load, person))  # 旧判据：框中心不在吊物框内
        self.assertEqual(len(geometry.find_person_under_load(dets)), 1)  # 新判据：命中

    def test_overhead_view_detected(self) -> None:
        """俯拍视角：人被吊物框包住 → 命中（branch=overhead）。"""
        dets = [
            _det("load", (0.2, 0.2, 0.8, 0.8)),
            _det("person", (0.4, 0.4, 0.6, 0.6)),
        ]
        findings = geometry.find_person_under_load(dets)
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].evidence["branch"], "overhead")

    def test_safe_scene_no_hit(self) -> None:
        """水平距超出梯形坠落区半宽 → 不命中。"""
        self.assertEqual(geometry.find_person_under_load(_safe_dets()), [])

    def test_vertical_gate_blocks_person_above(self) -> None:
        """脚点在载荷底边**上方**（垂直门控不通过）→ 不命中。"""
        dets = [
            _det("load", (0.4, 0.2, 0.6, 0.3)),
            _det("person", (0.45, 0.02, 0.55, 0.15)),  # 脚点 y=0.15 在载荷底边 y=0.3 之上
        ]
        self.assertEqual(geometry.find_person_under_load(dets), [])


# --------------------------------------------------------------------------- #
# 5. 净空比阈值 0.12
# --------------------------------------------------------------------------- #
class ClearanceRatioTest(unittest.TestCase):
    """用「高个工人」（像素高 432，使垂直门控不成为限制）隔离净空比判据。"""

    def _case(self, ratio: float) -> list[Detection]:
        # 载荷底边 y=0.72（388.8px）；工人像素高 432px → gap = ratio * 432。
        load = _det("load", (0.4, 0.6, 0.6, 0.72))
        gap_px = ratio * 432.0
        foot_norm = (0.72 * IMAGE_H + gap_px) / IMAGE_H
        top_norm = foot_norm - 432.0 / IMAGE_H
        person = _det("person", (0.45, top_norm, 0.55, foot_norm))
        return [load, person]

    def test_below_threshold_no_hit(self) -> None:
        # r_clr = 0.10 < 0.12 → 不命中（垂直门控 43.2px ≥ 40px，故确由净空比拦下）。
        self.assertEqual(geometry.find_person_under_load(self._case(0.10)), [])

    def test_above_threshold_hits(self) -> None:
        findings = geometry.find_person_under_load(self._case(0.14))
        self.assertEqual(len(findings), 1)
        self.assertGreaterEqual(
            findings[0].evidence["clearance_ratio"], geometry.CLEARANCE_RATIO_THRESHOLD
        )
        self.assertGreaterEqual(findings[0].evidence["gap_px"], geometry.VERTICAL_GATE_PX)


# --------------------------------------------------------------------------- #
# 6. 三级输出（确定 / 疑似 / 需人工看）
# --------------------------------------------------------------------------- #
class ThreeTierOutputTest(unittest.TestCase):
    def test_verdict_for_margin_mapping(self) -> None:
        self.assertEqual(geometry.verdict_for_margin(0.30), geometry.VERDICT_CONFIRMED)
        self.assertEqual(geometry.verdict_for_margin(0.25), geometry.VERDICT_CONFIRMED)
        self.assertEqual(geometry.verdict_for_margin(0.15), geometry.VERDICT_SUSPECTED)
        self.assertEqual(geometry.verdict_for_margin(0.10), geometry.VERDICT_SUSPECTED)
        self.assertEqual(geometry.verdict_for_margin(0.05), geometry.VERDICT_REVIEW)
        self.assertEqual(geometry.verdict_for_margin(0.0), "")
        self.assertEqual(geometry.verdict_for_margin(-0.5), "")

    def _lateral_case(self, dx_px: float) -> list[Detection]:
        load = _det("load", (0.4, 0.2, 0.6, 0.3))
        center_x = 0.5 + dx_px / IMAGE_W
        person = _det("person", (center_x - 0.05, 0.75, center_x + 0.05, 0.95))
        return [load, person]

    def test_confirmed_when_directly_under(self) -> None:
        findings = geometry.find_person_under_load(self._lateral_case(0.0))
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].verdict, geometry.VERDICT_CONFIRMED)
        self.assertGreaterEqual(findings[0].margin, geometry.MARGIN_CONFIRMED)

    def test_suspected_near_lateral_boundary(self) -> None:
        # dx ≈ 0.85 × 坠落区半宽 → 决策余量 ≈ 0.15 → 疑似（非二值）。
        findings = geometry.find_person_under_load(self._lateral_case(107.5))
        self.assertEqual(len(findings), 1)
        self.assertGreaterEqual(findings[0].margin, geometry.MARGIN_SUSPECT)
        self.assertLess(findings[0].margin, geometry.MARGIN_CONFIRMED)
        self.assertEqual(findings[0].verdict, geometry.VERDICT_SUSPECTED)

    def test_needs_review_at_boundary(self) -> None:
        # dx ≈ 0.97 × 坠落区半宽 → 决策余量 < 0.10 → 需人工看。
        findings = geometry.find_person_under_load(self._lateral_case(121.9))
        self.assertEqual(len(findings), 1)
        self.assertGreater(findings[0].margin, 0.0)
        self.assertLess(findings[0].margin, geometry.MARGIN_SUSPECT)
        self.assertEqual(findings[0].verdict, geometry.VERDICT_REVIEW)


# --------------------------------------------------------------------------- #
# 7. 梯形坠落区单调扩张
# --------------------------------------------------------------------------- #
class FallZoneTest(unittest.TestCase):
    def _half_width(self, person_bbox: tuple[float, float, float, float]) -> float:
        dets = [_det("load", (0.4, 0.05, 0.6, 0.1)), _det("person", person_bbox)]
        findings = geometry.find_person_under_load(dets)
        self.assertEqual(len(findings), 1, "该用例应命中，方能读出坠落区半宽")
        return float(findings[0].evidence["fall_zone_half_width_px"])

    def test_half_width_expands_toward_ground(self) -> None:
        # 高个工人（像素高 432）抬高 ρ，使 δ_lat 超过 30px 下限 → 坠落区真正扩张。
        near = self._half_width((0.45, 0.0, 0.55, 0.8))   # 脚点离载荷较近
        far = self._half_width((0.45, 0.2, 0.55, 1.0))    # 脚点更接近地面
        self.assertGreater(far, near)


# --------------------------------------------------------------------------- #
# 8. 时序持久
# --------------------------------------------------------------------------- #
class TemporalPersistenceTest(unittest.TestCase):
    def test_persistent_hits_produce_finding(self) -> None:
        frames = [_frame("clip", i, _ground_view_dets()) for i in range(3)]
        findings = geometry.find_person_under_load_sequence(frames)
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].hazard_id, "D-5-3")
        self.assertGreaterEqual(
            findings[0].evidence["persistence"]["hit_frames"],
            geometry.DEFAULT_SEQUENCE_MIN_HITS,
        )

    def test_single_hit_not_enough(self) -> None:
        frames = [
            _frame("clip", 0, _ground_view_dets()),
            _frame("clip", 1, _safe_dets()),
            _frame("clip", 2, _safe_dets()),
        ]
        self.assertEqual(geometry.find_person_under_load_sequence(frames), [])


# --------------------------------------------------------------------------- #
# 9. relations 命令
# --------------------------------------------------------------------------- #
class RelationsCommandTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_eats_annotation_boxes_not_predictions(self) -> None:
        """用「标注框 vs 模型预测」互为反向的构造，证明吃的是**标注框**。"""
        config = _make_config(self.tmp)
        # A：标注框=危险（地面视角），模型预测=安全 → 应命中。
        # B：标注框=安全，模型预测=危险 → 不应命中（不得回落到模型预测）。
        records = [
            _record("clipA", 0, annotations=_ground_view_dets(), predictions=_safe_dets()),
            _record("clipB", 0, annotations=_safe_dets(), predictions=_ground_view_dets()),
        ]
        _write_annotations(config, records)
        run_relations(config)

        rows = read_jsonl(config.workspace("relations", RELATIONS_FILENAME))
        hit_paths = {row["relative_path"] for row in rows if row["hazard_id"] == "D-5-3"}
        self.assertIn("clipA/frame_000000.png", hit_paths)
        self.assertNotIn("clipB/frame_000000.png", hit_paths)

    def test_output_is_independent_relations_jsonl(self) -> None:
        config = _make_config(self.tmp)
        _write_annotations(
            config, [_record("clip", 0, annotations=_ground_view_dets())]
        )
        summary = run_relations(config)
        out_path = config.workspace("relations", RELATIONS_FILENAME)
        self.assertTrue(out_path.exists())
        self.assertEqual(summary["stage"], "relations")
        self.assertIn("不接入交付指标", summary["note"])

        rows = read_jsonl(out_path)
        self.assertTrue(rows)
        for row in rows:
            for key in ("hazard_id", "relation", "verdict", "margin", "scope", "relative_path"):
                self.assertIn(key, row)
        self.assertEqual(summary["relation_count"], len(rows))

    def test_missing_annotations_raises(self) -> None:
        config = _make_config(self.tmp)
        self.assertFalse(config.annotations_path.exists())
        with self.assertRaises(ValueError) as ctx:
            run_relations(config)
        self.assertIn("标注记录", str(ctx.exception))

    def test_empty_annotations_raises(self) -> None:
        config = _make_config(self.tmp)
        config.annotations_path.parent.mkdir(parents=True, exist_ok=True)
        config.annotations_path.write_text("", encoding="utf-8")
        with self.assertRaises(ValueError) as ctx:
            run_relations(config)
        self.assertIn("为空", str(ctx.exception))

    def test_relations_for_records_pure(self) -> None:
        rows = relations_for_records(
            [_record("clip", 0, annotations=_ground_view_dets())]
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["hazard_id"], "D-5-3")
        self.assertEqual(rows[0]["scope"], "frame")

    def test_command_is_implemented_and_dispatched(self) -> None:
        self.assertIn("relations", IMPLEMENTED_COMMANDS)
        config = _make_config(self.tmp)
        _write_annotations(
            config, [_record("clip", 0, annotations=_ground_view_dets())]
        )
        summary = run_cycle(config, "relations")
        self.assertEqual(summary["stage"], "relations")

    def test_cli_end_to_end(self) -> None:
        config = _make_config(self.tmp)
        _write_annotations(
            config, [_record("clip", 0, annotations=_ground_view_dets())]
        )
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = cli.main(["relations", "--config", str(config.config_path)])
        self.assertEqual(code, cli.EXIT_READY, err.getvalue())
        self.assertTrue(config.workspace("relations", RELATIONS_FILENAME).exists())

    def test_cli_missing_annotations_returns_4(self) -> None:
        config = _make_config(self.tmp)
        stderr = io.StringIO()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(stderr):
            code = cli.main(["relations", "--config", str(config.config_path)])
        self.assertEqual(code, cli.EXIT_STAGE_FAILED)
        self.assertIn("执行失败", stderr.getvalue())


# --------------------------------------------------------------------------- #
# 9'. 隔离断言：evaluate 不得 import layers（relations 是唯一入口）
# --------------------------------------------------------------------------- #
class RelationsIsolationTest(unittest.TestCase):
    def test_evaluate_module_still_does_not_import_layers(self) -> None:
        import industrial_hazard_coldstart.evaluate as evaluate_module

        source = Path(evaluate_module.__file__).read_text(encoding="utf-8")
        self.assertNotIn("from industrial_hazard_coldstart.layers", source)
        self.assertNotIn("RelationFinding", source)

    def test_relations_is_the_layers_entrypoint(self) -> None:
        import industrial_hazard_coldstart.relations as relations_module

        source = Path(relations_module.__file__).read_text(encoding="utf-8")
        self.assertIn("from industrial_hazard_coldstart.layers import geometry", source)


# --------------------------------------------------------------------------- #
# 10. D-4-3 判据透视局限：结论一律降级 needs_review
# --------------------------------------------------------------------------- #
class ObliqueLiftPerspectiveTest(unittest.TestCase):
    """``find_oblique_lift`` 用 **2D 画面角度**、未考虑透视与深度差 → 结论**一律**需人工确认。

    真实素材上，吊物框大（近）、吊钩框小（远）时深度差大，**物理垂直**也会呈大角度斜线；
    故不论角度多大，``verdict`` 必须恒为 ``needs_review``，**绝不给** ``confirmed_danger``。
    """

    def _oblique_dets(self, angle_deg: float) -> list[Detection]:
        """构造使 ``vertical_angle_deg ≈ angle_deg`` 的 load/hook（aspect_ratio=1）。"""
        dy = 0.15
        dx = dy * math.tan(math.radians(angle_deg))
        load_cx = max(0.5, 0.05 + dx)  # 保证 hook_cx = load_cx - dx ≥ 0.05
        load_cy = 0.6
        hook_cx = load_cx - dx
        hook_cy = load_cy - dy
        return [
            _det("load", (load_cx - 0.1, load_cy - 0.05, load_cx + 0.1, load_cy + 0.05)),
            _det("hook", (hook_cx - 0.02, hook_cy - 0.02, hook_cx + 0.02, hook_cy + 0.02)),
        ]

    def test_extreme_angle_still_needs_review(self) -> None:
        findings = geometry.find_oblique_lift(self._oblique_dets(80.0))
        self.assertEqual(len(findings), 1)
        self.assertGreater(findings[0].evidence["angle_deg"], 70.0)
        self.assertEqual(findings[0].verdict, geometry.VERDICT_REVIEW)

    def test_moderate_angle_also_needs_review(self) -> None:
        findings = geometry.find_oblique_lift(self._oblique_dets(30.0))
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].verdict, geometry.VERDICT_REVIEW)

    def test_never_confirmed_regardless_of_angle(self) -> None:
        for angle in (20.0, 30.0, 45.0, 60.0, 75.0, 80.0):
            with self.subTest(angle=angle):
                findings = geometry.find_oblique_lift(self._oblique_dets(angle))
                self.assertEqual(len(findings), 1)
                self.assertEqual(findings[0].verdict, geometry.VERDICT_REVIEW)
                self.assertNotEqual(findings[0].verdict, geometry.VERDICT_CONFIRMED)

    def test_evidence_has_perspective_warning(self) -> None:
        findings = geometry.find_oblique_lift(self._oblique_dets(60.0))
        self.assertEqual(len(findings), 1)
        warning = findings[0].evidence.get("perspective_warning")
        self.assertIsInstance(warning, str)
        self.assertIn("透视", warning)
        self.assertIn("深度", warning)
        self.assertEqual(warning, geometry.OBLIQUE_PERSPECTIVE_WARNING)

    def test_margin_still_reflects_angle(self) -> None:
        small = geometry.find_oblique_lift(self._oblique_dets(20.0))[0]
        large = geometry.find_oblique_lift(self._oblique_dets(80.0))[0]
        self.assertGreater(large.margin, small.margin)

    def test_d53_criterion_unaffected(self) -> None:
        """D-5-3（射影解）不受本次 D-4-3 降级影响，仍能给出 confirmed_danger。"""
        findings = geometry.find_person_under_load(_ground_view_dets())
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].hazard_id, "D-5-3")
        self.assertEqual(findings[0].verdict, geometry.VERDICT_CONFIRMED)


if __name__ == "__main__":
    unittest.main()
