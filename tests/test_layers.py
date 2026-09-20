"""T05.3–T05.5 单测：几何关系层 / 时序层 / 图像质量层 + OCR 占位。

覆盖点：

1. **几何层**：``contains`` / ``vertical_angle_deg`` / ``find_person_under_load``（D-5-3）/
   ``find_oblique_lift``（D-4-3）；
2. **时序层**：静止 → ``suspended``、运动 → ``moving``、帧数不足 → 无结论；
3. **图像质量层**：``brightness_score`` 复用 ``histogram16``（黑≈0 / 白≈1，PGM 与 PNG 两路）；
4. **OCR 占位**：``ocr_weight_marking`` 抛 ``NotImplementedError`` 且带许可说明；
5. **留位隔离**：三层结论**不接入** ``EvalReport`` 验收判定（结构上无可达路径）。

约束：不联网、不调用真实 VLM / ultralytics、不读真实素材；临时文件用 ``tempfile``。
"""

from __future__ import annotations

import struct
import tempfile
import unittest
import zlib
from dataclasses import fields
from pathlib import Path

from industrial_hazard_coldstart import taxonomy
from industrial_hazard_coldstart.layers import geometry, semantic, temporal
from industrial_hazard_coldstart.models import (
    CandidateFrame,
    Detection,
    EvalReport,
    RelationFinding,
    TemporalFinding,
)


# --------------------------------------------------------------------------- #
# 夹具
# --------------------------------------------------------------------------- #
def _write_pgm(path: Path, value: int, width: int = 8, height: int = 8) -> Path:
    """写一张纯灰 P5 PGM（母项目 imageio 可直接读）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(f"P5\n{width} {height}\n255\n".encode("ascii") + bytes([value]) * (width * height))
    return path


def _write_png(path: Path, color: tuple[int, int, int]) -> Path:
    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    width = height = 8
    raw = (b"\x00" + bytes(color) * width) * height
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )
    return path


def _frame(source_id: str, index: int, load_bbox: tuple[float, float, float, float]) -> CandidateFrame:
    return CandidateFrame(
        source_id=source_id,
        relative_path=f"{source_id}/frame_{index:06d}.png",
        original_path="",
        extracted_path="",
        detections=[Detection(label="load", bbox=load_bbox, confidence=0.9)],
        scene="hoist",
        metadata={"frame_index": index},
    )


# --------------------------------------------------------------------------- #
# 几何层
# --------------------------------------------------------------------------- #
class GeometryLayerTest(unittest.TestCase):
    def test_contains_true(self) -> None:
        load = Detection(label="load", bbox=(0.2, 0.2, 0.8, 0.8), confidence=0.9)
        person = Detection(label="person", bbox=(0.4, 0.4, 0.6, 0.6), confidence=0.8)
        self.assertTrue(geometry.contains(load, person))

    def test_contains_false_when_outside(self) -> None:
        load = Detection(label="load", bbox=(0.0, 0.0, 0.3, 0.3), confidence=0.9)
        person = Detection(label="person", bbox=(0.6, 0.6, 0.9, 0.9), confidence=0.8)
        self.assertFalse(geometry.contains(load, person))

    def test_vertical_angle_zero_when_aligned(self) -> None:
        load = Detection(label="load", bbox=(0.4, 0.6, 0.6, 0.8), confidence=0.9)
        hook = Detection(label="hook", bbox=(0.45, 0.2, 0.55, 0.3), confidence=0.9)
        self.assertAlmostEqual(geometry.vertical_angle_deg(load, hook), 0.0, places=3)

    def test_vertical_angle_oblique(self) -> None:
        load = Detection(label="load", bbox=(0.6, 0.6, 0.8, 0.8), confidence=0.9)
        hook = Detection(label="hook", bbox=(0.1, 0.1, 0.2, 0.2), confidence=0.9)
        angle = geometry.vertical_angle_deg(load, hook)
        self.assertGreater(angle, geometry.OBLIQUE_ANGLE_THRESHOLD_DEG)

    def test_find_person_under_load(self) -> None:
        dets = [
            Detection(label="load", bbox=(0.2, 0.2, 0.8, 0.8), confidence=0.9),
            Detection(label="person", bbox=(0.4, 0.4, 0.6, 0.6), confidence=0.8),
        ]
        findings = geometry.find_person_under_load(dets)
        self.assertEqual(len(findings), 1)
        self.assertIsInstance(findings[0], RelationFinding)
        self.assertEqual(findings[0].hazard_id, "D-5-3")
        self.assertEqual(findings[0].relation, "person_under_load")
        self.assertEqual(len(findings[0].supporting), 2)

    def test_find_oblique_lift(self) -> None:
        dets = [
            Detection(label="load", bbox=(0.6, 0.6, 0.8, 0.8), confidence=0.9),
            Detection(label="rigging", bbox=(0.1, 0.1, 0.2, 0.2), confidence=0.85),
        ]
        findings = geometry.find_oblique_lift(dets)
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].hazard_id, "D-4-3")
        self.assertEqual(findings[0].relation, "oblique_lift")

    def test_no_finding_when_clean(self) -> None:
        dets = [
            Detection(label="load", bbox=(0.4, 0.6, 0.6, 0.8), confidence=0.9),
            Detection(label="person", bbox=(0.05, 0.05, 0.15, 0.15), confidence=0.9),
            Detection(label="hook", bbox=(0.45, 0.2, 0.55, 0.3), confidence=0.9),
        ]
        self.assertEqual(geometry.find_relations(dets), [])

    def test_layer_note_marks_not_integrated(self) -> None:
        self.assertIn("不接入交付指标", geometry.LAYER_NOTE)


# --------------------------------------------------------------------------- #
# 时序层
# --------------------------------------------------------------------------- #
class TemporalLayerTest(unittest.TestCase):
    def test_suspended_when_static(self) -> None:
        frames = [_frame("clip", i, (0.4, 0.4, 0.6, 0.6)) for i in range(4)]
        findings = temporal.detect_suspended_load(frames)
        self.assertEqual(len(findings), 1)
        self.assertIsInstance(findings[0], TemporalFinding)
        self.assertEqual(findings[0].hazard_id, "D-3-6")
        self.assertEqual(findings[0].finding, "suspended")
        self.assertEqual(findings[0].frame_indices, [0, 1, 2, 3])

    def test_moving_when_displaced(self) -> None:
        # 相邻帧中心位移 0.15（≤ 回落匹配上限 0.25）→ 仍成一条轨迹；
        # 但轨迹内最大位移 0.30 > 0.02 → 判「运动」。
        frames = [
            _frame("clip", 0, (0.25, 0.40, 0.35, 0.50)),
            _frame("clip", 1, (0.40, 0.40, 0.50, 0.50)),
            _frame("clip", 2, (0.55, 0.40, 0.65, 0.50)),
        ]
        findings = temporal.detect_suspended_load(frames)
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].finding, "moving")

    def test_too_few_frames_no_finding(self) -> None:
        frames = [_frame("clip", 0, (0.4, 0.4, 0.6, 0.6)), _frame("clip", 1, (0.4, 0.4, 0.6, 0.6))]
        self.assertEqual(temporal.detect_suspended_load(frames), [])

    def test_layer_note_marks_not_integrated(self) -> None:
        self.assertIn("不接入交付指标", temporal.LAYER_NOTE)


# --------------------------------------------------------------------------- #
# 图像质量层 + OCR 占位
# --------------------------------------------------------------------------- #
class SemanticLayerTest(unittest.TestCase):
    def test_brightness_black_pgm(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_pgm(Path(tmp) / "black.pgm", 0)
            self.assertAlmostEqual(semantic.brightness_score(path), 0.0, places=6)

    def test_brightness_white_pgm(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_pgm(Path(tmp) / "white.pgm", 255)
            # 全 255 → 最高桶（下标 15）→ 15/15 = 1.0
            self.assertAlmostEqual(semantic.brightness_score(path), 1.0, places=6)

    def test_brightness_black_png(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_png(Path(tmp) / "black.png", (0, 0, 0))
            self.assertAlmostEqual(semantic.brightness_score(path), 0.0, places=6)

    def test_brightness_white_png(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_png(Path(tmp) / "white.png", (255, 255, 255))
            self.assertAlmostEqual(semantic.brightness_score(path), 1.0, places=6)

    def test_brightness_between(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dark = _write_png(Path(tmp) / "d.png", (20, 20, 20))
            bright = _write_png(Path(tmp) / "b.png", (200, 200, 200))
            self.assertLess(semantic.brightness_score(dark), semantic.brightness_score(bright))

    def test_load_gray_bytes_pgm(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_pgm(Path(tmp) / "g.pgm", 128, 4, 4)
            gray = semantic.load_gray_bytes(path)
            self.assertEqual(len(gray), 16)
            self.assertEqual(set(gray), {128})

    def test_ocr_weight_marking_is_placeholder(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_png(Path(tmp) / "plate.png", (200, 200, 200))
            with self.assertRaises(NotImplementedError) as ctx:
                semantic.ocr_weight_marking(path)
            message = str(ctx.exception)
            self.assertIn("许可", message)
            self.assertIn("本期留位", message)

    def test_layer_note_marks_not_integrated(self) -> None:
        self.assertIn("不接入交付指标", semantic.LAYER_NOTE)


# --------------------------------------------------------------------------- #
# 留位隔离：三层结论不得混进 EvalReport
# --------------------------------------------------------------------------- #
class LayerIsolationTest(unittest.TestCase):
    def test_eval_report_has_no_relation_fields(self) -> None:
        names = {field.name for field in fields(EvalReport)}
        self.assertEqual(
            names,
            {"dataset_version", "holdout_hash", "per_class", "tier_verdict", "consumption_rate"},
        )
        self.assertNotIn("relations", names)
        self.assertNotIn("temporal", names)

    def test_taxonomy_marks_layers_as_not_supported(self) -> None:
        # 本期交付物只有检测权重：几何 / 时序层明确「留位不实现」。
        self.assertFalse(taxonomy.LAYER_SUPPORTED[taxonomy.LAYER_GEOMETRY])
        self.assertFalse(taxonomy.LAYER_SUPPORTED[taxonomy.LAYER_TEMPORAL])
        self.assertFalse(taxonomy.LAYER_SUPPORTED[taxonomy.LAYER_OCR])

    def test_evaluate_module_does_not_import_layers(self) -> None:
        import industrial_hazard_coldstart.evaluate as evaluate_module

        source = Path(evaluate_module.__file__).read_text(encoding="utf-8")
        self.assertNotIn("from industrial_hazard_coldstart.layers", source)
        self.assertNotIn("RelationFinding", source)
        self.assertNotIn("TemporalFinding", source)


if __name__ == "__main__":
    unittest.main()
