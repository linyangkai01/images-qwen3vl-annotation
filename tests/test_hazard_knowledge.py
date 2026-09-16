from __future__ import annotations

import unittest
from pathlib import Path

from hazard_annotation.knowledge import KnowledgeBase, KnowledgeCompressor, SceneRouter
from hazard_annotation.models import HazardEvent, HazardReport
from traffic_cv_closed_loop.models import Detection

ROOT = Path(__file__).resolve().parents[1]
KB_PATH = ROOT / "hazard-detection" / "knowledge_base.json"


class KnowledgeBaseTest(unittest.TestCase):
    def setUp(self) -> None:
        self.kb = KnowledgeBase.load(KB_PATH)

    def test_id_index_covers_all_hazards(self) -> None:
        # The shipped knowledge base actually contains 58 hazards (the meta
        # ``total_hazards`` field still says 53 and is stale); the index must
        # cover every normalised definition including the S3 hazard T-1.
        self.assertEqual(len(self.kb.id_index), len(self.kb.hazards))
        self.assertEqual(self.kb.total, 58)
        for hazard_id in ("D-1-1", "G-2-2", "T-1"):
            self.assertIn(hazard_id, self.kb.id_index)

    def test_normalisation_inherits_category(self) -> None:
        definition = self.kb.get("D-1-1")
        self.assertIsNotNone(definition)
        assert definition is not None
        self.assertEqual(definition.category, "吊索具缺陷与使用不当")
        self.assertEqual(definition.scene_id, "S1")
        self.assertEqual(definition.severity, "high")
        self.assertTrue(definition.norm_ref)

    def test_scene_ids(self) -> None:
        self.assertEqual(self.kb.scene_ids(), ["S1", "S2", "S3"])

    def test_severity_levels_loaded(self) -> None:
        self.assertEqual(self.kb.severity_levels["critical"]["color"], "#FF0000")


class KnowledgeCompressorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.kb = KnowledgeBase.load(KB_PATH)
        self.compressor = KnowledgeCompressor(self.kb)

    def test_build_brief_within_budget(self) -> None:
        brief = self.compressor.build_brief(["S1", "S2"], 4000)
        self.assertLessEqual(len(brief), 4000)
        self.assertIn("D-1-1", brief)
        self.assertIn("G-2-2", brief)

    def test_build_brief_cached(self) -> None:
        first = self.compressor.build_brief(["S1"], 4000)
        second = self.compressor.build_brief(["S1"], 4000)
        self.assertIs(first, second)

    def test_s1_brief_includes_crane_rules(self) -> None:
        brief = self.compressor.build_brief(["S1"], 8000)
        self.assertIn("十不吊", brief)
        self.assertIn("行车六严禁", brief)

    def test_s2_brief_excludes_crane_rules(self) -> None:
        brief = self.compressor.build_brief(["S2"], 8000)
        self.assertNotIn("十不吊", brief)

    def test_backfill_fills_authoritative_fields(self) -> None:
        report = HazardReport(
            scene_type="吊装作业",
            hazards=[
                HazardEvent(id="D-1-1", evidence="绳索交叉", confidence=0.9),
                HazardEvent(id="ZZ-999", evidence="fabricated"),
            ],
        )
        result = self.compressor.backfill(report)
        valid = result.hazards[0]
        self.assertEqual(valid.norm_reference, "GB 6067.1-2010")
        self.assertEqual(valid.severity, "high")
        self.assertEqual(valid.category, "吊索具缺陷与使用不当")
        self.assertEqual(valid.action, "停止作业，重新整理绳索")
        self.assertFalse(valid.invalid_id)
        self.assertTrue(result.hazards[1].invalid_id)


class SceneRouterTest(unittest.TestCase):
    def setUp(self) -> None:
        self.router = SceneRouter(mode="rules")

    @staticmethod
    def _detection(label: str) -> Detection:
        return Detection(label=label, bbox=(0.1, 0.1, 0.2, 0.2))

    def test_route_s1(self) -> None:
        detections = [self._detection("hook"), self._detection("load")]
        self.assertEqual(self.router.route(detections), ["S1"])

    def test_route_s2(self) -> None:
        self.assertEqual(self.router.route([self._detection("ladder")]), ["S2"])

    def test_route_both(self) -> None:
        detections = [self._detection("hook"), self._detection("scaffold")]
        self.assertEqual(self.router.route(detections), ["S1", "S2"])

    def test_route_unknown(self) -> None:
        self.assertEqual(self.router.route([self._detection("car")]), [])

    def test_route_by_video_votes(self) -> None:
        frames = [
            [self._detection("hook")],
            [self._detection("hook"), self._detection("load")],
            [self._detection("ladder")],
        ]
        self.assertEqual(self.router.route_by_video(frames), ["S1"])


if __name__ == "__main__":
    unittest.main()
