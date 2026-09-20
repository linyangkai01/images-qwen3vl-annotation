"""T01 共享契约测试：12 类体系 / S1 能力分层表 / 尺寸分档 / hoist 配置。

覆盖 team-lead 要求的断言：
- 12 类 class_id 唯一且顺序与约定一致；
- S1 的 27 条隐患在能力分层表中全部出现，一条不漏；
- 被排除的 4 类不出现在 12 类里；
- A/B/C 三档覆盖全部 12 类，无遗漏无重叠；
- hoist 配置能被加载，且 endpoint_policy.mode == "allowlist"、白名单含
  119.23.145.227:38091。
"""

from __future__ import annotations

import unittest
from pathlib import Path

from traffic_cv_closed_loop.config import ProjectConfig, endpoint_allowed

from industrial_hazard_coldstart import models, taxonomy

ROOT = Path(__file__).resolve().parents[1]
HOIST_CONFIG = ROOT / "own" / "hoist_project" / "config" / "project.hoist.json"

EXPECTED_CLASS_NAMES = (
    "person",
    "helmet",
    "vest",
    "hook",
    "rigging",
    "overhead_crane",
    "spreader",
    "balance_beam",
    "load",
    "pallet",
    "hopper",
    "ebike",
)


class TaxonomyClassTest(unittest.TestCase):
    """12 类体系与 class_id 顺序。"""

    def test_class_ids_unique_and_ordered(self) -> None:
        self.assertEqual(len(taxonomy.CLASS_NAMES), 12)
        self.assertEqual(len(set(taxonomy.CLASS_NAMES)), 12, "12 类存在重复")
        self.assertEqual(taxonomy.CLASS_NAMES, EXPECTED_CLASS_NAMES)
        for index, name in enumerate(EXPECTED_CLASS_NAMES):
            self.assertEqual(taxonomy.class_id_of(name), index)
            self.assertEqual(taxonomy.class_name_of(index), name)

    def test_excluded_classes_absent_from_taxonomy(self) -> None:
        for name in ("ladder", "scaffold", "guardrail", "harness"):
            self.assertIn(name, taxonomy.EXCLUDED_CLASS_NAMES)
            self.assertNotIn(name, taxonomy.CLASS_NAMES)

    def test_taxonomy_classes_built_consistently(self) -> None:
        classes = taxonomy.build_taxonomy_classes()
        self.assertEqual(len(classes), 12)
        self.assertEqual([c.class_id for c in classes], list(range(12)))
        self.assertEqual([c.name for c in classes], list(EXPECTED_CLASS_NAMES))
        for item in classes:
            self.assertEqual(item.tier, taxonomy.tier_of_class(item.name))


class HazardLayerTableTest(unittest.TestCase):
    """S1 的 27 条隐患 → 能力层（一条不漏）。"""

    def test_s1_has_27_unique_hazards(self) -> None:
        self.assertEqual(len(taxonomy.S1_HAZARD_IDS), 27)
        self.assertEqual(len(set(taxonomy.S1_HAZARD_IDS)), 27)

    def test_capability_table_covers_all_27(self) -> None:
        capabilities = taxonomy.build_hazard_capabilities()
        ids = [item.hazard_id for item in capabilities]
        self.assertEqual(len(ids), 27, "能力分层表条数不为 27")
        self.assertEqual(len(set(ids)), 27, "能力分层表存在重复")
        self.assertEqual(set(ids), set(taxonomy.S1_HAZARD_IDS))

    def test_layer_buckets_are_mutually_exclusive_and_exhaustive(self) -> None:
        buckets = {
            taxonomy.LAYER_DETECTION: set(taxonomy.S1_DETECTION_HAZARD_IDS),
            taxonomy.LAYER_GEOMETRY: set(taxonomy.S1_GEOMETRY_HAZARD_IDS),
            taxonomy.LAYER_TEMPORAL: set(taxonomy.S1_TEMPORAL_HAZARD_IDS),
            taxonomy.LAYER_OCR: set(taxonomy.S1_OCR_HAZARD_IDS),
            taxonomy.LAYER_SUPER_RESOLUTION: set(taxonomy.S1_SUPER_RESOLUTION_HAZARD_IDS),
            taxonomy.LAYER_IMAGE_QUALITY: set(taxonomy.S1_IMAGE_QUALITY_HAZARD_IDS),
        }
        total = sum(len(items) for items in buckets.values())
        self.assertEqual(total, 27, "各层条数之和不为 27（存在遗漏或重叠）")

        union: set[str] = set()
        for layer, items in buckets.items():
            self.assertEqual(union & items, set(), f"层 {layer} 与其他层存在重叠")
            union |= items
        self.assertEqual(union, set(taxonomy.S1_HAZARD_IDS), "各层并集不等于 S1 全量")

        # 明确写死的四条几何层隐患（team-lead 指定）
        self.assertEqual(
            set(taxonomy.S1_GEOMETRY_HAZARD_IDS), {"D-2-1", "D-2-4", "D-4-3", "D-5-3"}
        )
        self.assertEqual(set(taxonomy.S1_TEMPORAL_HAZARD_IDS), {"D-3-6", "D-3-7"})
        self.assertEqual(set(taxonomy.S1_OCR_HAZARD_IDS), {"D-1-6"})
        self.assertEqual(set(taxonomy.S1_SUPER_RESOLUTION_HAZARD_IDS), {"D-1-4"})
        self.assertEqual(set(taxonomy.S1_IMAGE_QUALITY_HAZARD_IDS), {"D-4-1"})
        self.assertEqual(len(taxonomy.S1_DETECTION_HAZARD_IDS), 18)

    def test_layer_of_hazard_matches_table(self) -> None:
        for hazard_id in taxonomy.S1_HAZARD_IDS:
            self.assertEqual(
                taxonomy.layer_of_hazard(hazard_id), taxonomy.S1_HAZARD_LAYERS[hazard_id]
            )

    def test_validate_taxonomy_passes(self) -> None:
        # 自检不得抛异常。
        taxonomy.validate_taxonomy()


class TierTest(unittest.TestCase):
    """A/B/C 尺寸分档覆盖全部 12 类，无遗漏无重叠。"""

    def test_tiers_cover_all_classes_without_overlap(self) -> None:
        tiered = [name for tier in taxonomy.TIER_ORDER for name in taxonomy.TIER_CLASSES[tier]]
        self.assertEqual(len(tiered), 12)
        self.assertEqual(len(set(tiered)), 12, "三档之间存在重叠")
        self.assertEqual(set(tiered), set(taxonomy.CLASS_NAMES), "三档未覆盖全部 12 类")
        for name in taxonomy.CLASS_NAMES:
            self.assertIn(taxonomy.tier_of_class(name), taxonomy.TIER_ORDER)

    def test_tier_thresholds_match_spec(self) -> None:
        self.assertEqual(taxonomy.tier_threshold("A"), {"map50": 0.70, "recall": 0.80, "pixel_risk": False})
        self.assertEqual(taxonomy.tier_threshold("B"), {"map50": 0.50, "recall": 0.65, "pixel_risk": False})
        self.assertEqual(taxonomy.tier_threshold("C"), {"map50": 0.35, "recall": 0.55, "pixel_risk": True})
        # C 档必须带像素下限风险标记。
        self.assertTrue(taxonomy.TIER_THRESHOLDS[taxonomy.TIER_C]["pixel_risk"])
        self.assertIn(taxonomy.TIER_C, taxonomy.TIER_RISK_NOTES)


class HoistConfigTest(unittest.TestCase):
    """hoist 配置加载 + 端点护栏（必须 allowlist）。"""

    def test_hoist_config_loads(self) -> None:
        config = ProjectConfig.load(HOIST_CONFIG)
        self.assertEqual(config.scene, "hoist")
        self.assertEqual(list(config.labels), list(EXPECTED_CLASS_NAMES))
        self.assertEqual(config.qwen_provider, "private_vlm")
        self.assertEqual(config.inference_backend, "private_vlm")

    def test_endpoint_policy_is_allowlist_with_lab_endpoint(self) -> None:
        config = ProjectConfig.load(HOIST_CONFIG)
        self.assertEqual(config.inference_endpoint_mode, "allowlist")
        self.assertIn("119.23.145.227:38091", config.inference_endpoint_allowlist)
        # 白名单命中 → 放行。
        allowed, reason = endpoint_allowed(
            config.qwen_chat_endpoint,
            config.inference_endpoint_mode,
            config.inference_endpoint_allowlist,
        )
        self.assertTrue(allowed, reason)

    def test_public_endpoint_requires_allowlist(self) -> None:
        # 锁定关键点：119.23.145.227 是公网 IP，local_only 会直接拒绝，
        # 因此**必须**用 allowlist 模式，否则合法端点被护栏拦死。
        config = ProjectConfig.load(HOIST_CONFIG)
        self.assertFalse(config.inference_endpoint_allowlist == [])

        from traffic_cv_closed_loop.config import is_local_endpoint_host

        self.assertFalse(is_local_endpoint_host("119.23.145.227"))

        rejected, reason = endpoint_allowed(
            config.qwen_chat_endpoint, "local_only", []
        )
        self.assertFalse(rejected, "local_only 不应放行公网端点")
        self.assertIn("local_only", reason)

        # allowlist 模式下，未列入白名单的公网端点仍被拒绝。
        blocked, block_reason = endpoint_allowed(
            "http://8.8.8.8:1234/api/generate",
            config.inference_endpoint_mode,
            config.inference_endpoint_allowlist,
        )
        self.assertFalse(blocked)
        self.assertIn("allowlist", block_reason)


class ModelsRoundTripTest(unittest.TestCase):
    """9 个新数据结构的 from_dict/to_dict 互逆 + 复用母项目结构。"""

    def test_reuses_mother_structures(self) -> None:
        from traffic_cv_closed_loop.models import Detection as BaseDetection

        self.assertIs(models.Detection, BaseDetection)
        self.assertEqual(models.CandidateFrame.__module__, "traffic_cv_closed_loop.models")

    def test_media_asset_round_trip_chinese_path(self) -> None:
        asset = models.MediaAsset(
            source_id="zhuhai_01",
            path="D:/素材/珠海/吊装01.mp4",
            scene="hoist",
            media_type="video",
            duration_s=12.5,
            width=960,
            height=540,
            fps=30.0,
            count=1,
        )
        self.assertEqual(models.MediaAsset.from_dict(asset.to_dict()), asset)

    def test_all_new_structures_round_trip(self) -> None:
        samples = [
            models.KeyframeRecord(
                source_id="s1", frame_index=3, timestamp_s=1.2,
                extracted_path="workspace/extracted/frames/s1/f003.png",
                phash=123, motion_ratio=0.5, kept_reason="motion",
            ),
            models.TaxonomyClass(class_id=3, name="hook", tier="B", served_hazards=["D-1-5"]),
            models.HazardCapability(hazard_id="D-5-3", layer="geometry", supported=False, notes="留位"),
            models.HoldoutEntry(
                dataset_version="v1", relative_path="images/val/a.png",
                sha256="abc", registered_at="2026-09-20T00:00:00",
            ),
            models.PriorityItem(
                relative_path="images/a.png", score=0.9,
                signals={"min_confidence": 0.1}, reasons=["low_confidence"],
            ),
            models.EvalReport(
                dataset_version="v1", holdout_hash="deadbeef",
                per_class={"person": {"map50": 0.8}}, tier_verdict={"A": "pass"},
                consumption_rate=0.95,
            ),
            models.RelationFinding(
                hazard_id="D-5-3", relation="person_under_load", confidence=0.7,
                supporting=[{"label": "person", "bbox": [0.1, 0.1, 0.2, 0.2]}], notes="",
            ),
            models.TemporalFinding(
                hazard_id="D-3-6", source_id="s1", finding="suspended",
                confidence=0.6, frame_indices=[1, 2, 3], notes="",
            ),
        ]
        for sample in samples:
            restored = type(sample).from_dict(sample.to_dict())
            self.assertEqual(restored, sample)

    def test_from_dict_tolerates_missing_fields(self) -> None:
        # 缺字段时也要能构造（默认值兜底）。
        self.assertEqual(models.MediaAsset.from_dict({"source_id": "x", "path": "p"}).scene, "hoist")
        self.assertEqual(models.HazardCapability.from_dict({"hazard_id": "D-1-1"}).supported, False)
        self.assertEqual(models.PriorityItem.from_dict({"relative_path": "a"}).score, 0.0)

    def test_to_dict_none_list_dict_fields_do_not_crash(self) -> None:
        # 缺陷 C：to_dict 侧对 None 列表/字典字段的守卫，须与 from_dict 对称。
        samples = [
            models.TemporalFinding(hazard_id="h", source_id="s", finding="f", frame_indices=None),
            models.RelationFinding(hazard_id="h", relation="r", supporting=None),
            models.PriorityItem(relative_path="p", reasons=None, signals=None),
            models.TaxonomyClass(class_id=0, name="person", served_hazards=None),
            models.EvalReport(dataset_version="v1", per_class=None, tier_verdict=None),
        ]
        for sample in samples:
            with self.subTest(cls=type(sample).__name__):
                payload = sample.to_dict()  # 不应抛 TypeError
                # 且与 from_dict 的宽容行为对称：None 回落为空容器。
                restored = type(sample).from_dict(payload)
                self.assertEqual(type(sample).from_dict(payload).to_dict(), payload)
                self.assertIsInstance(restored, type(sample))

    def test_from_dict_none_str_fields_become_empty_not_none_string(self) -> None:
        # 缺陷 E：str 字段遇 None 回落为 ""，绝不产出 "None" 中间态。
        asset = models.MediaAsset.from_dict(
            {"source_id": None, "path": None, "scene": None, "media_type": None}
        )
        self.assertEqual(asset.source_id, "")
        self.assertEqual(asset.path, "")
        self.assertEqual(asset.scene, "")
        self.assertEqual(asset.media_type, "")
        for value in (asset.source_id, asset.path, asset.scene, asset.media_type):
            self.assertNotEqual(value, "None")

        # 数值 / 布尔字段遇 None 回落默认值（不崩）。
        asset2 = models.MediaAsset.from_dict({"source_id": "s", "path": "p", "width": None, "fps": None})
        self.assertEqual(asset2.width, 0)
        self.assertEqual(asset2.fps, 0.0)
        self.assertEqual(models.HazardCapability.from_dict({"hazard_id": "h", "supported": None}).supported, False)
        self.assertEqual(models.HoldoutEntry.from_dict({"dataset_version": "v"}).relative_path, "")


if __name__ == "__main__":
    unittest.main()
