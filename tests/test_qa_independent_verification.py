"""T01 独立验证（QA / 严过关）。

与工程师的 ``test_taxonomy.py`` / ``test_coldstart_scaffold.py`` **互补**：
本文件不复跑其断言，而是从「它没想到的边界」入手，且**以真实知识库文件为真源**
（不是拿 ``taxonomy.S1_HAZARD_IDS`` 自己验自己）。

覆盖五件事：
1. class_id 是否真的等于数组下标（构造真实 YOLO 标签行反查类别名）；
2. 能力分层表与 ``hazard-detection/knowledge_base.json`` 的**双向**一致性；
3. 端点护栏的边界（端口不匹配 / 空格 / 大小写 / 未列入 / local_only）；
4. 9 个 dataclass 的 round-trip 无损性（中文路径 / None / 空 / 嵌套）；
5. CLI 与 doctor 的实际行为（退出码 / 缺失或损坏配置）。

缺陷处置（team-lead 决策，2026-09-20）：
- 缺陷 A/C/E 已在 ``industrial_hazard_coldstart`` 内修复，对应用例已转绿；
- 缺陷 B/D 位于**冻结母项目**（``traffic_cv_closed_loop/``，硬约束「一行不改」），
  其用例改为**行为刻画 + 绊线**：断言母项目**当前真实语义**，并在注释中写明
  「若断言某天失败，说明母项目已修正，请同步更新 doctor 告警与注释」。
硬约束：不修改任何生产代码，不读私有素材，不联网。
"""

from __future__ import annotations

import contextlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from traffic_cv_closed_loop.config import (
    ProjectConfig,
    endpoint_allowed,
    is_local_endpoint_host,
)

from industrial_hazard_coldstart import models, taxonomy
from industrial_hazard_coldstart.cli import main

ROOT = Path(__file__).resolve().parents[1]
HOIST_CONFIG = ROOT / "own" / "hoist_project" / "config" / "project.hoist.json"
KB_PATH = ROOT / "hazard-detection" / "knowledge_base.json"
LAB_ENDPOINT = "http://119.23.145.227:38091/api/generate"

EXPECTED_CLASS_NAMES = (
    "person", "helmet", "vest", "hook", "rigging", "overhead_crane",
    "spreader", "balance_beam", "load", "pallet", "hopper", "ebike",
)


def _load_kb_s1() -> tuple[list[str], dict[str, str]]:
    """从唯一真源知识库读取 S1 的隐患 id 顺序与 id→name。"""
    raw = json.loads(KB_PATH.read_text(encoding="utf-8"))
    scene = next(s for s in raw["scenes"] if s.get("scene_id") == "S1")
    ids: list[str] = []
    names: dict[str, str] = {}
    for category in scene["hazard_categories"]:
        for hazard in category["hazards"]:
            ids.append(hazard["id"])
            names[hazard["id"]] = hazard["name"]
    return ids, names


# --------------------------------------------------------------------------- #
# 1. class_id 真的等于数组下标吗？
# --------------------------------------------------------------------------- #
class ClassIdIndexTest(unittest.TestCase):
    def test_class_id_mapping_is_enumerate_index(self) -> None:
        # 直接与 enumerate 对照，而不是与硬编码数字对照。
        self.assertEqual(
            taxonomy.CLASS_ID_TO_NAME,
            {index: name for index, name in enumerate(taxonomy.CLASS_NAMES)},
        )
        self.assertEqual(
            taxonomy.CLASS_NAME_TO_ID,
            {name: index for index, name in enumerate(taxonomy.CLASS_NAMES)},
        )

    def test_real_yolo_label_line_maps_to_expected_class(self) -> None:
        # 关键：构造真实 YOLO 标签行（class_id cx cy w h），反查类别名。
        from traffic_cv_closed_loop.models import Detection

        config = ProjectConfig.load(HOIST_CONFIG)
        cases = {
            "overhead_crane": 5,
            "ebike": 11,
            "person": 0,
            "hopper": 10,
        }
        for name, expected_id in cases.items():
            line = Detection(label=name, bbox=[0.1, 0.2, 0.3, 0.4]).to_yolo_line(
                config.label_to_index
            )
            parts = line.split()
            self.assertEqual(len(parts), 5, f"YOLO 标签行格式不对：{line!r}")
            class_id = int(parts[0])
            self.assertEqual(class_id, expected_id, f"{name} 的 class_id 漂移")
            self.assertEqual(taxonomy.class_name_of(class_id), name)

    def test_config_labels_order_matches_taxonomy(self) -> None:
        # 配置 labels 顺序 == 契约顺序；两者若漂移，YOLO 标签就会错位。
        config = ProjectConfig.load(HOIST_CONFIG)
        self.assertEqual(list(config.labels), list(taxonomy.CLASS_NAMES))
        for name in config.labels:
            self.assertEqual(config.label_to_index[name], taxonomy.class_id_of(name))

    def test_class_id_5_and_11_exact(self) -> None:
        self.assertEqual(taxonomy.class_name_of(5), "overhead_crane")
        self.assertEqual(taxonomy.class_name_of(11), "ebike")


# --------------------------------------------------------------------------- #
# 2. 能力分层表 ↔ 知识库（双向 + 真源）
# --------------------------------------------------------------------------- #
class KnowledgeBaseBidirectionalTest(unittest.TestCase):
    def test_s1_ids_match_knowledge_base_order_and_set(self) -> None:
        kb_ids, _ = _load_kb_s1()
        self.assertEqual(len(kb_ids), 27)
        self.assertEqual(list(taxonomy.S1_HAZARD_IDS), kb_ids, "id 顺序与知识库不一致")

    def test_forward_every_kb_id_in_layer_table(self) -> None:
        kb_ids, _ = _load_kb_s1()
        missing = [hid for hid in kb_ids if hid not in taxonomy.S1_HAZARD_LAYERS]
        self.assertEqual(missing, [], "知识库有隐患未进分层表")

    def test_reverse_every_layer_id_in_knowledge_base(self) -> None:
        # 工程师可能只验了正向；这里验反向：分层表里不得出现知识库没有的 id。
        kb_ids, _ = _load_kb_s1()
        kb_set = set(kb_ids)
        extra = sorted(set(taxonomy.S1_HAZARD_LAYERS) - kb_set)
        self.assertEqual(extra, [], "分层表含知识库不存在的 id")

    def test_hazard_names_match_knowledge_base(self) -> None:
        kb_ids, kb_names = _load_kb_s1()
        mismatched = {
            hid: (taxonomy.S1_HAZARD_NAMES.get(hid), kb_names[hid])
            for hid in kb_ids
            if taxonomy.S1_HAZARD_NAMES.get(hid) != kb_names[hid]
        }
        self.assertEqual(mismatched, {}, "隐患中文名与知识库不一致")

    def test_buckets_are_exact_partition_of_knowledge_base(self) -> None:
        kb_ids, _ = _load_kb_s1()
        buckets = {
            taxonomy.LAYER_DETECTION: taxonomy.S1_DETECTION_HAZARD_IDS,
            taxonomy.LAYER_GEOMETRY: taxonomy.S1_GEOMETRY_HAZARD_IDS,
            taxonomy.LAYER_TEMPORAL: taxonomy.S1_TEMPORAL_HAZARD_IDS,
            taxonomy.LAYER_OCR: taxonomy.S1_OCR_HAZARD_IDS,
            taxonomy.LAYER_SUPER_RESOLUTION: taxonomy.S1_SUPER_RESOLUTION_HAZARD_IDS,
            taxonomy.LAYER_IMAGE_QUALITY: taxonomy.S1_IMAGE_QUALITY_HAZARD_IDS,
        }
        seen: list[str] = []
        for ids in buckets.values():
            seen.extend(ids)
        self.assertEqual(len(seen), len(set(seen)), "分桶之间存在重叠")
        self.assertEqual(set(seen), set(kb_ids), "分桶并集 ≠ 知识库 S1 全量")

    def test_layer_of_hazard_agrees_with_bucket_membership(self) -> None:
        for layer, ids in {
            taxonomy.LAYER_GEOMETRY: taxonomy.S1_GEOMETRY_HAZARD_IDS,
            taxonomy.LAYER_TEMPORAL: taxonomy.S1_TEMPORAL_HAZARD_IDS,
            taxonomy.LAYER_OCR: taxonomy.S1_OCR_HAZARD_IDS,
            taxonomy.LAYER_SUPER_RESOLUTION: taxonomy.S1_SUPER_RESOLUTION_HAZARD_IDS,
            taxonomy.LAYER_IMAGE_QUALITY: taxonomy.S1_IMAGE_QUALITY_HAZARD_IDS,
        }.items():
            for hid in ids:
                self.assertEqual(taxonomy.layer_of_hazard(hid), layer)

    def test_non_detection_layers_match_requirement_8_3(self) -> None:
        # 需求 §8.3 明列的 9 条「框做不了」隐患，逐条核对归层。
        self.assertEqual(set(taxonomy.S1_GEOMETRY_HAZARD_IDS), {"D-2-1", "D-2-4", "D-4-3", "D-5-3"})
        self.assertEqual(set(taxonomy.S1_TEMPORAL_HAZARD_IDS), {"D-3-6", "D-3-7"})
        self.assertEqual(set(taxonomy.S1_OCR_HAZARD_IDS), {"D-1-6"})
        self.assertEqual(set(taxonomy.S1_SUPER_RESOLUTION_HAZARD_IDS), {"D-1-4"})
        self.assertEqual(set(taxonomy.S1_IMAGE_QUALITY_HAZARD_IDS), {"D-4-1"})
        self.assertEqual(len(taxonomy.S1_DETECTION_HAZARD_IDS), 18)


# --------------------------------------------------------------------------- #
# 3. 端点护栏边界
# --------------------------------------------------------------------------- #
class EndpointGuardTest(unittest.TestCase):
    def test_allowlist_entry_without_port_still_allows(self) -> None:
        allowed, reason = endpoint_allowed(LAB_ENDPOINT, "allowlist", ["119.23.145.227"])
        self.assertTrue(allowed, reason)

    def test_port_mismatch_is_rejected(self) -> None:
        # 团队最关心的一条：白名单端口不匹配时，绝不能放行。
        allowed, reason = endpoint_allowed(LAB_ENDPOINT, "allowlist", ["119.23.145.227:9999"])
        self.assertFalse(allowed, f"端口不匹配却放行了：{reason}")

    def test_exact_host_port_match_allows(self) -> None:
        allowed, reason = endpoint_allowed(LAB_ENDPOINT, "allowlist", ["119.23.145.227:38091"])
        self.assertTrue(allowed, reason)

    def test_whitespace_in_allowlist_entry_is_tolerated(self) -> None:
        allowed, reason = endpoint_allowed(
            LAB_ENDPOINT, "allowlist", ["  119.23.145.227:38091  "]
        )
        self.assertTrue(allowed, reason)

    def test_unlisted_public_host_is_rejected(self) -> None:
        allowed, reason = endpoint_allowed(
            "http://8.8.8.8:1234/api/generate", "allowlist", ["119.23.145.227:38091"]
        )
        self.assertFalse(allowed)
        self.assertIn("allowlist", reason)

    def test_local_only_rejects_public_endpoint_with_empty_allowlist(self) -> None:
        self.assertFalse(is_local_endpoint_host("119.23.145.227"))
        allowed, reason = endpoint_allowed(LAB_ENDPOINT, "local_only", [])
        self.assertFalse(allowed, "local_only 不应放行公网端点")
        self.assertIn("local_only", reason)

    def test_local_only_with_stale_allowlist_currently_allows_documented_quirk(self) -> None:
        # 行为刻画（非缺陷修复）：母项目 ``endpoint_allowed()`` 的白名单命中判定发生在
        # mode 判定**之前**，因此 ``local_only`` + 非空白名单 → 公网端点仍被放行。
        # 即 ``local_only`` 并非最严格档，它等价于「本机/内网 **或** 白名单」。
        #
        # 不修的原因：``traffic_cv_closed_loop/`` 是冻结资产，硬约束「一行不改」；
        # 我们只能在自己的 doctor 里对「local_only + 白名单」发出告警兜住
        # （见 industrial_hazard_coldstart/doctor.py 的 LOCAL_ONLY_ALLOWLIST_WARNING）。
        #
        # 绊线：若本断言某天开始失败，说明母项目已修正该语义；届时请同步更新
        #       doctor 的告警与注释、以及本用例。
        allowed, reason = endpoint_allowed(
            LAB_ENDPOINT, "local_only", ["119.23.145.227:38091"]
        )
        self.assertTrue(
            allowed,
            f"母项目既有语义：白名单命中先于 mode 判定 → 放行。reason={reason}",
        )
        self.assertIn("命中白名单", reason)

    def test_local_only_bypass_reachable_through_config_validate_is_known_quirk(self) -> None:
        # 行为刻画：上述「local_only + 残留白名单」的配置能通过 ProjectConfig.validate()，
        # 证明该语义不是理论问题，而是可实际加载的配置。
        #
        # 绊线：若本断言某天开始失败（load 抛错或 allowed 变 False），说明母项目已收紧语义；
        #       届时请同步更新 doctor 告警与本注释。
        raw = json.loads(HOIST_CONFIG.read_text(encoding="utf-8"))
        raw["inference"]["endpoint_policy"]["mode"] = "local_only"
        raw["inference"]["endpoint_policy"]["allowlist"] = ["119.23.145.227:38091"]
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = Path(tmp) / "cfg.json"
            cfg_path.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
            cfg = ProjectConfig.load(cfg_path)  # 不抛错 = 该配置被母项目判定为合法
        allowed, _ = endpoint_allowed(
            cfg.qwen_chat_endpoint,
            cfg.inference_endpoint_mode,
            cfg.inference_endpoint_allowlist,
        )
        self.assertTrue(
            allowed, "母项目既有语义：local_only + 残留白名单仍放行公网端点"
        )


# --------------------------------------------------------------------------- #
# 4. dataclass round-trip 无损性
# --------------------------------------------------------------------------- #
class RoundTripTest(unittest.TestCase):
    NEW_CLASSES = (
        models.MediaAsset, models.KeyframeRecord, models.TaxonomyClass,
        models.HazardCapability, models.HoldoutEntry, models.PriorityItem,
        models.EvalReport, models.RelationFinding, models.TemporalFinding,
    )

    def _assert_dict_round_trip(self, obj) -> None:
        first = obj.to_dict()
        second = type(obj).from_dict(first).to_dict()
        self.assertEqual(first, second, f"{type(obj).__name__} round-trip 后 dict 不一致")

    def test_chinese_path_round_trip(self) -> None:
        asset = models.MediaAsset(
            source_id="珠海_01", path="D:/素材/珠海/吊装 01.mp4",
            scene="hoist", media_type="video", duration_s=12.5,
            width=960, height=540, fps=30.0, count=1,
        )
        self._assert_dict_round_trip(asset)
        self.assertEqual(models.MediaAsset.from_dict(asset.to_dict()).path, asset.path)
        self.assertEqual(asset.resolved_path.name, "吊装 01.mp4")

    def test_empty_and_none_defaults_round_trip(self) -> None:
        for obj in (
            models.PriorityItem(relative_path="", score=0.0, signals={}, reasons=[]),
            models.TaxonomyClass(class_id=0, name="person", tier="", served_hazards=[]),
            models.RelationFinding(hazard_id="h", relation="r", supporting=[], notes=""),
        ):
            self._assert_dict_round_trip(obj)

    def test_nested_detection_round_trip(self) -> None:
        # CandidateFrame 里含 Detection 列表（母项目结构，本链路复用）。
        frame = models.CandidateFrame(
            source_id="s1", relative_path="s1/f1.png", original_path="",
            extracted_path="", scene="hoist",
            detections=[
                models.Detection(label="person", bbox=(0.1, 0.1, 0.2, 0.2), confidence=0.9),
                models.Detection(label="load", bbox=(0.0, 0.0, 1.0, 1.0), confidence=0.5),
            ],
            metadata={"k": "v"},
        )
        self._assert_dict_round_trip(frame)
        restored = models.CandidateFrame.from_dict(frame.to_dict())
        self.assertEqual([d.label for d in restored.detections], ["person", "load"])

    def test_all_nine_new_dataclasses_round_trip(self) -> None:
        samples = [
            models.MediaAsset(source_id="s", path="p"),
            models.KeyframeRecord(source_id="s", frame_index=1),
            models.TaxonomyClass(class_id=3, name="hook"),
            models.HazardCapability(hazard_id="D-1-1", layer="detection"),
            models.HoldoutEntry(dataset_version="v1", relative_path="p", sha256="x"),
            models.PriorityItem(relative_path="p"),
            models.EvalReport(dataset_version="v1"),
            models.RelationFinding(hazard_id="D-5-3", relation="under"),
            models.TemporalFinding(hazard_id="D-3-6", source_id="s", finding="suspended"),
        ]
        self.assertEqual(len(samples), 9)
        for sample in samples:
            self._assert_dict_round_trip(sample)

    def test_confidence_round_trip_preserves_four_decimal_precision(self) -> None:
        # 真实契约（非缺陷）：母项目 ``Detection.to_dict`` 将 confidence 四舍五入到
        # **4 位小数**，这是既有精度约定。本用例验证该契约成立，而不是要求更高精度。
        # 用一个 4 位小数可精确表示的值做 round-trip，断言对象层无损。
        #
        # 绊线：若某天该断言失败，说明母项目改了精度约定；请同步更新本注释。
        detection = models.Detection(
            label="person", bbox=(0.1, 0.1, 0.2, 0.2), confidence=0.1234
        )
        restored = models.Detection.from_dict(detection.to_dict())
        self.assertEqual(
            restored, detection,
            f"4 位小数精度内 round-trip 应无损：0.1234 -> {restored.confidence}",
        )
        self.assertEqual(restored.confidence, 0.1234)

    def test_none_list_field_does_not_crash_to_dict(self) -> None:
        # [DEFECT] 直接以 None 构造列表字段后调用 to_dict 会抛 TypeError。
        # from_dict 侧对 None 是宽容的（回落 []），但 to_dict 侧不是。
        for obj in (
            models.TemporalFinding(hazard_id="h", source_id="s", finding="f", frame_indices=None),
            models.RelationFinding(hazard_id="h", relation="r", supporting=None),
            models.PriorityItem(relative_path="p", reasons=None),
        ):
            with self.subTest(cls=type(obj).__name__):
                obj.to_dict()  # 不应抛异常

    def test_from_dict_tolerates_none_list_fields(self) -> None:
        # 反向：from_dict 对 None 是宽容的（作为对照，说明只有 to_dict 侧不一致）。
        self.assertEqual(
            models.TemporalFinding.from_dict(
                {"hazard_id": "h", "source_id": "s", "finding": "f", "frame_indices": None}
            ).frame_indices,
            [],
        )


# --------------------------------------------------------------------------- #
# 5. CLI 与 doctor 实际行为
# --------------------------------------------------------------------------- #
class CliAndDoctorTest(unittest.TestCase):
    PENDING = (
        "inventory", "extract-video", "extract", "annotate", "prioritize",
        "curate", "freeze-holdout", "train", "evaluate",
    )

    def test_help_exit_code_zero(self) -> None:
        with self.assertRaises(SystemExit) as ctx, contextlib.redirect_stdout(io.StringIO()):
            main(["--help"])
        self.assertEqual(ctx.exception.code, 0)

    def test_every_pending_subcommand_returns_2_with_stderr(self) -> None:
        for command in self.PENDING:
            with self.subTest(command=command):
                out, err = io.StringIO(), io.StringIO()
                with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                    code = main([command, "--config", str(HOIST_CONFIG)])
                self.assertEqual(code, 2, f"{command} 退出码应为 2（尚未实现）")
                self.assertIn("尚未实现", err.getvalue(), f"{command} 未在 stderr 说明")

    def test_doctor_report_has_four_checks_and_honest_holdout(self) -> None:
        config = ProjectConfig.load(HOIST_CONFIG)
        from industrial_hazard_coldstart.doctor import build_hoist_doctor_report

        report = build_hoist_doctor_report(config)
        self.assertEqual(
            set(report["checks"]),
            {"ffmpeg", "vlm_endpoint", "endpoint_policy", "holdout"},
        )
        self.assertEqual(report["checks"]["holdout"]["status"], "not_implemented")
        self.assertIn("holdout_not_implemented", report["pending"])
        # VLM 探活不在本机执行（不联网）。
        self.assertFalse(report["checks"]["vlm_endpoint"]["probed"])

    def _run_cli(self, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-m", "industrial_hazard_coldstart", *args],
            cwd=str(ROOT), capture_output=True, text=True,
        )

    def test_doctor_missing_config_has_readable_error_not_traceback(self) -> None:
        # [DEFECT] 配置不存在时应给出可读错误；当前抛出未捕获的 FileNotFoundError。
        result = self._run_cli("doctor", "--config", "does/not/exist.json")
        combined = result.stdout + result.stderr
        self.assertNotIn("Traceback (most recent call last)", combined, combined[-800:])
        self.assertIn("Config file not found", combined)

    def test_doctor_corrupt_config_has_readable_error_not_traceback(self) -> None:
        # [DEFECT] 配置损坏（非法 JSON）时应给出可读错误；当前抛 JSONDecodeError 栈。
        with tempfile.TemporaryDirectory() as tmp:
            broken = Path(tmp) / "broken.json"
            broken.write_text("{ this is not valid json", encoding="utf-8")
            result = self._run_cli("doctor", "--config", str(broken))
        combined = result.stdout + result.stderr
        self.assertNotIn("Traceback (most recent call last)", combined, combined[-800:])


if __name__ == "__main__":
    unittest.main()
