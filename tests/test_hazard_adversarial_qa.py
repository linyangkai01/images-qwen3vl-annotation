"""Independent adversarial QA tests for the hazard-annotation delivery.

Written by the QA engineer to *attack* the highest-risk areas rather than
re-run the engineer's own tests. Two kinds of tests live here:

* normal tests (``test_*``) assert behaviour that currently works — they are
  the regression net proving the good parts hold;
* finding tests (``test_finding_*``) assert the *desired* behaviour and are
  expected to FAIL while the underlying defect is open. Each one is the
  concrete, reproducible evidence attached to ``QA-hazard-annotation.md``.

Nothing here touches the network, ffmpeg or yolo; every external dependency is
either absent (graceful-degradation checks) or replaced by a stub.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from hazard_annotation import imageio
from hazard_annotation.config import load_hazard_config
from hazard_annotation.diagnostics import build_doctor_report
from hazard_annotation.ffmpeg import FfmpegRunner, FfmpegUnavailableError
from hazard_annotation.imageio import (
    ImageStats,
    PgmImage,
    PpmImage,
    ahash,
    hamming,
    histogram16,
    histogram_delta,
    load_image,
    motion_ratio,
)
from hazard_annotation.keyframes import (
    KeyframeExtractor,
    KeyframeSource,
    _AnalysisFrame,
)
from hazard_annotation.knowledge import KnowledgeBase, KnowledgeCompressor, SceneRouter
from hazard_annotation.models import HazardEvent, HazardReport
from hazard_annotation.pipeline import annotation as annotation_module
from hazard_annotation.pipeline.extraction import run_extraction
from traffic_cv_closed_loop.adapters.qwen import (
    ThinkingJsonParser,
    _AnnotationParseError,
    _extract_json_from_text,
)
from traffic_cv_closed_loop.adapters.yolo import build_detector
from traffic_cv_closed_loop.cli import _build_parser as build_traffic_parser
from traffic_cv_closed_loop.config import ProjectConfig
from traffic_cv_closed_loop.io_utils import read_json, read_jsonl
from traffic_cv_closed_loop.models import Detection

ROOT = Path(__file__).resolve().parents[1]
KB_PATH = ROOT / "hazard-detection" / "knowledge_base.json"
MOCK_PROJECT = ROOT / "examples" / "hazard_mock_project"
LEGACY_MOCK = ROOT / "examples" / "mock_project" / "config" / "project.json"
LEGACY_PROD = ROOT / "examples" / "production_project" / "config" / "project.production.json"


# --------------------------------------------------------------------------- #
# 1. ThinkingJsonParser — the single most failure-prone component
# --------------------------------------------------------------------------- #
class ThinkingJsonParserAdversarialTest(unittest.TestCase):
    def setUp(self) -> None:
        self.parser = ThinkingJsonParser()

    # -- behaviours that must hold ---------------------------------------- #
    def test_takes_last_balanced_object_after_thinking_braces(self) -> None:
        text = (
            "思考：用户举例 {\"a\": 1} 只是示意，不要照抄。\n"
            "再想想 {一个未加引号的示例} 也要忽略。\n"
            "最终答案：{\"objects\": [], \"hazard_report\": {\"overall_risk\": \"none\"}}"
        )
        parsed = self.parser.parse(text)
        self.assertEqual(parsed["hazard_report"]["overall_risk"], "none")
        self.assertNotIn("a", parsed)

    def test_string_aware_state_machine_ignores_braces_in_strings(self) -> None:
        text = '说明\n{"note": "use { and } and \\" escaped quote", "objects": []}'
        parsed = self.parser.parse(text)
        self.assertEqual(parsed["note"], 'use { and } and " escaped quote')

    def test_json_fence_is_unwrapped(self) -> None:
        text = '推理过程……\n```json\n{"objects": [], "final": 1}\n```\n'
        self.assertEqual(self.parser.parse(text), {"objects": [], "final": 1})

    def test_think_tag_is_stripped(self) -> None:
        text = ' thinking内部推理 {"noise": true} 不要泄漏<｜end▁of▁thinking｜>{"final": 1}'
        parsed = self.parser.parse(text)
        self.assertEqual(parsed, {"final": 1})
        self.assertNotIn("noise", parsed)

    def test_pipe_thinking_tag_is_stripped(self) -> None:
        text = '<|thinking|>内部 {"noise": 1}<|/thinking|>{"final": 1}'
        self.assertEqual(self.parser.parse(text), {"final": 1})

    def test_thinking_fence_is_stripped(self) -> None:
        text = "```thinking\nreasoning with {braces}\n```\n{\"objects\": [], \"final\": 1}"
        self.assertEqual(self.parser.parse(text), {"objects": [], "final": 1})

    def test_repairs_trailing_commas(self) -> None:
        self.assertEqual(self.parser.parse('{"a": [1, 2, ], }'), {"a": [1, 2]})

    def test_repairs_single_quotes(self) -> None:
        self.assertEqual(self.parser.parse("{'a': 1, 'b': 'x'}"), {"a": 1, "b": "x"})

    def test_repairs_python_literals(self) -> None:
        parsed = self.parser.parse('{"a": True, "b": None, "c": False}')
        self.assertIs(parsed["a"], True)
        self.assertIsNone(parsed["b"])
        self.assertIs(parsed["c"], False)

    def test_repairs_comments(self) -> None:
        line = '{"a": 1, // comment\n "b": 2}'
        block = '{"a": 1, /* c */ "b": 2}'
        self.assertEqual(self.parser.parse(line), {"a": 1, "b": 2})
        self.assertEqual(self.parser.parse(block), {"a": 1, "b": 2})

    def test_repairs_truncated_object(self) -> None:
        text = '{"objects": [{"label": "person", "bbox": [0, 0, 1, 1]}'
        parsed = self.parser.parse(text)
        self.assertEqual(parsed["objects"][0]["label"], "person")

    def test_top_level_array_is_returned(self) -> None:
        self.assertEqual(self.parser.parse("prefix [1, 2, 3] suffix"), [1, 2, 3])

    def test_garbage_raises_instead_of_returning_empty(self) -> None:
        with self.assertRaises(ValueError):
            self.parser.parse("there is no json here at all")

    def test_empty_string_raises(self) -> None:
        with self.assertRaises(ValueError):
            self.parser.parse("")

    def test_repair_disabled_raises_on_broken_json(self) -> None:
        strict = ThinkingJsonParser(repair=False)
        with self.assertRaises(Exception):
            strict.parse("{'a': 1}")

    def test_legacy_helper_still_prefers_last_fragment(self) -> None:
        text = 'prefix {"first": 1} middle {"second": 2} suffix'
        self.assertEqual(_extract_json_from_text(text), {"second": 2})

    # -- findings ---------------------------------------------------------- #
    def test_finding_fence_draft_in_thinking_does_not_override_final_answer(self) -> None:
        """A ```json draft inside the chain-of-thought must not beat the final answer.

        Real thinking models frequently sketch a draft JSON inside a fence and
        then print the real answer afterwards without a fence. The parser's
        "prefer the fenced block" step short-circuits the "last balanced JSON"
        rule and returns the draft.
        """
        text = '```json\n{"draft": 1, "objects": []}\n```\n最终答案：{"final": 2, "objects": []}'
        self.assertEqual(self.parser.parse(text), {"final": 2, "objects": []})

    def test_finding_single_quoted_json_with_url_is_not_corrupted(self) -> None:
        """The ``//`` comment stripper must not eat ``//`` inside a string.

        A single-quoted payload containing a URL loses everything after ``//``
        and is returned *silently wrong* (no exception).
        """
        text = "{'url': 'http://example.com/x', 'objects': []}"
        parsed = self.parser.parse(text)
        self.assertEqual(parsed["url"], "http://example.com/x")


# --------------------------------------------------------------------------- #
# 2. Knowledge base normalisation
# --------------------------------------------------------------------------- #
class KnowledgeBaseAdversarialTest(unittest.TestCase):
    def setUp(self) -> None:
        self.kb = KnowledgeBase.load(KB_PATH)
        self.compressor = KnowledgeCompressor(self.kb)

    def test_id_index_covers_all_58_hazards(self) -> None:
        self.assertEqual(self.kb.total, 58)
        self.assertEqual(len(self.kb.id_index), 58)
        self.assertEqual(len(self.kb.id_index), len(self.kb.hazards))
        # 27 (S1) + 30 (S2) + 1 (S3)
        s1 = [h for h in self.kb.hazards if h.scene_id == "S1"]
        s2 = [h for h in self.kb.hazards if h.scene_id == "S2"]
        s3 = [h for h in self.kb.hazards if h.scene_id == "S3"]
        self.assertEqual((len(s1), len(s2), len(s3)), (27, 30, 1))

    def test_s3_direct_hazard_t1_is_normalised(self) -> None:
        definition = self.kb.get("T-1")
        self.assertIsNotNone(definition)
        assert definition is not None
        self.assertEqual(definition.scene_id, "S3")
        self.assertEqual(definition.name, "场地积水")
        self.assertEqual(definition.category, "通用隐患")

    def test_stale_meta_count_is_not_trusted(self) -> None:
        """``kb.total`` must come from the parsed hazards, never from ``meta``.

        The shipped knowledge base now keeps ``meta.total_hazards`` in sync
        (53 -> 58), so the original "meta says 53" premise no longer holds.
        The underlying invariant is still worth guarding, so this test
        deliberately corrupts ``meta`` in a temp copy and proves the count is
        still derived from the actual hazard list rather than the metadata.
        """
        self.assertEqual(self.kb.total, 58)
        self.assertEqual(self.kb.meta.get("total_hazards"), 58)

        with tempfile.TemporaryDirectory() as tmp:
            corrupt = Path(tmp) / "knowledge_base.json"
            raw = json.loads(KB_PATH.read_text(encoding="utf-8"))
            raw.setdefault("meta", {})["total_hazards"] = 999
            corrupt.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
            stale_kb = KnowledgeBase.load(corrupt)
            self.assertEqual(stale_kb.total, 58)
            self.assertNotEqual(stale_kb.total, stale_kb.meta.get("total_hazards"))

        brief = self.compressor.build_brief([], 100000)
        for hazard in self.kb.hazards:
            self.assertIn(hazard.id, brief)
        self.assertIn("T-1", brief)

    def test_build_brief_never_exceeds_budget(self) -> None:
        for budget in (200, 300, 1000, 4000, 8000):
            brief = self.compressor.build_brief(["S1", "S2"], budget)
            self.assertLessEqual(len(brief), budget, f"budget={budget}")

    def test_build_brief_truncation_keeps_highest_severity_first(self) -> None:
        brief = self.compressor.build_brief(["S1", "S2"], 220)
        self.assertIn("已按严重度截断", brief)
        # Everything that survived must be critical (the highest tier).
        kept_lines = [line for line in brief.splitlines() if line.startswith("- ") and " | " in line]
        self.assertTrue(kept_lines)
        for line in kept_lines:
            self.assertIn("critical", line)

    def test_build_brief_scene_scoping(self) -> None:
        s1 = self.compressor.build_brief(["S1"], 8000)
        s2 = self.compressor.build_brief(["S2"], 8000)
        self.assertIn("十不吊", s1)
        self.assertIn("行车六严禁", s1)
        self.assertNotIn("十不吊", s2)
        self.assertNotIn("G-", s1)

    def test_backfill_marks_invalid_id(self) -> None:
        report = HazardReport(hazards=[HazardEvent(id="FAKE-999", evidence="x")])
        self.compressor.backfill(report)
        self.assertTrue(report.hazards[0].invalid_id)
        self.assertEqual(report.hazards[0].category, "未知编号")

    def test_backfill_overwrites_model_supplied_norm_reference(self) -> None:
        report = HazardReport(
            hazards=[HazardEvent(id="D-1-1", norm_reference="模型编造的GB", severity="low")]
        )
        self.compressor.backfill(report)
        hazard = report.hazards[0]
        self.assertNotEqual(hazard.norm_reference, "模型编造的GB")
        self.assertEqual(hazard.norm_reference, self.kb.get("D-1-1").norm_ref)
        self.assertEqual(hazard.severity, self.kb.get("D-1-1").severity)

    def test_duplicate_ids_are_rejected(self) -> None:
        raw = json.loads(KB_PATH.read_text(encoding="utf-8"))
        raw["scenes"][0]["hazard_categories"][0]["hazards"].append(
            dict(raw["scenes"][0]["hazard_categories"][0]["hazards"][0])
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "kb.json"
            path.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
            with self.assertRaises(ValueError):
                KnowledgeBase.load(path)


class SceneRouterAdversarialTest(unittest.TestCase):
    def setUp(self) -> None:
        self.router = SceneRouter(mode="rules")

    @staticmethod
    def _det(label: str) -> Detection:
        return Detection(label=label, bbox=(0.0, 0.0, 1.0, 1.0))

    def test_routing_matrix(self) -> None:
        self.assertEqual(self.router.route([self._det("hook"), self._det("rigging"), self._det("load")]), ["S1"])
        self.assertEqual(self.router.route([self._det("ladder"), self._det("scaffold")]), ["S2"])
        self.assertEqual(self.router.route([self._det("hook"), self._det("ladder")]), ["S1", "S2"])
        self.assertEqual(self.router.route([self._det("car")]), [])
        self.assertEqual(self.router.route([]), [])

    def test_video_vote(self) -> None:
        frames = [
            [self._det("hook")],
            [self._det("hook"), self._det("load")],
            [self._det("ladder")],
        ]
        self.assertEqual(self.router.route_by_video(frames), ["S1"])

    def test_model_mode_uses_classifier_when_rules_are_uncertain(self) -> None:
        router = SceneRouter(mode="auto", classifier=lambda dets: ["S1"])
        self.assertEqual(router.route([self._det("car")]), ["S1"])
        # Rules still win when they are decisive.
        self.assertEqual(router.route([self._det("ladder")]), ["S2"])


# --------------------------------------------------------------------------- #
# 3. Pure-Python image pipeline
# --------------------------------------------------------------------------- #
class ImageIoAdversarialTest(unittest.TestCase):
    def _write(self, tmp: str, name: str, data: bytes) -> Path:
        path = Path(tmp) / name
        path.write_bytes(data)
        return path

    def test_p5_binary_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(tmp, "a.pgm", b"P5\n2 2\n255\n" + bytes([0, 128, 255, 64]))
            image = PgmImage.load(path)
            self.assertEqual((image.width, image.height), (2, 2))
            self.assertEqual(image.pixels, bytes([0, 128, 255, 64]))

    def test_p2_ascii_gray(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "a.pgm"
            path.write_text("P2\n2 1\n255\n10   20\n", encoding="ascii")
            self.assertEqual(PgmImage.load(path).pixels, bytes([10, 20]))

    def test_p3_ascii_rgb_and_p6_binary_rgb(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p3 = Path(tmp) / "a.ppm"
            p3.write_text("P3\n2 1\n255\n0 255 0\n255 0 0\n", encoding="ascii")
            self.assertEqual(PpmImage.load(p3).to_gray_bytes(), bytes([85, 85]))
            p6 = self._write(tmp, "b.ppm", b"P6\n2 1\n255\n" + bytes([0, 255, 0, 255, 0, 0]))
            self.assertEqual(PpmImage.load(p6).to_gray_bytes(), bytes([85, 85]))

    def test_header_comment_and_multi_whitespace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(tmp, "a.pgm", b"P5\n# comment line\n  2   2\n255\n" + bytes([1, 2, 3, 4]))
            image = PgmImage.load(path)
            self.assertEqual((image.width, image.height), (2, 2))
            self.assertEqual(image.pixels, bytes([1, 2, 3, 4]))

    def test_bad_magic_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(tmp, "a.pgm", b"P9\n1 1\n255\n\x00")
            with self.assertRaises(ValueError):
                load_image(path)

    def test_truncated_header_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(tmp, "a.pgm", b"P5\n")
            with self.assertRaises(ValueError):
                load_image(path)

    def test_ahash_identical_images_have_zero_distance(self) -> None:
        pixels = bytes(range(64))
        digest = ahash(pixels, 8, 8, 8)
        self.assertEqual(hamming(digest, digest), 0)
        stats = ImageStats(8)
        image = PgmImage(width=8, height=8, pixels=pixels)
        self.assertEqual(stats.hamming(stats.ahash(image), stats.ahash(image)), 0)

    def test_ahash_distinguishes_structured_images(self) -> None:
        ascending = bytes((i * 3) % 256 for i in range(64))
        descending = bytes(255 - (i * 3) % 256 for i in range(64))
        self.assertGreater(hamming(ahash(ascending, 8, 8, 8), ahash(descending, 8, 8, 8)), 0)

    def test_motion_ratio_hand_computed(self) -> None:
        self.assertAlmostEqual(motion_ratio(bytes([0, 0, 0, 0]), bytes([0, 0, 10, 30]), 25), 0.25)
        self.assertAlmostEqual(motion_ratio(bytes([0, 0, 0, 0]), bytes([0, 0, 10, 30]), 5), 0.5)
        self.assertEqual(motion_ratio(b"", b"", 25), 0.0)

    def test_motion_ratio_mismatched_sizes_uses_overlap(self) -> None:
        # No exception; compares the overlapping prefix only.
        self.assertAlmostEqual(motion_ratio(bytes([0, 0, 0, 0]), bytes([0, 30]), 25), 0.5)

    def test_histogram_hand_computed(self) -> None:
        histogram = histogram16(bytes([0, 16, 255]))
        self.assertAlmostEqual(sum(histogram), 1.0)
        self.assertAlmostEqual(histogram[0], 1 / 3)
        self.assertAlmostEqual(histogram[1], 1 / 3)
        self.assertAlmostEqual(histogram[15], 1 / 3)
        self.assertAlmostEqual(histogram_delta([1.0] + [0.0] * 15, [0.0] * 15 + [1.0]), 1.0)

    # -- findings ---------------------------------------------------------- #
    def test_finding_crlf_header_does_not_shift_the_raster(self) -> None:
        """A CRLF-terminated P5 header must not shift the raster by one byte.

        Netpbm files written on Windows commonly use ``\\r\\n``; the header
        parser skips exactly one whitespace byte after ``maxval``, so the first
        pixel becomes the ``\\n`` instead of the real sample.
        """
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(tmp, "crlf.pgm", b"P5\r\n1 1\r\n255\r\n" + bytes([7]))
            self.assertEqual(PgmImage.load(path).pixels, bytes([7]))

    def test_finding_pixel_count_mismatch_is_reported(self) -> None:
        """A truncated / oversized raster must raise rather than silently pad.

        Silently zero-padding a truncated analysis frame fabricates content and
        can produce spurious motion ratios; truncating an oversized one hides
        corruption.
        """
        with tempfile.TemporaryDirectory() as tmp:
            short = self._write(tmp, "short.pgm", b"P5\n2 2\n255\n" + bytes([1, 2]))
            long = self._write(tmp, "long.pgm", b"P5\n1 1\n255\n" + bytes([1, 2, 3, 4]))
            with self.assertRaises(ValueError):
                PgmImage.load(short)
            with self.assertRaises(ValueError):
                PgmImage.load(long)

    def test_finding_uniform_frames_are_indistinguishable_by_ahash(self) -> None:
        """aHash must not collapse two very different uniform frames.

        Solid black vs solid white are maximally different but both hash to 0,
        so the perceptual de-duplicator treats them as duplicates.
        """
        self.assertGreater(hamming(ahash(bytes([0] * 256), 16, 16, 8), ahash(bytes([255] * 256), 16, 16, 8)), 0)


# --------------------------------------------------------------------------- #
# 4. Config backward compatibility + profile overlay
# --------------------------------------------------------------------------- #
class ConfigAdversarialTest(unittest.TestCase):
    def test_legacy_configs_still_load(self) -> None:
        for path in (LEGACY_MOCK, LEGACY_PROD):
            config = ProjectConfig.load(path)
            self.assertTrue(config.labels)
            self.assertTrue(config.name)

    def test_missing_new_sections_fall_back_to_defaults(self) -> None:
        config = ProjectConfig.load(LEGACY_MOCK)
        self.assertEqual(config.video.sample_fps, 1.0)
        self.assertEqual(config.video.analysis_width, 160)
        self.assertEqual(config.keyframe.max_keyframes, 400)
        self.assertEqual(config.keyframe.phash_size, 8)
        self.assertEqual(config.keyframe.min_gap_seconds, 2.0)
        self.assertEqual(config.knowledge.kb_max_chars, 8000)
        self.assertEqual(config.knowledge.scene_routing, "rules")
        self.assertTrue(config.visualization.enable)
        self.assertEqual(config.run_mode, "quick")

    def test_kb_path_autodiscovery(self) -> None:
        config = ProjectConfig.load(LEGACY_MOCK)
        self.assertTrue(config.kb_path.exists())
        self.assertEqual(config.kb_path.name, "knowledge_base.json")

    def test_to_dict_exposes_new_sections(self) -> None:
        config = ProjectConfig.load(LEGACY_MOCK)
        payload = config.to_dict()
        for key in ("video", "keyframe", "knowledge", "hazard", "visualization", "run_mode", "profiles"):
            self.assertIn(key, payload)

    def test_apply_profile_quick_and_full(self) -> None:
        quick = load_hazard_config(ROOT / "examples" / "hazard_project" / "config" / "project.json", mode="quick")
        self.assertEqual(
            (quick.video.sample_fps, quick.keyframe.max_keyframes, quick.qwen_timeout_seconds),
            (0.5, 30, 90),
        )
        self.assertEqual(
            (quick.qwen_max_retries, quick.qwen_concurrency, quick.qwen_max_tokens),
            (1, 1, 1500),
        )
        self.assertEqual(quick.knowledge.kb_max_chars, 4000)

        full = load_hazard_config(ROOT / "examples" / "hazard_project" / "config" / "project.json", mode="full")
        self.assertEqual(
            (full.video.sample_fps, full.keyframe.max_keyframes, full.qwen_timeout_seconds),
            (1.0, 400, 300),
        )
        self.assertEqual((full.qwen_max_retries, full.qwen_concurrency, full.qwen_max_tokens), (2, 2, 2500))
        self.assertEqual(full.knowledge.kb_max_chars, 8000)

    def test_user_profile_overrides_are_layered(self) -> None:
        raw = json.loads((MOCK_PROJECT / "config" / "project.json").read_text(encoding="utf-8"))
        raw["profiles"] = {"quick": {"keyframe": {"max_keyframes": 7}, "qwen": {"concurrency": 4}}}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "project.json"
            path.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
            config = load_hazard_config(path, mode="quick")
            self.assertEqual(config.keyframe.max_keyframes, 7)
            self.assertEqual(config.qwen_concurrency, 4)

    def test_validate_rejects_illegal_values(self) -> None:
        base = json.loads((MOCK_PROJECT / "config" / "project.json").read_text(encoding="utf-8"))
        cases = [
            ("keyframe", "max_keyframes", 0),
            ("video", "sample_fps", 0),
            ("keyframe", "phash_size", 1),
            ("knowledge", "kb_max_chars", 10),
        ]
        for section, key, value in cases:
            raw = json.loads(json.dumps(base))
            raw.setdefault(section, {})[key] = value
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "project.json"
                path.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
                with self.assertRaises(ValueError, msg=f"{section}.{key}={value}"):
                    ProjectConfig.load(path)


# --------------------------------------------------------------------------- #
# 5. Six-step keyframe logic (pure-Python steps, ffmpeg bypassed)
# --------------------------------------------------------------------------- #
def _frame(index: int, gray: bytes, width: int = 16, height: int = 16, name: str | None = None) -> _AnalysisFrame:
    return _AnalysisFrame(
        index=index,
        timestamp=float(index),
        image_path=Path(name or f"f_{index:04d}.pgm"),
        width=width,
        height=height,
        gray=gray,
    )


class _StubDetector:
    """Minimal detector returning canned detections keyed by file name."""

    def __init__(self, mapping: dict[str, list[Detection]]) -> None:
        self.mapping = mapping

    def detect_many(self, image_paths: list[Path]) -> dict[Path, list[Detection]]:
        return {path.resolve(): self.mapping.get(path.name, []) for path in image_paths}


class KeyframeAdversarialTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = ProjectConfig.load(MOCK_PROJECT / "config" / "project.json")
        self.config.keyframe.motion_min_ratio = 0.002
        self.config.keyframe.dedup_hamming_threshold = 4
        self.config.keyframe.scene_change_threshold = 0.4
        self.config.keyframe.max_keyframes = 400
        self.config.keyframe.min_gap_seconds = 0.0
        self.extractor = KeyframeExtractor(self.config, detector=None)

    def test_motion_step_removes_static_frames(self) -> None:
        frames = [
            _frame(0, bytes([100] * 256)),
            _frame(1, bytes([100] * 256)),  # identical -> static
            _frame(2, bytes([200] * 256)),  # big change -> kept
        ]
        kept, removed = self.extractor._step_motion(frames)
        self.assertEqual([f.index for f in kept], [0, 2])
        self.assertEqual(removed, 1)

    def test_dedup_step_collapses_near_identical_frames(self) -> None:
        base = bytearray([100] * 256)
        variant = bytearray([100] * 256)
        variant[0] = 130  # passes motion, but aHash differs by 1 bit only
        frames = [_frame(0, bytes(base)), _frame(1, bytes(variant))]
        motion_kept, _ = self.extractor._step_motion(frames)
        self.assertEqual([f.index for f in motion_kept], [0, 1])
        dedup_kept, removed = self.extractor._step_dedup(motion_kept)
        self.assertEqual([f.index for f in dedup_kept], [0])
        self.assertEqual(removed, 1)

    def test_force_keep_protects_target_frame_under_cap(self) -> None:
        # F1 survives motion+dedup (very different pattern) and carries a person.
        frames = [
            _frame(0, bytes([100] * 256)),
            _frame(1, bytes([0, 255] * 128), name="f_0001.pgm"),
        ]
        detector = _StubDetector({"f_0001.pgm": [Detection(label="person", bbox=(0.1, 0.1, 0.2, 0.2))]})
        self.extractor.detector = detector
        self.config.keyframe.max_keyframes = 1
        force = self.extractor._step_force_keep_images(frames)
        self.assertEqual(force, [1])
        selected = self.extractor._step_cap_and_gap(frames, [], force)
        self.assertEqual([f.index for f in selected], [1])
        self.assertEqual(self.extractor._reason_for(1, [], force), "force_keep")

    def test_min_gap_and_cap(self) -> None:
        frames = [_frame(i, bytes([(i * 40) % 256] * 256)) for i in range(6)]
        self.config.keyframe.min_gap_seconds = 2.0
        self.config.keyframe.max_keyframes = 10
        selected = self.extractor._step_cap_and_gap(frames, [], [])
        timestamps = [f.timestamp for f in selected]
        self.assertEqual(timestamps, sorted(timestamps))
        for previous, current in zip(timestamps, timestamps[1:]):
            self.assertGreaterEqual(current - previous, 2.0)

        self.config.keyframe.max_keyframes = 2
        self.config.keyframe.min_gap_seconds = 0.0
        capped = self.extractor._step_cap_and_gap(frames, [], [])
        self.assertLessEqual(len(capped), 2)

    def test_image_directory_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            frames_dir = Path(tmp) / "frames"
            frames_dir.mkdir()
            for index, value in enumerate([10, 10, 200, 10]):
                (frames_dir / f"f_{index:04d}.pgm").write_bytes(b"P5\n16 16\n255\n" + bytes([value] * 256))
            config = ProjectConfig.load(MOCK_PROJECT / "config" / "project.json")
            config.workspace_dir = Path(tmp) / "ws"
            config.keyframe.min_gap_seconds = 0.0
            config.keyframe.motion_min_ratio = 0.002
            extractor = KeyframeExtractor(config, detector=None)
            records = extractor.extract(KeyframeSource(source_id="s", frames_dir=frames_dir))
            self.assertEqual(extractor.counts["total_frames"], 4)
            self.assertGreaterEqual(extractor.counts["motion_removed"], 1)
            self.assertTrue(records)

    # -- findings ---------------------------------------------------------- #
    def test_finding_scene_change_frame_survives_deduplication(self) -> None:
        """PRD P0-1.4: a scene-cut frame must not be eaten by de-duplication.

        black -> white is a maximal histogram change (delta = 1.0 > 0.4), yet
        the white frame is removed by the aHash de-duplicator *before* scene
        detection runs, so ``scene_cuts`` stays 0 and the frame is lost.
        """
        with tempfile.TemporaryDirectory() as tmp:
            frames_dir = Path(tmp) / "frames"
            frames_dir.mkdir()
            (frames_dir / "f_0000.pgm").write_bytes(b"P5\n16 16\n255\n" + bytes([0] * 256))
            (frames_dir / "f_0001.pgm").write_bytes(b"P5\n16 16\n255\n" + bytes([255] * 256))
            config = ProjectConfig.load(MOCK_PROJECT / "config" / "project.json")
            config.workspace_dir = Path(tmp) / "ws"
            config.keyframe.min_gap_seconds = 0.0
            config.keyframe.motion_min_ratio = 0.002
            config.keyframe.dedup_hamming_threshold = 4
            config.keyframe.scene_change_threshold = 0.4
            extractor = KeyframeExtractor(config, detector=None)
            records = extractor.extract(KeyframeSource(source_id="s", frames_dir=frames_dir))
            self.assertGreaterEqual(extractor.counts["scene_cuts"], 1)
            self.assertEqual(len(records), 2)

    def test_finding_target_frame_is_not_dropped_by_dedup(self) -> None:
        """PRD P0-1.5: a frame that hits a key object must not be deduped away.

        The person-bearing frame is a near-duplicate of the previous frame, so
        the de-duplicator drops it before the target-driven step can rescue it.
        """
        with tempfile.TemporaryDirectory() as tmp:
            frames_dir = Path(tmp) / "frames"
            frames_dir.mkdir()
            (frames_dir / "f_0000.pgm").write_bytes(b"P5\n16 16\n255\n" + bytes([100] * 256))
            variant = bytearray([100] * 256)
            variant[0] = 130
            (frames_dir / "f_0001.pgm").write_bytes(b"P5\n16 16\n255\n" + bytes(variant))
            config = ProjectConfig.load(MOCK_PROJECT / "config" / "project.json")
            config.workspace_dir = Path(tmp) / "ws"
            config.keyframe.min_gap_seconds = 0.0
            config.keyframe.motion_min_ratio = 0.002
            config.keyframe.dedup_hamming_threshold = 4
            detector = _StubDetector({"f_0001.pgm": [Detection(label="person", bbox=(0.1, 0.1, 0.2, 0.2))]})
            extractor = KeyframeExtractor(config, detector=detector)
            records = extractor.extract(KeyframeSource(source_id="s", frames_dir=frames_dir))
            names = {Path(record.relative_path).name for record in records}
            self.assertIn("f_0001.pgm", names)


# --------------------------------------------------------------------------- #
# 6. Graceful degradation & no-silent-drop contract
# --------------------------------------------------------------------------- #
class GracefulDegradationTest(unittest.TestCase):
    def _clean_config(self, tmp: str):
        project = Path(tmp) / "hazard_mock_project"
        shutil.copytree(MOCK_PROJECT, project)
        shutil.rmtree(project / "workspace", ignore_errors=True)
        shutil.rmtree(project / "input" / "frames", ignore_errors=True)
        shutil.copytree(ROOT / "hazard-detection", Path(tmp) / "hazard-detection")
        config = load_hazard_config(project / "config" / "project.json", mode="quick")
        config.qwen_provider = "mock"
        config.yolo_detector = "sidecar"
        config.yolo_trainer = "mock"
        return project, config

    def test_ffmpeg_runner_reports_unavailable(self) -> None:
        self.assertFalse(FfmpegRunner("definitely-not-a-real-binary").available())

    def test_video_without_ffmpeg_raises_clear_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _, config = self._clean_config(tmp)
            # Force ffmpeg to be "absent" instead of relying on the ambient PATH:
            # otherwise this test flips to a RuntimeError as soon as a real
            # ffmpeg is installed (it is not hermetic w.r.t. the environment).
            config.video.ffmpeg_binary = "definitely-not-a-real-ffmpeg-binary"
            dummy = Path(tmp) / "clip.mp4"
            dummy.write_bytes(b"not a real video")
            extractor = KeyframeExtractor(config, detector=None)
            with self.assertRaises(FfmpegUnavailableError) as ctx:
                extractor.extract(
                    KeyframeSource(source_id="clip", video_path=dummy, frames_dir=Path(tmp) / "empty")
                )
            self.assertIn("ffmpeg", str(ctx.exception).lower())

    def test_doctor_reports_video_keyframe_knowledge(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _, config = self._clean_config(tmp)
            report = build_doctor_report(config)
            self.assertIn("video", report)
            self.assertIn("keyframe", report)
            self.assertIn("knowledge", report)
            self.assertEqual(report["knowledge"]["hazard_count"], 58)
            self.assertFalse(report["knowledge"]["meta_count_stale"])
            self.assertIn(report["status"], {"ready", "issues_found"})

    def test_doctor_flags_stale_meta_count_when_meta_disagrees(self) -> None:
        """Doctor must still warn when ``meta`` disagrees with the real count.

        The shipped knowledge base is now consistent, so this test corrupts the
        copied knowledge base to keep the staleness warning covered.
        """
        with tempfile.TemporaryDirectory() as tmp:
            _, config = self._clean_config(tmp)
            kb_copy = Path(tmp) / "hazard-detection" / "knowledge_base.json"
            raw = json.loads(kb_copy.read_text(encoding="utf-8"))
            raw.setdefault("meta", {})["total_hazards"] = 999
            kb_copy.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")

            report = build_doctor_report(config)
            self.assertTrue(report["knowledge"]["meta_count_stale"])
            self.assertEqual(report["knowledge"]["hazard_count"], 58)
            self.assertTrue(
                any("meta_count_stale" in warning for warning in report["warnings"]),
                "应上浮 knowledge_base_meta_count_stale 告警",
            )

    def test_ultralytics_detector_without_binary_raises_clear_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _, config = self._clean_config(tmp)
            config.yolo_detector = "ultralytics"
            config.yolo_binary = "definitely-not-a-real-yolo-binary"
            detector = build_detector(config)
            with tempfile.TemporaryDirectory() as img_tmp:
                image = Path(img_tmp) / "a.pgm"
                image.write_bytes(b"P5\n1 1\n255\n\x00")
                with self.assertRaises(FileNotFoundError):
                    detector.detect_many([image])

    def test_unknown_detector_mode_raises_value_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _, config = self._clean_config(tmp)
            config.yolo_detector = "nonsense"
            with self.assertRaises(ValueError):
                build_detector(config)

    def test_failed_annotation_is_queued_and_raw_response_preserved(self) -> None:
        """No-silent-drop contract: a parse failure must be queued + retained."""
        with tempfile.TemporaryDirectory() as tmp:
            project, config = self._clean_config(tmp)
            # Regenerate the mock frames (removed above) and extract candidates.
            from hazard_annotation.demo import build_mock_assets

            build_mock_assets(project)
            run_extraction(config)

            class _BoomAnnotator:
                def annotate_combined(self, candidate, prompt):  # noqa: D401
                    raise _AnnotationParseError(
                        "simulated parse failure",
                        {"provider": "mock", "content": "<raw model response>", "parsed": None},
                    )

            with mock.patch.object(annotation_module, "build_qwen_annotator", lambda cfg: _BoomAnnotator()):
                summary = annotation_module.run_annotation(config)

            self.assertGreater(summary["annotated_frames"], 0)
            self.assertEqual(summary["review_queue_count"], summary["annotated_frames"])
            self.assertGreater(summary["annotation_errors"], 0)

            queue = read_jsonl(config.review_queue_path)
            self.assertEqual(len(queue), summary["annotated_frames"])
            self.assertTrue(all("json_parse_failed" in record["review_reasons"] for record in queue))

            raw_files = sorted(config.review_raw_responses_dir.rglob("*.json"))
            self.assertEqual(len(raw_files), summary["annotated_frames"])
            raw = read_json(raw_files[0])
            self.assertEqual(raw.get("content"), "<raw model response>")

    def test_mock_demo_runs_clean_end_to_end(self) -> None:
        from hazard_annotation.__main__ import main

        with tempfile.TemporaryDirectory() as tmp:
            project, _ = self._clean_config(tmp)
            config_path = project / "config" / "project.json"
            exit_code = main(["--config", str(config_path), "all", "--mock"])
            self.assertEqual(exit_code, 0)
            workspace = project / "workspace"
            self.assertTrue((workspace / "extracted" / "candidates.jsonl").exists())
            self.assertTrue((workspace / "hazards" / "report.json").exists())
            self.assertTrue(list((workspace / "hazards").rglob("*.json")))
            self.assertTrue(list((workspace / "review" / "labels").rglob("*.txt")))
            self.assertTrue((workspace / "visualization" / "report.html").exists())
            report = read_json(workspace / "hazards" / "report.json")
            self.assertEqual(report["overall_risk"], "critical")
            # every hazard id must hit the knowledge base
            kb = KnowledgeBase.load(KB_PATH)
            for record in read_jsonl(workspace / "review" / "annotations.jsonl"):
                for hazard in record["hazard_report"]["hazards"]:
                    self.assertIn(hazard["id"], kb.id_index)
                    self.assertFalse(hazard["invalid_id"])


# --------------------------------------------------------------------------- #
# 6b. PRD acceptance-criterion checks
# --------------------------------------------------------------------------- #
class PrdAcceptanceFindingsTest(unittest.TestCase):
    def _clean_config(self, tmp: str):
        project = Path(tmp) / "hazard_mock_project"
        shutil.copytree(MOCK_PROJECT, project)
        shutil.rmtree(project / "workspace", ignore_errors=True)
        shutil.rmtree(project / "input" / "frames", ignore_errors=True)
        shutil.copytree(ROOT / "hazard-detection", Path(tmp) / "hazard-detection")
        config = load_hazard_config(project / "config" / "project.json", mode="quick")
        config.qwen_provider = "mock"
        config.yolo_detector = "sidecar"
        config.yolo_trainer = "mock"
        return project, config

    def test_knowledge_base_norm_ref_coverage_meets_prd_target(self) -> None:
        """Knowledge-base ``norm_ref`` coverage must clear the PRD completeness bar.

        Originally this test *documented a gap*: only a minority of hazards
        carried ``norm_ref``, so local back-fill left ``norm_reference`` empty
        for most hazards and channel-B completeness stalled at ~27%.

        The gap has since been closed by filling all 47 missing entries, so the
        test now asserts the target is met instead of the gap existing.
        """
        kb = KnowledgeBase.load(KB_PATH)
        with_norm = [hazard.id for hazard in kb.hazards if hazard.norm_ref.strip()]
        self.assertGreaterEqual(len(with_norm) / kb.total, 0.95)
        self.assertEqual(len(with_norm), kb.total, "所有隐患条目都应具备 norm_ref")
        # spot-check a few well-known entries
        self.assertTrue(kb.get("D-1-1").norm_ref)
        self.assertTrue(kb.get("D-5-3").norm_ref)

    def test_finding_demo_required_field_completeness_meets_prd(self) -> None:
        """PRD P0-4: channel-B required-field completeness must be >= 95%.

        The shipped demo reports ~27% because 47/58 knowledge-base hazards have
        no ``norm_ref``; ``norm_reference`` therefore stays empty after
        back-fill and the field is counted as missing.
        """
        from hazard_annotation.__main__ import main

        with tempfile.TemporaryDirectory() as tmp:
            project, _ = self._clean_config(tmp)
            exit_code = main(["--config", str(project / "config" / "project.json"), "all", "--mock"])
            self.assertEqual(exit_code, 0)
            summary = read_json(project / "workspace" / "review" / "summary.json")
            self.assertGreaterEqual(summary["required_field_completeness"], 0.95)


# --------------------------------------------------------------------------- #
# 7. cli.py --config regression (verify the engineer's justification)
# --------------------------------------------------------------------------- #
class CliConfigRegressionTest(unittest.TestCase):
    def test_pre_subcommand_config_is_preserved(self) -> None:
        parser = build_traffic_parser()
        args = parser.parse_args(["--config", "/tmp/example/project.json", "run-loop"])
        self.assertEqual(args.config, "/tmp/example/project.json")

    def test_post_subcommand_config_is_honoured(self) -> None:
        parser = build_traffic_parser()
        args = parser.parse_args(["run-loop", "--config", "/tmp/other/project.json"])
        self.assertEqual(args.config, "/tmp/other/project.json")

    def test_default_config_when_none_given(self) -> None:
        parser = build_traffic_parser()
        args = parser.parse_args(["status"])
        self.assertEqual(args.config, "examples/mock_project/config/project.json")

    def test_legacy_subparser_default_clobbers_pre_subcommand_config(self) -> None:
        """Documents *why* the SUPPRESS change is necessary, not cosmetic.

        Reconstructing the previous parser (subparser ``--config`` with a
        literal default) shows argparse overwrites the pre-subcommand value with
        the subparser default. That is exactly the order-dependence the engineer
        described: ``main(["--config", temp, "run-loop"])`` silently ran against
        the repository's ``examples/mock_project`` instead of the temp copy.
        """
        parser = argparse.ArgumentParser(prog="legacy")
        parser.add_argument("--config", default="examples/mock_project/config/project.json")
        subparsers = parser.add_subparsers(dest="command", required=True)
        run_loop = subparsers.add_parser("run-loop")
        run_loop.add_argument("--config", default="examples/mock_project/config/project.json")

        args = parser.parse_args(["--config", "/tmp/temp_project/config/project.json", "run-loop"])
        # The passed path is lost — this is the bug the SUPPRESS default fixes.
        self.assertEqual(args.config, "examples/mock_project/config/project.json")


if __name__ == "__main__":
    unittest.main()
