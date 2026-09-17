"""抽帧断点续跑（checkpoint / resume）的测试。

核心不变量：**绝不拿旧结果冒充新配置的产物**。

抽帧是小时级任务，复用能省大量时间；但复用必须建立在"输入和配置都没变"
的证明之上。这组测试就盯这一点：

* 配置指纹对抽帧参数敏感 —— 改一个阈值就必须重算；
* 数据源指纹对输入敏感 —— 换了文件/改了目录内容就必须重算；
* 断点文件版本或指纹不符时整体作废；
* 分片文件缺失时不得假装可复用；
* 写入是原子的，不留半截文件。
"""

from __future__ import annotations

import json
import os
import tempfile
import time
import unittest
from pathlib import Path

from hazard_annotation.checkpoint import (
    CHECKPOINT_VERSION,
    ExtractionCheckpoint,
    config_fingerprint,
    safe_part_name,
    source_signature,
)
from hazard_annotation.keyframes import KeyframeSource
from hazard_annotation.models import KeyframeRecord
from traffic_cv_closed_loop.config import ProjectConfig

ROOT = Path(__file__).resolve().parents[1]
MOCK_PROJECT = ROOT / "examples" / "hazard_mock_project"


def _config(tmp: str) -> ProjectConfig:
    config = ProjectConfig.load(MOCK_PROJECT / "config" / "project.json")
    config.workspace_dir = Path(tmp) / "ws"
    return config


def _write_frames(frames_dir: Path, frames: list[tuple[str, bytes]]) -> None:
    frames_dir.mkdir(parents=True, exist_ok=True)
    for name, pixels in frames:
        (frames_dir / name).write_bytes(b"P5\n16 16\n255\n" + pixels)


def _record(source_id: str, name: str) -> KeyframeRecord:
    return KeyframeRecord(
        source_id=source_id,
        relative_path=f"{source_id}/{name}",
        original_path=f"/orig/{name}",
        extracted_path=f"/extracted/{name}",
        timestamp=1.5,
        reason="scene_change",
        metadata={"probe": True},
    )


class ConfigFingerprintTest(unittest.TestCase):
    def test_stable_for_identical_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = _config(tmp)
            self.assertEqual(config_fingerprint(config), config_fingerprint(config))

    def test_changes_when_keyframe_threshold_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = _config(tmp)
            before = config_fingerprint(config)
            config.keyframe.motion_min_ratio = 0.5
            self.assertNotEqual(before, config_fingerprint(config))

    def test_changes_when_labels_change(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = _config(tmp)
            before = config_fingerprint(config)
            config.labels = list(config.labels) + ["extra_label"]
            self.assertNotEqual(before, config_fingerprint(config))

    def test_changes_when_detector_mode_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = _config(tmp)
            before = config_fingerprint(config)
            config.yolo_detector = "ultralytics"
            self.assertNotEqual(before, config_fingerprint(config))

    def test_video_path_is_not_part_of_config_fingerprint(self) -> None:
        """视频路径由 source_signature 负责，不该让配置指纹跟着变。"""
        with tempfile.TemporaryDirectory() as tmp:
            config = _config(tmp)
            before = config_fingerprint(config)
            config.video.video_path = "some/other/video.mp4"
            self.assertEqual(before, config_fingerprint(config))


class SourceSignatureTest(unittest.TestCase):
    def test_video_signature_changes_with_mtime(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            video = Path(tmp) / "clip.mp4"
            video.write_bytes(b"fake-video-bytes")
            source = KeyframeSource(source_id="v", video_path=video)
            before = source_signature(source)
            os.utime(video, (time.time() + 60, time.time() + 60))
            self.assertNotEqual(before, source_signature(source))

    def test_video_signature_changes_with_size(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            video = Path(tmp) / "clip.mp4"
            video.write_bytes(b"short")
            source = KeyframeSource(source_id="v", video_path=video)
            before = source_signature(source)
            video.write_bytes(b"a-much-longer-payload")
            self.assertNotEqual(before, source_signature(source))

    def test_missing_video_is_still_signable(self) -> None:
        source = KeyframeSource(source_id="v", video_path=Path("/nope/missing.mp4"))
        self.assertTrue(source_signature(source))

    def test_frames_dir_signature_changes_when_content_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            frames_dir = Path(tmp) / "frames"
            _write_frames(frames_dir, [("f_0000.pgm", bytes([1] * 256))])
            source = KeyframeSource(source_id="d", frames_dir=frames_dir)
            before = source_signature(source)
            _write_frames(frames_dir, [("f_0001.pgm", bytes([2] * 256))])
            self.assertNotEqual(before, source_signature(source))

    def test_frames_dir_signature_stable_without_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            frames_dir = Path(tmp) / "frames"
            _write_frames(frames_dir, [("f_0000.pgm", bytes([1] * 256))])
            source = KeyframeSource(source_id="d", frames_dir=frames_dir)
            self.assertEqual(source_signature(source), source_signature(source))


class SafePartNameTest(unittest.TestCase):
    def test_plain_name_is_preserved(self) -> None:
        self.assertEqual(safe_part_name("hoist_scene"), "hoist_scene")

    def test_path_separators_are_neutralised(self) -> None:
        self.assertNotIn("/", safe_part_name("../../etc/passwd"))
        self.assertNotIn("\\", safe_part_name("..\\..\\windows\\system32"))

    def test_spaces_and_unicode_are_replaced(self) -> None:
        self.assertEqual(safe_part_name("吊装 作业"), "source")

    def test_overlong_name_is_truncated_with_digest(self) -> None:
        result = safe_part_name("x" * 200)
        self.assertLessEqual(len(result), 64)
        self.assertIn("_", result)

    def test_empty_name_falls_back(self) -> None:
        self.assertEqual(safe_part_name(""), "source")
        self.assertEqual(safe_part_name("..."), "source")


class CheckpointRoundTripTest(unittest.TestCase):
    def test_mark_done_then_reuse(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = _config(tmp)
            frames_dir = Path(tmp) / "frames"
            _write_frames(frames_dir, [("f_0000.pgm", bytes([1] * 256))])
            source = KeyframeSource(source_id="s1", frames_dir=frames_dir)

            first = ExtractionCheckpoint(config)
            first.load()
            self.assertFalse(first.is_reusable(source))
            first.mark_done(source, [_record("s1", "f_0000.pgm")], {"final_keyframes": 1}, [])
            self.assertTrue(first.part_path("s1").exists())
            self.assertTrue(first.progress_path.exists())

            second = ExtractionCheckpoint(config)
            second.load()
            self.assertTrue(second.is_reusable(source))
            records = second.read_records("s1")
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].relative_path, "s1/f_0000.pgm")
            self.assertEqual(records[0].reason, "scene_change")
            self.assertTrue(records[0].metadata.get("probe"))
            self.assertEqual(second.progress_of("s1").counts["final_keyframes"], 1)

    def test_config_change_invalidates_everything(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = _config(tmp)
            frames_dir = Path(tmp) / "frames"
            _write_frames(frames_dir, [("f_0000.pgm", bytes([1] * 256))])
            source = KeyframeSource(source_id="s1", frames_dir=frames_dir)

            first = ExtractionCheckpoint(config)
            first.load()
            first.mark_done(source, [_record("s1", "f_0000.pgm")], {}, [])

            config.keyframe.motion_min_ratio = 0.9
            second = ExtractionCheckpoint(config)
            second.load()
            self.assertFalse(second.is_reusable(source))
            self.assertEqual(second.invalidation_reason, "config_changed")

    def test_version_mismatch_invalidates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = _config(tmp)
            frames_dir = Path(tmp) / "frames"
            _write_frames(frames_dir, [("f_0000.pgm", bytes([1] * 256))])
            source = KeyframeSource(source_id="s1", frames_dir=frames_dir)

            checkpoint = ExtractionCheckpoint(config)
            checkpoint.load()
            checkpoint.mark_done(source, [_record("s1", "f_0000.pgm")], {}, [])

            payload = json.loads(checkpoint.progress_path.read_text(encoding="utf-8"))
            payload["version"] = CHECKPOINT_VERSION + 1
            checkpoint.progress_path.write_text(
                json.dumps(payload, ensure_ascii=False), encoding="utf-8"
            )

            reloaded = ExtractionCheckpoint(config)
            reloaded.load()
            self.assertFalse(reloaded.is_reusable(source))
            self.assertEqual(reloaded.invalidation_reason, "checkpoint_version_changed")

    def test_source_change_invalidates_that_source_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = _config(tmp)
            dir_a = Path(tmp) / "a"
            dir_b = Path(tmp) / "b"
            _write_frames(dir_a, [("f_0000.pgm", bytes([1] * 256))])
            _write_frames(dir_b, [("f_0000.pgm", bytes([2] * 256))])
            source_a = KeyframeSource(source_id="a", frames_dir=dir_a)
            source_b = KeyframeSource(source_id="b", frames_dir=dir_b)

            first = ExtractionCheckpoint(config)
            first.load()
            first.mark_done(source_a, [_record("a", "f_0000.pgm")], {}, [])
            first.mark_done(source_b, [_record("b", "f_0000.pgm")], {}, [])

            # 只改 A 的内容
            _write_frames(dir_a, [("f_0000.pgm", bytes([3] * 256))])

            second = ExtractionCheckpoint(config)
            second.load()
            self.assertFalse(second.is_reusable(source_a))
            self.assertTrue(second.is_reusable(source_b))

    def test_missing_part_file_is_not_reusable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = _config(tmp)
            frames_dir = Path(tmp) / "frames"
            _write_frames(frames_dir, [("f_0000.pgm", bytes([1] * 256))])
            source = KeyframeSource(source_id="s1", frames_dir=frames_dir)

            checkpoint = ExtractionCheckpoint(config)
            checkpoint.load()
            checkpoint.mark_done(source, [_record("s1", "f_0000.pgm")], {}, [])
            checkpoint.part_path("s1").unlink()

            reloaded = ExtractionCheckpoint(config)
            reloaded.load()
            self.assertFalse(reloaded.is_reusable(source))

    def test_disabled_checkpoint_never_reuses_and_never_writes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = _config(tmp)
            frames_dir = Path(tmp) / "frames"
            _write_frames(frames_dir, [("f_0000.pgm", bytes([1] * 256))])
            source = KeyframeSource(source_id="s1", frames_dir=frames_dir)

            writer = ExtractionCheckpoint(config)
            writer.load()
            writer.mark_done(source, [_record("s1", "f_0000.pgm")], {}, [])

            # 换一个干净工作区，确认「关闭」状态下不读也不写
            fresh_config = _config(tempfile.mkdtemp())
            disabled = ExtractionCheckpoint(fresh_config, enabled=False)
            disabled.load()
            self.assertFalse(disabled.is_reusable(source))
            disabled.mark_done(source, [_record("s1", "f_0000.pgm")], {}, [])
            self.assertFalse(disabled.progress_path.exists())
            self.assertFalse(disabled.part_path("s1").exists())

            # 同一个工作区里，关闭状态也不应复用已有断点
            disabled_same_ws = ExtractionCheckpoint(config, enabled=False)
            disabled_same_ws.load()
            self.assertFalse(disabled_same_ws.is_reusable(source))

    def test_flush_leaves_no_temp_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = _config(tmp)
            checkpoint = ExtractionCheckpoint(config)
            checkpoint.load()
            checkpoint.flush()
            leftovers = [
                path.name
                for path in checkpoint.progress_path.parent.iterdir()
                if path.name.startswith(".tmp-")
            ]
            self.assertEqual(leftovers, [])

    def test_corrupt_progress_file_is_tolerated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = _config(tmp)
            checkpoint = ExtractionCheckpoint(config)
            checkpoint.progress_path.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.progress_path.write_text("{not valid json", encoding="utf-8")

            reloaded = ExtractionCheckpoint(config)
            reloaded.load()  # 不应抛异常
            self.assertEqual(reloaded.summary()["reused_count"], 0)

    def test_summary_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = _config(tmp)
            checkpoint = ExtractionCheckpoint(config)
            summary = checkpoint.summary()
            for key in (
                "enabled",
                "fingerprint",
                "invalidation_reason",
                "reused_sources",
                "recomputed_sources",
                "reused_count",
                "recomputed_count",
            ):
                self.assertIn(key, summary)


class ExtractionResumeIntegrationTest(unittest.TestCase):
    """跑真实的 run_extraction，验证第二次调用复用而不是重算。"""

    def _project(self, tmp: str) -> Path:
        project = Path(tmp) / "proj"
        (project / "config").mkdir(parents=True, exist_ok=True)
        (project / "input" / "frames").mkdir(parents=True, exist_ok=True)

        raw = json.loads((MOCK_PROJECT / "config" / "project.json").read_text(encoding="utf-8"))
        raw["qwen"]["provider"] = "mock"
        raw["yolo"]["detector"] = "sidecar"
        raw["yolo"]["trainer"] = "mock"
        raw["video"]["video_path"] = ""
        raw["keyframe"]["min_gap_seconds"] = 0.0
        raw["keyframe"]["motion_min_ratio"] = 0.0
        (project / "config" / "project.json").write_text(
            json.dumps(raw, ensure_ascii=False), encoding="utf-8"
        )

        _write_frames(
            project / "input" / "frames" / "scene_a",
            [("f_0000.pgm", bytes([10] * 256)), ("f_0001.pgm", bytes([240] * 256))],
        )
        return project

    def test_second_run_reuses_first_run(self) -> None:
        from hazard_annotation.pipeline.extraction import run_extraction

        with tempfile.TemporaryDirectory() as tmp:
            project = self._project(tmp)
            config = ProjectConfig.load(project / "config" / "project.json")

            first = run_extraction(config)
            self.assertEqual(first["checkpoint"]["reused_count"], 0)
            self.assertGreaterEqual(first["checkpoint"]["recomputed_count"], 1)

            second = run_extraction(config)
            self.assertGreater(second["checkpoint"]["reused_count"], 0)
            self.assertEqual(second["checkpoint"]["recomputed_count"], 0)
            # 复用后产出必须一致
            self.assertEqual(
                first["totals"]["final_keyframes"], second["totals"]["final_keyframes"]
            )

    def test_no_resume_forces_recompute(self) -> None:
        from hazard_annotation.pipeline.extraction import run_extraction

        with tempfile.TemporaryDirectory() as tmp:
            project = self._project(tmp)
            config = ProjectConfig.load(project / "config" / "project.json")

            run_extraction(config)
            forced = run_extraction(config, resume=False)
            self.assertFalse(forced["checkpoint"]["enabled"])
            self.assertEqual(forced["checkpoint"]["reused_count"], 0)
            self.assertGreaterEqual(forced["checkpoint"]["recomputed_count"], 1)

    def test_changing_keyframe_config_forces_recompute(self) -> None:
        from hazard_annotation.pipeline.extraction import run_extraction

        with tempfile.TemporaryDirectory() as tmp:
            project = self._project(tmp)
            config = ProjectConfig.load(project / "config" / "project.json")

            run_extraction(config)
            # mock 配置里 motion_min_ratio 是 0.0，这里改成一个确实不同的值
            self.assertEqual(config.keyframe.motion_min_ratio, 0.0)
            config.keyframe.motion_min_ratio = 0.4
            after = run_extraction(config)
            self.assertEqual(after["checkpoint"]["reused_count"], 0)
            self.assertEqual(after["checkpoint"]["invalidation_reason"], "config_changed")
            self.assertGreaterEqual(after["checkpoint"]["recomputed_count"], 1)


if __name__ == "__main__":
    unittest.main()
