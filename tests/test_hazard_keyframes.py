"""Regression tests for the six-step keyframe extraction ordering (finding F2).

Scene-change detection and target force-keep must run on the *original* sampled
sequence, and motion/de-dup filtering must never drop a protected frame. These
tests lock that invariant at both the step level and end-to-end.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from hazard_annotation.keyframes import KeyframeExtractor, KeyframeSource
from traffic_cv_closed_loop.config import ProjectConfig
from traffic_cv_closed_loop.models import Detection

ROOT = Path(__file__).resolve().parents[1]
MOCK_PROJECT = ROOT / "examples" / "hazard_mock_project"


class _StubDetector:
    def __init__(self, mapping: dict[str, list[Detection]]) -> None:
        self.mapping = mapping

    def detect_many(self, image_paths: list[Path]) -> dict[Path, list[Detection]]:
        return {path.resolve(): self.mapping.get(path.name, []) for path in image_paths}


def _write_frames(frames_dir: Path, frames: list[tuple[str, bytes]]) -> None:
    frames_dir.mkdir(parents=True, exist_ok=True)
    for name, pixels in frames:
        (frames_dir / name).write_bytes(b"P5\n16 16\n255\n" + pixels)


class KeyframeOrderingRegressionTest(unittest.TestCase):
    def _extractor(self, tmp: str, detector: object | None = None) -> KeyframeExtractor:
        config = ProjectConfig.load(MOCK_PROJECT / "config" / "project.json")
        config.workspace_dir = Path(tmp) / "ws"
        config.keyframe.min_gap_seconds = 0.0
        config.keyframe.motion_min_ratio = 0.002
        config.keyframe.dedup_hamming_threshold = 4
        config.keyframe.scene_change_threshold = 0.4
        return KeyframeExtractor(config, detector=detector)

    def test_scene_change_frame_survives_dedup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            frames_dir = Path(tmp) / "frames"
            _write_frames(
                frames_dir,
                [("f_0000.pgm", bytes([0] * 256)), ("f_0001.pgm", bytes([255] * 256))],
            )
            extractor = self._extractor(tmp)
            records = extractor.extract(KeyframeSource(source_id="s", frames_dir=frames_dir))
            self.assertGreaterEqual(extractor.counts["scene_cuts"], 1)
            self.assertEqual(len(records), 2)
            reasons = {Path(record.relative_path).name: record.reason for record in records}
            self.assertIn("scene_change", reasons["f_0001.pgm"])

    def test_target_frame_is_not_dropped_by_dedup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            frames_dir = Path(tmp) / "frames"
            variant = bytearray([100] * 256)
            variant[0] = 130  # near-duplicate of f_0000 -> would be de-duped
            _write_frames(
                frames_dir,
                [("f_0000.pgm", bytes([100] * 256)), ("f_0001.pgm", bytes(variant))],
            )
            detector = _StubDetector(
                {"f_0001.pgm": [Detection(label="person", bbox=(0.1, 0.1, 0.2, 0.2))]}
            )
            extractor = self._extractor(tmp, detector=detector)
            records = extractor.extract(KeyframeSource(source_id="s", frames_dir=frames_dir))
            names = {Path(record.relative_path).name for record in records}
            self.assertIn("f_0001.pgm", names)

    def test_protected_frames_are_never_removed_by_motion_or_dedup(self) -> None:
        extractor = self._extractor(tempfile.mkdtemp())
        from hazard_annotation.keyframes import _AnalysisFrame

        frames = [
            _AnalysisFrame(index=0, timestamp=0.0, image_path=Path("f0.pgm"), width=16, height=16, gray=bytes([10] * 256)),
            _AnalysisFrame(index=1, timestamp=1.0, image_path=Path("f1.pgm"), width=16, height=16, gray=bytes([10] * 256)),
        ]
        motion_kept, _ = extractor._step_motion(frames, protected={1})
        self.assertEqual([frame.index for frame in motion_kept], [0, 1])
        dedup_kept, _ = extractor._step_dedup(frames, protected={1})
        self.assertEqual([frame.index for frame in dedup_kept], [0, 1])


class KeyframeMinGapRegressionTest(unittest.TestCase):
    """N2: ``min_gap`` must never silently veto a protected (scene/target) frame."""

    def _extractor(self, tmp: str) -> KeyframeExtractor:
        config = ProjectConfig.load(MOCK_PROJECT / "config" / "project.json")
        config.workspace_dir = Path(tmp) / "ws"
        config.keyframe.motion_min_ratio = 0.002
        config.keyframe.dedup_hamming_threshold = 4
        config.keyframe.scene_change_threshold = 0.4
        return KeyframeExtractor(config, detector=None)

    def test_protected_frames_are_exempt_from_min_gap(self) -> None:
        from hazard_annotation.keyframes import _AnalysisFrame

        extractor = self._extractor(tempfile.mkdtemp())
        extractor.config.keyframe.min_gap_seconds = 2.0
        extractor.config.keyframe.max_keyframes = 100
        frames = [
            _AnalysisFrame(
                index=index,
                timestamp=float(index),
                image_path=Path(f"f{index}.pgm"),
                width=16,
                height=16,
                gray=bytes([(index * 40) % 256] * 256),
            )
            for index in range(5)
        ]
        # Protected frames 1 and 2 are only 1.0 s apart (< min_gap=2.0): both must
        # survive. Frame 0 sits within min_gap of frame 1 and is not protected,
        # so it is still dropped; frame 4 clears the gap and is kept.
        selected = extractor._step_cap_and_gap(frames, scene_indices=[1, 2], force_indices=[])
        self.assertEqual([frame.index for frame in selected], [1, 2, 4])

    def test_scene_change_frames_survive_min_gap_end_to_end(self) -> None:
        """QA repro: 4 uniform frames + min_gap=2.0 must keep every scene cut."""
        with tempfile.TemporaryDirectory() as tmp:
            frames_dir = Path(tmp) / "frames"
            _write_frames(
                frames_dir,
                [
                    ("f_0000.pgm", bytes([100] * 256)),
                    ("f_0001.pgm", bytes([255] * 256)),
                    ("f_0002.pgm", bytes([200] * 256)),
                    ("f_0003.pgm", bytes([255] * 256)),
                ],
            )
            extractor = self._extractor(tmp)
            extractor.config.keyframe.min_gap_seconds = 2.0
            records = extractor.extract(KeyframeSource(source_id="s", frames_dir=frames_dir))
            names = {Path(record.relative_path).name for record in records}
            # 100->255, 255->200, 200->255 are all maximal histogram changes.
            self.assertEqual(extractor.counts["scene_cuts"], 3)
            self.assertIn("f_0001.pgm", names)
            self.assertIn("f_0002.pgm", names)  # the frame QA saw silently dropped
            self.assertIn("f_0003.pgm", names)
            self.assertEqual(extractor.counts["final_keyframes"], len(records))
            self.assertEqual(extractor.counts["protected_dropped"], 0)

    def test_protected_drop_is_counted_and_warned(self) -> None:
        """If max_keyframes forces a protected frame out it must be counted + warned."""
        with tempfile.TemporaryDirectory() as tmp:
            frames_dir = Path(tmp) / "frames"
            _write_frames(
                frames_dir,
                [
                    ("f_0000.pgm", bytes([100] * 256)),
                    ("f_0001.pgm", bytes([255] * 256)),
                    ("f_0002.pgm", bytes([200] * 256)),
                    ("f_0003.pgm", bytes([255] * 256)),
                ],
            )
            extractor = self._extractor(tmp)
            extractor.config.keyframe.min_gap_seconds = 0.0
            extractor.config.keyframe.max_keyframes = 1  # forces protected frames out
            records = extractor.extract(KeyframeSource(source_id="s", frames_dir=frames_dir))
            self.assertEqual(len(records), 1)
            self.assertGreaterEqual(extractor.counts["protected_dropped"], 1)
            self.assertTrue(
                any("protected_dropped" in warning for warning in extractor.warnings),
                extractor.warnings,
            )


if __name__ == "__main__":
    unittest.main()
