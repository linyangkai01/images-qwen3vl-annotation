"""Real-ffmpeg regression tests for the video keyframe path (findings F9-F12).

These tests synthesise a short clip with ffmpeg (``lavfi`` sources) and exercise
the real CLI. When no ffmpeg binary is available the whole class is skipped, so
the suite still passes on machines without ffmpeg.

Synthetic clip: 7 s @ 1 fps -> black (t=0..2), white (t=3..4), red (t=5..6);
scene cuts happen at t=3 and t=5, i.e. 1-based analysis-frame indices 4 and 6.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from hazard_annotation.ffmpeg import FfmpegRunner
from hazard_annotation.keyframes import KeyframeExtractor, KeyframeSource
from traffic_cv_closed_loop.config import ProjectConfig

ROOT = Path(__file__).resolve().parents[1]
MOCK_PROJECT = ROOT / "examples" / "hazard_mock_project"

_KNOWN_FFMPEG = Path(
    r"C:\Users\81945\.workbuddy-ai\binaries\ffmpeg\ffmpeg-9.0.1-essentials_build\bin\ffmpeg.exe"
)


def _find_ffmpeg() -> str | None:
    found = shutil.which("ffmpeg")
    if found:
        return found
    env = os.environ.get("FFMPEG_BINARY", "")
    if env and Path(env).exists():
        return env
    if _KNOWN_FFMPEG.exists():
        return str(_KNOWN_FFMPEG)
    return None


FFMPEG = _find_ffmpeg()


def _run_ffmpeg(binary: str, args: list[str]) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run([binary, "-hide_banner", "-y", *args], capture_output=True, text=False)


def _build_scene_video(binary: str, out_path: Path) -> None:
    completed = _run_ffmpeg(
        binary,
        [
            "-f", "lavfi", "-i", "color=c=black:s=64x64:r=1:d=3",
            "-f", "lavfi", "-i", "color=c=white:s=64x64:r=1:d=2",
            "-f", "lavfi", "-i", "color=c=red:s=64x64:r=1:d=2",
            "-filter_complex", "[0:v][1:v][2:v]concat=n=3:v=1:a=0[v]",
            "-map", "[v]", "-pix_fmt", "yuv420p", str(out_path),
        ],
    )
    if completed.returncode != 0 or not out_path.exists():
        raise RuntimeError(f"failed to build test video: {completed.stderr[-500:]!r}")


def _gray_brightness(binary: str, image_path: Path) -> float:
    """Mean luma of an image file, decoded through ffmpeg to raw gray."""
    completed = subprocess.run(
        [binary, "-hide_banner", "-i", str(image_path), "-vf", "format=gray", "-f", "rawvideo", "-"],
        capture_output=True,
        text=False,
    )
    data = completed.stdout
    return sum(data) / len(data) if data else -1.0


@unittest.skipUnless(FFMPEG, "ffmpeg binary not available")
class FfmpegVideoPipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        cls.tmp = Path(cls._tmp.name)
        cls.video = cls.tmp / "cut.mp4"
        _build_scene_video(FFMPEG, cls.video)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    # -- F10: 1-based scene indices ---------------------------------------- #
    def test_scene_cut_indices_are_one_based(self) -> None:
        runner = FfmpegRunner(FFMPEG)
        indices = runner.scene_cut_indices(
            self.video, fps=1.0, threshold=0.4, available_indices=list(range(1, 8))
        )
        self.assertEqual(indices, [4, 6])
        self.assertEqual([(index - 1) / 1.0 for index in indices], [3.0, 5.0])

    def test_scene_cut_indices_at_2fps(self) -> None:
        runner = FfmpegRunner(FFMPEG)
        indices = runner.scene_cut_indices(
            self.video, fps=2.0, threshold=0.4, available_indices=list(range(1, 15))
        )
        self.assertEqual(indices, [7, 11])

    # -- F11: 0-based selector / no silent mis-pairing --------------------- #
    def test_extract_indices_produces_every_requested_frame(self) -> None:
        runner = FfmpegRunner(FFMPEG)
        produced = runner.extract_indices(
            self.video, self.tmp / "display", fps=1.0, indices=list(range(1, 8)), width=64
        )
        self.assertEqual(len(produced), 7)
        # N4: explicit (index, path) pairs — the index must travel with the file.
        self.assertEqual([index for index, _ in produced], list(range(1, 8)))
        paths = [path for _, path in produced]
        # frame_0001 must be t=0 (black), not t=1; index 4 (t=3) must be white.
        self.assertLess(_gray_brightness(FFMPEG, paths[0]), 60)
        self.assertLess(_gray_brightness(FFMPEG, paths[1]), 60)
        self.assertGreater(_gray_brightness(FFMPEG, paths[3]), 200)
        self.assertGreater(_gray_brightness(FFMPEG, paths[4]), 200)
        self.assertLess(_gray_brightness(FFMPEG, paths[5]), 140)  # index 6 (t=5, red)

    def test_extract_indices_pairs_unsorted_duplicate_input_correctly(self) -> None:
        """N4: unsorted + duplicated indices must not silently mis-pair.

        ``indices=[5, 1, 5, 3]`` collapses to ``{1, 3, 5}``; the returned pairs
        must associate each *index* with its own frame, so the caller never has
        to rely on positional zip against the original list.
        """
        runner = FfmpegRunner(FFMPEG)
        produced = runner.extract_indices(
            self.video, self.tmp / "unsorted", fps=1.0, indices=[5, 1, 5, 3], width=64
        )
        self.assertEqual([index for index, _ in produced], [1, 3, 5])
        brightness = {index: _gray_brightness(FFMPEG, path) for index, path in produced}
        # clip: black 0-3s, white 3-5s, red 5-7s -> idx1(t=0)=black,
        # idx3(t=2)=black, idx5(t=4)=white.
        self.assertLess(brightness[1], 60)
        self.assertLess(brightness[3], 60)
        self.assertGreater(brightness[5], 200)

    def test_extract_indices_empty_input_returns_empty(self) -> None:
        """N4: no indices requested -> no pairs (no ffmpeg call needed)."""
        runner = FfmpegRunner(FFMPEG)
        self.assertEqual(runner.extract_indices(self.video, self.tmp / "empty", fps=1.0, indices=[], width=64), [])

    # -- F12: counts match produced records -------------------------------- #
    def test_full_video_extraction_counts_match_records(self) -> None:
        config = ProjectConfig.load(MOCK_PROJECT / "config" / "project.json")
        config.workspace_dir = self.tmp / "ws"
        config.video.video_path = str(self.video)
        config.video.ffmpeg_binary = FFMPEG
        config.video.sample_fps = 1.0
        config.keyframe.motion_min_ratio = 0.0
        config.keyframe.dedup_hamming_threshold = 0
        config.keyframe.min_gap_seconds = 0.0
        config.keyframe.max_keyframes = 30

        extractor = KeyframeExtractor(config, detector=None)
        records = extractor.extract(KeyframeSource(source_id="clip", video_path=self.video))

        self.assertEqual(len(records), 7)
        self.assertEqual(extractor.counts["final_keyframes"], len(records))
        self.assertEqual(extractor.counts["display_missing"], 0)
        self.assertFalse(extractor.warnings)
        self.assertEqual(extractor.counts["scene_cuts"], 2)
        for record in records:
            self.assertTrue(Path(record.extracted_path).exists())


if __name__ == "__main__":
    unittest.main()
