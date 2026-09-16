"""QA round-2 adversarial regression tests (independent verification of F2-F12).

Written by the QA engineer as the *second* round of adversarial verification
after the engineer fixed findings F2-F12. It deliberately goes beyond the
engineer's own regression tests:

* the real-ffmpeg tests build a **different** clip from the engineer's
  (four solid-colour segments: black / white / red / lime, 8 s @ 25 fps with
  hard cuts at t=2, 4, 6) so the brightness fingerprint can pin every sampled
  index to an unambiguous segment;
* ``sample_fps`` is exercised at fractional values (0.5, 1.5) that the
  engineer's suite does not cover;
* ``extract_indices`` is exercised at the max index and beyond the frame count;
* the ``protected`` de-duplication guarantee is probed with a scene-cut frame
  whose aHash is *identical* to the previous frame (the worst case);
* the display-extraction failure path (F12) is forced and must warn, not skip.

Nothing here touches the network; ffmpeg-dependent tests skip when unavailable.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from hazard_annotation.ffmpeg import _FPS_MODE_ARGS, FfmpegRunner
from hazard_annotation.imageio import PgmImage, PpmImage, ahash, hamming
from hazard_annotation.keyframes import (
    KeyframeExtractor,
    KeyframeSource,
    _AnalysisFrame,
)
from traffic_cv_closed_loop.adapters.qwen import ThinkingJsonParser
from traffic_cv_closed_loop.config import ProjectConfig
from traffic_cv_closed_loop.models import Detection

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

#: (index -> expected mean luma) for my 8 s clip at 1 fps. black=0, white=255,
#: red=76, lime=150. The 1-based analysis index k samples t=(k-1)/1.
_EXPECTED_LUMA_1FPS = [0, 0, 255, 255, 76, 76, 150, 150]


def _gray_brightness(binary: str, image_path: Path) -> float:
    completed = subprocess.run(
        [binary, "-hide_banner", "-i", str(image_path), "-vf", "format=gray", "-f", "rawvideo", "-"],
        capture_output=True,
        text=False,
    )
    data = completed.stdout
    return sum(data) / len(data) if data else -1.0


def _build_four_segment_video(binary: str, out_path: Path) -> None:
    completed = subprocess.run(
        [
            binary, "-hide_banner", "-y",
            "-f", "lavfi", "-i", "color=c=black:s=320x240:r=25:d=2",
            "-f", "lavfi", "-i", "color=c=white:s=320x240:r=25:d=2",
            "-f", "lavfi", "-i", "color=c=0xFF0000:s=320x240:r=25:d=2",
            "-f", "lavfi", "-i", "color=c=lime:s=320x240:r=25:d=2",
            "-filter_complex", "[0:v][1:v][2:v][3:v]concat=n=4:v=1:a=0[v]",
            "-map", "[v]", "-pix_fmt", "yuv420p", "-c:v", "libx264", "-crf", "18", str(out_path),
        ],
        capture_output=True,
        text=False,
    )
    if completed.returncode != 0 or not out_path.exists():
        raise RuntimeError(f"failed to build clip: {completed.stderr[-500:]!r}")


class _StubDetector:
    def __init__(self, mapping: dict[str, list[Detection]]) -> None:
        self.mapping = mapping

    def detect_many(self, paths: list[Path]) -> dict[Path, list[Detection]]:
        return {path.resolve(): self.mapping.get(path.name, []) for path in paths}


def _write_uniform(directory: Path, values: list[int]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for index, value in enumerate(values):
        (directory / f"f_{index:04d}.pgm").write_bytes(
            b"P5\n16 16\n255\n" + bytes([value] * 256)
        )


def _image_config(workspace: Path) -> ProjectConfig:
    config = ProjectConfig.load(MOCK_PROJECT / "config" / "project.json")
    config.workspace_dir = workspace
    config.keyframe.min_gap_seconds = 0.0
    config.keyframe.motion_min_ratio = 0.002
    config.keyframe.dedup_hamming_threshold = 4
    config.keyframe.scene_change_threshold = 0.4
    config.keyframe.max_keyframes = 400
    return config


# --------------------------------------------------------------------------- #
# 1. Real-ffmpeg video path (F9 / F10 / F11)
# --------------------------------------------------------------------------- #
@unittest.skipUnless(FFMPEG, "ffmpeg binary not available")
class FfmpegRound2Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        cls.tmp = Path(cls._tmp.name)
        cls.video = cls.tmp / "mine.mp4"
        _build_four_segment_video(FFMPEG, cls.video)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def test_fps_mode_replaces_removed_vsync_option(self) -> None:
        """F9: ``-vsync`` is gone in ffmpeg 9.x; the runner must use -fps_mode."""
        self.assertEqual(_FPS_MODE_ARGS, ("-fps_mode", "passthrough"))
        # the old spelling is rejected by this ffmpeg build
        completed = subprocess.run(
            [FFMPEG, "-hide_banner", "-y", "-i", str(self.video), "-vf", "fps=1",
             "-vsync", "0", "-f", "null", "-"],
            capture_output=True, text=False,
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn(b"vsync", completed.stderr)
        # the new spelling works
        runner = FfmpegRunner(FFMPEG)
        out = self.tmp / "fpsmode"
        out.mkdir(exist_ok=True)
        self.assertEqual(len(runner.coarse_sample(self.video, out, fps=1.0, width=64)), 8)

    def test_scene_indices_align_at_integer_and_fractional_fps(self) -> None:
        """F10: cut at t=2/4/6 must map to the post-cut sampled index for any fps."""
        runner = FfmpegRunner(FFMPEG)
        for fps in (0.5, 1.0, 1.5, 2.0):
            with self.subTest(fps=fps):
                out = self.tmp / f"coarse{fps}"
                out.mkdir(exist_ok=True)
                frames = runner.coarse_sample(self.video, out, fps=fps, width=160)
                available = [int(p.stem) for p in frames]
                indices = runner.scene_cut_indices(
                    self.video, fps=fps, threshold=0.4, available_indices=available
                )
                self.assertEqual(
                    [round((index - 1) / fps, 3) for index in indices],
                    [2.0, 4.0, 6.0],
                )
                self.assertTrue(set(indices).issubset(set(available)))
                # every returned index must sit on the frame right after a cut:
                # luma jumps to white(255) / red(76) / lime(150) respectively.
                luma = {i: _gray_brightness(FFMPEG, out / f"{i:06d}.pgm") for i in indices}
                self.assertGreater(luma[indices[0]], 200.0)   # white
                self.assertLess(luma[indices[1]], 120.0)      # red
                self.assertGreater(luma[indices[2]], 120.0)   # lime

    def test_extract_indices_includes_max_index_with_correct_pairing(self) -> None:
        """F11: the 1-based max index must be produced (the original blind spot).

        N4: ``extract_indices`` now returns explicit ``(index, path)`` pairs, so
        the index/path association is asserted directly instead of assumed by
        position.
        """
        runner = FfmpegRunner(FFMPEG)
        out = self.tmp / "all"
        out.mkdir(exist_ok=True)
        indices = list(range(1, 9))
        produced = runner.extract_indices(self.video, out, fps=1.0, indices=indices, width=64)
        self.assertEqual(len(produced), 8)
        self.assertEqual([index for index, _ in produced], indices)
        luma = [round(_gray_brightness(FFMPEG, path)) for _, path in produced]
        for got, expected in zip(luma, _EXPECTED_LUMA_1FPS):
            self.assertAlmostEqual(got, expected, delta=6, msg=f"luma={luma}")

    def test_extract_indices_boundaries_are_graceful(self) -> None:
        """F11 boundary: max-only, overrun and fully-beyond requests.

        N4: results are ``(index, path)`` pairs, so boundary indices are checked
        explicitly and can never be mis-paired.
        """
        runner = FfmpegRunner(FFMPEG)
        max_only = self.tmp / "maxonly"
        max_only.mkdir(exist_ok=True)
        produced = runner.extract_indices(self.video, max_only, fps=1.0, indices=[8], width=64)
        self.assertEqual(len(produced), 1)
        self.assertEqual(produced[0][0], 8)
        self.assertGreater(_gray_brightness(FFMPEG, produced[0][1]), 120.0)  # lime

        overrun = self.tmp / "overrun"
        overrun.mkdir(exist_ok=True)
        produced = runner.extract_indices(self.video, overrun, fps=1.0, indices=list(range(1, 13)), width=64)
        self.assertEqual(len(produced), 8)  # only existing frames
        self.assertEqual([index for index, _ in produced], list(range(1, 9)))

        beyond = self.tmp / "beyond"
        beyond.mkdir(exist_ok=True)
        self.assertEqual(runner.extract_indices(self.video, beyond, fps=1.0, indices=[9, 10], width=64), [])

        empty = self.tmp / "empty"
        empty.mkdir(exist_ok=True)
        self.assertEqual(runner.extract_indices(self.video, empty, fps=1.0, indices=[], width=64), [])

        # a mixed in-range / out-of-range request keeps exactly the produced
        # indices and can never mis-pair them.
        mixed = self.tmp / "mixed"
        mixed.mkdir(exist_ok=True)
        produced = runner.extract_indices(self.video, mixed, fps=1.0, indices=[1, 9], width=64)
        self.assertEqual([index for index, _ in produced], [1])

    def test_full_video_extraction_counts_and_records(self) -> None:
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

        self.assertEqual(len(records), 8)
        self.assertEqual(extractor.counts["final_keyframes"], 8)
        self.assertEqual(extractor.counts["display_missing"], 0)
        self.assertEqual(extractor.counts["scene_cuts"], 3)
        self.assertFalse(extractor.warnings)
        for record in records:
            self.assertTrue(Path(record.extracted_path).exists())


# --------------------------------------------------------------------------- #
# 2. F2 — protected frames survive motion + de-dup (worst case)
# --------------------------------------------------------------------------- #
class KeyframeProtectedRound2Test(unittest.TestCase):
    def test_scene_cut_frame_with_identical_ahash_survives(self) -> None:
        """F2 worst case: a scene-cut frame whose aHash equals the previous frame.

        Uniform 100 -> uniform 120 differ by a full histogram bucket (delta 1.0)
        yet hash to the *same* value (neither has a block above the mean and both
        means are below 128), so without ``protected`` the de-duplicator removes
        it. Both filters must be shown to drop it *and* the protected run must
        keep it.
        """
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            frames_dir = tmp_path / "frames"
            _write_uniform(frames_dir, [100, 120])

            extractor = KeyframeExtractor(_image_config(tmp_path / "ws"), detector=None)
            frames = [
                _AnalysisFrame(0, 0.0, Path("f_0000.pgm"), 16, 16, bytes([100] * 256)),
                _AnalysisFrame(1, 1.0, Path("f_0001.pgm"), 16, 16, bytes([120] * 256)),
            ]
            self.assertEqual(hamming(ahash(bytes([100] * 256), 16, 16, 8),
                                     ahash(bytes([120] * 256), 16, 16, 8)), 0)
            motion_kept, _ = extractor._step_motion(frames)
            dedup_kept, _ = extractor._step_dedup(frames)
            self.assertEqual([f.index for f in motion_kept], [0])  # motion would drop it
            self.assertEqual([f.index for f in dedup_kept], [0])   # de-dup would drop it

            records = extractor.extract(KeyframeSource(source_id="s", frames_dir=frames_dir))
            names = {Path(r.relative_path).name for r in records}
            self.assertIn("f_0001.pgm", names)
            self.assertGreaterEqual(extractor.counts["scene_cuts"], 1)

    def test_target_frame_identical_to_previous_survives(self) -> None:
        """F2 worst case: an identical frame carrying a key target must survive."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            frames_dir = tmp_path / "frames"
            _write_uniform(frames_dir, [100, 100])
            detector = _StubDetector(
                {"f_0001.pgm": [Detection(label="person", bbox=(0.1, 0.1, 0.2, 0.2))]}
            )
            extractor = KeyframeExtractor(_image_config(tmp_path / "ws"), detector=detector)
            records = extractor.extract(KeyframeSource(source_id="s", frames_dir=frames_dir))
            names = {Path(r.relative_path).name for r in records}
            self.assertIn("f_0001.pgm", names)
            self.assertEqual(extractor.counts["force_kept"], 1)

    def test_image_directory_path_still_works_after_signature_change(self) -> None:
        """Regression: the image path must still work with the new protected arg."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            frames_dir = tmp_path / "frames"
            _write_uniform(frames_dir, [10, 10, 200, 10])
            config = _image_config(tmp_path / "ws")
            extractor = KeyframeExtractor(config, detector=None)
            records = extractor.extract(KeyframeSource(source_id="s", frames_dir=frames_dir))
            self.assertEqual(extractor.counts["total_frames"], 4)
            self.assertEqual(extractor.counts["final_keyframes"], len(records))
            self.assertTrue(records)
            for record in records:
                self.assertTrue(Path(record.extracted_path).exists())


# --------------------------------------------------------------------------- #
# 3. F12 — display extraction failure must warn, never skip silently
# --------------------------------------------------------------------------- #
class DisplayWarningRound2Test(unittest.TestCase):
    def _extractor_with_stubbed_video(self, tmp: Path, frames_dir: Path):
        config = _image_config(tmp / "ws")
        (tmp / "fake.mp4").write_bytes(b"not-a-real-video")
        extractor = KeyframeExtractor(config, detector=None)
        extractor._step_coarse_sample_video = lambda *a, **k: [frames_dir / f"f_{i:04d}.pgm" for i in range(4)]
        extractor.ffmpeg.available = lambda: True
        extractor.ffmpeg.scene_cut_indices = lambda **k: []
        return extractor

    def test_display_extract_failure_warns_and_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            frames_dir = tmp_path / "frames"
            _write_uniform(frames_dir, [0, 255, 0, 255])
            extractor = self._extractor_with_stubbed_video(tmp_path, frames_dir)

            def _boom(**kwargs):
                raise RuntimeError("boom: no decoder")

            extractor.ffmpeg.extract_indices = _boom
            records = extractor.extract(
                KeyframeSource(source_id="v", video_path=tmp_path / "fake.mp4")
            )
            self.assertEqual(records, [])
            self.assertEqual(extractor.counts["final_keyframes"], 0)
            self.assertGreater(extractor.counts["display_missing"], 0)
            self.assertTrue(any("display_extract_failed" in w for w in extractor.warnings))
            self.assertTrue(any("display_missing" in w for w in extractor.warnings))

    def test_display_count_mismatch_warns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            frames_dir = tmp_path / "frames"
            _write_uniform(frames_dir, [0, 255, 0, 255])
            extractor = self._extractor_with_stubbed_video(tmp_path, frames_dir)
            outdir = tmp_path / "disp"
            outdir.mkdir()
            (outdir / "frame_0001.png").write_bytes(b"png")
            extractor.ffmpeg.extract_indices = lambda **k: [(1, outdir / "frame_0001.png")]
            extractor.extract(KeyframeSource(source_id="v", video_path=tmp_path / "fake.mp4"))
            self.assertTrue(
                any("display_count_mismatch" in w for w in extractor.warnings),
                extractor.warnings,
            )
            self.assertEqual(extractor.counts["final_keyframes"], 1)
            self.assertEqual(extractor.counts["display_missing"], 3)


# --------------------------------------------------------------------------- #
# 4. F5 / F6 / F7 — image IO edges
# --------------------------------------------------------------------------- #
class ImageIoRound2Test(unittest.TestCase):
    def _write(self, tmp: str, name: str, data: bytes) -> Path:
        path = Path(tmp) / name
        path.write_bytes(data)
        return path

    def test_p6_crlf_header_does_not_shift_raster(self) -> None:
        """F5 also applies to P6 (colour)."""
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(tmp, "c.ppm", b"P6\r\n1 1\r\n255\r\n" + bytes([10, 20, 30]))
            self.assertEqual(PpmImage.load(path).pixels, bytes([10, 20, 30]))

    def test_p6_raster_length_mismatch_raises(self) -> None:
        """F6 also applies to P6: short and oversized rasters must raise."""
        with tempfile.TemporaryDirectory() as tmp:
            short = self._write(tmp, "s.ppm", b"P6\n2 1\n255\n" + bytes([1, 2, 3]))
            long = self._write(tmp, "l.ppm", b"P6\n1 1\n255\n" + bytes([1, 2, 3, 4, 5, 6]))
            with self.assertRaises(ValueError):
                PpmImage.load(short)
            with self.assertRaises(ValueError):
                PpmImage.load(long)

    def test_p5_single_trailing_whitespace_is_tolerated(self) -> None:
        """The Netpbm spec allows one trailing whitespace byte after the raster."""
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(tmp, "t.pgm", b"P5\n1 1\n255\n" + bytes([7]) + b"\n")
            self.assertEqual(PgmImage.load(path).pixels, bytes([7]))

    def test_ahash_distinguishes_uniform_black_and_white(self) -> None:
        """F7: the brightness high bit separates uniform black from white."""
        self.assertGreater(hamming(ahash(bytes([0] * 256), 16, 16, 8),
                                   ahash(bytes([255] * 256), 16, 16, 8)), 0)


# --------------------------------------------------------------------------- #
# 5. qwen parser — F3 / F4 and regression
# --------------------------------------------------------------------------- #
class QwenParserRound2Test(unittest.TestCase):
    def setUp(self) -> None:
        self.parser = ThinkingJsonParser()

    def test_fence_draft_does_not_override_later_unfenced_answer(self) -> None:
        """F3: fenced draft then an unfenced final answer."""
        text = '```json\n{"draft": 1, "objects": []}\n```\n最终答案：{"final": 2, "objects": []}'
        self.assertEqual(self.parser.parse(text), {"final": 2, "objects": []})

    def test_later_fenced_answer_wins_over_earlier_fenced_draft(self) -> None:
        text = '```json\n{"draft": 1}\n```\n说明\n```json\n{"final": 2}\n```'
        self.assertEqual(self.parser.parse(text), {"final": 2})

    def test_single_quoted_url_is_not_corrupted(self) -> None:
        """F4: the comment stripper must be string-aware."""
        parsed = self.parser.parse("{'url': 'http://example.com/x', 'objects': []}")
        self.assertEqual(parsed["url"], "http://example.com/x")

    def test_double_quoted_string_with_comment_marker_is_preserved(self) -> None:
        parsed = self.parser.parse('{"note": "it\'s fine // ok", "objects": []}')
        self.assertEqual(parsed["note"], "it's fine // ok")

    def test_real_comments_are_still_stripped(self) -> None:
        self.assertEqual(self.parser.parse('{"a": 1, // c\n "b": 2}'), {"a": 1, "b": 2})
        self.assertEqual(self.parser.parse('{"a": 1, /* c */ "b": 2}'), {"a": 1, "b": 2})

    def test_original_capabilities_are_intact(self) -> None:
        """Regression net: the parse() rewrite must not break prior behaviour."""
        self.assertEqual(self.parser.parse(' thinkingx {"n": 1} 不要<｜end▁of▁thinking｜>{"final": 1}'), {"final": 1})
        self.assertEqual(self.parser.parse('<|thinking|>x {"n": 1}<|/thinking|>{"final": 1}'), {"final": 1})
        self.assertEqual(self.parser.parse('{"objects": [], "final": 1}'), {"objects": [], "final": 1})
        self.assertEqual(self.parser.parse("prefix [1, 2, 3] suffix"), [1, 2, 3])
        self.assertEqual(
            self.parser.parse('{"objects": [{"label": "person", "bbox": [0, 0, 1, 1]}')["objects"][0]["label"],
            "person",
        )
        self.assertEqual(self.parser.parse('prefix {"first": 1} middle {"second": 2} suffix'), {"second": 2})

    def test_stray_degenerate_span_does_not_override_fenced_answer(self) -> None:
        """N3: a trailing degenerate span must not shadow the real answer.

        The F3 fix ("keep whichever balanced span starts *later*") could be
        defeated by a stray ``{}`` / ``[]`` trailing the correct fenced answer,
        silently degrading a good result to an empty one. Degenerate spans now
        win only when *every* candidate is degenerate.
        """
        self.assertEqual(self.parser.parse('```json\n{"real": 1}\n```\n说明 {}'), {"real": 1})
        self.assertEqual(self.parser.parse('```json\n{"real": 1}\n```\n说明 []'), {"real": 1})
        # a lone degenerate payload is still returned when nothing else exists
        self.assertEqual(self.parser.parse("{}"), {})
        self.assertEqual(self.parser.parse("[]"), [])
        # an "empty but structured" payload (e.g. a no-object answer) is NOT
        # degenerate and must be returned verbatim
        self.assertEqual(self.parser.parse('{"objects": []}'), {"objects": []})
        self.assertEqual(self.parser.parse('```json\n{"objects": []}\n```'), {"objects": []})
        # F3 positive case still holds: an earlier draft must not beat a later
        # non-degenerate answer.
        self.assertEqual(
            self.parser.parse('```json\n{"draft": 1}\n```\n最终 {"final": 2}'), {"final": 2}
        )

    def test_characterization_later_degenerate_answer_shadowed_by_earlier_span(self) -> None:
        """Known trade-off of the N3 fix (LOW severity) — recorded, not endorsed.

        The N3 rule ("skip a trailing degenerate span") has a reverse edge: when
        the model's *real* final answer is itself an empty ``{}`` / ``[]`` and an
        earlier non-degenerate span exists (e.g. an example), the parser returns
        the earlier span instead of the final answer. This is unlikely for the
        hazard pipeline (whose answer is always a populated
        ``{"objects": [...], "hazard_report": {...}}`` object) but it is a real
        behavioural edge worth tracking. If the parser is changed to fix it,
        update this test deliberately.
        """
        self.assertEqual(self.parser.parse('示例 {"a": 1}\n最终答案：{}'), {"a": 1})
        self.assertEqual(self.parser.parse("说明 [1, 2]\n最终：[]"), [1, 2])


if __name__ == "__main__":
    unittest.main()
