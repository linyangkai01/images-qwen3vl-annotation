"""Regression tests for the ffmpeg version self-check (ARCH: ffmpeg >= 5.1).

``-vsync`` was removed in ffmpeg 9.x and its replacement ``-fps_mode`` requires
ffmpeg >= 5.1, so the pipeline exposes ``FfmpegRunner.version`` /
``check_min_version`` and surfaces the result in the ``doctor`` video section.
These tests use the real binary when present and a synthetic "fake ffmpeg" for
the malformed/old-version paths; ffmpeg-dependent cases skip when it is absent.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from hazard_annotation.config import load_hazard_config
from hazard_annotation.diagnostics import _video_section
from hazard_annotation.ffmpeg import FfmpegRunner

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


def _make_fake_ffmpeg(directory: Path, output_line: str) -> Path:
    """Create a runnable stub that ignores its args and echoes ``output_line``."""
    directory.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        path = directory / "fake_ffmpeg.cmd"
        path.write_text("@echo off\r\necho {}\r\n".format(output_line), encoding="ascii")
    else:
        path = directory / "fake_ffmpeg.sh"
        path.write_text("#!/bin/sh\necho '{}'\n".format(output_line), encoding="ascii")
        path.chmod(0o755)
    return path


class FfmpegVersionParsingTest(unittest.TestCase):
    """Deterministic parsing of ``ffmpeg -version`` first lines (no binary)."""

    def test_standard_release_version(self) -> None:
        text = "ffmpeg version 9.0.1-essentials_build-www.gyan.dev Copyright (c) 2000-2025"
        self.assertEqual(FfmpegRunner._parse_version(text), (9, 0, 1))

    def test_two_component_version(self) -> None:
        self.assertEqual(FfmpegRunner._parse_version("ffmpeg version 5.1\nbuilt with gcc"), (5, 1))

    def test_nightly_n_prefix_is_tolerated(self) -> None:
        self.assertEqual(FfmpegRunner._parse_version("ffmpeg version n6.1.1 Copyright"), (6, 1, 1))

    def test_unparseable_build_tag_returns_none(self) -> None:
        self.assertIsNone(
            FfmpegRunner._parse_version("ffmpeg version N-108972-g6b6c3d0d17 Copyright")
        )

    def test_empty_or_garbage_output_returns_none(self) -> None:
        self.assertIsNone(FfmpegRunner._parse_version(""))
        self.assertIsNone(FfmpegRunner._parse_version("   \n\n"))
        self.assertIsNone(FfmpegRunner._parse_version("not an ffmpeg banner at all"))


class FfmpegVersionThresholdTest(unittest.TestCase):
    def test_check_min_version_boundaries(self) -> None:
        runner = FfmpegRunner("ffmpeg")
        for parsed, expected in (
            ((9, 0, 1), True),
            ((5, 1), True),
            ((5, 1, 2), True),
            ((5, 0), False),
            ((4, 4), False),
            (None, False),
        ):
            with self.subTest(version=parsed):
                with mock.patch.object(runner, "version", return_value=parsed):
                    self.assertIs(runner.check_min_version(5, 1), expected)

    def test_missing_binary_is_none_and_false(self) -> None:
        runner = FfmpegRunner("definitely-not-a-real-ffmpeg-binary-xyz")
        self.assertFalse(runner.available())
        self.assertIsNone(runner.version())
        self.assertFalse(runner.check_min_version(5, 1))


class FfmpegFakeBinaryTest(unittest.TestCase):
    """The version probe must degrade to ``None`` instead of raising."""

    def test_malformed_version_returns_none_without_raising(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fake = _make_fake_ffmpeg(Path(tmp), "ffmpeg version N-108972-g6b6c3d0d17")
            runner = FfmpegRunner(str(fake))
            self.assertTrue(runner.available())
            self.assertIsNone(runner.version())
            self.assertFalse(runner.check_min_version(5, 1))

    def test_old_version_is_reported_and_warns_but_is_not_blocking(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fake = _make_fake_ffmpeg(Path(tmp), "ffmpeg version 4.4.2")
            config = load_hazard_config(MOCK_PROJECT / "config" / "project.json", mode="quick")
            config.video.ffmpeg_binary = str(fake)
            section, issues, warnings = _video_section(config)
            self.assertTrue(section["ffmpeg_available"])
            self.assertEqual(section["ffmpeg_version"], "4.4.2")
            self.assertFalse(section["ffmpeg_min_version_ok"])
            self.assertTrue(any("ffmpeg_version_too_old" in item for item in warnings))
            # a version problem is a warning, never a blocking issue
            self.assertFalse(any("ffmpeg_version" in item for item in issues))

    def test_unparseable_version_warns_in_doctor(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fake = _make_fake_ffmpeg(Path(tmp), "ffmpeg version N-108972-g6b6c3d0d17")
            config = load_hazard_config(MOCK_PROJECT / "config" / "project.json", mode="quick")
            config.video.ffmpeg_binary = str(fake)
            section, issues, warnings = _video_section(config)
            self.assertEqual(section["ffmpeg_version"], "")
            self.assertFalse(section["ffmpeg_min_version_ok"])
            self.assertTrue(any("ffmpeg_version_unparseable" in item for item in warnings))
            self.assertFalse(any("ffmpeg_version" in item for item in issues))


@unittest.skipUnless(FFMPEG, "ffmpeg binary not available")
class FfmpegRealVersionTest(unittest.TestCase):
    def test_real_ffmpeg_version_and_min_check(self) -> None:
        runner = FfmpegRunner(FFMPEG)
        parsed = runner.version()
        self.assertIsNotNone(parsed)
        assert parsed is not None  # narrow for type checkers
        self.assertGreaterEqual(parsed[0], 5)
        self.assertTrue(runner.check_min_version(5, 1))

    def test_real_ffmpeg_9_0_1_exact_tuple(self) -> None:
        parsed = FfmpegRunner(FFMPEG).version()
        # The local/CI binary is 9.0.1; only pin the exact tuple when major is 9
        # so the suite stays portable to other (>= 5.1) ffmpeg builds.
        if parsed and parsed[0] == 9:
            self.assertEqual(parsed, (9, 0, 1))

    def test_doctor_video_section_reports_version(self) -> None:
        config = load_hazard_config(MOCK_PROJECT / "config" / "project.json", mode="quick")
        config.video.ffmpeg_binary = FFMPEG
        section, _issues, warnings = _video_section(config)
        self.assertTrue(section["ffmpeg_available"])
        self.assertNotEqual(section["ffmpeg_version"], "")
        self.assertTrue(section["ffmpeg_min_version_ok"])
        self.assertFalse(any("ffmpeg_version" in item for item in warnings))


if __name__ == "__main__":
    unittest.main()
