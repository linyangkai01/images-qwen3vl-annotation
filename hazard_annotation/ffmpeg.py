"""Thin wrapper around the external ``ffmpeg`` CLI.

``ffmpeg`` is the only external binary used for video handling. When it is not
installed every method raises a clear :class:`FfmpegUnavailableError` pointing
the user at ``python -m hazard_annotation doctor`` instead of crashing or
silently producing empty results.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

_FPS_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\s*fps")
_PTS_TIME_PATTERN = re.compile(r"pts_time:(\d+(?:\.\d+)?)")
#: First line of ``ffmpeg -version``, e.g. ``ffmpeg version 9.0.1-...``; the
#: capture is the version token right after the word ``version``.
_VERSION_TOKEN = re.compile(r"\bversion\b\s+(\S+)", re.IGNORECASE)
#: A leading dotted version, tolerating a nightly ``n`` prefix (``n6.1.1``).
_DOTTED_VERSION = re.compile(r"^n?(\d+)(?:\.(\d+))?(?:\.(\d+))?", re.IGNORECASE)

#: ``-vsync`` was removed in ffmpeg 9.x; ``-fps_mode passthrough`` is the
#: supported replacement (ffmpeg >= 5.1) and keeps the same "do not duplicate
#: or drop frames" semantics. Centralised so both call sites stay in sync.
_FPS_MODE_ARGS: tuple[str, str] = ("-fps_mode", "passthrough")


class FfmpegUnavailableError(RuntimeError):
    """Raised when the configured ffmpeg binary cannot be found."""


def _decode_process_output(payload: bytes | str | None) -> str:
    if payload is None:
        return ""
    if isinstance(payload, str):
        return payload
    for encoding in ("utf-8", "gbk"):
        try:
            return payload.decode(encoding)
        except UnicodeDecodeError:
            continue
    return payload.decode("utf-8", errors="replace")


class FfmpegRunner:
    """Run the minimal set of ffmpeg commands the pipeline needs."""

    def __init__(self, binary: str = "ffmpeg") -> None:
        self.binary = binary or "ffmpeg"

    def available(self) -> bool:
        return shutil.which(self.binary) is not None

    def version(self) -> tuple[int, ...] | None:
        """Return the ffmpeg version as a tuple, or ``None`` when unreadable.

        Parses the first line of ``ffmpeg -version`` (e.g.
        ``ffmpeg version 9.0.1-essentials_build-...`` -> ``(9, 0, 1)``). Nightly
        style strings are tolerated (``ffmpeg version n6.1.1`` -> ``(6, 1, 1)``)
        while an unparseable build tag such as ``N-108972-g6b6c3d0d17`` yields
        ``None``. A missing binary or any parse failure returns ``None`` instead
        of raising, so callers can degrade gracefully.
        """
        if not self.available():
            return None
        try:
            completed = subprocess.run(
                [self.binary, "-version"],
                check=False,
                capture_output=True,
                text=False,
            )
        except OSError:
            return None
        text = _decode_process_output(completed.stdout) or _decode_process_output(completed.stderr)
        return self._parse_version(text)

    @staticmethod
    def _parse_version(text: str) -> tuple[int, ...] | None:
        """Extract a dotted version tuple from ``ffmpeg -version`` output."""
        first_line = next((line.strip() for line in text.splitlines() if line.strip()), "")
        if not first_line:
            return None
        match = _VERSION_TOKEN.search(first_line)
        token = match.group(1) if match else first_line
        token_match = _DOTTED_VERSION.match(token)
        if not token_match:
            return None
        parts = tuple(int(group) for group in token_match.groups() if group is not None)
        return parts or None

    def check_min_version(self, major: int = 5, minor: int = 1) -> bool:
        """Return ``True`` when the ffmpeg version is at least ``major.minor``.

        ``-fps_mode passthrough`` (the replacement for the ``-vsync`` option
        removed in ffmpeg 9.x) requires ffmpeg >= 5.1. Returns ``False`` when the
        version cannot be parsed, so a deployment can be flagged before a run.
        """
        parsed = self.version()
        if parsed is None:
            return False
        padded = tuple(parsed) + (0,) * max(0, 2 - len(parsed))
        return (padded[0], padded[1]) >= (major, minor)

    def _ensure_available(self) -> None:
        if not self.available():
            raise FfmpegUnavailableError(
                f"ffmpeg binary not found in PATH: '{self.binary}'. "
                "Install ffmpeg or run `python -m hazard_annotation doctor` for a full report."
            )

    def _run(self, command: list[str]) -> subprocess.CompletedProcess[bytes]:
        self._ensure_available()
        return subprocess.run(command, check=False, capture_output=True, text=False)

    def probe_source_fps(self, video: str | Path) -> float:
        """Return the source frame rate parsed from ``ffmpeg -i`` stderr."""
        self._ensure_available()
        completed = subprocess.run(
            [self.binary, "-hide_banner", "-i", str(video)],
            check=False,
            capture_output=True,
            text=False,
        )
        stderr_text = _decode_process_output(completed.stderr)
        match = _FPS_PATTERN.search(stderr_text)
        if not match:
            return 0.0
        try:
            return float(match.group(1))
        except ValueError:
            return 0.0

    def coarse_sample(
        self,
        video: str | Path,
        out_dir: str | Path,
        fps: float,
        width: int,
    ) -> list[Path]:
        """Decode ``video`` at ``fps`` into grayscale PGM analysis frames."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        command = [
            self.binary,
            "-hide_banner",
            "-y",
            "-i",
            str(video),
            "-vf",
            f"fps={fps},scale={width}:-1,format=gray",
            *_FPS_MODE_ARGS,
            "-f",
            "image2",
            "-pix_fmt",
            "gray",
            str(out_dir / "%06d.pgm"),
        ]
        completed = self._run(command)
        if completed.returncode != 0:
            stderr_text = _decode_process_output(completed.stderr)
            raise RuntimeError(
                f"ffmpeg coarse sampling failed (exit {completed.returncode}): {stderr_text[-800:]}"
            )
        return sorted(out_dir.glob("*.pgm"))

    def extract_indices(
        self,
        video: str | Path,
        out_dir: str | Path,
        fps: float,
        indices: list[int],
        width: int,
    ) -> list[tuple[int, Path]]:
        """Extract specific post-``fps`` frame indices as PNG keyframes.

        ``indices`` are the 1-based analysis-frame indices (``%06d`` naming);
        ffmpeg's ``n`` is 0-based, so the selector uses ``n = index - 1``.

        Returns one ``(index, path)`` pair per produced frame so a caller can
        never mis-pair an image with the wrong index: the ``select`` filter
        emits matches in ascending frame order, i.e. in the order of
        ``sorted(set(indices))`` — duplicates collapse and the caller's input
        order is irrelevant. When an index lies beyond the end of the video,
        ffmpeg simply emits fewer frames and only the produced pairs are
        returned (a caller comparing counts can then warn).
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        if not indices:
            return []
        unique = sorted(set(indices))
        selector = "+".join(f"eq(n\\,{index - 1})" for index in unique)
        command = [
            self.binary,
            "-hide_banner",
            "-y",
            "-i",
            str(video),
            "-vf",
            f"fps={fps},select='{selector}',scale={width}:-1",
            *_FPS_MODE_ARGS,
            str(out_dir / "frame_%04d.png"),
        ]
        completed = self._run(command)
        if completed.returncode != 0:
            stderr_text = _decode_process_output(completed.stderr)
            raise RuntimeError(
                f"ffmpeg keyframe extraction failed (exit {completed.returncode}): {stderr_text[-800:]}"
            )
        produced = sorted(out_dir.glob("*.png"))
        # ffmpeg numbers outputs sequentially (frame_0001.png, ...), so the k-th
        # file corresponds to the k-th requested (ascending, de-duplicated)
        # index. Zip them explicitly to make the pairing unambiguous.
        return list(zip(unique, produced))

    def scene_cut_indices(
        self,
        video: str | Path,
        fps: float,
        threshold: float,
        available_indices: list[int] | None = None,
    ) -> list[int]:
        """Return 1-based sampled-frame indices where scene changes exceed threshold.

        The analysis frames are named ``%06d`` (1-based) and the extractor uses
        ``timestamp = (index - 1) / fps``; the raw ``pts_time`` therefore maps to
        ``index = round(seconds * fps) + 1``. When ``available_indices`` is given
        each result is snapped to the nearest existing frame so the caller never
        receives an index that is absent from the analysis sequence.
        """
        command = [
            self.binary,
            "-hide_banner",
            "-i",
            str(video),
            "-vf",
            f"select='gt(scene,{threshold})',showinfo",
            "-f",
            "null",
            "-",
        ]
        completed = self._run(command)
        stderr_text = _decode_process_output(completed.stderr)
        indices: set[int] = set()
        for match in _PTS_TIME_PATTERN.finditer(stderr_text):
            try:
                seconds = float(match.group(1))
            except ValueError:
                continue
            indices.add(int(round(seconds * fps)) + 1)
        if available_indices:
            indices = {self._nearest_index(available_indices, index) for index in indices}
        return sorted(indices)

    @staticmethod
    def _nearest_index(indices: list[int], value: int) -> int:
        return min(indices, key=lambda candidate: (abs(candidate - value), candidate))
