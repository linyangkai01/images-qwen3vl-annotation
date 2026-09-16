"""Six-step keyframe extraction.

The extractor turns either a video (via ffmpeg) or an already-sliced image
directory into a bounded list of :class:`KeyframeRecord`. The six steps are:

1. coarse sampling (ffmpeg ``fps`` filter, or direct image enumeration),
2. motion detection (pure-Python frame difference),
3. perceptual de-duplication (average hash + hamming distance),
4. scene-change forced retention (ffmpeg ``scene`` filter or histogram),
5. object-driven forced retention (prior YOLO detections),
6. cap + minimum gap greedy selection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hazard_annotation.ffmpeg import FfmpegRunner, FfmpegUnavailableError
from hazard_annotation.imageio import ImageStats, ahash, histogram16, histogram_delta, hamming, motion_ratio
from hazard_annotation.models import KeyframeRecord
from traffic_cv_closed_loop.config import ProjectConfig
from traffic_cv_closed_loop.io_utils import copy_file, ensure_dir, iter_image_files


@dataclass(slots=True)
class KeyframeSource:
    """A single video or image-directory input."""

    source_id: str
    video_path: Path | None = None
    frames_dir: Path | None = None


@dataclass(slots=True)
class _AnalysisFrame:
    index: int
    timestamp: float
    image_path: Path
    width: int
    height: int
    gray: bytes
    display_path: Path | None = None
    motion: float = 1.0
    detections: list[Any] = field(default_factory=list)


class KeyframeExtractor:
    """Orchestrate the six-step keyframe selection."""

    def __init__(self, config: ProjectConfig, detector: Any | None = None) -> None:
        self.config = config
        self.detector = detector
        self.ffmpeg = FfmpegRunner(config.video.ffmpeg_binary)
        self.stats = ImageStats(config.keyframe.phash_size)
        self.counts: dict[str, int] = {}
        self.warnings: list[str] = []

    # -- public API ---------------------------------------------------------
    def extract(self, source: KeyframeSource) -> list[KeyframeRecord]:
        self.counts = {
            "total_frames": 0,
            "sampled_frames": 0,
            "motion_removed": 0,
            "dedup_removed": 0,
            "scene_cuts": 0,
            "force_kept": 0,
            "protected_dropped": 0,
            "display_missing": 0,
            "final_keyframes": 0,
        }
        self.warnings = []
        if source.video_path is not None and self.ffmpeg.available() and self.config.keyframe.enable:
            return self._extract_from_video(source)
        if source.video_path is not None and not self.ffmpeg.available():
            if source.frames_dir is not None and iter_image_files(source.frames_dir):
                return self._extract_from_images(source)
            raise FfmpegUnavailableError(
                "ffmpeg is required to decode the configured video but was not found in PATH. "
                "Install ffmpeg or provide an image directory; run "
                "`python -m hazard_annotation doctor` for a full report."
            )
        return self._extract_from_images(source)

    # -- video pipeline -----------------------------------------------------
    def _extract_from_video(self, source: KeyframeSource) -> list[KeyframeRecord]:
        assert source.video_path is not None
        video_path = source.video_path
        sample_fps = self.config.video.sample_fps
        analysis_dir = ensure_dir(self.config.workspace("extracted", "analysis", source.source_id))
        display_dir = ensure_dir(self.config.extracted_frames_dir / source.source_id)

        analysis_files = self._step_coarse_sample_video(video_path, analysis_dir)
        frames = self._load_video_analysis_frames(analysis_files, sample_fps)
        self.counts["total_frames"] = len(frames)
        self.counts["sampled_frames"] = len(frames)
        if not frames:
            return []

        # Steps 4 (scene change) and 5 (target force-keep) run on the ORIGINAL
        # sampled sequence so a scene cut or a key object can never be hidden by
        # an earlier motion / de-dup filter.
        scene_indices = self._scene_indices(frames, video_path=video_path, sample_fps=sample_fps)
        self.counts["scene_cuts"] = len(scene_indices)

        display_map = self._extract_display_frames(video_path, display_dir, sample_fps, frames)
        force_indices = self._step_force_keep(frames, display_map)
        self.counts["force_kept"] = len(force_indices)

        # Steps 2 (motion) and 3 (de-dup) filter the remaining frames but must
        # never drop a frame protected by a scene cut or a forced target.
        protected = set(scene_indices) | set(force_indices)
        frames, motion_removed = self._step_motion(frames, protected)
        self.counts["motion_removed"] = motion_removed
        frames, dedup_removed = self._step_dedup(frames, protected)
        self.counts["dedup_removed"] = dedup_removed

        selected = self._step_cap_and_gap(frames, scene_indices, force_indices)

        records: list[KeyframeRecord] = []
        display_missing = 0
        for frame in selected:
            display_path = display_map.get(frame.index)
            if display_path is None:
                # Never silently drop a selected frame: count it and warn.
                display_missing += 1
                continue
            records.append(
                self._build_record(
                    source_id=source.source_id,
                    original_path=video_path,
                    extracted_path=display_path,
                    relative_name=display_path.name,
                    timestamp=frame.timestamp,
                    reason=self._reason_for(frame.index, scene_indices, force_indices),
                    detections=frame.detections,
                    metadata={"mode": "video", "sample_index": frame.index},
                )
            )
        self.counts["display_missing"] = display_missing
        self.counts["final_keyframes"] = len(records)
        if display_missing:
            self.warnings.append(
                f"display_missing:{display_missing} selected frames had no extracted image"
            )
        return records

    def _step_coarse_sample_video(self, video_path: Path, analysis_dir: Path) -> list[Path]:
        return self.ffmpeg.coarse_sample(
            video=video_path,
            out_dir=analysis_dir,
            fps=self.config.video.sample_fps,
            width=self.config.video.analysis_width,
        )

    def _load_video_analysis_frames(self, files: list[Path], sample_fps: float) -> list[_AnalysisFrame]:
        from hazard_annotation.imageio import PgmImage

        frames: list[_AnalysisFrame] = []
        for path in files:
            try:
                image = PgmImage.load(path)
            except (ValueError, OSError):
                continue
            index = self._parse_sequence_index(path)
            timestamp = (index - 1) / sample_fps if sample_fps > 0 else float(index)
            frames.append(
                _AnalysisFrame(
                    index=index,
                    timestamp=timestamp,
                    image_path=path,
                    width=image.width,
                    height=image.height,
                    gray=image.to_gray_bytes(),
                )
            )
        return frames

    def _extract_display_frames(
        self,
        video_path: Path,
        display_dir: Path,
        sample_fps: float,
        frames: list[_AnalysisFrame],
    ) -> dict[int, Path]:
        if not frames:
            return {}
        indices = sorted({frame.index for frame in frames})
        try:
            produced = self.ffmpeg.extract_indices(
                video=video_path,
                out_dir=display_dir,
                fps=sample_fps,
                indices=indices,
                width=self.config.video.scale_width,
            )
        except (RuntimeError, FfmpegUnavailableError) as error:
            self.warnings.append(f"display_extract_failed:{error}")
            return {}
        # ``extract_indices`` returns explicit ``(index, path)`` pairs, so an
        # image can never be paired with the wrong analysis index regardless of
        # the caller's input order or duplicates. A count mismatch means ffmpeg
        # produced fewer frames than requested; warn loudly instead of silently
        # shrinking the output.
        if len(produced) != len(indices):
            self.warnings.append(
                f"display_count_mismatch:requested={len(indices)},produced={len(produced)}"
            )
        return {index: path for index, path in produced}

    def _step_scene_cut_video(
        self,
        video_path: Path,
        sample_fps: float,
        available_indices: list[int] | None = None,
    ) -> list[int]:
        try:
            return self.ffmpeg.scene_cut_indices(
                video=video_path,
                fps=sample_fps,
                threshold=self.config.keyframe.scene_change_threshold,
                available_indices=available_indices,
            )
        except (RuntimeError, FfmpegUnavailableError):
            return []

    def _scene_indices(
        self,
        frames: list[_AnalysisFrame],
        *,
        video_path: Path | None = None,
        sample_fps: float | None = None,
    ) -> list[int]:
        """Scene-cut indices computed on the original sampled frame sequence.

        The pure-Python histogram detector always runs (it is the only detector
        available for image directories and when ffmpeg is absent); a video
        additionally unions in the ffmpeg ``scene`` filter result, snapped to the
        nearest existing sampled index so every returned index is valid.
        """
        indices = set(self._step_scene_cut_histogram(frames))
        if video_path is not None and sample_fps is not None:
            available = [frame.index for frame in frames]
            indices.update(self._step_scene_cut_video(video_path, sample_fps, available))
        return sorted(indices)

    # -- image-directory pipeline ------------------------------------------
    def _extract_from_images(self, source: KeyframeSource) -> list[KeyframeRecord]:
        assert source.frames_dir is not None
        frames_dir = source.frames_dir
        images = self._step_coarse_sample_images(frames_dir)
        self.counts["total_frames"] = len(iter_image_files(frames_dir))
        self.counts["sampled_frames"] = len(images)
        if not images:
            return []

        frames = self._load_image_analysis_frames(images)

        # Steps 4 (scene change) and 5 (target force-keep) run on the ORIGINAL
        # sampled sequence so a scene cut or a key object can never be hidden by
        # an earlier motion / de-dup filter.
        scene_indices = self._step_scene_cut_histogram(frames)
        self.counts["scene_cuts"] = len(scene_indices)

        force_indices = self._step_force_keep_images(frames)
        self.counts["force_kept"] = len(force_indices)

        # Steps 2 (motion) and 3 (de-dup) filter the remaining frames but must
        # never drop a frame protected by a scene cut or a forced target.
        protected = set(scene_indices) | set(force_indices)
        frames, motion_removed = self._step_motion(frames, protected)
        self.counts["motion_removed"] = motion_removed
        frames, dedup_removed = self._step_dedup(frames, protected)
        self.counts["dedup_removed"] = dedup_removed

        selected = self._step_cap_and_gap(frames, scene_indices, force_indices)

        records: list[KeyframeRecord] = []
        for frame in selected:
            extracted_path = self.config.extracted_frames_dir / source.source_id / frame.image_path.name
            copy_file(frame.image_path, extracted_path)
            records.append(
                self._build_record(
                    source_id=source.source_id,
                    original_path=frame.image_path,
                    extracted_path=extracted_path,
                    relative_name=frame.image_path.name,
                    timestamp=frame.timestamp,
                    reason=self._reason_for(frame.index, scene_indices, force_indices),
                    detections=frame.detections,
                    metadata={"mode": "images", "sequence_index": frame.index},
                )
            )
        # ``final_keyframes`` must reflect the records actually produced.
        self.counts["final_keyframes"] = len(records)
        return records

    def _step_coarse_sample_images(self, frames_dir: Path) -> list[Path]:
        images = iter_image_files(frames_dir)
        selected: list[Path] = []
        for index, image_path in enumerate(images):
            if index % max(1, self.config.frame_step) != 0:
                continue
            if len(selected) >= self.config.max_frames_per_source:
                break
            selected.append(image_path)
        return selected

    def _load_image_analysis_frames(self, images: list[Path]) -> list[_AnalysisFrame]:
        frames: list[_AnalysisFrame] = []
        for index, path in enumerate(images):
            width = height = 0
            gray = b""
            try:
                from hazard_annotation.imageio import load_image

                image = load_image(path)
                width, height = image.width, image.height
                gray = image.to_gray_bytes()
            except (ValueError, OSError):
                pass
            frames.append(
                _AnalysisFrame(
                    index=index,
                    timestamp=float(index),
                    image_path=path,
                    width=width,
                    height=height,
                    gray=gray,
                )
            )
        return frames

    # -- shared steps -------------------------------------------------------
    def _step_motion(
        self,
        frames: list[_AnalysisFrame],
        protected: set[int] | None = None,
    ) -> tuple[list[_AnalysisFrame], int]:
        """Drop near-static frames, but never a protected (scene/target) frame."""
        protected = protected or set()
        kept: list[_AnalysisFrame] = []
        removed = 0
        previous: _AnalysisFrame | None = None
        threshold = self.config.keyframe.pixel_delta_threshold
        minimum = self.config.keyframe.motion_min_ratio
        for frame in frames:
            if previous is None or not frame.gray or not previous.gray:
                frame.motion = 1.0
                kept.append(frame)
            else:
                frame.motion = motion_ratio(previous.gray, frame.gray, threshold)
                if frame.index in protected or frame.motion >= minimum:
                    kept.append(frame)
                else:
                    removed += 1
            previous = frame
        return kept, removed

    def _step_dedup(
        self,
        frames: list[_AnalysisFrame],
        protected: set[int] | None = None,
    ) -> tuple[list[_AnalysisFrame], int]:
        """Collapse perceptually identical frames, but never a protected frame."""
        protected = protected or set()
        kept: list[_AnalysisFrame] = []
        removed = 0
        last_hash: int | None = None
        phash_size = self.config.keyframe.phash_size
        threshold = self.config.keyframe.dedup_hamming_threshold
        for frame in frames:
            if not frame.gray:
                kept.append(frame)
                continue
            digest = ahash(frame.gray, frame.width, frame.height, phash_size)
            if (
                frame.index in protected
                or last_hash is None
                or hamming(last_hash, digest) >= threshold
            ):
                kept.append(frame)
                last_hash = digest
            else:
                removed += 1
        return kept, removed

    def _step_scene_cut_histogram(self, frames: list[_AnalysisFrame]) -> list[int]:
        # Image directories carry no ffmpeg scene scores, so scene changes are
        # always detected with the pure-Python 16-bucket histogram difference.
        scene_indices: list[int] = []
        previous_histogram: list[float] | None = None
        for frame in frames:
            if not frame.gray:
                previous_histogram = None
                continue
            histogram = histogram16(frame.gray)
            if previous_histogram is not None:
                if histogram_delta(previous_histogram, histogram) >= self.config.keyframe.scene_change_threshold:
                    scene_indices.append(frame.index)
            previous_histogram = histogram
        return scene_indices

    def _step_force_keep(
        self,
        frames: list[_AnalysisFrame],
        display_map: dict[int, Path],
    ) -> list[int]:
        if self.detector is None or not display_map:
            return []
        paths = list(display_map.values())
        detections_map = self.detector.detect_many(paths)
        force: list[int] = []
        keep_labels = set(self.config.keyframe.force_keep_labels)
        for index, path in display_map.items():
            detections = detections_map.get(path.resolve(), detections_map.get(path, []))
            for frame in frames:
                if frame.index == index:
                    frame.detections = detections
                    break
            if any(item.label in keep_labels for item in detections):
                force.append(index)
        return force

    def _step_force_keep_images(self, frames: list[_AnalysisFrame]) -> list[int]:
        if self.detector is None:
            return []
        paths = [frame.image_path for frame in frames]
        detections_map = self.detector.detect_many(paths)
        keep_labels = set(self.config.keyframe.force_keep_labels)
        force: list[int] = []
        for frame in frames:
            detections = detections_map.get(frame.image_path.resolve(), detections_map.get(frame.image_path, []))
            frame.detections = detections
            if any(item.label in keep_labels for item in detections):
                force.append(frame.index)
        return force

    def _step_cap_and_gap(
        self,
        frames: list[_AnalysisFrame],
        scene_indices: list[int],
        force_indices: list[int],
    ) -> list[_AnalysisFrame]:
        """Apply the max-keyframe cap and the minimum-gap spacing filter.

        Frames protected by a scene cut or a forced target are **exempt from the
        ``min_gap`` filter**: PRD P0-1.4 promises a scene-cut frame is *forced*
        to survive, so a spacing constraint must never silently veto it. If the
        ``max_keyframes`` cap still forces a protected frame out, the drop is
        counted in ``self.counts['protected_dropped']`` and surfaced as a
        warning — never silent (architecture "严禁静默丢帧" rule).

        The signature is intentionally unchanged (returns just the selected
        frames) so existing direct callers keep working; the drop count travels
        through ``self.counts`` / ``self.warnings``.
        """
        scene_set = set(scene_indices)
        force_set = set(force_indices)
        protected_set = scene_set | force_set

        def priority(frame: _AnalysisFrame) -> int:
            if frame.index in scene_set:
                return 0
            if frame.index in force_set:
                return 1
            return 2

        ordered = sorted(frames, key=lambda frame: (priority(frame), -frame.motion, frame.timestamp))
        selected: list[_AnalysisFrame] = []
        protected_dropped = 0
        min_gap = self.config.keyframe.min_gap_seconds
        max_keyframes = self.config.keyframe.max_keyframes
        for frame in ordered:
            is_protected = frame.index in protected_set
            if len(selected) >= max_keyframes:
                # The cap is full: every remaining frame is dropped. A dropped
                # *protected* frame is a contract violation worth reporting, so
                # keep iterating to count them instead of breaking early.
                if is_protected:
                    protected_dropped += 1
                continue
            if (
                not is_protected
                and min_gap > 0
                and any(abs(frame.timestamp - chosen.timestamp) < min_gap for chosen in selected)
            ):
                continue
            selected.append(frame)
        selected.sort(key=lambda frame: frame.timestamp)
        self.counts["protected_dropped"] = protected_dropped
        if protected_dropped:
            self.warnings.append(
                f"protected_dropped:{protected_dropped} scene/target frames exceeded "
                f"max_keyframes={max_keyframes}"
            )
        return selected

    # -- helpers ------------------------------------------------------------
    @staticmethod
    def _parse_sequence_index(path: Path) -> int:
        stem = path.stem
        digits = "".join(character for character in stem if character.isdigit())
        return int(digits) if digits else 0

    @staticmethod
    def _reason_for(index: int, scene_indices: list[int], force_indices: list[int]) -> str:
        reasons: list[str] = []
        if index in set(scene_indices):
            reasons.append("scene_change")
        if index in set(force_indices):
            reasons.append("force_keep")
        if not reasons:
            reasons.append("motion")
        return "+".join(reasons)

    def _build_record(
        self,
        source_id: str,
        original_path: Path,
        extracted_path: Path,
        relative_name: str,
        timestamp: float,
        reason: str,
        detections: list[Any],
        metadata: dict[str, Any],
    ) -> KeyframeRecord:
        return KeyframeRecord(
            source_id=source_id,
            relative_path=f"{source_id}/{relative_name}",
            original_path=str(original_path.resolve()),
            extracted_path=str(extracted_path.resolve()),
            timestamp=timestamp,
            reason=reason,
            detections=list(detections),
            scene=self.config.scene,
            metadata=metadata,
        )
