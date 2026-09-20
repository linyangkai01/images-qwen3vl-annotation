"""T02 单测：素材编目 + 视频解码与关键帧筛选 + 母项目初检串联。

覆盖要点（对应 T02.4 判据）：

- **关键帧去重**：用合成图构造近似重复帧，验证被去重；
- **KeyframeConfig 各字段确实生效**：``enable / phash_size / dedup_hamming_threshold /
  motion_min_ratio + pixel_delta_threshold / scene_change_threshold / min_gap_seconds /
  max_keyframes / force_keep_labels`` 逐一验证；
- **中文路径**：输入视频与输出目录含中文，全链路可用；
- **ffmpeg 缺失降级**：注入「查不到二进制」的替身，验证给出**明确错误**而非静默空结果；
- **probe_video 对不可解码文件**：给出可读 ``ValueError``。

约束：
- 只用**合成视频 / 合成图**（ffmpeg ``lavfi`` 测试源 + 手写 PGM / ``imageio.encode_png``），
  **绝不触碰 ``industrial-data/`` 下的真实素材**；
- 需要 ffmpeg 的用例在本机无 ffmpeg 时**跳过**（不因环境缺失而失败）；
- ffmpeg 路径优先取环境变量 ``COLDSTART_FFMPEG``，其次本机实测路径，最后 ``PATH``。
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from traffic_cv_closed_loop.config import KeyframeConfig, ProjectConfig
from traffic_cv_closed_loop.imageio import ImageStats, encode_png, load_image
from traffic_cv_closed_loop.io_utils import read_json, read_jsonl

from industrial_hazard_coldstart.inventory import run_inventory, scan_media
from industrial_hazard_coldstart.orchestrator import run_cycle
from industrial_hazard_coldstart.video_decode import (
    FfmpegMissingError,
    probe_video,
    run_video_extract,
    sample_frames,
    select_keyframes,
)

ROOT = Path(__file__).resolve().parents[1]
HOIST_CONFIG = ROOT / "own" / "hoist_project" / "config" / "project.hoist.json"


def _find_ffmpeg() -> str | None:
    """定位 ffmpeg：环境变量 → 本机实测路径 → PATH。"""
    env = os.environ.get("COLDSTART_FFMPEG")
    if env and (shutil.which(env) or Path(env).exists()):
        return env
    known = r"D:\WorkPlace\tools\ffmpeg\bin\ffmpeg.exe"
    if Path(known).exists():
        return known
    return shutil.which("ffmpeg")


FFMPEG = _find_ffmpeg()
requires_ffmpeg = unittest.skipUnless(FFMPEG, "本机无 ffmpeg，跳过需要真实解码的用例")


# --------------------------------------------------------------------------- #
# 合成素材工具（零真实素材）
# --------------------------------------------------------------------------- #
def _write_pgm(path: Path, width: int, height: int, value_fn) -> Path:
    """写一张灰度 PGM（P5）合成图。"""
    pixels = bytes(value_fn(x, y) for y in range(height) for x in range(width))
    path.write_bytes(f"P5\n{width} {height}\n255\n".encode("ascii") + pixels)
    return path


def _gradient(offset: int):
    """返回一个随 (x, y, offset) 变化的灰度取值函数（用于造互不相同的帧）。"""
    return lambda x, y: (x * 8 + y * 3 + offset) % 256


def _make_video(path: Path, seconds: int = 6, fps: int = 10, size: str = "320x240") -> Path:
    """用 ffmpeg ``lavfi`` 测试源生成一段合成视频。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        FFMPEG,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"testsrc=duration={seconds}:size={size}:rate={fps}",
        "-pix_fmt",
        "yuv420p",
        str(path),
    ]
    subprocess.run(command, check=True)
    return path


def _make_config(
    tmp: Path,
    *,
    ffmpeg_binary: str | None = None,
    sample_fps: float = 1.0,
    keyframe: dict | None = None,
    include_empty_frames: bool = True,
    detector: str = "sidecar",
) -> ProjectConfig:
    """在临时项目里写一份 hoist 配置并加载。"""
    raw = json.loads(HOIST_CONFIG.read_text(encoding="utf-8"))
    raw["video"]["ffmpeg_binary"] = (
        ffmpeg_binary if ffmpeg_binary is not None else (FFMPEG or "ffmpeg")
    )
    raw["video"]["sample_fps"] = sample_fps
    if keyframe:
        raw["keyframe"].update(keyframe)
    raw["yolo"]["detector"] = detector
    raw["pipeline"]["include_empty_frames"] = include_empty_frames

    config_dir = tmp / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "project.json"
    config_path.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
    return ProjectConfig.load(config_path)


def _no_ffmpeg():
    """让 ``video_decode`` 找不到任何二进制（注入替身，验证降级）。"""
    return mock.patch(
        "industrial_hazard_coldstart.video_decode.shutil.which", return_value=None
    )


# --------------------------------------------------------------------------- #
# 1. 关键帧筛选：KeyframeConfig 各字段生效
# --------------------------------------------------------------------------- #
class SelectKeyframesTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _frames(self, count: int, step: int = 40) -> list[Path]:
        frames = []
        for index in range(count):
            path = self.tmp / f"g_{index:06d}_{index:09.3f}.pgm"
            _write_pgm(path, 16, 16, _gradient(index * step))
            frames.append(path)
        return frames

    def test_dedup_removes_near_duplicates(self) -> None:
        # 第 1、2 帧完全相同 → 第二帧应被判重丢弃；第三帧不同 → 保留。
        frames = [
            _write_pgm(self.tmp / "frame_000000_00000.000.pgm", 16, 16, _gradient(0)),
            _write_pgm(self.tmp / "frame_000001_00001.000.pgm", 16, 16, _gradient(0)),
            _write_pgm(self.tmp / "frame_000002_00002.000.pgm", 16, 16, _gradient(120)),
        ]
        kf = KeyframeConfig(
            enable=True,
            motion_min_ratio=0.0,
            dedup_hamming_threshold=4,
            phash_size=8,
            scene_change_threshold=1.1,
            max_keyframes=100,
            min_gap_seconds=0.0,
        )
        records = select_keyframes(frames, kf, ImageStats(kf.phash_size))
        self.assertEqual([r.frame_index for r in records], [0, 2])

    def test_max_keyframes_truncates(self) -> None:
        frames = self._frames(6)
        kf = KeyframeConfig(
            enable=True,
            motion_min_ratio=0.0,
            dedup_hamming_threshold=-1,  # 关闭去重（汉明距离恒 ≥ 0）
            scene_change_threshold=1.1,
            max_keyframes=2,
            min_gap_seconds=0.0,
        )
        records = select_keyframes(frames, kf, ImageStats(kf.phash_size))
        self.assertEqual(len(records), 2)
        self.assertEqual([r.frame_index for r in records], [0, 1])

    def test_min_gap_seconds_filters(self) -> None:
        # 时间戳 0/1/2/3/4/5 秒，min_gap=2 → 保留 0、2、4。
        frames = self._frames(6)
        kf = KeyframeConfig(
            enable=True,
            motion_min_ratio=0.0,
            dedup_hamming_threshold=-1,
            scene_change_threshold=1.1,
            max_keyframes=100,
            min_gap_seconds=2.0,
        )
        records = select_keyframes(frames, kf, ImageStats(kf.phash_size))
        self.assertEqual([r.timestamp_s for r in records], [0.0, 2.0, 4.0])

    def test_motion_min_ratio_filters_static(self) -> None:
        static = _write_pgm(self.tmp / "m_000000_00000.000.pgm", 16, 16, lambda x, y: 50)
        # 约 10% 像素变化（远低于 motion_min_ratio=0.5）→ 判静丢弃。
        small = bytearray([50] * 256)
        for index in range(26):
            small[index] = 200
        slight = self.tmp / "m_000001_00001.000.pgm"
        slight.write_bytes(b"P5\n16 16\n255\n" + bytes(small))
        # 约 78% 像素变化 → 保留。
        big = bytearray([50] * 256)
        for index in range(200):
            big[index] = 200
        moving = self.tmp / "m_000002_00002.pgm"
        moving.write_bytes(b"P5\n16 16\n255\n" + bytes(big))

        kf = KeyframeConfig(
            enable=True,
            motion_min_ratio=0.5,
            pixel_delta_threshold=25,
            dedup_hamming_threshold=-1,
            scene_change_threshold=1.1,
            max_keyframes=100,
            min_gap_seconds=0.0,
        )
        records = select_keyframes([static, slight, moving], kf, ImageStats(kf.phash_size))
        self.assertEqual([r.frame_index for r in records], [0, 2])
        self.assertGreater(records[-1].motion_ratio, 0.5)

    def test_scene_change_forces_keep(self) -> None:
        # scene_change_threshold=0 → 任何帧都判为场景切换，近似重复帧也被强制保留。
        first = _write_pgm(self.tmp / "s_000000_00000.000.pgm", 16, 16, _gradient(0))
        second = _write_pgm(self.tmp / "s_000001_00001.000.pgm", 16, 16, _gradient(0))
        kf = KeyframeConfig(
            enable=True,
            motion_min_ratio=0.0,
            dedup_hamming_threshold=4,
            scene_change_threshold=0.0,
            max_keyframes=100,
            min_gap_seconds=0.0,
        )
        records = select_keyframes([first, second], kf, ImageStats(kf.phash_size))
        self.assertEqual(len(records), 2)
        self.assertTrue(records[-1].kept_reason.startswith("scene_change"))

    def test_phash_size_flows_into_imageio(self) -> None:
        path = _write_pgm(self.tmp / "p_000000_00000.000.pgm", 16, 16, _gradient(0))
        image = load_image(path)
        self.assertEqual(len(ImageStats(4).block_downscale(image)), 4)
        self.assertEqual(len(ImageStats(8).block_downscale(image)), 8)
        hash4 = select_keyframes([path], KeyframeConfig(phash_size=4), ImageStats(4))[0].phash
        hash8 = select_keyframes([path], KeyframeConfig(phash_size=8), ImageStats(8))[0].phash
        self.assertNotEqual(hash4, hash8)

    def test_force_keep_labels_override(self) -> None:
        first = _write_pgm(self.tmp / "f_000000_00000.000.pgm", 16, 16, _gradient(0))
        duplicate = _write_pgm(self.tmp / "f_000001_00001.000.pgm", 16, 16, _gradient(0))
        # 给「重复帧」挂上强制保留标签的侧车。
        (self.tmp / "f_000001_00001.000.labels.json").write_text(
            json.dumps(["person"], ensure_ascii=False), encoding="utf-8"
        )
        kf = KeyframeConfig(
            enable=True,
            motion_min_ratio=0.0,
            dedup_hamming_threshold=4,
            scene_change_threshold=1.1,
            max_keyframes=100,
            min_gap_seconds=0.0,
            force_keep_labels=["person"],
        )
        records = select_keyframes([first, duplicate], kf, ImageStats(kf.phash_size))
        self.assertEqual(len(records), 2)
        self.assertTrue(records[-1].kept_reason.startswith("force_keep_label"))

    def test_disable_keeps_all_frames(self) -> None:
        frames = self._frames(5)
        kf = KeyframeConfig(enable=False, dedup_hamming_threshold=4, max_keyframes=1)
        records = select_keyframes(frames, kf, ImageStats(kf.phash_size))
        # enable=False 时不做任何筛选（含不施加 max_keyframes 上限）。
        self.assertEqual(len(records), 5)
        self.assertTrue(all(r.kept_reason == "keyframe_disabled" for r in records))

    def test_records_round_trip(self) -> None:
        frames = self._frames(3)
        kf = KeyframeConfig(
            enable=True,
            motion_min_ratio=0.0,
            dedup_hamming_threshold=-1,
            scene_change_threshold=1.1,
            max_keyframes=100,
            min_gap_seconds=0.0,
        )
        records = select_keyframes(frames, kf, ImageStats(kf.phash_size))
        for record in records:
            self.assertEqual(
                type(record).from_dict(record.to_dict()).to_dict(), record.to_dict()
            )


# --------------------------------------------------------------------------- #
# 2. ffmpeg 缺失降级（不依赖真实 ffmpeg）
# --------------------------------------------------------------------------- #
class FfmpegDegradationTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_sample_frames_raises_when_ffmpeg_missing(self) -> None:
        video = self.tmp / "dummy.mp4"
        video.write_bytes(b"")
        with _no_ffmpeg():
            with self.assertRaises(FfmpegMissingError) as ctx:
                sample_frames(video, self.tmp / "out", 1.0, "coldstart-no-such-ffmpeg")
        message = str(ctx.exception)
        self.assertIn("ffmpeg", message)
        self.assertIn("ffmpeg_binary", message)
        # 绝不能静默产出空结果。
        self.assertFalse((self.tmp / "out").exists() and any((self.tmp / "out").iterdir()))

    def test_probe_video_raises_when_ffmpeg_missing(self) -> None:
        video = self.tmp / "dummy.mp4"
        video.write_bytes(b"")
        with _no_ffmpeg():
            with self.assertRaises(FfmpegMissingError):
                probe_video(video, "coldstart-no-such-ffmpeg")

    def test_run_video_extract_raises_when_ffmpeg_missing(self) -> None:
        config = _make_config(self.tmp, ffmpeg_binary="coldstart-no-such-ffmpeg")
        video = self.tmp / "dummy.mp4"
        video.write_bytes(b"")
        with _no_ffmpeg():
            with self.assertRaises(FfmpegMissingError):
                run_video_extract(config, video_paths=[video])

    def test_run_video_extract_errors_when_no_videos(self) -> None:
        config = _make_config(self.tmp, ffmpeg_binary="coldstart-no-such-ffmpeg")
        # 默认媒体根不存在 → 找不到任何视频 → 明确报错（而非空结果）。
        with self.assertRaises(ValueError):
            run_video_extract(config)

    def test_unsupported_decode_backend_raises(self) -> None:
        config = _make_config(self.tmp)
        config.video.decode_backend = "opencv"  # type: ignore[misc]
        with self.assertRaises(ValueError):
            run_video_extract(config, video_paths=[self.tmp / "x.mp4"])


# --------------------------------------------------------------------------- #
# 3. probe_video 容器头 / 不可解码（需 ffmpeg）
# --------------------------------------------------------------------------- #
@requires_ffmpeg
class ProbeVideoTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_probe_reads_container_header(self) -> None:
        video = _make_video(self.tmp / "clip.mp4", seconds=6, fps=10, size="320x240")
        info = probe_video(video, FFMPEG)
        self.assertEqual(info["width"], 320)
        self.assertEqual(info["height"], 240)
        self.assertAlmostEqual(info["fps"], 10.0, places=3)
        self.assertGreaterEqual(info["duration_s"], 5.5)
        self.assertGreater(info["nb_frames"], 0)

    def test_probe_rejects_undecodable_file(self) -> None:
        broken = self.tmp / "garbage.mp4"
        broken.write_bytes(b"this is definitely not a video container" * 8)
        with self.assertRaises(ValueError) as ctx:
            probe_video(broken, FFMPEG)
        self.assertIn("无法解码", str(ctx.exception))

    def test_probe_missing_file_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            probe_video(self.tmp / "missing.mp4", FFMPEG)


# --------------------------------------------------------------------------- #
# 4. 中文路径全链路（需 ffmpeg）
# --------------------------------------------------------------------------- #
@requires_ffmpeg
class ChinesePathTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_run_video_extract_with_chinese_paths(self) -> None:
        config = _make_config(
            self.tmp,
            sample_fps=1.0,
            keyframe={
                "max_keyframes": 100,
                "min_gap_seconds": 0.0,
                "motion_min_ratio": 0.0,
                "scene_change_threshold": 1.1,
            },
        )
        # 输入与输出路径均含中文。
        video = _make_video(self.tmp / "input" / "media" / "珠海素材" / "吊装视频.mp4")

        summary = run_video_extract(config, video_paths=[video])

        self.assertEqual(summary["total_videos"], 1)
        self.assertGreater(summary["total_sampled_frames"], 0)
        self.assertGreater(summary["total_kept_frames"], 0)

        output_dir = config.input_frames_dir / "吊装视频"
        self.assertTrue(output_dir.exists())
        kept = sorted(output_dir.glob("*.png"))
        self.assertEqual(len(kept), summary["total_kept_frames"])
        # 关键帧清单落盘（含中文路径）。
        keyframe_manifest = config.workspace("extracted", "keyframes", "吊装视频.json")
        self.assertTrue(keyframe_manifest.exists())
        payload = read_json(keyframe_manifest)
        self.assertEqual(len(payload["keyframes"]), summary["total_kept_frames"])


# --------------------------------------------------------------------------- #
# 5. 素材编目（F1）
# --------------------------------------------------------------------------- #
class InventoryTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_scan_media_registers_images_with_chinese_paths(self) -> None:
        media = self.tmp / "media" / "中文目录"
        media.mkdir(parents=True)
        (media / "照片一.png").write_bytes(encode_png(12, 7, bytes([1, 2, 3] * (12 * 7))))
        _write_pgm(media / "照片二.pgm", 5, 9, lambda x, y: x + y)

        assets = scan_media(self.tmp / "media", "hoist", ffmpeg_binary=None)
        by_type = {a.media_type for a in assets}
        self.assertEqual(by_type, {"image"})
        self.assertEqual(len(assets), 2)
        dimensions = {(a.width, a.height) for a in assets}
        self.assertIn((12, 7), dimensions)  # PNG 走 IHDR 头部解析
        self.assertIn((5, 9), dimensions)  # PGM 走 imageio
        self.assertTrue(all(a.count == 1 for a in assets))

    def test_run_inventory_writes_manifest(self) -> None:
        media = self.tmp / "input" / "media" / "宿州照片"
        media.mkdir(parents=True)
        (media / "a.png").write_bytes(encode_png(10, 10, bytes([9, 9, 9] * 100)))
        # 用一个不存在的 ffmpeg：编目应**降级但不失败**。
        config = _make_config(self.tmp, ffmpeg_binary="coldstart-no-such-ffmpeg")

        manifest = run_inventory(config)

        self.assertEqual(manifest["counts"]["image"], 1)
        self.assertEqual(manifest["counts"]["video"], 0)
        self.assertTrue(manifest["media_root_exists"])
        required = {
            "source_id",
            "path",
            "scene",
            "media_type",
            "duration_s",
            "width",
            "height",
            "fps",
            "count",
        }
        for asset in manifest["assets"]:
            self.assertTrue(required.issubset(asset.keys()))
            self.assertEqual(asset["scene"], "hoist")
        manifest_path = config.workspace("inventory", "manifest.json")
        self.assertTrue(manifest_path.exists())
        self.assertEqual(read_json(manifest_path)["counts"]["image"], 1)

    @requires_ffmpeg
    def test_video_metadata_via_ffprobe(self) -> None:
        media = self.tmp / "input" / "media"
        _make_video(media / "clip.mp4", seconds=4, fps=12, size="160x120")
        config = _make_config(self.tmp, ffmpeg_binary=FFMPEG)

        manifest = run_inventory(config)

        self.assertEqual(manifest["counts"]["video"], 1)
        asset = manifest["assets"][0]
        self.assertEqual(asset["media_type"], "video")
        self.assertEqual(asset["width"], 160)
        self.assertEqual(asset["height"], 120)
        self.assertAlmostEqual(asset["fps"], 12.0, places=3)
        self.assertGreater(asset["duration_s"], 0.0)
        self.assertGreater(asset["count"], 0)


# --------------------------------------------------------------------------- #
# 6. 编排：关键帧 → 母项目初检（T02.3）
# --------------------------------------------------------------------------- #
@requires_ffmpeg
class OrchestratorTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_extract_video_then_extract_produces_candidates(self) -> None:
        config = _make_config(
            self.tmp,
            sample_fps=1.0,
            keyframe={
                "max_keyframes": 100,
                "min_gap_seconds": 0.0,
                "motion_min_ratio": 0.0,
                "scene_change_threshold": 1.1,
            },
            include_empty_frames=False,
        )
        video = _make_video(self.tmp / "input" / "media" / "clip.mp4", seconds=6)

        video_result = run_cycle(config, "extract-video", video_paths=[video])
        self.assertEqual(video_result["stage"], "extract-video")
        self.assertGreater(video_result["total_kept_frames"], 0)

        # 给其中一帧挂上 sidecar 检测，验证 detections 能贯穿到 candidates.jsonl。
        frame = sorted((config.input_frames_dir / "clip").glob("*.png"))[0]
        frame.with_suffix(".detections.json").write_text(
            json.dumps(
                {
                    "objects": [
                        {
                            "label": "person",
                            "bbox": [0.1, 0.1, 0.4, 0.5],
                            "confidence": 0.9,
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )

        extract_result = run_cycle(config, "extract", auto_extract_video=False)
        self.assertEqual(extract_result["stage"], "extract")
        self.assertGreaterEqual(extract_result["selected_candidates"], 1)

        candidates = read_jsonl(config.candidates_path)
        self.assertTrue(candidates)
        # CandidateFrame 结构与母项目兼容。
        required = {
            "source_id",
            "relative_path",
            "original_path",
            "extracted_path",
            "detections",
            "scene",
            "metadata",
        }
        self.assertTrue(required.issubset(candidates[0].keys()))
        self.assertEqual(candidates[0]["detections"][0]["label"], "person")

    def test_extract_auto_chains_video_extract(self) -> None:
        config = _make_config(
            self.tmp,
            sample_fps=1.0,
            keyframe={
                "max_keyframes": 100,
                "min_gap_seconds": 0.0,
                "motion_min_ratio": 0.0,
                "scene_change_threshold": 1.1,
            },
            include_empty_frames=True,
        )
        _make_video(self.tmp / "input" / "media" / "clip.mp4", seconds=4)

        # input_frames_dir 尚无帧 → extract 应先自动抽帧再初检。
        result = run_cycle(config, "extract")

        self.assertIsNotNone(result["video_extract"])
        self.assertGreater(result["video_extract"]["total_kept_frames"], 0)
        self.assertGreaterEqual(result["selected_candidates"], 1)
        self.assertTrue(config.candidates_path.exists())

    def test_unknown_stage_raises(self) -> None:
        config = _make_config(self.tmp)
        with self.assertRaises(ValueError):
            run_cycle(config, "definitely-not-a-stage")


# --------------------------------------------------------------------------- #
# 7. probe_video 有效性判定（#2）：图片冒充视频不得被放行为假视频
# --------------------------------------------------------------------------- #
def _kf(**overrides) -> KeyframeConfig:
    """宽松的 ``KeyframeConfig``（默认关闭去重，便于隔离被测维度）。"""
    base = dict(
        enable=True,
        motion_min_ratio=0.0,
        pixel_delta_threshold=25,
        dedup_hamming_threshold=-1,
        phash_size=8,
        scene_change_threshold=1.1,
        max_keyframes=1000,
        min_gap_seconds=0.0,
    )
    base.update(overrides)
    return KeyframeConfig(**base)


@requires_ffmpeg
class ProbeValidityTest(unittest.TestCase):
    """#2：``probe_video`` 以 ``duration_s`` 为主判据，拒绝图片冒充的视频。"""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_png_disguised_as_mp4_is_rejected(self) -> None:
        """一张 PNG 改名成 .mp4：必须 ValueError 拒绝（不得放行为假视频）。"""
        png = self.tmp / "src.png"
        png.write_bytes(encode_png(64, 64, bytes([7, 7, 7] * (64 * 64))))
        fake = self.tmp / "fake.mp4"
        fake.write_bytes(png.read_bytes())

        with self.assertRaises(ValueError) as ctx:
            probe_video(fake, FFMPEG)
        self.assertIn("没有可解码的视频时长", str(ctx.exception))

    def test_single_frame_video_still_accepted(self) -> None:
        """对照：真正的单帧视频时长 > 0，不得被有效性判定误伤。"""
        single = self.tmp / "single.mp4"
        subprocess.run(
            [FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi",
             "-i", "testsrc=duration=2:size=160x120:rate=10", "-frames:v", "1",
             "-pix_fmt", "yuv420p", str(single)],
            check=True,
        )
        info = probe_video(single, FFMPEG)
        self.assertEqual(info["nb_frames"], 1)
        self.assertGreater(info["duration_s"], 0.0)

    def test_disguised_video_not_registered_as_fake_fps(self) -> None:
        """编目层：图片冒充视频不得登记出 ``fps=25、count=0`` 的假视频。"""
        media = self.tmp / "input" / "media"
        media.mkdir(parents=True)
        png = media / "src.png"
        png.write_bytes(encode_png(32, 32, bytes([3, 3, 3] * (32 * 32))))
        (media / "fake.mp4").write_bytes(png.read_bytes())
        config = _make_config(self.tmp, ffmpeg_binary=FFMPEG)

        manifest = run_inventory(config)

        fake_assets = [a for a in manifest["assets"] if a["path"].endswith("fake.mp4")]
        self.assertEqual(len(fake_assets), 1)
        asset = fake_assets[0]
        self.assertEqual(asset["fps"], 0.0)  # 不得为 25.0
        self.assertEqual(asset["count"], 0)
        self.assertEqual(asset["duration_s"], 0.0)


# --------------------------------------------------------------------------- #
# 8. 子进程返回码显示为有符号（#3）
# --------------------------------------------------------------------------- #
class SignedReturnCodeTest(unittest.TestCase):
    """#3：负返回码不得显示成 ``4294967274`` 这类无符号数。"""

    def test_negative_codes_are_restored(self) -> None:
        from industrial_hazard_coldstart.video_decode import _signed_returncode

        self.assertEqual(_signed_returncode(0), 0)
        self.assertEqual(_signed_returncode(1), 1)
        self.assertEqual(_signed_returncode(0x7FFFFFFF), 0x7FFFFFFF)
        # -22（EINVAL）在 Windows 上表现为 4294967274。
        self.assertEqual(_signed_returncode(4294967274), -22)
        # -9（SIGKILL）表现为 4294967287。
        self.assertEqual(_signed_returncode(4294967287), -9)
        self.assertEqual(_signed_returncode(0x80000000), -2147483648)


# --------------------------------------------------------------------------- #
# 9. dedup_hamming_threshold 三种取值语义（#4）
# --------------------------------------------------------------------------- #
class DedupSemanticsTest(unittest.TestCase):
    """#4：``>0`` 近似判重 / ``=0`` 仅完全相同判重 / ``<0`` 完全关闭。"""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_three_value_semantics(self) -> None:
        # 用渐变帧（而非纯色帧）：纯色帧之间 aHash 只差「亮度高位」，距离恒为 1，
        # 无法体现「>0 近似判重」；渐变帧的距离远大于 4。
        identical_a = _write_pgm(self.tmp / "d_000000_00000.000.pgm", 16, 16, _gradient(0))
        identical_b = _write_pgm(self.tmp / "d_000001_00001.000.pgm", 16, 16, _gradient(0))
        different = _write_pgm(self.tmp / "d_000002_00002.000.pgm", 16, 16, _gradient(120))
        frames = [identical_a, identical_b, different]

        # >0：近似重复去重。
        kept_pos = select_keyframes(frames, _kf(dedup_hamming_threshold=4), ImageStats(8))
        self.assertEqual([r.frame_index for r in kept_pos], [0, 2])
        # =0：仅完全相同的 aHash 去重（相同帧距离 0 → 仍去重）。
        kept_zero = select_keyframes(frames, _kf(dedup_hamming_threshold=0), ImageStats(8))
        self.assertEqual([r.frame_index for r in kept_zero], [0, 2])
        # <0：完全关闭去重。
        kept_neg = select_keyframes(frames, _kf(dedup_hamming_threshold=-1), ImageStats(8))
        self.assertEqual([r.frame_index for r in kept_neg], [0, 1, 2])

    def test_docstring_documents_three_value_semantics(self) -> None:
        doc = select_keyframes.__doc__ or ""
        self.assertIn("完全关闭去重", doc)
        self.assertIn("完全相同", doc)
        self.assertIn("< 0", doc)


# --------------------------------------------------------------------------- #
# 10. 批量容错（#5）：一个坏视频不得拖垮整批，且绝不静默
# --------------------------------------------------------------------------- #
@requires_ffmpeg
class BatchFaultToleranceTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _good_and_bad(self) -> tuple[Path, Path]:
        media = self.tmp / "input" / "media"
        good = _make_video(media / "good.mp4", seconds=6)
        bad = media / "bad.mp4"
        bad.write_bytes(b"")  # 0 字节：不可解码
        return good, bad

    def _config(self) -> ProjectConfig:
        return _make_config(
            self.tmp,
            sample_fps=1.0,
            keyframe={
                "max_keyframes": 100,
                "min_gap_seconds": 0.0,
                "motion_min_ratio": 0.0,
                "scene_change_threshold": 1.1,
            },
        )

    def test_one_bad_video_does_not_abort_batch(self) -> None:
        good, bad = self._good_and_bad()
        config = self._config()

        summary = run_video_extract(config, video_paths=[good, bad])

        self.assertEqual(summary["total_videos"], 2)
        self.assertEqual(summary["succeeded_videos"], 1)
        self.assertTrue(summary["has_failures"])
        self.assertEqual(len(summary["failed_videos"]), 1)
        self.assertEqual(summary["failed_videos"][0]["source_id"], "bad")
        # 好视频仍被正常处理（整批未被拖垮）。
        self.assertGreater(summary["total_kept_frames"], 0)
        self.assertTrue((config.input_frames_dir / "good").exists())

    def test_partial_failure_is_not_silent_via_summary(self) -> None:
        """「跳过并记录」而非「悄悄跳过」：失败视频与原因必须在摘要里可见。"""
        good, bad = self._good_and_bad()
        summary = run_video_extract(self._config(), video_paths=[good, bad])

        failed = summary["failed_videos"][0]
        self.assertEqual(set(failed), {"source_id", "video", "error_type", "reason"})
        self.assertTrue(failed["reason"])  # 原因非空

    def test_partial_failure_cli_returns_4(self) -> None:
        self._good_and_bad()
        config = self._config()
        from industrial_hazard_coldstart.cli import EXIT_STAGE_FAILED, main

        code = main(["--config", str(config.config_path), "extract-video"])
        self.assertEqual(code, EXIT_STAGE_FAILED)


# --------------------------------------------------------------------------- #
# 11. 编排幂等性（#1 工程师侧镜像）
# --------------------------------------------------------------------------- #
@requires_ffmpeg
class IdempotencyTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_rerun_with_tighter_threshold_removes_stale_keyframes(self) -> None:
        video = _make_video(self.tmp / "input" / "media" / "clip.mp4", seconds=8)
        config = _make_config(
            self.tmp,
            sample_fps=1.0,
            keyframe={
                "max_keyframes": 100,
                "min_gap_seconds": 0.0,
                "motion_min_ratio": 0.0,
                "scene_change_threshold": 1.1,
            },
        )
        run_video_extract(config, video_paths=[video])  # 第一轮：宽松
        config.keyframe.min_gap_seconds = 4.0  # 第二轮：收紧，保留数变少
        second = run_video_extract(config, video_paths=[video])

        output_dir = config.input_frames_dir / "clip"
        on_disk = {p.name for p in output_dir.glob("*.png")}
        manifest = read_json(config.workspace("extracted", "keyframes", "clip.json"))
        listed = {Path(item["extracted_path"]).name for item in manifest["keyframes"]}
        self.assertEqual(second["total_kept_frames"], len(listed))
        self.assertEqual(on_disk, listed)  # 无陈旧帧残留


if __name__ == "__main__":
    unittest.main()
