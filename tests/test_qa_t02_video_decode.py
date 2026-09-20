"""T02 独立验证（QA）：素材编目 + 视频解码与关键帧筛选的边界与降级行为。

本文件由 QA 独立编写，**不复用**工程师的 ``tests/test_video_decode.py``，
刻意从「对抗性输入」角度验证 T02 的实现，重点回答：

1. **双产物（PNG + PGM 孪生）设计是否会导致「静默降级」** —— 删除 PGM 后
   ``select_keyframes`` 必须**显式报错**，绝不能静默放行全部帧（否则去重失效，
   下游 VLM 调用量暴增而调用方无感）。见 :class:`NoSilentDegradationTest`。
2. **视频解码异常输入** —— 单帧 / 极短 / 音频-only / 图片冒充视频 / 空文件 /
   不存在路径 / 0 时长，逐一确认「抛什么、是否可读、是否静默返回空」。
3. **``KeyframeConfig`` 边界值** —— ``max_keyframes=1`` / ``min_gap`` 超时长 /
   ``motion_min_ratio`` 两端 / ``phash_size`` 两端 / ``dedup_hamming_threshold`` 语义。
4. **帧序号→秒换算** —— ``ts = i / sample_fps`` 是否与 ffmpeg 实际 PTS 对齐。
5. **中文多层路径**（含空格）。
6. **编排幂等性** —— 同项目连跑两次 ``extract-video`` 是否残留陈旧关键帧。
7. **CLI 退出码**（0 / 3 / 4）。

约束：不修改任何生产代码；只用合成视频；不读真实素材；不联网；临时文件用
``tempfile``。
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from traffic_cv_closed_loop.config import KeyframeConfig, ProjectConfig
from traffic_cv_closed_loop.imageio import ImageStats, load_image
from traffic_cv_closed_loop.io_utils import read_json, read_jsonl

from industrial_hazard_coldstart import video_decode as vd
from industrial_hazard_coldstart.orchestrator import run_cycle

ROOT = Path(__file__).resolve().parents[1]
HOIST_CONFIG = ROOT / "own" / "hoist_project" / "config" / "project.hoist.json"


# --------------------------------------------------------------------------- #
# 环境定位（与工程师测试一致的口径：环境变量 → 本机实测路径 → PATH）
# --------------------------------------------------------------------------- #
def _find_ffmpeg() -> str | None:
    env = os.environ.get("COLDSTART_FFMPEG")
    if env and (shutil.which(env) or Path(env).exists()):
        return env
    known = r"D:\WorkPlace\tools\ffmpeg\bin\ffmpeg.exe"
    if Path(known).exists():
        return known
    return shutil.which("ffmpeg")


FFMPEG = _find_ffmpeg()
requires_ffmpeg = unittest.skipUnless(FFMPEG, "本机无 ffmpeg，跳过需要真实解码的用例")

#: 宽松的 ``KeyframeConfig``：关闭一切门控，只保留「首帧」，便于隔离被测维度。
def _open_kf(**overrides) -> KeyframeConfig:
    base = dict(
        enable=True,
        motion_min_ratio=0.0,
        pixel_delta_threshold=25,
        dedup_hamming_threshold=-1,  # <0 = 关闭去重（实测语义，见 DedupThresholdTest）
        phash_size=8,
        scene_change_threshold=1.1,
        max_keyframes=1000,
        min_gap_seconds=0.0,
    )
    base.update(overrides)
    return KeyframeConfig(**base)


def _make_video(path: Path, seconds: float = 6, fps: int = 10, size: str = "320x240") -> Path:
    """用 ffmpeg lavfi 测试源造一段合成视频（绝不使用真实素材）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
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
        ],
        check=True,
    )
    return path


def _write_pgm(path: Path, width: int, height: int, value_fn) -> Path:
    pixels = bytes(value_fn(x, y) for y in range(height) for x in range(width))
    path.write_bytes(f"P5\n{width} {height}\n255\n".encode("ascii") + pixels)
    return path


def _make_config(tmp: Path, *, ffmpeg_binary: str | None = None, sample_fps: float = 1.0,
                 keyframe: dict | None = None) -> ProjectConfig:
    raw = json.loads(HOIST_CONFIG.read_text(encoding="utf-8"))
    raw["video"]["ffmpeg_binary"] = ffmpeg_binary if ffmpeg_binary is not None else (FFMPEG or "ffmpeg")
    raw["video"]["sample_fps"] = sample_fps
    if keyframe:
        raw["keyframe"].update(keyframe)
    config_dir = tmp / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "project.json"
    config_path.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
    return ProjectConfig.load(config_path)


# --------------------------------------------------------------------------- #
# 1. 双产物设计是否导致「静默降级」（本批首要验证目标）
# --------------------------------------------------------------------------- #
@requires_ffmpeg
class NoSilentDegradationTest(unittest.TestCase):
    """验证 PGM 缺失时是「显式报错」而非「静默放行全部帧」。"""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.video = _make_video(self.tmp / "clip.mp4", seconds=8, fps=10)
        self.frames_dir = self.tmp / "frames"
        vd.sample_frames(self.video, self.frames_dir, 1.0, FFMPEG)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_png_and_pgm_are_always_paired(self) -> None:
        pngs = sorted(self.frames_dir.glob("frame_*.png"))
        pgms = sorted(self.frames_dir.glob("frame_*.pgm"))
        self.assertTrue(pngs)
        self.assertEqual([p.stem for p in pngs], [p.stem for p in pgms])

    def test_imageio_cannot_decode_png_only_netpbm(self) -> None:
        # 这是「PGM 是去重必需」的根因：母项目 imageio 不认 PNG。
        png = sorted(self.frames_dir.glob("frame_*.png"))[0]
        with self.assertRaises(ValueError):
            load_image(png)

    def test_delete_all_pgm_raises_not_silent(self) -> None:
        """删掉全部 PGM 后调 select_keyframes：必须抛错，绝不能返回「全部帧」。"""
        pngs = sorted(self.frames_dir.glob("frame_*.png"))
        total = len(pngs)
        for pgm in self.frames_dir.glob("frame_*.pgm"):
            pgm.unlink()

        with self.assertRaises(ValueError) as ctx:
            vd.select_keyframes(pngs, _open_kf(), ImageStats(8))
        message = str(ctx.exception)
        self.assertIn(".pgm", message)  # 错误信息明确指向缺失的 PGM 分析帧
        self.assertIn("Netpbm", message)
        # 关键：绝不能「静默返回空」或「静默返回全部帧」。
        self.assertNotEqual(total, 0)

    def test_partial_pgm_deletion_raises_not_partial_pass(self) -> None:
        """只删一半 PGM：也必须抛错，而不是「仅对缺 PGM 的帧静默放行」。"""
        pngs = sorted(self.frames_dir.glob("frame_*.png"))
        pgms = sorted(self.frames_dir.glob("frame_*.pgm"))
        for index, pgm in enumerate(pgms):
            if index % 2 == 1:
                pgm.unlink()

        with self.assertRaises(ValueError) as ctx:
            vd.select_keyframes(pngs, _open_kf(), ImageStats(8))
        self.assertIn(".pgm", str(ctx.exception))

    def test_baseline_with_pgm_present_dedups(self) -> None:
        """对照：PGM 齐全时去重正常（避免「因为报错才通过」的假阴性）。"""
        pngs = sorted(self.frames_dir.glob("frame_*.png"))
        kept = vd.select_keyframes(pngs, _open_kf(dedup_hamming_threshold=4), ImageStats(8))
        self.assertLess(len(kept), len(pngs))
        self.assertGreater(len(kept), 0)


# --------------------------------------------------------------------------- #
# 2. 视频解码异常输入
# --------------------------------------------------------------------------- #
@requires_ffmpeg
class DecodeAnomalyTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_missing_path_raises_filenotfound(self) -> None:
        missing = self.tmp / "nope.mp4"
        with self.assertRaises(FileNotFoundError):
            vd.probe_video(missing, FFMPEG)
        with self.assertRaises(FileNotFoundError):
            vd.sample_frames(missing, self.tmp / "o", 1.0, FFMPEG)

    def test_empty_file_is_rejected_readably(self) -> None:
        empty = self.tmp / "empty.mp4"
        empty.write_bytes(b"")
        with self.assertRaises(ValueError) as ctx:
            vd.probe_video(empty, FFMPEG)
        self.assertIn("无法解码", str(ctx.exception))
        with self.assertRaises(RuntimeError):
            vd.sample_frames(empty, self.tmp / "o", 1.0, FFMPEG)

    def test_audio_only_is_rejected_readably(self) -> None:
        audio = self.tmp / "audio.mp3"
        subprocess.run(
            [FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi",
             "-i", "sine=frequency=440:duration=3", str(audio)],
            check=True,
        )
        with self.assertRaises(ValueError) as ctx:
            vd.probe_video(audio, FFMPEG)
        self.assertIn("没有可解码的视频流", str(ctx.exception))

    def test_png_renamed_as_mp4_does_not_silently_succeed(self) -> None:
        """一张 PNG 改名成 .mp4：链路必须最终报错，不得静默返回空结果。"""
        png = self.tmp / "src.png"
        subprocess.run(
            [FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi",
             "-i", "testsrc=duration=1:size=64x64:rate=1", "-frames:v", "1", str(png)],
            check=True,
        )
        fake = self.tmp / "fake.mp4"
        shutil.copy(png, fake)
        # probe 可能把它当作 png 流「探测成功」，但抽帧阶段必须失败（非静默）。
        with self.assertRaises(RuntimeError):
            vd.sample_frames(fake, self.tmp / "o", 1.0, FFMPEG)

    def test_single_frame_video_fails_loudly(self) -> None:
        """单帧视频在 sample_fps=1.0 下抽不出帧：应给出可读 RuntimeError，而非空结果。"""
        single = self.tmp / "single.mp4"
        subprocess.run(
            [FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi",
             "-i", "testsrc=duration=2:size=160x120:rate=10", "-frames:v", "1",
             "-pix_fmt", "yuv420p", str(single)],
            check=True,
        )
        info = vd.probe_video(single, FFMPEG)
        self.assertEqual(info["nb_frames"], 1)
        with self.assertRaises(RuntimeError) as ctx:
            vd.sample_frames(single, self.tmp / "o", 1.0, FFMPEG)
        self.assertIn("未抽取到任何帧", str(ctx.exception))

    def test_very_short_video_yields_at_least_one_frame(self) -> None:
        short = _make_video(self.tmp / "short.mp4", seconds=0.5, fps=10, size="160x120")
        frames = vd.sample_frames(short, self.tmp / "o", 1.0, FFMPEG)
        self.assertGreaterEqual(len(frames), 1)


# --------------------------------------------------------------------------- #
# 3. KeyframeConfig 边界值
# --------------------------------------------------------------------------- #
class KeyframeBoundaryTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _frames(self, count: int, step: int = 37) -> list[Path]:
        frames = []
        for index in range(count):
            path = self.tmp / f"frame_{index:06d}_{index:09.3f}.pgm"
            _write_pgm(path, 16, 16, lambda x, y, o=index * step: (x * 8 + y * 3 + o) % 256)
            frames.append(path)
        return frames

    def test_max_keyframes_one(self) -> None:
        records = vd.select_keyframes(self._frames(6), _open_kf(max_keyframes=1), ImageStats(8))
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].frame_index, 0)

    def test_min_gap_larger_than_duration_keeps_only_first(self) -> None:
        frames = self._frames(6)  # ts = 0..5
        records = vd.select_keyframes(frames, _open_kf(min_gap_seconds=100.0), ImageStats(8))
        self.assertEqual([r.timestamp_s for r in records], [0.0])

    def test_motion_min_ratio_zero_keeps_all_and_upper_bound_drops(self) -> None:
        # 首帧全 50；次帧一半像素变 200（delta=150>25）→ motion_ratio≈0.5。
        base = bytearray([50] * 256)
        changed = bytearray(base)
        for index in range(128):
            changed[index] = 200
        first = self.tmp / "m_000000_00000.000.pgm"
        second = self.tmp / "m_000001_00001.000.pgm"
        first.write_bytes(b"P5\n16 16\n255\n" + bytes(base))
        second.write_bytes(b"P5\n16 16\n255\n" + bytes(changed))
        frames = [first, second]

        # 下界 0.0：任何运动都保留。
        kept_low = vd.select_keyframes(frames, _open_kf(motion_min_ratio=0.0), ImageStats(8))
        self.assertEqual([r.frame_index for r in kept_low], [0, 1])
        self.assertAlmostEqual(kept_low[-1].motion_ratio, 0.5, places=2)

        # 上界 1.0：motion(0.5) < 1.0 → 仅首帧保留。
        kept_high = vd.select_keyframes(frames, _open_kf(motion_min_ratio=1.0), ImageStats(8))
        self.assertEqual([r.frame_index for r in kept_high], [0])

    def test_phash_size_extremes(self) -> None:
        frames = self._frames(3)
        for size in (2, 64):
            records = vd.select_keyframes(frames, _open_kf(phash_size=size), ImageStats(size))
            self.assertEqual(len(records), 3)
            self.assertIsInstance(records[0].phash, int)

    def test_dedup_threshold_zero_does_not_disable(self) -> None:
        """澄清语义：``dedup_hamming_threshold=0`` **不能**关闭去重（仅 ``<0`` 能）。

        实现是「汉明距离 ≤ 阈值 → 判重丢弃」；相同帧距离=0，``0 <= 0`` 成立，仍被判重。
        """
        identical_a = _write_pgm(self.tmp / "d_000000_00000.000.pgm", 16, 16, lambda x, y: 10)
        identical_b = _write_pgm(self.tmp / "d_000001_00001.000.pgm", 16, 16, lambda x, y: 10)
        different = _write_pgm(self.tmp / "d_000002_00002.000.pgm", 16, 16, lambda x, y: 200)
        frames = [identical_a, identical_b, different]

        kept_at_zero = vd.select_keyframes(frames, _open_kf(dedup_hamming_threshold=0), ImageStats(8))
        self.assertEqual([r.frame_index for r in kept_at_zero], [0, 2])  # 相同帧仍被去重

        kept_at_negative = vd.select_keyframes(frames, _open_kf(dedup_hamming_threshold=-1), ImageStats(8))
        self.assertEqual([r.frame_index for r in kept_at_negative], [0, 1, 2])  # 真正关闭

    @requires_ffmpeg
    def test_static_video_dedups_to_single_frame(self) -> None:
        static = self.tmp / "static.mp4"
        # 静态视频：纯色源（testsrc 是动态的，无法表达「所有帧几乎相同」）。
        subprocess.run(
            [FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi",
             "-i", "color=c=gray:s=320x240:d=8:r=10", "-pix_fmt", "yuv420p", str(static)],
            check=True,
        )
        frames = vd.sample_frames(static, self.tmp / "sf", 1.0, FFMPEG)
        kf = KeyframeConfig(
            enable=True, motion_min_ratio=0.002, dedup_hamming_threshold=4, phash_size=8,
            scene_change_threshold=0.4, max_keyframes=400, min_gap_seconds=2.0,
        )
        records = vd.select_keyframes(frames, kf, ImageStats(8))
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].kept_reason, "first_frame")


# --------------------------------------------------------------------------- #
# 4. 帧序号 → 秒换算（ARCH 列为必验项）
# --------------------------------------------------------------------------- #
@requires_ffmpeg
class TimestampAlignmentTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.video = _make_video(self.tmp / "clip.mp4", seconds=8, fps=10)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _ffmpeg_pts(self, sample_fps: float) -> list[float]:
        result = subprocess.run(
            [FFMPEG, "-hide_banner", "-loglevel", "info", "-y", "-i", str(self.video),
             "-vf", f"fps={sample_fps},showinfo", "-f", "null", "-"],
            capture_output=True, text=True,
        )
        return [float(x) for x in re.findall(r"pts_time:([0-9.]+)", result.stderr)]

    def test_embedded_timestamp_matches_real_pts(self) -> None:
        pattern = re.compile(r"_(\d+\.\d{3})\.png$")
        for sample_fps in (0.5, 1.0, 2.0):
            frames = vd.sample_frames(self.video, self.tmp / f"f{sample_fps}", sample_fps, FFMPEG)
            embedded = [float(pattern.search(p.name).group(1)) for p in frames]
            self.assertEqual(embedded, [index / sample_fps for index in range(len(frames))])
            pts = self._ffmpeg_pts(sample_fps)
            self.assertEqual(len(embedded), len(pts))
            for got, want in zip(embedded, pts):
                self.assertAlmostEqual(got, want, places=6)

    def test_min_gap_uses_embedded_timestamp(self) -> None:
        frames = vd.sample_frames(self.video, self.tmp / "gap", 1.0, FFMPEG)  # ts 0..7
        records = vd.select_keyframes(frames, _open_kf(min_gap_seconds=3.0), ImageStats(8))
        self.assertEqual([r.timestamp_s for r in records], [0.0, 3.0, 6.0])


# --------------------------------------------------------------------------- #
# 5. 中文多层路径（含空格）
# --------------------------------------------------------------------------- #
@requires_ffmpeg
class ChinesePathTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_multi_level_chinese_paths_with_spaces(self) -> None:
        video = _make_video(self.tmp / "吊装" / "珠海 2026" / "吊装 视频 A.mp4", seconds=6)
        config = _make_config(
            self.tmp,
            sample_fps=1.0,
            keyframe={"max_keyframes": 100, "min_gap_seconds": 0.0,
                      "motion_min_ratio": 0.0, "scene_change_threshold": 1.1},
        )
        summary = run_cycle(config, "extract-video", video_paths=[video])

        self.assertEqual(summary["total_videos"], 1)
        self.assertGreater(summary["total_kept_frames"], 0)
        output_dir = config.input_frames_dir / "吊装 视频 A"
        self.assertTrue(output_dir.exists())
        self.assertEqual(
            len(list(output_dir.glob("*.png"))), summary["total_kept_frames"]
        )
        manifest = config.workspace("extracted", "keyframes", "吊装 视频 A.json")
        self.assertTrue(manifest.exists())


# --------------------------------------------------------------------------- #
# 6. 编排幂等性
# --------------------------------------------------------------------------- #
@requires_ffmpeg
class OrchestratorIdempotencyTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_same_params_rerun_does_not_accumulate(self) -> None:
        video = _make_video(self.tmp / "clip.mp4", seconds=6)
        config = _make_config(
            self.tmp,
            sample_fps=1.0,
            keyframe={"max_keyframes": 100, "min_gap_seconds": 0.0,
                      "motion_min_ratio": 0.0, "scene_change_threshold": 1.1},
        )
        first = run_cycle(config, "extract-video", video_paths=[video])
        second = run_cycle(config, "extract-video", video_paths=[video])
        self.assertEqual(first["total_kept_frames"], second["total_kept_frames"])
        output_dir = config.input_frames_dir / "clip"
        self.assertEqual(len(list(output_dir.glob("*.png"))), second["total_kept_frames"])

    def test_rerun_with_tighter_threshold_leaves_no_stale_keyframes(self) -> None:
        """BUG 复现：收紧阈值重跑后，``input_frames_dir`` 不得残留上一轮的关键帧。

        预期（正确行为）：输出目录内容 == 本轮 keyframes.json 清单。
        实测（当前实现）：输出目录不清理 → 陈旧 PNG 残留，清单与实际文件不一致，
        下游 ``run_extraction``（``iter_image_files`` 递归读目录）会把陈旧帧一并送检。
        """
        video = _make_video(self.tmp / "clip.mp4", seconds=8)
        config = _make_config(
            self.tmp,
            sample_fps=1.0,
            keyframe={"max_keyframes": 100, "min_gap_seconds": 0.0,
                      "motion_min_ratio": 0.0, "scene_change_threshold": 1.1},
        )
        run_cycle(config, "extract-video", video_paths=[video])  # 第一轮：宽松

        config.keyframe.min_gap_seconds = 4.0  # 第二轮：收紧，保留数变少
        second = run_cycle(config, "extract-video", video_paths=[video])

        output_dir = config.input_frames_dir / "clip"
        on_disk = {p.name for p in output_dir.glob("*.png")}
        manifest = read_json(config.workspace("extracted", "keyframes", "clip.json"))
        listed = {Path(item["extracted_path"]).name for item in manifest["keyframes"]}

        self.assertEqual(second["total_kept_frames"], len(listed))
        self.assertEqual(
            on_disk, listed,
            f"输出目录残留了未在本轮清单中的陈旧关键帧：{sorted(on_disk - listed)}",
        )

    def test_media_root_missing_raises_readable_error(self) -> None:
        config = _make_config(self.tmp, ffmpeg_binary=FFMPEG or "ffmpeg")
        with self.assertRaises(ValueError) as ctx:
            run_cycle(config, "extract-video")  # 无 video_paths，默认媒体根不存在
        self.assertIn("未找到可解码的视频素材", str(ctx.exception))


# --------------------------------------------------------------------------- #
# 7. CLI 退出码
# --------------------------------------------------------------------------- #
class CliExitCodeTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _run_cli(self, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-m", "industrial_hazard_coldstart", *args],
            cwd=str(ROOT), capture_output=True, text=True,
        )

    def test_pending_command_returns_2(self) -> None:
        result = self._run_cli("--config", str(HOIST_CONFIG), "train")
        self.assertEqual(result.returncode, 2)

    def test_missing_config_returns_3(self) -> None:
        result = self._run_cli("--config", str(self.tmp / "nope.json"), "inventory")
        self.assertEqual(result.returncode, 3)

    def test_malformed_config_returns_3(self) -> None:
        bad = self.tmp / "bad.json"
        bad.write_text("{not json", encoding="utf-8")
        result = self._run_cli("--config", str(bad), "inventory")
        self.assertEqual(result.returncode, 3)

    def test_help_returns_0(self) -> None:
        result = self._run_cli("--help")
        self.assertEqual(result.returncode, 0)

    @requires_ffmpeg
    def test_inventory_returns_0(self) -> None:
        config = _make_config(self.tmp)
        result = self._run_cli("--config", str(config.config_path), "inventory")
        self.assertEqual(result.returncode, 0)

    @requires_ffmpeg
    def test_missing_ffmpeg_returns_4(self) -> None:
        config = _make_config(self.tmp, ffmpeg_binary=r"D:\no\such\ffmpeg.exe")
        # 默认媒体根需存在且含视频，才能越过「未找到视频」检查、直达 ffmpeg 缺失分支。
        media = config.input_frames_dir.parent / "media"
        media.mkdir(parents=True, exist_ok=True)
        (media / "clip.mp4").write_bytes(b"placeholder-not-decoded")
        result = self._run_cli("--config", str(config.config_path), "extract-video")
        self.assertEqual(result.returncode, 4)
        self.assertIn("ffmpeg", result.stderr)


if __name__ == "__main__":
    unittest.main()
