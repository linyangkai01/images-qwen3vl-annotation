"""视频解码与关键帧筛选（功能点 F2）—— 本项目第一个真实缺口。

母项目 ``pipeline/extraction.py`` 只读图片目录、**从不调用 ffmpeg**；
``VideoConfig`` / ``KeyframeConfig`` 虽被加载校验，却**没有任何代码消费它们**。
本模块从零补齐这条链路：

1. :func:`probe_video` —— 用 ``ffprobe`` 读容器头（时长 / 分辨率 / 帧率 / 帧数）；
2. :func:`sample_frames` —— 用 ``ffmpeg`` 按 ``sample_fps`` 抽帧；
3. :func:`select_keyframes` —— 复用母项目 ``imageio`` 的 aHash / 汉明距离 /
   直方图差 / 运动比，做关键帧筛选（去重 + 运动门 + 场景切换 + 时间间隔 + 上限）；
4. :func:`run_video_extract` —— 把保留的关键帧写入
   ``config.input_frames_dir/<source_id>/``，供母项目 ``run_extraction`` 消费。

设计要点（与硬约束一致）：

- **中文路径全链路支持**：``subprocess`` 以参数列表（非 shell）传路径，Windows 下
  经 ``CreateProcessW`` 传 Unicode，输入 / 输出路径含中文均可用；
- **ffmpeg 缺失必须显式降级报错**（:class:`FfmpegMissingError`），**绝不静默产出空结果**；
- **抽帧 / 解码失败给出可读错误**（含 ffmpeg 返回码与 stderr），不吞异常；
- 抽帧产物为 **PNG**（可视化 / 交付用）；因母项目 ``imageio`` 只能解码 Netpbm，
  同一次 ffmpeg 调用**并行产出灰度 PGM**（分析帧），``select_keyframes`` 用它做去重分析；
- 仅标准库 + 母项目 + 自身，零新增第三方依赖（ffmpeg 是外部二进制，非 Python 依赖）。
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from traffic_cv_closed_loop import imageio as _imageio
from traffic_cv_closed_loop.config import KeyframeConfig, ProjectConfig
from traffic_cv_closed_loop.imageio import ImageStats, load_image
from traffic_cv_closed_loop.io_utils import ensure_dir, read_json, stem_sidecar, write_json

from industrial_hazard_coldstart.inventory import (
    VIDEO_SUFFIXES,
    default_media_root,
    is_ignorable_media_path,
)
from industrial_hazard_coldstart.models import KeyframeRecord

__all__ = [
    "FfmpegMissingError",
    "VIDEO_SUFFIXES",
    "probe_video",
    "sample_frames",
    "select_keyframes",
    "discover_videos",
    "run_video_extract",
]

#: ffmpeg / ffprobe 探测的短超时（秒）。
_FFPROBE_TIMEOUT_S = 30
#: 抽帧超时（秒）；小样本足够，避免异常视频卡死。
_SAMPLE_TIMEOUT_S = 600

#: 时间戳从帧文件名尾部解析（形如 ``frame_000003_00003.000.png``）。
_TIMESTAMP_RE = re.compile(r"_(\d+\.\d{3})$")

_MISSING_FFMPEG_MSG = (
    "未找到 ffmpeg 可执行文件：{binary!r}。\n"
    "视频解码与抽帧依赖外部 ffmpeg / ffprobe（不属于 Python 依赖）。\n"
    "请在配置 `config.video.ffmpeg_binary` 指定绝对路径"
    "（本机实测：D:\\WorkPlace\\tools\\ffmpeg\\bin\\ffmpeg.exe），或将其加入 PATH。\n"
    "注意：本阶段不会静默产出空结果，而是显式报错。"
)


class FfmpegMissingError(RuntimeError):
    """ffmpeg / ffprobe 不可用时抛出的明确降级错误。"""


# --------------------------------------------------------------------------- #
# 二进制解析与子进程工具
# --------------------------------------------------------------------------- #
def _decode(payload: bytes | str | None) -> str:
    """把子进程输出解码为可读文本（utf-8 → gbk → 兜底 replace）。"""
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


def _signed_returncode(code: int) -> int:
    """把子进程退出码还原为**有符号**整数。

    Windows 上被信号终止的进程退出码是 32 位无符号（如 ``-22`` 显示为
    ``4294967274``、``-9`` 显示为 ``3199971767``），直接拼进错误信息会**难以排障**。
    这里把 ``> 0x7FFFFFFF`` 的值按二补数还原回负值，其余原样返回。
    """
    if code > 0x7FFFFFFF:
        return code - 0x100000000
    return code


def _resolve_executable(binary: str) -> str | None:
    """按 ``PATH`` 查找，再回退为显式路径；找不到返回 ``None``。"""
    if not binary:
        return None
    found = shutil.which(binary)
    if found:
        return found
    candidate = Path(binary)
    if candidate.exists():
        return str(candidate)
    return None


def _ffprobe_binary(ffmpeg_binary: str) -> str:
    """从 ffmpeg 路径推导 ffprobe（同目录同名，扩展名保留）。"""
    name = Path(ffmpeg_binary).name
    if name.lower().startswith("ffmpeg"):
        probe_name = "ffprobe" + Path(name).suffix
        sibling = Path(ffmpeg_binary).with_name(probe_name)
        if sibling.exists():
            return str(sibling)
        return "ffprobe"
    return "ffprobe"


def _require(binary: str) -> str:
    """解析出可用的 ffmpeg 路径，否则抛 :class:`FfmpegMissingError`。"""
    resolved = _resolve_executable(binary)
    if resolved is None:
        raise FfmpegMissingError(_MISSING_FFMPEG_MSG.format(binary=binary))
    return resolved


def _parse_fraction(raw: Any) -> float:
    """把 ffprobe 的 ``"30000/1001"`` 之类的帧率串解析为浮点。"""
    text = str(raw or "").strip()
    if not text:
        return 0.0
    if "/" in text:
        numerator, _, denominator = text.partition("/")
        try:
            denominator_value = float(denominator)
            if denominator_value == 0:
                return 0.0
            return float(numerator) / denominator_value
        except ValueError:
            return 0.0
    try:
        return float(text)
    except ValueError:
        return 0.0


def _first_float(*values: Any) -> float:
    for value in values:
        if value is None or value == "":
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return 0.0


# --------------------------------------------------------------------------- #
# 1. 容器头探测
# --------------------------------------------------------------------------- #
def probe_video(path: Path | str, ffmpeg_binary: str) -> dict[str, Any]:
    """用 ``ffprobe`` 读取视频容器头信息。

    Args:
        path: 视频路径（可含中文）。
        ffmpeg_binary: ffmpeg 路径（用于推导同目录 ffprobe）。

    Returns:
        含 ``duration_s / width / height / fps / nb_frames / codec / format_name``
        的字典。

    Raises:
        FileNotFoundError: 文件不存在。
        FfmpegMissingError: ffprobe / ffmpeg 不可用。
        ValueError: 文件无法解码 / 不含视频流 / **无可解码的视频时长**
            （图片冒充视频、损坏文件；附 ffprobe 返回码与 stderr）。

    Note:
        **有效性判定以 ``duration_s`` 为主判据**：正常的视频文件一定有
        ``duration > 0``；而一张 PNG 改名成 ``.mp4`` 时，ffprobe 会把它当作
        ``png_pipe`` 视频流「探测成功」，但**时长缺失（``duration = 0``）**——
        若放行，``inventory`` 会登记出 ``fps=25、count=0`` 的**假视频**。
        故 ``duration_s <= 0`` 一律判为「非可解码视频」并抛 :class:`ValueError`。
        ``nb_frames`` 仅作辅证（容器未报帧数的合法视频仍以时长为准，不被误伤）。
    """
    video = Path(path)
    if not video.exists():
        raise FileNotFoundError(f"视频文件不存在：{video}")

    probe = _ffprobe_binary(ffmpeg_binary)
    resolved = _resolve_executable(probe)
    if resolved is None:
        # ffprobe 缺失：若 ffmpeg 本身也缺失，给出同一条明确降级信息。
        raise FfmpegMissingError(_MISSING_FFMPEG_MSG.format(binary=ffmpeg_binary))

    command = [
        resolved,
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(video),
    ]
    completed = subprocess.run(command, capture_output=True, timeout=_FFPROBE_TIMEOUT_S)
    stderr_text = _decode(completed.stderr).strip()
    if completed.returncode != 0:
        raise ValueError(
            f"无法解码媒体文件（ffprobe 返回码 {_signed_returncode(completed.returncode)}）："
            f"{video}\n"
            f"ffprobe 输出：{stderr_text or '(空)'}"
        )

    try:
        payload = json.loads(_decode(completed.stdout) or "{}")
    except json.JSONDecodeError as error:
        raise ValueError(f"ffprobe 输出不是合法 JSON：{video}（{error}）") from error

    streams = payload.get("streams") or []
    video_stream = next(
        (item for item in streams if item.get("codec_type") == "video"), None
    )
    if video_stream is None:
        raise ValueError(f"文件中没有可解码的视频流：{video}")

    fmt = payload.get("format") or {}
    fps = _parse_fraction(video_stream.get("avg_frame_rate") or video_stream.get("r_frame_rate"))
    duration_s = _first_float(fmt.get("duration"), video_stream.get("duration"))
    nb_frames = int(_first_float(video_stream.get("nb_frames")))
    if nb_frames <= 0 and duration_s > 0 and fps > 0:
        nb_frames = int(round(duration_s * fps))

    codec = str(video_stream.get("codec_name") or "")
    format_name = str(fmt.get("format_name") or "")
    if duration_s <= 0:
        # 图片冒充视频 / 损坏文件：容器无有效时长 → 明确拒绝（绝不登记为假视频）。
        raise ValueError(
            f"文件没有可解码的视频时长（duration_s={duration_s}）：{video}\n"
            "该文件可能是图片、音频或已损坏，而非可解码的视频。\n"
            f"（codec={codec or '(无)'}，format={format_name or '(无)'}，"
            f"nb_frames={nb_frames}）"
        )

    return {
        "path": str(video),
        "duration_s": duration_s,
        "width": int(_first_float(video_stream.get("width"))),
        "height": int(_first_float(video_stream.get("height"))),
        "fps": fps,
        "nb_frames": nb_frames,
        "codec": codec,
        "format_name": format_name,
        "size_bytes": int(_first_float(fmt.get("size"))),
    }


# --------------------------------------------------------------------------- #
# 2. 抽帧
# --------------------------------------------------------------------------- #
def _build_sample_command(
    resolved: str,
    video: Path,
    png_pattern: Path,
    pgm_pattern: Path,
    sample_fps: float,
    scale_width: int,
    analysis_width: int,
) -> list[str]:
    """构造单次 ffmpeg 调用：同时产出 PNG（交付）与灰度 PGM（分析）。"""
    png_filter = f"fps={sample_fps}"
    if scale_width > 0:
        png_filter += f",scale={scale_width}:-2"
    pgm_filter = f"fps={sample_fps}"
    if analysis_width > 0:
        pgm_filter += f",scale={analysis_width}:-2"

    return [
        resolved,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video),
        "-map",
        "0:v:0",
        "-vf",
        png_filter,
        str(png_pattern),
        "-map",
        "0:v:0",
        "-vf",
        pgm_filter,
        "-pix_fmt",
        "gray",
        str(pgm_pattern),
    ]


def sample_frames(
    video: Path | str,
    out_dir: Path | str,
    sample_fps: float,
    ffmpeg_binary: str,
    scale_width: int = 0,
    analysis_width: int = 0,
) -> list[Path]:
    """用 ffmpeg 按 ``sample_fps`` 抽帧。

    每帧同时产出：

    - ``frame_<i>_<ts>.png`` —— 交付 / 可视化用的**关键帧交付物**（全分辨率）；
    - ``frame_<i>_<ts>.pgm`` —— 灰度**分析中间产物**（供母项目 ``imageio`` 去重）。

    ⚠️ **PGM 是中间产物，不是交付物**：母项目 ``imageio`` 只能解码 Netpbm，
    故去重分析依赖 PGM；保留它可让「调整去重阈值」时**不必重新解码视频**。
    若磁盘紧张，可安全删除 ``*.pgm``（删除后仅需重新抽帧才能再次分析）。

    文件名尾部内嵌时间戳（``<i>/sample_fps`` 秒），使 :func:`select_keyframes`
    能在不额外传参的情况下实现 ``min_gap_seconds`` 过滤。

    Args:
        video: 视频路径（可含中文）。
        out_dir: 抽帧输出目录（可含中文）。
        sample_fps: 采样帧率（> 0）。
        ffmpeg_binary: ffmpeg 路径。
        scale_width: PNG 输出宽度；``<= 0`` 表示不缩放。
        analysis_width: PGM 分析帧宽度；``<= 0`` 表示不缩放。

    Returns:
        按帧序排序的 PNG 路径列表。

    Raises:
        FfmpegMissingError: ffmpeg 不可用。
        ValueError: ``sample_fps`` 非正。
        RuntimeError: ffmpeg 抽帧失败或未产出任何帧（附可读错误）。
    """
    video_path = Path(video)
    out_path = Path(out_dir)
    if sample_fps <= 0:
        raise ValueError(f"sample_fps 必须 > 0，实际为 {sample_fps}")
    if not video_path.exists():
        raise FileNotFoundError(f"视频文件不存在：{video_path}")

    resolved = _require(ffmpeg_binary)
    ensure_dir(out_path)

    # 清理上一轮遗留的原始帧，避免新旧混在一起。
    for stale in out_path.glob("frame_*"):
        if stale.is_file():
            stale.unlink()

    png_pattern = out_path / "frame_%06d.png"
    pgm_pattern = out_path / "frame_%06d.pgm"
    command = _build_sample_command(
        resolved, video_path, png_pattern, pgm_pattern, sample_fps, scale_width, analysis_width
    )
    completed = subprocess.run(command, capture_output=True, timeout=_SAMPLE_TIMEOUT_S)
    stderr_text = _decode(completed.stderr).strip()
    if completed.returncode != 0:
        raise RuntimeError(
            f"ffmpeg 抽帧失败（返回码 {_signed_returncode(completed.returncode)}）："
            f"{video_path}\n"
            f"命令：{' '.join(command)}\n"
            f"ffmpeg 输出：{stderr_text or '(空)'}"
        )

    raw_pngs = sorted(out_path.glob("frame_*.png"))
    if not raw_pngs:
        raise RuntimeError(
            f"ffmpeg 未抽取到任何帧：{video_path}（sample_fps={sample_fps}）。"
            "请确认视频可解码且时长足够。"
        )

    # 重命名为内嵌时间戳的稳定名字（index 从 0 起，ts = index / sample_fps）。
    results: list[Path] = []
    for index, png in enumerate(raw_pngs):
        old_pgm = png.with_suffix(".pgm")
        timestamp = index / sample_fps
        stem = f"frame_{index:06d}_{timestamp:09.3f}"
        new_png = out_path / f"{stem}.png"
        new_pgm = out_path / f"{stem}.pgm"
        if png != new_png:
            png.replace(new_png)
        if old_pgm.exists():
            old_pgm.replace(new_pgm)
        results.append(new_png)
    return results


# --------------------------------------------------------------------------- #
# 3. 关键帧筛选
# --------------------------------------------------------------------------- #
def _analysis_path(path: Path) -> Path:
    """返回用于 ``imageio`` 解码的分析帧路径。

    ``imageio`` 仅支持 Netpbm；若传入的是 PNG，则回退到同名的 ``.pgm`` 分析帧。
    """
    if path.suffix.lower() in (".pgm", ".ppm", ".pnm", ".pbm"):
        return path
    sibling = path.with_suffix(".pgm")
    return sibling if sibling.exists() else path


def _load_frame(path: Path):
    """加载一帧（自动回退到同名 PGM 分析帧）；失败给出可读错误。"""
    target = _analysis_path(path)
    try:
        return load_image(target)
    except (ValueError, OSError) as error:
        raise ValueError(
            "无法解析帧图像（母项目 imageio 仅支持 Netpbm；PNG 需同目录同名 .pgm 分析帧）："
            f"{path} -> {target}（{error}）"
        ) from error


def _timestamp_for(path: Path, index: int) -> float:
    """从文件名尾部解析时间戳；解析不出时退化为帧序号（按 1 秒/帧）。"""
    match = _TIMESTAMP_RE.search(path.stem)
    if match:
        return float(match.group(1))
    return float(index)


def _source_id_of(path: Path) -> str:
    """关键帧的 ``source_id``：取所在目录名（抽帧目录按 ``<source_id>`` 命名）。"""
    return path.parent.name


def _frame_labels(path: Path) -> list[str]:
    """读取帧的可选标签侧车 ``<stem>.labels.json``（供 ``force_keep_labels`` 使用）。"""
    payload = read_json(stem_sidecar(path, "labels"), default=None)
    if payload is None:
        return []
    if isinstance(payload, list):
        return [str(item) for item in payload]
    if isinstance(payload, dict):
        raw = payload.get("labels", [])
        if isinstance(raw, list):
            return [str(item) for item in raw]
    return []


def select_keyframes(
    frames: list[Path],
    kf: KeyframeConfig,
    stats: ImageStats | None = None,
) -> list[KeyframeRecord]:
    """从抽出的帧中筛选关键帧。

    真正消费 ``KeyframeConfig`` 的**全部字段**：

    - ``enable``：关闭时直接保留全部帧（不做任何筛选）；
    - ``phash_size``：aHash 的降采样网格尺寸（构造 ``ImageStats``）；
    - ``dedup_hamming_threshold``：与**上一个保留帧**的 aHash 汉明距离 ≤ 阈值 → 判重丢弃。
      判定式为「汉明距离 ≤ 阈值」，据此三种取值语义如下：

      * ``> 0``：距离 ≤ 阈值的近似帧判重（常规去重）；
      * ``= 0``：**仅「完全相同的 aHash」判重**（相同帧距离 = 0，``0 <= 0`` 成立 → 仍去重）；
      * ``< 0``：**完全关闭去重**（距离恒 ≥ 0，``≥ 0`` 不可能 ≤ 负数 → 永不判重）。

      ⚠️ 澄清口径：``= 0`` **不是**「关闭去重」，真正关闭需要 ``< 0``。相同帧本就应被去重，
      故保持此行为（仅澄清文档，不改实现）。

    - ``motion_min_ratio`` + ``pixel_delta_threshold``：与**相邻帧**的运动比低于阈值 → 判静丢弃；
    - ``scene_change_threshold``：相邻帧直方图差 ≥ 阈值 → 场景切换，**强制保留**；
    - ``min_gap_seconds``：与上一个保留帧的时间间隔不足 → 丢弃（场景切换 / 强制保留可越过）；
    - ``max_keyframes``：保留数达到上限即停止；
    - ``force_keep_labels``：帧侧车标签命中这些标签 → **强制保留**。

    ⚠️ **``force_keep_labels`` 是预置钩子**：本阶段（抽帧→筛选）尚无检测结果，
    标签来源是帧侧车 ``<stem>.labels.json``。**真实链路当前不会生成该侧车**——
    它是为 T03（先初检、再按标签回补）预留的接口，当前只在单测里被触发。
    请勿假设线上存在 ``.labels.json`` 文件。

    Args:
        frames: 抽帧路径列表（PNG，或直接是 Netpbm）。
        kf: 关键帧配置。
        stats: 图像统计器；缺省按 ``kf.phash_size`` 构造。

    Returns:
        保留的 :class:`KeyframeRecord` 列表（按帧序）。
    """
    resolved_stats = stats if stats is not None else ImageStats(kf.phash_size)
    records: list[KeyframeRecord] = []

    if not kf.enable:
        # 关闭筛选：原样保留全部帧，仅登记哈希，便于追溯。
        for index, path in enumerate(frames):
            image = _load_frame(path)
            records.append(
                KeyframeRecord(
                    source_id=_source_id_of(path),
                    frame_index=index,
                    timestamp_s=_timestamp_for(path, index),
                    extracted_path=str(path),
                    phash=resolved_stats.ahash(image),
                    motion_ratio=0.0,
                    kept_reason="keyframe_disabled",
                )
            )
        return records

    forced_labels = set(kf.force_keep_labels)
    prev_gray: bytes | None = None
    prev_hist: list[float] | None = None
    last_kept_hash: int | None = None
    last_kept_ts: float | None = None

    for index, path in enumerate(frames):
        image = _load_frame(path)
        gray = image.to_gray_bytes()
        timestamp = _timestamp_for(path, index)
        phash = resolved_stats.ahash(image)
        histogram = resolved_stats.histogram16(image)

        motion = (
            0.0
            if prev_gray is None
            else _imageio.motion_ratio(prev_gray, gray, kf.pixel_delta_threshold)
        )
        scene_delta = (
            0.0 if prev_hist is None else resolved_stats.histogram_delta(prev_hist, histogram)
        )
        prev_gray, prev_hist = gray, histogram

        matched = [label for label in _frame_labels(path) if label in forced_labels]

        if matched:
            reason = "force_keep_label:" + ",".join(matched)
        elif index == 0:
            reason = "first_frame"
        elif scene_delta >= kf.scene_change_threshold:
            reason = f"scene_change:{scene_delta:.3f}"
        else:
            if (
                last_kept_hash is not None
                and resolved_stats.hamming(phash, last_kept_hash) <= kf.dedup_hamming_threshold
            ):
                continue  # 与上一个保留帧近似重复 → 去重
            if motion < kf.motion_min_ratio:
                continue  # 画面几乎静止 → 丢弃
            if last_kept_ts is not None and (timestamp - last_kept_ts) < kf.min_gap_seconds:
                continue  # 距上一个保留帧太近 → 丢弃
            reason = f"motion:{motion:.4f}"

        records.append(
            KeyframeRecord(
                source_id=_source_id_of(path),
                frame_index=index,
                timestamp_s=timestamp,
                extracted_path=str(path),
                phash=phash,
                motion_ratio=motion,
                kept_reason=reason,
            )
        )
        last_kept_hash = phash
        last_kept_ts = timestamp
        if len(records) >= kf.max_keyframes:
            break

    return records


# --------------------------------------------------------------------------- #
# 4. 编排：解码 → 筛选 → 落盘到 input_frames_dir
# --------------------------------------------------------------------------- #
def discover_videos(
    config: ProjectConfig,
    video_paths: list[Path] | None = None,
) -> list[Path]:
    """确定待处理的视频列表。

    优先级：显式 ``video_paths`` > ``config.video.video_path`` > 扫描默认媒体根。
    扫描默认媒体根时**跳过**隐藏文件 / 隐藏目录 / macOS 元数据（``._*`` /
    ``__MACOSX``）——见 :func:`inventory.is_ignorable_media_path`；显式传入的路径
    视为调用方有意为之，不做跳过。
    """
    if video_paths:
        return [Path(item) for item in video_paths]
    if config.video.video_path:
        return [Path(config.video.video_path)]
    root = default_media_root(config)
    if not root.exists():
        return []
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in VIDEO_SUFFIXES
        # 输入卫生：跳过隐藏文件 / 隐藏目录 / macOS 元数据（._* / __MACOSX）。
        and not is_ignorable_media_path(path, root)
    )


def _extract_one_video(
    config: ProjectConfig,
    video: Path,
    stats: ImageStats,
) -> dict[str, Any]:
    """处理**单个**视频：探测 → 抽帧 → 筛选 → 落盘到 ``input_frames_dir``。

    该函数对单个视频是 **fail-loud** 的（失败即抛异常）；批量容错由
    :func:`run_video_extract` 负责（记录失败视频并继续处理下一个）。

    **幂等性（关键）**：写关键帧前会**清空该 source 子目录内上一轮遗留的
    ``frame_*``**（与 :func:`sample_frames` 的清理口径一致）。否则收紧阈值重跑时，
    上一轮的关键帧会残留在 ``input_frames_dir/<source_id>/``，与
    ``keyframes/<source_id>.json`` 清单**不一致**；而下游 ``run_extraction`` 用
    ``io_utils.iter_image_files()`` **递归读整个目录**，会把陈旧帧一并送检 →
    VLM 调用量虚增、标注对象与清单不符。**只清该 source 自己的子目录**，绝不误删其他 source。

    Returns:
        该视频的摘要字典（含 ``sampled_frames`` / ``kept_frames`` / ``output_dir``）。
    """
    video_cfg = config.video
    source_id = video.stem
    probe = probe_video(video, video_cfg.ffmpeg_binary)
    frames_dir = config.workspace("extracted", "video_frames", source_id)
    sampled = sample_frames(
        video,
        frames_dir,
        video_cfg.sample_fps,
        video_cfg.ffmpeg_binary,
        scale_width=video_cfg.scale_width,
        analysis_width=video_cfg.analysis_width,
    )
    kept = select_keyframes(sampled, config.keyframe, stats)

    output_dir = config.input_frames_dir / source_id
    ensure_dir(output_dir)
    # 清理该 source 子目录中上一轮遗留的关键帧，避免陈旧帧残留（见函数 docstring）。
    for stale in output_dir.glob("frame_*"):
        if stale.is_file():
            stale.unlink()

    kept_records: list[KeyframeRecord] = []
    for record in kept:
        source_frame = Path(record.extracted_path)
        destination = output_dir / source_frame.name
        shutil.copy2(source_frame, destination)
        record.source_id = source_id
        record.extracted_path = str(destination)
        kept_records.append(record)

    write_json(
        config.workspace("extracted", "keyframes", f"{source_id}.json"),
        {
            "source_id": source_id,
            "video": str(video),
            "probe": probe,
            "sampled_frames": len(sampled),
            "kept_frames": len(kept_records),
            "keyframes": [item.to_dict() for item in kept_records],
        },
    )

    return {
        "source_id": source_id,
        "video": str(video),
        "duration_s": probe["duration_s"],
        "width": probe["width"],
        "height": probe["height"],
        "fps": probe["fps"],
        "sampled_frames": len(sampled),
        "kept_frames": len(kept_records),
        "output_dir": str(output_dir),
    }


def run_video_extract(
    config: ProjectConfig,
    video_paths: list[Path] | None = None,
) -> dict[str, Any]:
    """执行「视频解码 + 关键帧筛选」，把保留的关键帧写入 ``input_frames_dir``。

    产出：
    - ``config.input_frames_dir/<source_id>/*.png`` —— 保留的关键帧（**重跑前先清空**）；
    - ``workspace/extracted/video_frames/<source_id>/`` —— 全部抽样帧（含分析 PGM）；
    - ``workspace/extracted/keyframes/<source_id>.json`` —— 每段视频的关键帧清单；
    - ``workspace/status/extract_video.json`` —— 阶段摘要。

    **批量容错（关键，但绝不静默）**：单个视频失败（不可解码 / 抽不出帧 / 探测失败）时，
    **记录该视频与失败原因后继续处理下一个**，避免「一个 0.1 秒的坏视频拖垮整批」。
    失败信息通过摘要显式暴露：
    - ``failed_videos``：``[{source_id, video, error_type, reason}, ...]``；
    - ``has_failures``：是否有失败（供调用方 / CLI 判定退出码）。

    ⚠️ **容错 ≠ 静默**：「跳过并记录」与「悄悄跳过」的区别，就是有没有让调用方看见——
    故本函数**永远**在摘要里回报失败清单，绝不隐藏。

    Args:
        config: 项目配置（消费 ``video`` 与 ``keyframe`` 两段全部字段）。
        video_paths: 显式视频列表；缺省按 :func:`discover_videos` 推导。

    Returns:
        阶段摘要字典（含 ``failed_videos`` / ``has_failures``）。

    Raises:
        ValueError: ``decode_backend`` 非 ffmpeg、或未找到任何视频（**硬错误**）。
        FfmpegMissingError: ffmpeg / ffprobe 不可用（明确降级，不静默）。
    """
    video_cfg = config.video
    if video_cfg.decode_backend.lower() != "ffmpeg":
        raise ValueError(
            f"不支持的 decode_backend：{video_cfg.decode_backend!r}（本链路仅支持 ffmpeg）"
        )

    videos = discover_videos(config, video_paths)
    if not videos:
        raise ValueError(
            "未找到可解码的视频素材："
            f"video_path={video_cfg.video_path!r}，默认媒体根={default_media_root(config)}。"
            "请通过 video_paths 显式指定视频。"
        )

    # 尽早校验 ffmpeg 可用性（缺失即显式报错，绝不产出空结果）。
    _require(video_cfg.ffmpeg_binary)

    ensure_dir(config.input_frames_dir)
    stats = ImageStats(config.keyframe.phash_size)

    source_summaries: list[dict[str, Any]] = []
    failed_videos: list[dict[str, Any]] = []
    total_sampled = 0
    total_kept = 0

    for video in videos:
        try:
            source_summary = _extract_one_video(config, video, stats)
        except FfmpegMissingError:
            # ffmpeg 在批量中途消失 → 环境级故障，不可静默跳过，直接上抛。
            raise
        except Exception as error:  # noqa: BLE001 - 单视频失败 → 记录并继续，绝不静默
            failed_videos.append(
                {
                    "source_id": video.stem,
                    "video": str(video),
                    "error_type": type(error).__name__,
                    "reason": str(error),
                }
            )
            continue
        total_sampled += source_summary["sampled_frames"]
        total_kept += source_summary["kept_frames"]
        source_summaries.append(source_summary)

    summary: dict[str, Any] = {
        "stage": "extract-video",
        "input_frames_dir": str(config.input_frames_dir),
        "sample_fps": video_cfg.sample_fps,
        "videos": source_summaries,
        "total_videos": len(videos),
        "succeeded_videos": len(source_summaries),
        "total_sampled_frames": total_sampled,
        "total_kept_frames": total_kept,
        "failed_videos": failed_videos,
        "has_failures": bool(failed_videos),
    }
    write_json(config.workspace("status", "extract_video.json"), summary)
    return summary
