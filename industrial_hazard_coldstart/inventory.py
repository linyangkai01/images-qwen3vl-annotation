"""素材编目（功能点 F1）。

只做**元数据登记**，不解码画面内容：

- 视频：用 ``ffprobe`` 读容器头（时长 / 分辨率 / 帧率 / 帧数），不解码；
- 照片：用轻量头部解析（PNG IHDR / JPEG SOF）或母项目 ``imageio``（Netpbm）
  取分辨率，不解码整幅画面；
- **只登记、不产出**任何派生素材。

产出 ``workspace/inventory/manifest.json``，字段与需求 §F1 对齐：
``source_id / path / scene / media_type / duration_s / width / height / fps / count``。

约定：
- **中文路径全链路支持**（素材本身是中文命名）；
- ffprobe 缺失或读不到时**降级**为 0，但仍登记该条（编目不因缺 ffmpeg 而失败）；
- 仅标准库 + 母项目 + 自身，零新增第三方依赖。

目录约定：媒体素材根目录默认取 ``config.input_frames_dir`` 的同级 ``media/``
（即 ``input/media``）；可用 ``run_inventory(..., media_root=...)`` 显式覆盖，
这样测试可注入临时目录，**绝不触碰真实私有素材**。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from traffic_cv_closed_loop.config import ProjectConfig
from traffic_cv_closed_loop.imageio import load_image
from traffic_cv_closed_loop.io_utils import (
    SUPPORTED_IMAGE_SUFFIXES,
    ensure_dir,
    write_json,
)

from industrial_hazard_coldstart.models import MediaAsset

__all__ = [
    "VIDEO_SUFFIXES",
    "HIDDEN_METADATA_DIRNAME",
    "MAX_SKIPPED_SAMPLES",
    "SKIP_REASON_HIDDEN",
    "SKIP_REASON_UNSUPPORTED",
    "SKIP_REASON_NOT_MEDIA",
    "default_media_root",
    "is_ignorable_media_path",
    "scan_media",
    "scan_media_detailed",
    "run_inventory",
]

#: 视为「视频」的扩展名（编目分流用；与 imageio 支持的照片扩展名互斥）。
#:
#: **必须覆盖真实素材里出现的容器**——漏一个就会**静默漏掉整段素材**（R2 缺陷：
#: 珠海目录的 ``.qt`` 真视频占总时长 52.7% 却被漏搜）。故这里覆盖主流容器：
#: MP4/MOV/QuickTime、AVI、MKV/WebM、WMV/ASF、FLV/F4V、MPEG 家族、传输流、
#: MXF/VOB/DV、RealMedia、Ogg、3GP/3G2 等。
VIDEO_SUFFIXES: frozenset[str] = frozenset(
    {
        ".mp4",
        ".m4v",
        ".mov",
        ".qt",          # QuickTime（Mac 常见；R2 补入）
        ".avi",
        ".mkv",
        ".flv",
        ".f4v",         # Flash MP4
        ".wmv",
        ".asf",         # Advanced Systems Format（WMV 容器）
        ".webm",
        ".ogv",         # Ogg video
        ".mpg",
        ".mpeg",
        ".m2v",         # MPEG-2 视频基本流
        ".ts",
        ".mts",
        ".m2ts",
        ".3gp",
        ".3g2",         # 3GPP2
        ".mxf",         # Material Exchange Format（专业视频）
        ".vob",         # DVD Video Object
        ".dv",          # DV
        ".rm",          # RealMedia
        ".rmvb",        # RealMedia 可变码率
        ".divx",        # DivX
    }
)

#: macOS 压缩 / 拷贝时生成的元数据目录名（内含 AppleDouble 等垃圾，**不是**用户素材）。
HIDDEN_METADATA_DIRNAME: str = "__MACOSX"

#: 跳过清单里最多记录多少条样例（避免 manifest 膨胀；数量仍为全量）。
MAX_SKIPPED_SAMPLES: int = 20

#: 未登记文件的原因标签（**每个未登记文件都必须有原因**，绝不静默）。
SKIP_REASON_HIDDEN: str = "hidden_or_appledouble"   # 隐藏文件/目录 / macOS 元数据
SKIP_REASON_UNSUPPORTED: str = "unsupported_suffix"  # 扩展名不在视频/图片白名单
SKIP_REASON_NOT_MEDIA: str = "not_media"             # 无扩展名的文件

#: Netpbm 系列（母项目 ``imageio`` 可直接读，无需 ffprobe）。
_NETPBM_SUFFIXES: frozenset[str] = frozenset({".pgm", ".ppm", ".pnm", ".pbm"})


def is_ignorable_media_path(path: Path, root: Path | None = None) -> bool:
    """判断某路径是否应被**跳过**（隐藏文件 / 隐藏目录 / macOS 元数据）。

    macOS 用 ``ditto`` / Finder 压缩目录时会写入 AppleDouble 资源分叉（``._*``）
    与 ``__MACOSX`` 目录；这些文件**不是**用户素材，若被登记会污染候选集，下游
    抽帧 / 标注**必然全部失败**（需求 F1 的输入卫生）。

    跳过判据（对相对 ``root`` 的**每一段**路径名逐段判断）：

    - 任一段以 ``.`` 开头 → 隐藏文件 / 隐藏目录（覆盖 ``._*`` / ``.DS_Store`` 等）；
    - 任一段等于 ``__MACOSX`` → macOS 元数据目录。

    该判据对**正常文件零误伤**：中文名、正常扩展名、非隐藏路径一律照常登记。

    Args:
        path: 待判断的路径。
        root: 素材根目录；给定时按相对路径逐段判断，``None`` 时只看文件名。

    Returns:
        ``True`` 表示应跳过。
    """
    candidate = Path(path)
    if root is not None:
        try:
            candidate = candidate.relative_to(Path(root))
        except ValueError:
            candidate = Path(path.name)
    for part in candidate.parts:
        if part.startswith(".") or part == HIDDEN_METADATA_DIRNAME:
            return True
    return False


def _skipped_relative(path: Path, root: Path) -> str:
    """跳过样例的可读标识：优先相对路径（``/`` 分隔），否则退回文件名。"""
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.name



def default_media_root(config: ProjectConfig) -> Path:
    """返回默认的媒体素材根目录。

    - 若配置了 ``video.video_path``，取其所在目录；
    - 否则取 ``input_frames_dir`` 的同级 ``media/``（即 ``<project>/input/media``）。

    该约定使「原始素材」与「抽帧产物（input/frames）」分离，互不污染。
    """
    if config.video.video_path:
        return Path(config.video.video_path).parent
    return Path(config.input_frames_dir).parent / "media"


# --------------------------------------------------------------------------- #
# 轻量图像头部解析（只读尺寸，不解码像素）
# --------------------------------------------------------------------------- #
def _png_dimensions(data: bytes) -> tuple[int, int] | None:
    """从 PNG 头部（IHDR）读取 ``(width, height)``；非法则 ``None``。"""
    if len(data) < 24 or data[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    if data[12:16] != b"IHDR":
        return None
    width = int.from_bytes(data[16:20], "big")
    height = int.from_bytes(data[20:24], "big")
    if width <= 0 or height <= 0:
        return None
    return width, height


#: JPEG SOF（Start Of Frame）标记，排除 DHT(0xC4) / JPG(0xC8) / DAC(0xCC)。
_JPEG_SOF_MARKERS: frozenset[int] = frozenset(
    {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}
)


def _jpeg_dimensions(data: bytes) -> tuple[int, int] | None:
    """从 JPEG 的 SOF 段读取 ``(width, height)``；非法则 ``None``。

    只扫描标记段，不解码熵编码数据。
    """
    if len(data) < 4 or data[0:2] != b"\xff\xd8":
        return None
    index = 2
    length = len(data)
    while index + 9 < length:
        if data[index] != 0xFF:
            index += 1
            continue
        marker = data[index + 1]
        if marker in (0xFF, 0x00) or 0xD0 <= marker <= 0xD9:
            index += 2
            continue
        segment_length = int.from_bytes(data[index + 2 : index + 4], "big")
        if segment_length < 2:
            return None
        if marker in _JPEG_SOF_MARKERS:
            height = int.from_bytes(data[index + 5 : index + 7], "big")
            width = int.from_bytes(data[index + 7 : index + 9], "big")
            if width <= 0 or height <= 0:
                return None
            return width, height
        index += 2 + segment_length
    return None


def _image_dimensions(path: Path) -> tuple[int, int]:
    """读取照片分辨率；读不出时返回 ``(0, 0)``（降级，不抛异常）。"""
    suffix = path.suffix.lower()
    try:
        if suffix in _NETPBM_SUFFIXES:
            image = load_image(path)
            return int(image.width), int(image.height)
        data = path.read_bytes()
    except (OSError, ValueError):
        return 0, 0

    if suffix == ".png":
        return _png_dimensions(data) or (0, 0)
    if suffix in (".jpg", ".jpeg"):
        return _jpeg_dimensions(data) or (0, 0)
    return 0, 0


# --------------------------------------------------------------------------- #
# 单条素材 → MediaAsset
# --------------------------------------------------------------------------- #
def _image_asset(path: Path, root: Path, scene: str) -> MediaAsset:
    width, height = _image_dimensions(path)
    return MediaAsset(
        source_id=_relative_source_id(path, root),
        path=str(path),
        scene=scene,
        media_type="image",
        duration_s=0.0,
        width=width,
        height=height,
        fps=0.0,
        count=1,
    )


def _video_asset(path: Path, scene: str, ffmpeg_binary: str | None) -> MediaAsset:
    """登记一段视频；用 ffprobe 读容器头，失败则降级为 0 但不丢弃该条。"""
    duration_s = 0.0
    width = 0
    height = 0
    fps = 0.0
    count = 0

    if ffmpeg_binary:
        # 延迟 import，避免仅做照片编目时也拉起视频解码模块。
        from industrial_hazard_coldstart.video_decode import probe_video

        try:
            info = probe_video(path, ffmpeg_binary)
        except Exception:  # noqa: BLE001 - 编目阶段对任何探测失败一律降级
            info = {}
        duration_s = float(info.get("duration_s") or 0.0)
        width = int(info.get("width") or 0)
        height = int(info.get("height") or 0)
        fps = float(info.get("fps") or 0.0)
        nb_frames = int(info.get("nb_frames") or 0)
        if nb_frames > 0:
            count = nb_frames
        elif duration_s > 0 and fps > 0:
            count = int(round(duration_s * fps))

    return MediaAsset(
        source_id=path.stem,
        path=str(path),
        scene=scene,
        media_type="video",
        duration_s=duration_s,
        width=width,
        height=height,
        fps=fps,
        count=count,
    )


def _relative_source_id(path: Path, root: Path) -> str:
    """照片的 ``source_id``：相对素材根的路径（去后缀），保证跨目录唯一。"""
    try:
        relative = path.relative_to(root)
    except ValueError:
        relative = Path(path.name)
    return relative.with_suffix("").as_posix().replace("/", "__")


# --------------------------------------------------------------------------- #
# 扫描与编目
# --------------------------------------------------------------------------- #
def scan_media_detailed(
    root: Path,
    scene: str,
    ffmpeg_binary: str | None = None,
) -> tuple[list[MediaAsset], dict[str, Any]]:
    """递归扫描 ``root``，返回 ``(素材清单, 跳过统计)``。

    与 :func:`scan_media` 同口径，但额外返回**被跳过**文件的分类统计，使调用方
    **看得见**少登记了什么（本项目「绝不静默」的既定规矩）。

    **统计闭合性（根本判据）**：``len(assets) + skipped["count"] == skipped["files_total"]``，
    即目录内**每个文件**要么被登记、要么被计入某个 ``by_reason``——**不存在
    「既没登记、也没跳过」的第三态**（R2 缺陷的根因：``.qt`` 真视频两头都不在）。

    Args:
        root: 媒体素材根目录（可含中文）。
        scene: 场景标签。
        ffmpeg_binary: ffmpeg 可执行路径；``None`` 时视频只登记路径、不读容器头。

    Returns:
        ``(assets, skipped)``：``assets`` 按路径排序；``skipped`` 形如::

            {
              "count": int,                    # == by_reason 之和
              "files_total": int,              # 目录内文件总数（含隐藏）
              "by_reason": {
                  "hidden_or_appledouble": int,
                  "unsupported_suffix": int,
                  "not_media": int,
              },
              "unsupported_suffixes": {".xyz": 2, ...},   # 未知扩展名 → 数量
              "samples": [相对路径, ...],       # 至多 MAX_SKIPPED_SAMPLES 条
            }
    """
    root = Path(root)
    if not root.exists():
        return [], {
            "count": 0,
            "files_total": 0,
            "by_reason": {
                SKIP_REASON_HIDDEN: 0,
                SKIP_REASON_UNSUPPORTED: 0,
                SKIP_REASON_NOT_MEDIA: 0,
            },
            "unsupported_suffixes": {},
            "samples": [],
        }

    assets: list[MediaAsset] = []
    files_total = 0
    by_reason: dict[str, int] = {
        SKIP_REASON_HIDDEN: 0,
        SKIP_REASON_UNSUPPORTED: 0,
        SKIP_REASON_NOT_MEDIA: 0,
    }
    unsupported_suffixes: dict[str, int] = {}
    samples: list[str] = []

    def _record_sample(path: Path) -> None:
        if len(samples) < MAX_SKIPPED_SAMPLES:
            samples.append(_skipped_relative(path, root))

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        files_total += 1
        # 输入卫生：跳过隐藏文件 / 隐藏目录 / macOS 元数据（._* / __MACOSX）。
        if is_ignorable_media_path(path, root):
            by_reason[SKIP_REASON_HIDDEN] += 1
            _record_sample(path)
            continue
        suffix = path.suffix.lower()
        if suffix in VIDEO_SUFFIXES:
            assets.append(_video_asset(path, scene, ffmpeg_binary))
            continue
        if suffix in SUPPORTED_IMAGE_SUFFIXES:
            assets.append(_image_asset(path, root, scene))
            continue
        # 未登记：**必须给出原因**（否则就是 R2 那种「凭空消失」）。
        if suffix:
            by_reason[SKIP_REASON_UNSUPPORTED] += 1
            unsupported_suffixes[suffix] = unsupported_suffixes.get(suffix, 0) + 1
        else:
            by_reason[SKIP_REASON_NOT_MEDIA] += 1
        _record_sample(path)

    skipped: dict[str, Any] = {
        "count": sum(by_reason.values()),
        "files_total": files_total,
        "by_reason": dict(by_reason),
        "unsupported_suffixes": dict(sorted(unsupported_suffixes.items())),
        "samples": samples,
    }
    return assets, skipped


def scan_media(
    root: Path,
    scene: str,
    ffmpeg_binary: str | None = None,
) -> list[MediaAsset]:
    """递归扫描 ``root``，把视频与照片登记为 :class:`MediaAsset` 列表。

    跳过隐藏文件 / 隐藏目录 / macOS 元数据（``._*`` / ``__MACOSX``）——见
    :func:`is_ignorable_media_path`。需要跳过统计时用 :func:`scan_media_detailed`。

    Args:
        root: 媒体素材根目录（可含中文）。
        scene: 场景标签（本期恒为 ``hoist``；字段保留以便高空素材到位后复用）。
        ffmpeg_binary: ffmpeg 可执行路径；``None`` 时视频只登记路径、不读容器头。

    Returns:
        按路径排序的素材清单；``root`` 不存在时返回空列表（不抛异常）。
    """
    assets, _skipped = scan_media_detailed(root, scene, ffmpeg_binary)
    return assets


def run_inventory(
    config: ProjectConfig,
    media_root: Path | None = None,
    scene: str | None = None,
) -> dict[str, Any]:
    """执行素材编目，产出 ``workspace/inventory/manifest.json``。

    Args:
        config: 项目配置（提供 workspace 路径、ffmpeg 二进制、默认场景）。
        media_root: 素材根目录；缺省用 :func:`default_media_root`。
        scene: 场景标签；缺省用 ``config.scene``。

    Returns:
        manifest 字典（含 ``assets`` / ``counts`` / ``skipped``）。
    """
    resolved_scene = scene or config.scene
    root = Path(media_root) if media_root is not None else default_media_root(config)
    assets, skipped = scan_media_detailed(root, resolved_scene, config.video.ffmpeg_binary)

    counts = {
        "total": len(assets),
        "video": sum(1 for item in assets if item.media_type == "video"),
        "image": sum(1 for item in assets if item.media_type == "image"),
        # 被跳过的文件（不计入 total，但必须可见）。
        "skipped": skipped["count"],
        # 目录内文件总数（含隐藏）；闭合性：total + skipped == files_total。
        "files_total": skipped["files_total"],
    }
    # 闭合性自检：每个文件要么登记、要么有跳过原因，绝不「凭空消失」。
    closure_ok = counts["total"] + counts["skipped"] == counts["files_total"]
    manifest: dict[str, Any] = {
        "scene": resolved_scene,
        "media_root": str(root),
        "media_root_exists": root.exists(),
        "assets": [item.to_dict() for item in assets],
        "counts": counts,
        "skipped": skipped,
        "closure_ok": closure_ok,
    }

    manifest_path = config.workspace("inventory", "manifest.json")
    ensure_dir(manifest_path.parent)
    write_json(manifest_path, manifest)

    status = {
        "stage": "inventory",
        "manifest_path": str(manifest_path),
        "media_root": str(root),
        "counts": counts,
        "skipped": skipped,
        "closure_ok": closure_ok,
    }
    write_json(config.workspace("status", "inventory.json"), status)
    return manifest
