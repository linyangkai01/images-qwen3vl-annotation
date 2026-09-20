"""阶段编排：把素材层与母项目初检串起来（T02.3）。

本批实现两个阶段（外加一个只读的 ``inventory``）：

- ``extract-video`` —— 调 :func:`video_decode.run_video_extract`，把关键帧写入
  ``config.input_frames_dir/<source_id>/``；
- ``extract`` —— 调母项目 ``pipeline.extraction.run_extraction``，在关键帧上做
  通用 YOLO 零样本初检，产出 ``candidates.jsonl``。

**串联逻辑**：``extract`` 阶段默认先确保关键帧已就位——若 ``input_frames_dir``
尚无任何图片且能发现视频，则自动先跑一遍 ``extract-video``，再调
``run_extraction``。该行为可用 ``auto_extract_video=False`` 关闭。

产出的 ``CandidateFrame`` **直接复用母项目类**（不重定义），结构与母项目完全兼容。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from traffic_cv_closed_loop.config import ProjectConfig
from traffic_cv_closed_loop.io_utils import iter_image_files
from traffic_cv_closed_loop.pipeline.extraction import run_extraction

from industrial_hazard_coldstart.inventory import run_inventory
from industrial_hazard_coldstart.video_decode import discover_videos, run_video_extract

__all__ = ["SUPPORTED_STAGES", "run_cycle"]

#: 本批实现的阶段名。
SUPPORTED_STAGES: tuple[str, ...] = ("inventory", "extract-video", "extract")


def _needs_video_extract(
    config: ProjectConfig,
    video_paths: list[Path] | None,
) -> bool:
    """判断 ``extract`` 前是否需要先抽帧：无图片、且有可发现的视频。"""
    if iter_image_files(config.input_frames_dir):
        return False
    return bool(discover_videos(config, video_paths))


def run_cycle(
    config: ProjectConfig,
    stage: str,
    *,
    video_paths: list[Path] | None = None,
    media_root: Path | None = None,
    auto_extract_video: bool = True,
) -> dict[str, Any]:
    """执行一个链路阶段。

    Args:
        config: 项目配置。
        stage: 阶段名，见 :data:`SUPPORTED_STAGES`。
        video_paths: ``extract-video`` / ``extract`` 使用的显式视频列表。
        media_root: ``inventory`` 使用的素材根目录。
        auto_extract_video: ``extract`` 阶段是否在需要时自动先抽帧。

    Returns:
        阶段摘要字典（``extract`` 结果含母项目 ``run_extraction`` 的全部字段）。

    Raises:
        ValueError: 阶段名不受支持。
    """
    normalized = (stage or "").strip().lower()

    if normalized == "inventory":
        return {"stage": "inventory", **run_inventory(config, media_root=media_root)}

    if normalized == "extract-video":
        return {"stage": "extract-video", **run_video_extract(config, video_paths=video_paths)}

    if normalized == "extract":
        chained: dict[str, Any] | None = None
        if auto_extract_video and _needs_video_extract(config, video_paths):
            chained = run_video_extract(config, video_paths=video_paths)
        summary = run_extraction(config)
        return {"stage": "extract", "video_extract": chained, **summary}

    raise ValueError(
        f"未知阶段：{stage!r}（本批支持：{', '.join(SUPPORTED_STAGES)}）"
    )
