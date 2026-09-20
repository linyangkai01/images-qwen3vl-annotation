"""命令行入口与子命令骨架。

子命令覆盖完整链路：``inventory / extract-video / extract / annotate /
prioritize / curate / freeze-holdout / train / evaluate / doctor``。

**本批（T01）只落地 ``doctor``**；其余子命令一律打印明确的「尚未实现」提示
并返回**非 0** 退出码（``2``），**绝不静默成功**。这样既保证 ``--help`` 可用，
也让上游编排脚本能立刻发现「该阶段还没接上」。

退出码约定：
- ``0``：成功；
- ``1``：``doctor`` 发现阻塞性问题；
- ``2``：子命令尚未实现（或参数错误）。
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from industrial_hazard_coldstart.doctor import build_hoist_doctor_report

#: 默认配置指向自有吊装子树。
DEFAULT_CONFIG = "own/hoist_project/config/project.hoist.json"

#: 尚未实现的子命令 → 计划落地的子任务（用于给出可操作的提示）。
_PENDING_COMMANDS: dict[str, str] = {
    "inventory": "T02.1（素材编目 F1）",
    "extract-video": "T02.2（视频解码与关键帧筛选 F2）",
    "extract": "T02.3（接入母项目初检 F3）",
    "annotate": "T03.2（两段式标注 F4）",
    "prioritize": "T03.4（主动学习 F6）",
    "curate": "T04.1（质量过滤与版本化数据集 F8）",
    "freeze-holdout": "T04.2（固定 holdout 冻结）",
    "train": "T04.3（YOLO 冷启动训练 F9）",
    "evaluate": "T05.1（固定 holdout 逐类评估 F10）",
}

_COMMAND_HELP: dict[str, str] = {
    "inventory": "扫描素材目录，登记时长/分辨率/帧率，产出 manifest.json。",
    "extract-video": "ffmpeg 抽帧 + 关键帧去重，产出候选关键帧。",
    "extract": "复用母项目 run_extraction 做通用 YOLO 零样本初检。",
    "annotate": "复用母项目两段式标注，私有 VLM 复核纠偏。",
    "prioritize": "按不确定性排序，产出 priority_queue.jsonl。",
    "curate": "质量过滤 + 版本化数据集构建。",
    "freeze-holdout": "冻结固定 holdout 并登记 sha256。",
    "train": "在数据集上执行 YOLO 冷启动训练。",
    "evaluate": "在固定 holdout 上逐类评估并按 A/B/C 三档判定。",
    "doctor": "环境探活：ffmpeg / VLM 端点 / 端点护栏 / holdout 完整性。",
}


def _build_parser() -> argparse.ArgumentParser:
    """构造 argparse 解析器（``--help`` 必须可用）。"""
    parser = argparse.ArgumentParser(
        prog="industrial-hazard-coldstart",
        description="工业隐患单场景（吊装）YOLO 冷启动标注与训练。",
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help=f"项目配置 JSON 路径（默认 {DEFAULT_CONFIG}）。",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_config_argument(command_parser: argparse.ArgumentParser) -> None:
        # ``argparse.SUPPRESS`` 防止子命令默认值覆盖子命令之前传入的 ``--config``。
        command_parser.add_argument(
            "--config",
            default=argparse.SUPPRESS,
            help="项目配置 JSON 路径。",
        )

    for command, help_text in _COMMAND_HELP.items():
        command_parser = subparsers.add_parser(command, help=help_text)
        add_config_argument(command_parser)

    return parser


def _print_payload(payload: Any) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _pending_message(command: str) -> str:
    task = _PENDING_COMMANDS.get(command, "后续任务")
    return (
        f"子命令 `{command}` 尚未实现（计划落地于 {task}）。\n"
        "本批（T01）只交付项目基础设施与共享契约，仅 `doctor` 可用。\n"
        "该子命令不会静默成功——请在上游编排脚本中等待对应任务完成后接入。"
    )


def main(argv: list[str] | None = None) -> int:
    """命令行主入口，返回进程退出码。"""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "doctor":
        # 延迟 import，避免 ``import industrial_hazard_coldstart`` 时拉起母项目配置。
        from traffic_cv_closed_loop.config import ProjectConfig

        config = ProjectConfig.load(args.config)
        report = build_hoist_doctor_report(config)
        _print_payload(report)
        return 0 if report["status"] == "ready" else 1

    if args.command in _PENDING_COMMANDS:
        print(_pending_message(args.command), file=sys.stderr)
        return 2

    parser.error(f"Unsupported command: {args.command}")
    return 2  # pragma: no cover - argparse.error 已抛出 SystemExit
