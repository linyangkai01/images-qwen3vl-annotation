"""命令行入口与子命令骨架。

子命令覆盖完整链路：``inventory / extract-video / extract / annotate /
prioritize / curate / freeze-holdout / train / evaluate / doctor``。

**本批（T01）只落地 ``doctor``**；其余子命令一律打印明确的「尚未实现」提示
并返回**非 0** 退出码（``2``），**绝不静默成功**。这样既保证 ``--help`` 可用，
也让上游编排脚本能立刻发现「该阶段还没接上」。

退出码约定：
- ``0``：成功 / ``doctor`` 一切就绪；
- ``1``：``doctor`` 读到配置但发现问题（环境问题）；
- ``2``：子命令尚未实现（或参数错误）；
- ``3``：配置**无法读取**（路径不存在 / 非法 JSON / 校验失败）。

``1`` 与 ``3`` 刻意区分：脚本据此能分辨「配置根本读不到」与「配置读到了但环境有问题」。
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from industrial_hazard_coldstart.doctor import build_hoist_doctor_report

#: 默认配置指向自有吊装子树。
DEFAULT_CONFIG = "own/hoist_project/config/project.hoist.json"

#: 退出码常量（与文档字符串一致）。
EXIT_READY = 0
EXIT_ISSUES = 1
EXIT_NOT_IMPLEMENTED = 2
EXIT_CONFIG_UNREADABLE = 3

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
        epilog=(
            "退出码：0=就绪；1=doctor 发现问题；"
            "2=子命令尚未实现/参数错误；3=配置无法读取。"
        ),
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


def _config_error_message(config_path: str, error: Exception) -> str:
    """把配置读取失败转成可读的中文错误（不打印栈）。"""
    return (
        f"错误：无法读取配置文件 {config_path}\n"
        f"原因：{error}\n"
        "提示：请确认 --config 指向的文件存在、是合法 JSON，且通过 ProjectConfig 校验。\n"
        f"（退出码 {EXIT_CONFIG_UNREADABLE} = 配置无法读取，与退出码 "
        f"{EXIT_ISSUES} = doctor 发现问题 区分。）"
    )


def main(argv: list[str] | None = None) -> int:
    """命令行主入口，返回进程退出码。"""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "doctor":
        # 延迟 import，避免 ``import industrial_hazard_coldstart`` 时拉起母项目配置。
        from traffic_cv_closed_loop.config import ProjectConfig

        try:
            config = ProjectConfig.load(args.config)
        except (FileNotFoundError, ValueError, json.JSONDecodeError, OSError) as error:
            # 配置读不到 → 可读错误 + 专用退出码，绝不抛栈。
            print(_config_error_message(str(args.config), error), file=sys.stderr)
            return EXIT_CONFIG_UNREADABLE

        report = build_hoist_doctor_report(config)
        _print_payload(report)
        return EXIT_READY if report["status"] == "ready" else EXIT_ISSUES

    if args.command in _PENDING_COMMANDS:
        print(_pending_message(args.command), file=sys.stderr)
        return EXIT_NOT_IMPLEMENTED

    parser.error(f"Unsupported command: {args.command}")
    return EXIT_NOT_IMPLEMENTED  # pragma: no cover - argparse.error 已抛出 SystemExit
