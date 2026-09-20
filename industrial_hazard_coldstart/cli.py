"""命令行入口与子命令。

子命令覆盖完整链路：``inventory / extract-video / extract / annotate /
prioritize / curate / freeze-holdout / train / evaluate / doctor``。

**已实现**：``doctor``（T01）、``inventory / extract-video / extract``（T02）、
``annotate / prioritize``（T03）、``curate / freeze-holdout``（T04.1 / T04.2）、
``train``（T04.3）、``evaluate``（T05.1）——数据阶段均接线到 :func:`orchestrator.run_cycle`。
**仍未实现**的子命令一律打印明确的「尚未实现」提示并返回**非 0** 退出码（``2``），
**绝不静默成功**——既保证 ``--help`` 可用，也让上游编排脚本能立刻发现「该阶段还没接上」。
（T05.1 落地后 :data:`PENDING_COMMANDS` 为**空**：全部子命令均已实现。）

已实现 / 未实现两个集合**显式声明**（见 :data:`IMPLEMENTED_COMMANDS` /
:data:`PENDING_COMMANDS`）：**每实现一个阶段，只改这一处**；测试引用常量而非硬编码
子命令名，避免「阶段转正后测试假红」。

退出码约定：
- ``0``：成功 / ``doctor`` 一切就绪；
- ``1``：``doctor`` 读到配置但发现问题（环境问题）；
- ``2``：子命令尚未实现（或参数错误）；
- ``3``：配置**无法读取**（路径不存在 / 非法 JSON / 校验失败）；
- ``4``：已实现阶段的**执行失败**（如 ffmpeg 缺失、素材缺失）。

``1`` 与 ``3`` 刻意区分：脚本据此能分辨「配置根本读不到」与「配置读到了但环境有问题」。
``4`` 与 ``2`` 区分「实现好了但跑失败」与「压根还没实现」。
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
EXIT_STAGE_FAILED = 4

#: 已实现并**真正接线**的子命令（每实现一个阶段就加入本集合）。
IMPLEMENTED_COMMANDS: tuple[str, ...] = (
    "doctor",
    "inventory",
    "extract-video",
    "extract",
    "annotate",
    "prioritize",
    "curate",
    "freeze-holdout",
    "train",
    "evaluate",
)

#: 尚未实现的子命令 → 计划落地的子任务（用于给出可操作的提示）。
#: **每实现一个阶段，就把对应条目从这里移入 :data:`IMPLEMENTED_COMMANDS`。**
#: T05.1 已落地 ``evaluate``，本映射现为**空**——全部子命令均已实现。
_PENDING_COMMANDS: dict[str, str] = {}

#: 仍未实现的子命令名集合（供测试引用，避免硬编码子命令名）。
PENDING_COMMANDS: frozenset[str] = frozenset(_PENDING_COMMANDS)

#: 需要「加载配置 + 跑 orchestrator.run_cycle」的阶段（``doctor`` 单独处理）。
#: 由 :data:`IMPLEMENTED_COMMANDS` 派生，保证单一真源。
DATA_STAGE_COMMANDS: tuple[str, ...] = tuple(
    name for name in IMPLEMENTED_COMMANDS if name != "doctor"
)

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

#: **全部**子命令（单一真源，派生自 :data:`_COMMAND_HELP`，即 argparse 实际注册的子命令）。
#: 用途：验收断言「已实现 ∪ 未实现 == 全部子命令」——将来新增子命令若忘了归类，
#: 该覆盖性断言会立刻失败（而不是被「精确快照」悄悄放过）。
ALL_COMMANDS: tuple[str, ...] = tuple(_COMMAND_HELP)


def _build_parser() -> argparse.ArgumentParser:
    """构造 argparse 解析器（``--help`` 必须可用）。"""
    parser = argparse.ArgumentParser(
        prog="industrial-hazard-coldstart",
        description="工业隐患单场景（吊装）YOLO 冷启动标注与训练。",
        epilog=(
            "退出码：0=就绪；1=doctor 发现问题；"
            "2=子命令尚未实现/参数错误；3=配置无法读取；4=阶段执行失败。"
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
        # 版本号由调用方指定（不用时间戳）→ 跨轮可比；holdout 占比同理。
        # 这些阶段都产出 / 消费版本化数据集与固定 holdout。
        if command in {"curate", "freeze-holdout", "train", "evaluate"}:
            command_parser.add_argument(
                "--version",
                default=argparse.SUPPRESS,
                help="数据集 / 运行版本号（默认 coldstart-v1，非时间戳）。",
            )
        if command in {"curate", "freeze-holdout", "train"}:
            command_parser.add_argument(
                "--holdout-ratio",
                type=float,
                default=argparse.SUPPRESS,
                help="首次冻结 holdout 的占比（默认 0.2）。",
            )
        if command in {"train", "evaluate"}:
            command_parser.add_argument(
                "--execute",
                action="store_true",
                help="真正执行（默认只准备命令 / 走 mock）。",
            )

    return parser


def _print_payload(payload: Any) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _pending_message(command: str) -> str:
    task = _PENDING_COMMANDS.get(command, "后续任务")
    implemented = "、".join(f"`{name}`" for name in IMPLEMENTED_COMMANDS)
    return (
        f"子命令 `{command}` 尚未实现（计划落地于 {task}）。\n"
        f"当前已实现的子命令：{implemented}。\n"
        "该子命令不会静默成功——请在上游编排脚本中等待对应任务完成后接入。"
    )


def _stage_error_message(command: str, error: Exception) -> str:
    """把已实现阶段的执行失败转成可读的中文错误（不打印栈）。"""
    return (
        f"错误：阶段 `{command}` 执行失败\n"
        f"原因：{error}\n"
        f"（退出码 {EXIT_STAGE_FAILED} = 已实现阶段执行失败，"
        f"与退出码 {EXIT_NOT_IMPLEMENTED} = 尚未实现 区分。）"
    )


def _payload_failures(payload: Any) -> list[dict[str, Any]]:
    """从阶段摘要中提取「部分失败」清单（含 ``extract`` 内嵌的 ``video_extract``）。

    容错（跳过并记录）不等于静默：只要摘要带 ``has_failures``，本函数就把它暴露给
    CLI，由 CLI 以非 0 退出码让调用方看见。
    """
    failures: list[dict[str, Any]] = []
    if not isinstance(payload, dict):
        return failures
    if payload.get("has_failures"):
        failures.extend(payload.get("failed_videos") or [])
    nested = payload.get("video_extract")
    if isinstance(nested, dict) and nested.get("has_failures"):
        failures.extend(nested.get("failed_videos") or [])
    return failures


def _partial_failure_message(command: str, failures: list[dict[str, Any]]) -> str:
    """把「部分视频失败」转成可读的中文提示（不打印栈）。"""
    lines = [
        f"警告：阶段 `{command}` 完成，但有 {len(failures)} 个视频失败（已跳过并记录）："
    ]
    for item in failures:
        lines.append(
            f"  - {item.get('source_id', '?')}"
            f"（{item.get('error_type', 'Error')}）：{item.get('reason', '')}"
        )
    lines.append(
        f"（退出码 {EXIT_STAGE_FAILED} = 阶段部分失败；失败清单亦见阶段摘要 "
        f"的 failed_videos 字段。）"
    )
    return "\n".join(lines)


def _payload_warnings(payload: Any) -> list[str]:
    """从阶段摘要中提取**可见告警**（``warnings`` 列表）。

    告警是「合法但可能误配」的信号（如 sidecar 零命中），**不影响退出码**（仍为 0），
    但必须打到 stderr，避免调用方以为「一切正常」。
    """
    if not isinstance(payload, dict):
        return []
    raw = payload.get("warnings")
    if not isinstance(raw, list):
        return []
    return [str(item) for item in raw]


def _warnings_message(command: str, warnings: list[str]) -> str:
    """把阶段告警转成可读的中文提示（不打印栈，退出码不变）。"""
    lines = [f"提示：阶段 `{command}` 完成，但有 {len(warnings)} 条告警（请复核）："]
    lines.extend(f"  - {item}" for item in warnings)
    return "\n".join(lines)


def _config_error_message(config_path: str, error: Exception) -> str:
    """把配置读取失败转成可读的中文错误（不打印栈）。"""
    return (
        f"错误：无法读取配置文件 {config_path}\n"
        f"原因：{error}\n"
        "提示：请确认 --config 指向的文件存在、是合法 JSON，且通过 ProjectConfig 校验。\n"
        f"（退出码 {EXIT_CONFIG_UNREADABLE} = 配置无法读取，与退出码 "
        f"{EXIT_ISSUES} = doctor 发现问题 区分。）"
    )


def _load_config_or_report(config_path: str) -> ProjectConfig | int:
    """加载配置；失败时打印可读错误并返回专用退出码（int）。"""
    from traffic_cv_closed_loop.config import ProjectConfig

    try:
        return ProjectConfig.load(config_path)
    except (FileNotFoundError, ValueError, json.JSONDecodeError, OSError) as error:
        # 配置读不到 → 可读错误 + 专用退出码，绝不抛栈。
        print(_config_error_message(config_path, error), file=sys.stderr)
        return EXIT_CONFIG_UNREADABLE


def main(argv: list[str] | None = None) -> int:
    """命令行主入口，返回进程退出码。"""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "doctor":
        config = _load_config_or_report(args.config)
        if isinstance(config, int):
            return config
        report = build_hoist_doctor_report(config)
        _print_payload(report)
        return EXIT_READY if report["status"] == "ready" else EXIT_ISSUES

    if args.command in DATA_STAGE_COMMANDS:
        config = _load_config_or_report(args.config)
        if isinstance(config, int):
            return config
        # 延迟 import，避免 ``import industrial_hazard_coldstart`` 时拉起视频解码链路。
        from industrial_hazard_coldstart.orchestrator import run_cycle

        # 仅 ``train`` 携带可选的版本 / holdout / execute 参数（其它阶段没有这些开关）。
        stage_kwargs: dict[str, Any] = {}
        version_arg = getattr(args, "version", None)
        if version_arg is not None:
            stage_kwargs["version"] = version_arg
        holdout_ratio_arg = getattr(args, "holdout_ratio", None)
        if holdout_ratio_arg is not None:
            stage_kwargs["holdout_ratio"] = holdout_ratio_arg
        if getattr(args, "execute", False):
            stage_kwargs["execute"] = True

        try:
            payload = run_cycle(config, args.command, **stage_kwargs)
        except Exception as error:  # noqa: BLE001 - 阶段失败 → 可读错误 + 专用退出码
            print(_stage_error_message(args.command, error), file=sys.stderr)
            return EXIT_STAGE_FAILED
        _print_payload(payload)
        # 可见告警（如 sidecar 零命中）：打到 stderr，但不改变退出码（仍为 0）。
        warnings = _payload_warnings(payload)
        if warnings:
            print(_warnings_message(args.command, warnings), file=sys.stderr)
        # 部分视频失败（容错跳过）→ 非 0 退出码，让调用方看见，绝不静默。
        failures = _payload_failures(payload)
        if failures:
            print(_partial_failure_message(args.command, failures), file=sys.stderr)
            return EXIT_STAGE_FAILED
        return EXIT_READY

    if args.command in _PENDING_COMMANDS:
        print(_pending_message(args.command), file=sys.stderr)
        return EXIT_NOT_IMPLEMENTED

    parser.error(f"Unsupported command: {args.command}")
    return EXIT_NOT_IMPLEMENTED  # pragma: no cover - argparse.error 已抛出 SystemExit
