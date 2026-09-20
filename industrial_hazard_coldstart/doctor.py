"""环境探活与依赖诊断扩展（对应功能点 F12）。

在母项目 ``build_doctor_report`` 之外，本模块补充**吊装链路专属**的四项检查：

1. **ffmpeg 可用性** —— 视频抽帧链路依赖的外部二进制；
2. **VLM 端点（经 lab-server）** —— 只校验配置，**不在本机联网探活**
   （实测本机到端点不可达，调用必须从 ``lab-server`` 发起）；
3. **端点护栏判定** —— 用母项目 ``endpoint_allowed`` 复算一次放行/拒绝结论，
   并额外检测**危险组合**：``mode == "local_only"`` 且 ``allowlist`` 非空时给出告警
   （母项目 ``endpoint_allowed`` 先判白名单命中、后判 mode，故 ``local_only``
   **并非**最严格档，白名单仍会生效——母项目代码冻结不能改，只能在此告警兜住）；
4. **holdout 完整性**（T04.2 已实现）—— 若 ``workspace/holdout/index.json`` 存在，
   读回冻结清单并重算每帧 sha256，给出**真实结论**（``ok`` / ``tampered`` / ``missing``）；
   若尚未冻结则报 ``not_frozen``。**绝不**再假装「尚未实现」。

设计立场：**依赖尚未实现时给出「尚未实现」，绝不假装通过**；
危险但非阻塞的组合放入 ``warnings``，不混进阻塞项 ``issues``。
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

from traffic_cv_closed_loop.config import ProjectConfig, endpoint_allowed

from industrial_hazard_coldstart.holdout import (
    holdout_dir_of,
    load_holdout_entries,
    verify_holdout,
)

__all__ = ["probe_ffmpeg", "build_hoist_doctor_report"]

#: 探活 ffmpeg 的短超时（秒），避免 doctor 卡死。
_FFMPEG_PROBE_TIMEOUT_S = 10

#: ``local_only`` 与白名单并存时的告警（母项目护栏可被残留白名单绕过，冻结不改）。
LOCAL_ONLY_ALLOWLIST_WARNING = (
    "endpoint_policy.mode=local_only 与 allowlist 同时存在："
    "母项目 endpoint_allowed() 先判白名单命中、后判 mode，"
    "因此白名单仍会生效，local_only 并非最严格档。"
    "若确实只想允许本机/内网，请清空 allowlist。"
)


def probe_ffmpeg(binary: str) -> dict[str, Any]:
    """探测 ffmpeg 二进制是否可用。

    先按 ``PATH`` 查找，再回退到把 ``binary`` 当作显式路径判断（母项目约定
    ``ffmpeg_binary`` 既可以是名字也可以是绝对路径）。可用时尝试取版本号；
    任何失败都降级为「不可用/版本未知」，不抛异常。
    """
    resolved = shutil.which(binary)
    if resolved is None:
        candidate = Path(binary)
        if candidate.exists():
            resolved = str(candidate)

    available = resolved is not None
    version = ""
    if available:
        try:
            completed = subprocess.run(
                [resolved, "-version"],
                capture_output=True,
                text=True,
                timeout=_FFMPEG_PROBE_TIMEOUT_S,
            )
            stdout = (completed.stdout or "").strip()
            if stdout:
                version = stdout.splitlines()[0].strip()
        except (OSError, subprocess.SubprocessError):
            version = ""

    return {
        "binary": binary,
        "resolved_path": resolved,
        "available": available,
        "version": version,
    }


def build_hoist_doctor_report(config: ProjectConfig) -> dict[str, Any]:
    """构造吊装链路探活报告（含四项检查）。"""
    issues: list[str] = []
    warnings: list[str] = []
    pending: list[str] = []

    # --- 1. ffmpeg 可用性 ---
    ffmpeg = probe_ffmpeg(config.video.ffmpeg_binary)
    if not ffmpeg["available"]:
        issues.append("ffmpeg_missing")

    # --- 2. VLM 端点（经 lab-server）---
    vlm_endpoint = {
        "provider": config.qwen_provider,
        "endpoint": config.qwen_chat_endpoint,
        "invocation_host": "lab-server",
        "probed": False,
        "reachable": None,
        "note": (
            "本机不联网，VLM 调用必须从 lab-server 发起（见架构 §7.2）。"
            "此处只校验配置与端点护栏，不做真实探活。"
        ),
    }

    # --- 3. 端点护栏判定 ---
    allowed, reason = endpoint_allowed(
        config.qwen_chat_endpoint,
        config.inference_endpoint_mode,
        config.inference_endpoint_allowlist,
    )
    # 危险组合：local_only 与白名单并存时，白名单仍生效（母项目判定顺序所致）。
    local_only_with_allowlist = (
        config.inference_endpoint_mode == "local_only"
        and bool(config.inference_endpoint_allowlist)
    )
    endpoint_policy = {
        "mode": config.inference_endpoint_mode,
        "allowlist": list(config.inference_endpoint_allowlist),
        "endpoint": config.qwen_chat_endpoint,
        "allowed": allowed,
        "reason": reason,
        "local_only_with_allowlist": local_only_with_allowlist,
        "warning": LOCAL_ONLY_ALLOWLIST_WARNING if local_only_with_allowlist else "",
    }
    if not allowed:
        issues.append("endpoint_policy_rejected")
    if local_only_with_allowlist:
        warnings.append(LOCAL_ONLY_ALLOWLIST_WARNING)

    # --- 4. holdout 完整性（T04.2 已实现：读回冻结清单 + 重算 sha256）---
    holdout_index = config.workspace("holdout", "index.json")
    if holdout_index.exists():
        holdout_dir = holdout_dir_of(config)
        entries = load_holdout_entries(holdout_dir)
        verification = verify_holdout(entries, holdout_dir)
        if verification["hash_ok"]:
            holdout_status = "ok"
            detail = (
                f"固定 holdout 已冻结且逐字节校验通过"
                f"（{verification['verified']}/{verification['total']} 帧）。"
            )
        elif verification["tampered"]:
            holdout_status = "tampered"
            detail = (
                f"固定 holdout 完整性受损：{len(verification['tampered'])} 帧内容被篡改。"
            )
        else:
            holdout_status = "missing"
            detail = (
                f"固定 holdout 完整性受损：{len(verification['missing'])} 帧缺失。"
            )
        holdout = {
            "status": holdout_status,
            "detail": detail,
            "index_path": str(holdout_index),
            "index_exists": True,
            "total": verification["total"],
            "verified": verification["verified"],
            "tampered": verification["tampered"],
            "missing": verification["missing"],
            "hash_ok": verification["hash_ok"],
        }
        if not verification["hash_ok"]:
            issues.append("holdout_integrity_broken")
    else:
        holdout = {
            "status": "not_frozen",
            "detail": (
                "尚未冻结固定 holdout（index.json 不存在）；"
                "请先运行 `train` / `dataset` 阶段完成首次冻结。"
            ),
            "index_path": str(holdout_index),
            "index_exists": False,
            "total": 0,
            "verified": 0,
            "tampered": [],
            "missing": [],
            "hash_ok": None,
        }

    return {
        "status": "issues_found" if issues else "ready",
        "issues": issues,
        "warnings": warnings,
        "pending": pending,
        "project_name": config.name,
        "config_path": str(config.config_path),
        "scene": config.scene,
        "labels": list(config.labels),
        "checks": {
            "ffmpeg": ffmpeg,
            "vlm_endpoint": vlm_endpoint,
            "endpoint_policy": endpoint_policy,
            "holdout": holdout,
        },
    }
