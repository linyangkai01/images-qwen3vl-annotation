"""统一推理途径（inference path）抽象与视频时长校验。

本模块把「解析」这件事收敛到三条可选途径，由 ``inference.backend`` 选择：

* ``local_yolo``   —— 本机 Ultralytics YOLO（复用 ``UltralyticsDetector``）
* ``server_yolo``  —— lab-server 上的 YOLO（经 ssh 委派执行）
* ``private_vlm``  —— 实验室私有多模态模型 API（Ollama-native ``/api/generate``）
* ``mock``         —— 离线 sidecar，不联网

同时负责**视频时长校验**：私有多模态模型受算力限制，只能处理 1 分钟以内的
视频。超限时按 ``inference.on_over_limit`` 处置（``warn`` / ``error`` / ``segment``）。

设计原则：
* 不引入第三方依赖，只用标准库；
* 远程执行通过可注入的 ``runner`` 完成，便于单测替身；
* 任何一条途径缺少必要配置时，报错信息必须指明缺什么、怎么补。
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from hazard_annotation.ffmpeg import FfmpegRunner
from traffic_cv_closed_loop.adapters.qwen import (
    MockQwenAnnotator,
    OpenAICompatibleQwenAnnotator,
    PrivateVlmAnnotator,
    build_qwen_annotator,
)
from traffic_cv_closed_loop.adapters.yolo import (
    SidecarDetector,
    UltralyticsDetector,
    build_detector,
)
from traffic_cv_closed_loop.config import (
    ProjectConfig,
    endpoint_allowed,
    endpoint_host_of,
    is_local_endpoint_host,
)
from traffic_cv_closed_loop.io_utils import ensure_dir, write_json

#: 允许的途径取值（与 ``ProjectConfig.validate`` 保持一致）。
INFERENCE_BACKENDS: tuple[str, ...] = ("mock", "local_yolo", "server_yolo", "private_vlm")

#: 需要真正调用模型/检测器的途径（``mock`` 不算）。
REMOTE_BACKENDS: tuple[str, ...] = ("server_yolo", "private_vlm")


class VideoDurationError(RuntimeError):
    """视频超出私有多模态模型的时长上限，且处置策略为 ``error``。"""


class InferenceConfigError(RuntimeError):
    """所选推理途径缺少必要配置。"""


# --------------------------------------------------------------------------- #
# 视频时长校验
# --------------------------------------------------------------------------- #
@dataclass
class VideoDurationReport:
    """一次视频时长校验的结果，可直接序列化进 summary.json。"""

    video_path: str
    duration_seconds: float
    limit_seconds: float
    status: str  # ok | over_limit | unknown
    action: str  # proceed | warn | error | segment
    segment_count: int = 0
    message: str = ""
    warnings: list[str] = field(default_factory=list)

    @property
    def over_limit(self) -> bool:
        return self.status == "over_limit"

    def to_dict(self) -> dict[str, Any]:
        return {
            "video_path": self.video_path,
            "duration_seconds": round(self.duration_seconds, 3),
            "limit_seconds": self.limit_seconds,
            "status": self.status,
            "action": self.action,
            "segment_count": self.segment_count,
            "message": self.message,
            "warnings": list(self.warnings),
        }


def check_video_duration(
    video_path: str | Path,
    config: ProjectConfig,
    runner: FfmpegRunner | None = None,
) -> VideoDurationReport:
    """探测视频时长并按配置判定是否超限。

    * ``duration == 0`` 视为**未知**（ffprobe/ffmpeg 都没给出时长）：记 warning 但放行，
      因为把「探测失败」当成「超限」会误伤。
    * 超限时按 ``config.inference_on_over_limit`` 给出 ``action``：
      ``warn`` 放行并告警、``error`` 抛 :class:`VideoDurationError`、
      ``segment`` 计算需要切成几段（每段 ``segment_seconds``）。
    """
    path = Path(video_path)
    limit = float(config.inference_max_video_seconds)
    policy = config.inference_on_over_limit
    segment_seconds = float(config.inference_segment_seconds)

    probe = runner or FfmpegRunner(config.video.ffmpeg_binary)
    try:
        duration = probe.probe_duration(path)
    except Exception as error:  # ffmpeg 不可用等 —— 不因探测失败而中断
        return VideoDurationReport(
            video_path=str(path),
            duration_seconds=0.0,
            limit_seconds=limit,
            status="unknown",
            action="proceed",
            message=f"无法探测视频时长（{error}），跳过时长校验",
            warnings=["video_duration_unknown"],
        )

    if duration <= 0:
        return VideoDurationReport(
            video_path=str(path),
            duration_seconds=0.0,
            limit_seconds=limit,
            status="unknown",
            action="proceed",
            message="无法探测视频时长，跳过时长校验",
            warnings=["video_duration_unknown"],
        )

    if duration <= limit:
        return VideoDurationReport(
            video_path=str(path),
            duration_seconds=duration,
            limit_seconds=limit,
            status="ok",
            action="proceed",
            message=f"时长 {duration:.1f}s，在上限 {limit:.0f}s 内",
        )

    # --- 超限 ---
    if policy == "error":
        raise VideoDurationError(
            f"视频时长 {duration:.1f}s 超过上限 {limit:.0f}s；"
            f"当前 inference.on_over_limit=error。"
            f"可改为 warn（告警放行）或 segment（自动分段）。"
        )

    if policy == "segment":
        segment_count = int(-(-duration // segment_seconds))  # 向上取整
        return VideoDurationReport(
            video_path=str(path),
            duration_seconds=duration,
            limit_seconds=limit,
            status="over_limit",
            action="segment",
            segment_count=segment_count,
            message=(
                f"时长 {duration:.1f}s 超过上限 {limit:.0f}s，"
                f"建议切成 {segment_count} 段（每段 ≤ {segment_seconds:.0f}s）后再送模型"
            ),
            warnings=["video_over_limit_segmented"],
        )

    return VideoDurationReport(
        video_path=str(path),
        duration_seconds=duration,
        limit_seconds=limit,
        status="over_limit",
        action="warn",
        message=(
            f"视频时长 {duration:.1f}s 超过私有多模态模型上限 {limit:.0f}s；"
            f"已按 on_over_limit=warn 放行，但结果可能不完整，建议改用 segment 模式"
        ),
        warnings=["video_over_limit_warned"],
    )


def split_duration_segments(
    duration_seconds: float, segment_seconds: float
) -> list[tuple[float, float]]:
    """把 ``duration_seconds`` 切成 ``(start, end)`` 段列表，供 ffmpeg 裁剪用。"""
    if segment_seconds <= 0:
        raise ValueError("segment_seconds must be > 0")
    segments: list[tuple[float, float]] = []
    start = 0.0
    while start < duration_seconds:
        end = min(start + segment_seconds, duration_seconds)
        segments.append((round(start, 3), round(end, 3)))
        start = end
    return segments


# --------------------------------------------------------------------------- #
# 远程 YOLO（lab-server）
# --------------------------------------------------------------------------- #
def _default_runner(command: list[str]) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(command, check=False, capture_output=True, text=False)


@dataclass
class RemoteYoloSettings:
    """lab-server 上执行 YOLO 所需的连接与路径参数。"""

    host: str = "lab-server"
    remote_dir: str = "/data/users/lyk/.tmp/yolo-inference"
    activate_script: str = "/data/users/lyk/activate-hazard.sh"
    yolo_binary: str = "yolo"
    weights: str = ""
    confidence: float = 0.25
    iou: float = 0.45
    image_size: int = 640
    device: str = ""
    timeout_seconds: int = 1800


class RemoteYoloDetector:
    """在 lab-server 上跑 YOLO，本地只负责搬运图片与读回标签。

    流程：``scp`` 上传图片 → ``ssh`` 执行 ``yolo predict`` → ``scp`` 下载 labels。

    远程命令通过可注入的 ``runner`` 执行，便于在无 SSH 环境下单测命令构造。
    与本地 :class:`UltralyticsDetector` 保持同一 ``detect_many`` 契约。
    """

    def __init__(
        self,
        config: ProjectConfig,
        settings: RemoteYoloSettings | None = None,
        runner: Callable[[list[str]], subprocess.CompletedProcess[bytes]] | None = None,
    ) -> None:
        self.config = config
        self.settings = settings or build_remote_yolo_settings(config)
        self._runner = runner or _default_runner
        self.warnings: list[str] = []

    # -- 命令构造（单测关注点） ------------------------------------------- #
    def build_predict_command(self) -> str:
        """构造在 lab-server 上执行的 shell 片段（不含 ssh 前缀）。"""
        parts = [
            f"source {self.settings.activate_script} > /dev/null 2>&1",
            f"cd {self.settings.remote_dir}",
            (
                f"{self.settings.yolo_binary} predict "
                f"model={self.settings.weights or 'yolov8n.pt'} "
                f"source=images "
                f"conf={self.settings.confidence} "
                f"iou={self.settings.iou} "
                f"imgsz={self.settings.image_size} "
                f"save_txt=True save_conf=True project=out name=run exist_ok=True"
            ),
        ]
        if self.settings.device:
            parts[-1] += f" device={self.settings.device}"
        return " && ".join(parts)

    def build_ssh_command(self, remote_command: str) -> list[str]:
        return ["ssh", self.settings.host, remote_command]

    def build_upload_commands(self, image_paths: list[Path]) -> list[list[str]]:
        remote = f"{self.settings.host}:{self.settings.remote_dir}/images/"
        return [["scp", "-q", str(path), remote] for path in image_paths]

    def build_download_command(self) -> list[str]:
        remote = f"{self.settings.host}:{self.settings.remote_dir}/out/run/labels/"
        return ["scp", "-q", "-r", remote, str(self._local_labels_dir())]

    def _local_labels_dir(self) -> Path:
        return Path(self.config.workspace_dir) / "remote_yolo" / "labels"

    # -- 执行 -------------------------------------------------------------- #
    def detect_many(self, image_paths: list[Path]) -> dict[Path, list[Detection]]:
        from traffic_cv_closed_loop.models import Detection  # 局部导入避免循环

        resolved = [path.resolve() for path in image_paths]
        if not resolved:
            return {}

        ensure_dir(self._local_labels_dir())
        self._run_or_raise(
            self.build_ssh_command(f"mkdir -p {self.settings.remote_dir}/images"),
            "准备远程目录失败",
        )
        for command in self.build_upload_commands(resolved):
            self._run_or_raise(command, "上传图片失败")

        self._run_or_raise(
            self.build_ssh_command(self.build_predict_command()),
            "远程 YOLO 推理失败",
        )
        self._run_or_raise(self.build_download_command(), "下载标签失败")

        results: dict[Path, list[Detection]] = {}
        for image_path in resolved:
            label_path = self._local_labels_dir() / f"{image_path.stem}.txt"
            results[image_path] = self._parse_label_file(label_path)
        return results

    def detect(self, image_path: Path) -> list[Detection]:
        return self.detect_many([image_path]).get(Path(image_path).resolve(), [])

    def _parse_label_file(self, label_path: Path) -> list[Detection]:
        from traffic_cv_closed_loop.models import Detection

        if not label_path.exists():
            self.warnings.append(f"remote_label_missing:{label_path.name}")
            return []
        detections: list[Detection] = []
        labels = self.config.labels
        for raw_line in label_path.read_text(encoding="utf-8").splitlines():
            parts = raw_line.split()
            if len(parts) < 5:
                continue
            try:
                class_index = int(float(parts[0]))
                x_center, y_center, width, height = (float(value) for value in parts[1:5])
            except ValueError:
                continue
            confidence = float(parts[5]) if len(parts) > 5 else 0.0
            if not 0 <= class_index < len(labels):
                self.warnings.append(f"remote_class_out_of_range:{class_index}")
                continue
            # YOLO 标签是归一化 (cx, cy, w, h)，Detection 需要 xyxy
            x1 = x_center - width / 2.0
            y1 = y_center - height / 2.0
            x2 = x_center + width / 2.0
            y2 = y_center + height / 2.0
            detections.append(
                Detection(
                    label=labels[class_index],
                    bbox=(x1, y1, x2, y2),
                    confidence=confidence,
                    source="yolo-lab-server",
                )
            )
        return detections

    def _run_or_raise(
        self, command: list[str], failure_message: str
    ) -> subprocess.CompletedProcess[bytes]:
        completed = self._runner(command)
        if completed.returncode != 0:
            stderr = (completed.stderr or b"").decode("utf-8", errors="replace")
            raise RuntimeError(f"{failure_message}：{stderr.strip()[:400]}")
        return completed


def build_remote_yolo_settings(config: ProjectConfig) -> RemoteYoloSettings:
    """从配置组装 lab-server YOLO 的执行参数。"""
    raw = getattr(config, "raw_inference", {}) or {}
    server = raw.get("server", {}) if isinstance(raw, dict) else {}
    return RemoteYoloSettings(
        host=str(server.get("host", "lab-server")),
        remote_dir=str(server.get("remote_dir", "/data/users/lyk/.tmp/yolo-inference")),
        activate_script=str(
            server.get("activate_script", "/data/users/lyk/activate-hazard.sh")
        ),
        yolo_binary=str(server.get("yolo_binary", config.yolo_binary)),
        weights=str(server.get("weights", config.yolo_detector_weights)),
        confidence=float(server.get("confidence", config.yolo_detector_confidence)),
        iou=float(server.get("iou", config.yolo_detector_iou)),
        image_size=int(server.get("image_size", config.yolo_detector_image_size)),
        device=str(server.get("device", config.yolo_device)),
        timeout_seconds=int(server.get("timeout_seconds", 1800)),
    )


# --------------------------------------------------------------------------- #
# 途径路由
# --------------------------------------------------------------------------- #
class InferenceRouter:
    """按 ``inference.backend`` 选择解析途径，并对外暴露统一句柄。

    三条途径的分工：

    * ``private_vlm`` → :attr:`annotator`（产出目标框 + 隐患事件）
    * ``local_yolo`` → :attr:`detector`（只产出目标框，隐患靠知识库推断）
    * ``server_yolo`` → :attr:`detector`（同上，但在 lab-server 上执行）
    * ``mock`` → 两者都可用，均为离线替身
    """

    def __init__(self, config: ProjectConfig) -> None:
        self.config = config
        self.path = config.inference_backend.lower()
        if self.path not in INFERENCE_BACKENDS:
            raise InferenceConfigError(
                f"未知的推理途径 inference.backend={self.path}；"
                f"可选：{' / '.join(INFERENCE_BACKENDS)}"
            )
        self._annotator: Any = None
        self._detector: Any = None
        self.warnings: list[str] = []

    # -- 句柄 -------------------------------------------------------------- #
    @property
    def annotator(self) -> (
        MockQwenAnnotator | OpenAICompatibleQwenAnnotator | PrivateVlmAnnotator
    ):
        if self._annotator is None:
            self._annotator = self._build_annotator()
        return self._annotator

    @property
    def detector(self) -> SidecarDetector | UltralyticsDetector | RemoteYoloDetector:
        if self._detector is None:
            self._detector = self._build_detector()
        return self._detector

    def _build_annotator(self) -> Any:
        if self.path in ("private_vlm", "mock"):
            return build_qwen_annotator(self.config)
        if self.path in ("local_yolo", "server_yolo"):
            raise InferenceConfigError(
                f"inference.backend={self.path} 不提供多模态标注能力；"
                f"隐患事件（通道 B）需要 private_vlm 或 mock。"
                f"若只想跑通道 A，请直接使用 detector。"
            )
        raise InferenceConfigError(f"未知途径：{self.path}")

    def _build_detector(self) -> Any:
        if self.path == "server_yolo":
            return RemoteYoloDetector(self.config)
        if self.path == "local_yolo":
            return build_detector(self.config)
        if self.path == "private_vlm":
            # 私有 VLM 途径不依赖本地 YOLO 权重；用 sidecar 兜底初检先验
            return build_detector(self.config)
        if self.path == "mock":
            return build_detector(self.config)
        raise InferenceConfigError(f"未知途径：{self.path}")

    # -- 自检与说明 -------------------------------------------------------- #
    def describe(self) -> dict[str, Any]:
        """给 ``doctor`` 用的途径说明与配置检查结果。"""
        info: dict[str, Any] = {
            "backend": self.path,
            "available_backends": list(INFERENCE_BACKENDS),
            "max_video_seconds": self.config.inference_max_video_seconds,
            "on_over_limit": self.config.inference_on_over_limit,
            "segment_seconds": self.config.inference_segment_seconds,
            "endpoint_policy": {
                "mode": self.config.inference_endpoint_mode,
                "allowlist": list(self.config.inference_endpoint_allowlist),
            },
            "issues": [],
            "notes": [],
        }

        # 端点出网策略：把判定结果显式暴露出来，便于 doctor 审查
        endpoint = self.config.qwen_chat_endpoint
        if endpoint:
            allowed, reason = endpoint_allowed(
                endpoint,
                self.config.inference_endpoint_mode,
                self.config.inference_endpoint_allowlist,
            )
            info["endpoint_policy"]["endpoint"] = endpoint
            info["endpoint_policy"]["allowed"] = allowed
            info["endpoint_policy"]["reason"] = reason
            if not allowed:
                info["issues"].append("endpoint_blocked_by_policy")
            elif not is_local_endpoint_host(endpoint_host_of(endpoint)):
                info["notes"].append(
                    f"端点 {endpoint_host_of(endpoint)} 是公网地址，"
                    f"已由白名单显式放行；请确认该端点可信且素材可外发"
                )

        if self.path == "private_vlm":
            info["endpoint"] = self.config.qwen_chat_endpoint
            info["model"] = self.config.qwen_model
            info["images_mode"] = self.config.qwen_images_mode
            info["auth_env"] = self.config.qwen_api_key_env
            key_file = getattr(self.config, "qwen_api_key_file", "") or ""
            info["auth_file"] = key_file
            import os

            env_token = os.environ.get(self.config.qwen_api_key_env, "").strip()
            file_ready = bool(key_file) and Path(key_file).exists()
            info["auth_token_present"] = bool(env_token) or file_ready
            info["auth_token_source"] = (
                "env" if env_token else ("file" if file_ready else "none")
            )
            if not self.config.qwen_chat_endpoint:
                info["issues"].append("private_vlm_base_url_missing")
            if not info["auth_token_present"]:
                info["issues"].append(f"auth_token_missing:{self.config.qwen_api_key_env}")
                info["notes"].append(
                    f"请设置环境变量 {self.config.qwen_api_key_env}=<Bearer Token>，"
                    f"或把 token 写入文件并在配置里指定 qwen.api_key_file"
                )
        elif self.path == "server_yolo":
            settings = build_remote_yolo_settings(self.config)
            info["server"] = {
                "host": settings.host,
                "remote_dir": settings.remote_dir,
                "activate_script": settings.activate_script,
                "weights": settings.weights or "(未配置，将回退 yolov8n.pt)",
            }
            if not shutil.which("ssh"):
                info["issues"].append("ssh_binary_missing")
            if not settings.weights:
                info["notes"].append("未配置 yolo.detector_weights，远程将使用 yolov8n.pt")
        elif self.path == "local_yolo":
            info["detector_mode"] = self.config.yolo_detector
            info["weights"] = self.config.yolo_detector_weights or "(未配置)"
            info["device"] = self.config.yolo_device
            if self.config.yolo_detector != "ultralytics":
                info["notes"].append(
                    f"当前 yolo.detector={self.config.yolo_detector}；"
                    f"local_yolo 途径建议设为 ultralytics"
                )
        else:
            info["notes"].append("mock 途径不联网，仅用于离线验证链路")

        return info

    def write_report(self, path: str | Path) -> Path:
        target = Path(path)
        ensure_dir(target.parent)
        write_json(target, self.describe())
        return target


def build_inference_router(config: ProjectConfig) -> InferenceRouter:
    return InferenceRouter(config)


def render_duration_notice(report: VideoDurationReport) -> str:
    """把时长校验结果渲染成一句人话，供 CLI 打印。"""
    if report.status == "ok":
        return f"[时长校验] 通过：{report.message}"
    if report.status == "unknown":
        return f"[时长校验] 跳过：{report.message}"
    if report.action == "segment":
        return f"[时长校验] 超限：{report.message}"
    if report.action == "warn":
        return f"[时长校验] 告警：{report.message}"
    return f"[时长校验] 拒绝：{report.message}"


# --------------------------------------------------------------------------- #
# 连通性探测（不发送任何真实素材）
# --------------------------------------------------------------------------- #
def parse_endpoint(url: str) -> tuple[str, int]:
    """从 ``http://host:port/path`` 里拆出 ``(host, port)``。"""
    from urllib.parse import urlparse

    parsed = urlparse(url if "://" in url else f"http://{url}")
    host = parsed.hostname or ""
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    return host, port


def probe_tcp(host: str, port: int, timeout: float = 8.0) -> dict[str, Any]:
    """只做 TCP 握手，不发送任何业务数据。"""
    import socket

    started = time.perf_counter()
    try:
        with socket.create_connection((host, port), timeout=timeout):
            elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
            return {"reachable": True, "elapsed_ms": elapsed_ms}
    except OSError as error:
        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        return {"reachable": False, "elapsed_ms": elapsed_ms, "error": str(error)}


def probe_http_status(url: str, timeout: float = 8.0) -> dict[str, Any]:
    """发一个不带凭据、不带图片的 GET，看服务是否在线以及是否强制鉴权。

    403/401 是好信号：说明服务在线且认证层在工作。
    """
    import urllib.error
    import urllib.request

    started = time.perf_counter()
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            status = response.status
            body = response.read(200)
        return {
            "status_code": status,
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
            "body_preview": body.decode("utf-8", errors="replace")[:120],
        }
    except urllib.error.HTTPError as error:
        return {
            "status_code": error.code,
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
            "auth_enforced": error.code in (401, 403),
            "body_preview": error.read(200).decode("utf-8", errors="replace")[:120],
        }
    except Exception as error:  # noqa: BLE001 - 探测失败要作为数据返回
        return {
            "status_code": None,
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
            "error": str(error),
        }


def build_synthetic_probe_image(path: str | Path) -> Path:
    """在本地合成一张测试图，**不读取任何真实素材**。

    画面上有明确的可描述内容（几个色块 + 一个近似人形的矩形），
    便于人工判断模型是否真的"看到了"图片，而不是在编造。
    """
    from hazard_annotation.imageio import encode_png

    width, height = 160, 120
    pixels = bytearray()
    for y in range(height):
        for x in range(width):
            # 背景：蓝灰渐变
            r = 40 + int(60 * x / width)
            g = 50 + int(60 * y / height)
            b = 70
            # 左下方：一个"人形"浅色矩形
            if 20 <= x < 45 and 60 <= y < 110:
                r, g, b = 220, 210, 190
            # 中部：一个橙色"吊钩"方块
            if 80 <= x < 110 and 30 <= y < 55:
                r, g, b = 230, 140, 40
            # 右下方：一个灰色"吊物"方块
            if 115 <= x < 150 and 75 <= y < 105:
                r, g, b = 150, 150, 155
            pixels.extend((r, g, b))

    target = Path(path)
    ensure_dir(target.parent)
    target.write_bytes(encode_png(width, height, bytes(pixels)))
    return target


def probe_inference_endpoint(
    config: ProjectConfig,
    image_path: str | Path | None = None,
    send_request: bool = False,
) -> dict[str, Any]:
    """分层探测私有 VLM 端点。

    三层，逐层加深，**任一层失败都返回结构化结果而不是抛异常**：

    1. TCP 握手 —— 不发送任何数据
    2. 无凭据 GET —— 看服务是否在线、是否强制鉴权
    3. 可选：带合成图的一次真实请求（``send_request=True``，需 token）

    第 3 层默认关闭；即便开启，用的也是本地合成的图，**不涉及任何真实素材**。
    """
    import os

    endpoint = config.qwen_chat_endpoint
    key_file = getattr(config, "qwen_api_key_file", "") or ""
    env_token = os.environ.get(config.qwen_api_key_env, "").strip()
    file_ready = bool(key_file) and Path(key_file).exists()
    token_present = bool(env_token) or file_ready

    report: dict[str, Any] = {
        "backend": config.inference_backend,
        "endpoint": endpoint,
        "model": config.qwen_model,
        "images_mode": config.qwen_images_mode,
        "auth_env": config.qwen_api_key_env,
        "auth_file": key_file,
        "auth_token_present": token_present,
        "auth_token_source": "env" if env_token else ("file" if file_ready else "none"),
        "layers": {},
        "warnings": [],
        "issues": [],
    }

    if not endpoint:
        report["issues"].append("endpoint_missing")
        report["verdict"] = "配置缺失：qwen.base_url 或 qwen.endpoint 未设置"
        return report

    host, port = parse_endpoint(endpoint)
    report["layers"]["tcp"] = probe_tcp(host, port)

    # 无凭据 GET 根路径：探活 + 看鉴权是否强制
    root_url = f"{'https' if port == 443 else 'http'}://{host}:{port}/"
    report["layers"]["http_anonymous"] = probe_http_status(root_url)

    if not report["auth_token_present"]:
        report["issues"].append(f"auth_token_missing:{config.qwen_api_key_env}")
        report["warnings"].append(
            f"未设置环境变量 {config.qwen_api_key_env}，也未找到可用的 token 文件，"
            f"无法做带认证的真实请求"
        )

    if send_request:
        if not report["auth_token_present"]:
            report["issues"].append("cannot_send_request_without_token")
        else:
            target = (
                Path(image_path)
                if image_path
                else build_synthetic_probe_image(
                    Path(config.workspace_dir) / "probe" / "synthetic_probe.png"
                )
            )
            report["layers"]["request"] = _probe_real_request(config, target)

    report["verdict"] = _render_probe_verdict(report)
    return report


def _probe_real_request(config: ProjectConfig, image_path: Path) -> dict[str, Any]:
    """发一次真实请求（用合成图），把异常转成数据返回。"""
    from traffic_cv_closed_loop.models import CandidateFrame

    annotator = build_qwen_annotator(config)
    candidate = CandidateFrame(
        source_id="probe",
        relative_path=image_path.name,
        original_path=str(image_path.resolve()),
        extracted_path=str(image_path.resolve()),
        detections=[],
        scene=config.scene,
        metadata={"probe": True, "synthetic": True},
    )
    prompt = (
        "这是一张用于连通性测试的合成图片。请只输出 JSON，"
        '格式为 {"objects": [], "hazard_report": {"scene_type": "无", '
        '"hazards": [], "overall_risk": "none", "recommendation": ""}}。'
        "不需要真的识别隐患。"
    )
    started = time.perf_counter()
    try:
        objects, hazard_report, payload = annotator.annotate_combined(candidate, prompt)
        return {
            "ok": True,
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
            "image": str(image_path),
            "synthetic": True,
            "object_count": len(objects),
            "overall_risk": hazard_report.get("overall_risk"),
            "provider": payload.get("provider"),
            "raw_response_preview": str(payload.get("response", ""))[:200],
        }
    except Exception as error:  # noqa: BLE001 - 探测失败即数据
        return {
            "ok": False,
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
            "image": str(image_path),
            "synthetic": True,
            "error": f"{type(error).__name__}: {error}",
        }


def _render_probe_verdict(report: dict[str, Any]) -> str:
    layers = report["layers"]
    tcp = layers.get("tcp", {})
    http = layers.get("http_anonymous", {})

    if not tcp.get("reachable"):
        return f"不可达：TCP {report['endpoint']} 连接失败"
    if http.get("auth_enforced"):
        if report["auth_token_present"]:
            return "服务在线且强制鉴权；token 已就位，可发真实请求"
        return "服务在线且强制鉴权；但 token 未设置，带认证的请求还发不了"
    if http.get("status_code") == 200:
        return "服务在线；未强制鉴权（可能不需要 token）"
    if http.get("status_code") is None:
        return f"TCP 可达但 HTTP 探测失败：{http.get('error', '未知原因')}"
    return f"TCP 可达；HTTP 返回 {http['status_code']}"


def render_probe_report(report: dict[str, Any]) -> str:
    """把探测报告渲染成便于阅读的多行文本。"""
    layers = report["layers"]
    source_label = {
        "env": "环境变量",
        "file": "token 文件",
        "none": "未配置",
    }.get(report.get("auth_token_source", "none"), "未配置")
    lines = [
        "=== 私有 VLM 端点探测 ===",
        f"  途径      : {report['backend']}",
        f"  端点      : {report['endpoint']}",
        f"  模型      : {report['model']}",
        f"  images    : {report['images_mode']}",
        f"  token     : {source_label}（环境变量 {report['auth_env']}）",
    ]
    if report.get("auth_file"):
        lines.append(f"  token 文件 : {report['auth_file']}")
    lines.append("")
    tcp = layers.get("tcp", {})
    if tcp:
        state = "可连接" if tcp.get("reachable") else "不可连接"
        extra = f"（{tcp.get('elapsed_ms')}ms）" if tcp.get("reachable") else f"（{tcp.get('error')}）"
        lines.append(f"  [1/3] TCP 握手      : {state}{extra}")
    http = layers.get("http_anonymous", {})
    if http:
        code = http.get("status_code")
        note = "（强制鉴权，正常）" if http.get("auth_enforced") else ""
        lines.append(f"  [2/3] 无凭据 GET    : HTTP {code}{note}")
    req = layers.get("request")
    if req:
        if req.get("ok"):
            lines.append(
                f"  [3/3] 合成图请求    : 成功（{req['elapsed_ms']}ms，"
                f"检出 {req['object_count']} 个目标，overall_risk={req['overall_risk']}）"
            )
        else:
            lines.append(f"  [3/3] 合成图请求    : 失败（{req.get('error')}）")
    else:
        lines.append("  [3/3] 合成图请求    : 未执行（加 --send-request 可执行）")

    if report["issues"]:
        lines.append("")
        lines.append("  待处理：")
        for issue in report["issues"]:
            lines.append(f"    - {issue}")
    if report["warnings"]:
        for warning in report["warnings"]:
            lines.append(f"    ! {warning}")

    lines.append("")
    lines.append(f"  结论：{report['verdict']}")
    return "\n".join(lines)
