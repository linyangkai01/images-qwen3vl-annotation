from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from traffic_cv_closed_loop.io_utils import read_json


def _resolve_path(project_dir: Path, raw_path: str | None, default: str) -> Path:
    path_value = raw_path or default
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (project_dir / path).resolve()


def _resolve_model_reference(project_dir: Path, raw_value: str | None, default: str) -> str:
    value = str(raw_value or default)
    if not value:
        return value
    if value.startswith("http://") or value.startswith("https://"):
        return value

    path = Path(value)
    looks_like_local_path = (
        path.is_absolute()
        or value.startswith(".")
        or "/" in value
        or "\\" in value
    )
    if looks_like_local_path:
        if path.is_absolute():
            return str(path)
        return str((project_dir / path).resolve())
    return value


def _resolve_video_path(project_dir: Path, raw_path: str | None) -> str:
    """Resolve an optional video path against the project directory.

    Returns an empty string when no video is configured (image-directory mode).
    """
    if not raw_path:
        return ""
    path = Path(str(raw_path))
    if path.is_absolute():
        return str(path)
    return str((project_dir / path).resolve())


def _resolve_kb_path(project_dir: Path, raw_path: str | None) -> str:
    """Resolve the hazard knowledge-base path.

    When no explicit path is configured we search the project directory and then
    every ancestor directory for ``hazard-detection/knowledge_base.json`` so that
    the bundled knowledge base is found regardless of where the project lives.
    """
    if raw_path:
        path = Path(str(raw_path))
        if path.is_absolute():
            return str(path)
        return str((project_dir / path).resolve())

    candidate = project_dir / "hazard-detection" / "knowledge_base.json"
    if candidate.exists():
        return str(candidate.resolve())
    for parent in project_dir.parents:
        candidate = parent / "hazard-detection" / "knowledge_base.json"
        if candidate.exists():
            return str(candidate.resolve())
    return str(candidate.resolve())


def _string_dict(raw: Any) -> dict[str, str]:
    if not raw:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"Expected a JSON object, got {type(raw).__name__}")
    return {str(key): str(value) for key, value in raw.items()}


def _json_dict(raw: Any) -> dict[str, Any]:
    if not raw:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"Expected a JSON object, got {type(raw).__name__}")
    return dict(raw)


DEFAULT_FORCE_KEEP_LABELS: list[str] = [
    "person",
    "hook",
    "rigging",
    "overhead_crane",
    "ladder",
    "scaffold",
    "harness",
    "load",
]

#: Two-tier execution profiles. Keys are dotted paths resolved by
#: :meth:`ProjectConfig.apply_profile`; ``qwen.*`` maps onto the flat
#: ``qwen_*`` fields, the other prefixes map onto the nested config objects.
_BUILTIN_PROFILES: dict[str, dict[str, Any]] = {
    "quick": {
        "video.sample_fps": 0.5,
        "keyframe.max_keyframes": 30,
        "qwen.timeout_seconds": 90,
        "qwen.max_retries": 1,
        "qwen.concurrency": 1,
        "qwen.max_tokens": 1500,
        "qwen.combined_call": True,
        "knowledge.kb_max_chars": 4000,
    },
    "full": {
        "video.sample_fps": 1.0,
        "keyframe.max_keyframes": 400,
        "qwen.timeout_seconds": 300,
        "qwen.max_retries": 2,
        "qwen.concurrency": 2,
        "qwen.max_tokens": 2500,
        "qwen.combined_call": True,
        "knowledge.kb_max_chars": 8000,
    },
}

_PROFILE_SECTIONS = ("video", "keyframe", "knowledge", "hazard", "visualization", "qwen")


def _flatten_profile(profile: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in profile.items():
        path = f"{prefix}{key}"
        if isinstance(value, dict) and key in _PROFILE_SECTIONS:
            flat.update(_flatten_profile(value, prefix=f"{path}."))
        else:
            flat[path] = value
    return flat


def _coerce_like(current: Any, value: Any) -> Any:
    if isinstance(current, bool):
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)
    if isinstance(current, int) and not isinstance(current, bool):
        return int(value)
    if isinstance(current, float):
        return float(value)
    if isinstance(current, list) and isinstance(value, list):
        return list(value)
    return value


@dataclass(slots=True)
class VideoConfig:
    """Video decoding parameters (ffmpeg is the only supported backend)."""

    video_path: str = ""
    sample_fps: float = 1.0
    scale_width: int = 640
    analysis_width: int = 160
    decode_backend: str = "ffmpeg"
    ffmpeg_binary: str = "ffmpeg"

    def to_dict(self) -> dict[str, Any]:
        return {
            "video_path": self.video_path,
            "sample_fps": self.sample_fps,
            "scale_width": self.scale_width,
            "analysis_width": self.analysis_width,
            "decode_backend": self.decode_backend,
            "ffmpeg_binary": self.ffmpeg_binary,
        }


@dataclass(slots=True)
class KeyframeConfig:
    """Keyframe extraction thresholds (six-step strategy)."""

    enable: bool = True
    motion_min_ratio: float = 0.002
    pixel_delta_threshold: int = 25
    dedup_hamming_threshold: int = 4
    phash_size: int = 8
    scene_change_threshold: float = 0.4
    scene_detect_mode: str = "ffmpeg"
    max_keyframes: int = 400
    min_gap_seconds: float = 2.0
    force_keep_labels: list[str] = field(default_factory=lambda: list(DEFAULT_FORCE_KEEP_LABELS))

    def to_dict(self) -> dict[str, Any]:
        return {
            "enable": self.enable,
            "motion_min_ratio": self.motion_min_ratio,
            "pixel_delta_threshold": self.pixel_delta_threshold,
            "dedup_hamming_threshold": self.dedup_hamming_threshold,
            "phash_size": self.phash_size,
            "scene_change_threshold": self.scene_change_threshold,
            "scene_detect_mode": self.scene_detect_mode,
            "max_keyframes": self.max_keyframes,
            "min_gap_seconds": self.min_gap_seconds,
            "force_keep_labels": list(self.force_keep_labels),
        }


@dataclass(slots=True)
class KnowledgeConfig:
    """Hazard knowledge-base injection parameters."""

    kb_path: str = ""
    kb_mode: str = "compressed"
    kb_max_chars: int = 8000
    scene_routing: str = "rules"
    include_ten_no_lift: bool = True
    include_crane_six_forbidden: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "kb_path": self.kb_path,
            "kb_mode": self.kb_mode,
            "kb_max_chars": self.kb_max_chars,
            "scene_routing": self.scene_routing,
            "include_ten_no_lift": self.include_ten_no_lift,
            "include_crane_six_forbidden": self.include_crane_six_forbidden,
        }


@dataclass(slots=True)
class HazardConfig:
    """Hazard report output parameters."""

    output_dir: str = ""
    severity_levels: dict[str, Any] = field(default_factory=dict)
    overall_risk_policy: str = "max"

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_dir": self.output_dir,
            "severity_levels": self.severity_levels,
            "overall_risk_policy": self.overall_risk_policy,
        }


@dataclass(slots=True)
class VisualizationConfig:
    """Self-contained HTML report parameters."""

    enable: bool = True
    output_dir: str = ""
    thumbnail_size: int = 160

    def to_dict(self) -> dict[str, Any]:
        return {
            "enable": self.enable,
            "output_dir": self.output_dir,
            "thumbnail_size": self.thumbnail_size,
        }


@dataclass(slots=True)
class ProjectConfig:
    config_path: Path
    project_dir: Path
    name: str
    scene: str
    labels: list[str]
    input_frames_dir: Path
    workspace_dir: Path
    prompt_template_path: Path
    frame_step: int
    max_frames_per_source: int
    min_detection_confidence: float
    review_confidence_threshold: float
    include_empty_frames: bool
    include_review_queue_in_training: bool
    train_split: float
    val_split: float
    test_split: float
    random_seed: int
    qwen_provider: str
    qwen_base_url: str
    qwen_endpoint: str
    qwen_request_path: str
    qwen_model: str
    qwen_api_key_env: str
    qwen_require_api_key: bool
    qwen_timeout_seconds: int
    qwen_max_retries: int
    qwen_retry_backoff_seconds: float
    qwen_temperature: float
    qwen_top_p: float
    qwen_max_tokens: int
    qwen_response_format: str
    qwen_auth_scheme: str
    qwen_headers: dict[str, str]
    qwen_header_env: dict[str, str]
    qwen_extra_body: dict[str, Any]
    yolo_binary: str
    yolo_task: str
    yolo_detector: str
    yolo_trainer: str
    yolo_weights: str
    yolo_detector_weights: str
    yolo_detector_confidence: float
    yolo_detector_iou: float
    yolo_detector_image_size: int
    yolo_image_size: int
    yolo_epochs: int
    yolo_device: str
    yolo_batch: int
    yolo_workers: int
    yolo_patience: int
    yolo_cache: bool
    yolo_resume: bool
    yolo_close_mosaic: int
    yolo_save_period: int
    # --- hazard-annotation extension (all defaulted for backward compatibility) ---
    run_mode: str = "quick"
    profiles: dict[str, Any] = field(default_factory=dict)
    video: VideoConfig = field(default_factory=VideoConfig)
    keyframe: KeyframeConfig = field(default_factory=KeyframeConfig)
    knowledge: KnowledgeConfig = field(default_factory=KnowledgeConfig)
    hazard: HazardConfig = field(default_factory=HazardConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    qwen_enable_thinking: bool = True
    qwen_strip_thinking: bool = True
    qwen_json_repair: bool = True
    qwen_combined_call: bool = True
    qwen_concurrency: int = 1

    @classmethod
    def load(cls, config_path: str | Path) -> "ProjectConfig":
        config_path = Path(config_path).resolve()
        raw = read_json(config_path)
        if raw is None:
            raise FileNotFoundError(f"Config file not found: {config_path}")

        project_dir = config_path.parent.parent.resolve()
        input_cfg = raw.get("input", {})
        output_cfg = raw.get("output", {})
        pipeline_cfg = raw.get("pipeline", {})
        qwen_cfg = raw.get("qwen", {})
        yolo_cfg = raw.get("yolo", {})
        video_cfg = raw.get("video", {})
        keyframe_cfg = raw.get("keyframe", {})
        knowledge_cfg = raw.get("knowledge", {})
        hazard_cfg = raw.get("hazard", {})
        visualization_cfg = raw.get("visualization", {})

        train_split = float(pipeline_cfg.get("train_split", 0.7))
        val_split = float(pipeline_cfg.get("val_split", 0.2))
        test_split = float(pipeline_cfg.get("test_split", 0.1))
        if min(train_split, val_split, test_split) < 0:
            raise ValueError("train_split, val_split and test_split must be >= 0")
        if train_split + val_split + test_split <= 0:
            raise ValueError("At least one of train_split, val_split or test_split must be > 0")

        video = VideoConfig(
            video_path=_resolve_video_path(project_dir, video_cfg.get("video_path")),
            sample_fps=float(video_cfg.get("sample_fps", 1.0)),
            scale_width=int(video_cfg.get("scale_width", 640)),
            analysis_width=int(video_cfg.get("analysis_width", 160)),
            decode_backend=str(video_cfg.get("decode_backend", "ffmpeg")),
            ffmpeg_binary=str(video_cfg.get("ffmpeg_binary", "ffmpeg")),
        )
        keyframe = KeyframeConfig(
            enable=bool(keyframe_cfg.get("enable", True)),
            motion_min_ratio=float(keyframe_cfg.get("motion_min_ratio", 0.002)),
            pixel_delta_threshold=int(keyframe_cfg.get("pixel_delta_threshold", 25)),
            dedup_hamming_threshold=int(keyframe_cfg.get("dedup_hamming_threshold", 4)),
            phash_size=int(keyframe_cfg.get("phash_size", 8)),
            scene_change_threshold=float(keyframe_cfg.get("scene_change_threshold", 0.4)),
            scene_detect_mode=str(keyframe_cfg.get("scene_detect_mode", "ffmpeg")),
            max_keyframes=int(keyframe_cfg.get("max_keyframes", 400)),
            min_gap_seconds=float(keyframe_cfg.get("min_gap_seconds", 2.0)),
            force_keep_labels=[
                str(item)
                for item in keyframe_cfg.get("force_keep_labels", DEFAULT_FORCE_KEEP_LABELS)
            ],
        )
        knowledge = KnowledgeConfig(
            kb_path=_resolve_kb_path(project_dir, knowledge_cfg.get("kb_path")),
            kb_mode=str(knowledge_cfg.get("kb_mode", "compressed")),
            kb_max_chars=int(knowledge_cfg.get("kb_max_chars", 8000)),
            scene_routing=str(knowledge_cfg.get("scene_routing", "rules")),
            include_ten_no_lift=bool(knowledge_cfg.get("include_ten_no_lift", True)),
            include_crane_six_forbidden=bool(knowledge_cfg.get("include_crane_six_forbidden", True)),
        )
        hazard = HazardConfig(
            output_dir=str(_resolve_path(project_dir, hazard_cfg.get("output_dir"), "workspace/hazards")),
            severity_levels=_json_dict(hazard_cfg.get("severity_levels")),
            overall_risk_policy=str(hazard_cfg.get("overall_risk_policy", "max")),
        )
        visualization = VisualizationConfig(
            enable=bool(visualization_cfg.get("enable", True)),
            output_dir=str(
                _resolve_path(project_dir, visualization_cfg.get("output_dir"), "workspace/visualization")
            ),
            thumbnail_size=int(visualization_cfg.get("thumbnail_size", 160)),
        )

        config = cls(
            config_path=config_path,
            project_dir=project_dir,
            name=str(raw.get("project_name", "traffic-cv-closed-loop")),
            scene=str(raw.get("scene", "Traffic intersection vision project")),
            labels=[str(item) for item in raw.get("labels", ["person", "car"])],
            input_frames_dir=_resolve_path(project_dir, input_cfg.get("frames_dir"), "input/frames"),
            workspace_dir=_resolve_path(project_dir, output_cfg.get("workspace_dir"), "workspace"),
            prompt_template_path=_resolve_path(
                project_dir,
                qwen_cfg.get("prompt_template"),
                "assets/prompts/qwen3vl_traffic_zh.md",
            ),
            frame_step=int(pipeline_cfg.get("frame_step", 1)),
            max_frames_per_source=int(pipeline_cfg.get("max_frames_per_source", 300)),
            min_detection_confidence=float(pipeline_cfg.get("min_detection_confidence", 0.25)),
            review_confidence_threshold=float(pipeline_cfg.get("review_confidence_threshold", 0.8)),
            include_empty_frames=bool(pipeline_cfg.get("include_empty_frames", False)),
            include_review_queue_in_training=bool(pipeline_cfg.get("include_review_queue_in_training", False)),
            train_split=train_split,
            val_split=val_split,
            test_split=test_split,
            random_seed=int(pipeline_cfg.get("seed", 7)),
            qwen_provider=str(qwen_cfg.get("provider", "mock")),
            qwen_base_url=str(qwen_cfg.get("base_url", "") or ""),
            qwen_endpoint=str(qwen_cfg.get("endpoint", "") or ""),
            qwen_request_path=str(qwen_cfg.get("request_path", "/chat/completions")),
            qwen_model=str(qwen_cfg.get("model", "Qwen/Qwen2.5-VL-72B-Instruct")),
            qwen_api_key_env=str(qwen_cfg.get("api_key_env", "QWEN_API_KEY")),
            qwen_require_api_key=bool(qwen_cfg.get("require_api_key", True)),
            qwen_timeout_seconds=int(qwen_cfg.get("timeout_seconds", 60)),
            qwen_max_retries=int(qwen_cfg.get("max_retries", 2)),
            qwen_retry_backoff_seconds=float(qwen_cfg.get("retry_backoff_seconds", 2.0)),
            qwen_temperature=float(qwen_cfg.get("temperature", 0.1)),
            qwen_top_p=float(qwen_cfg.get("top_p", 0.9)),
            qwen_max_tokens=int(qwen_cfg.get("max_tokens", 1200)),
            qwen_response_format=str(qwen_cfg.get("response_format", "json_object")),
            qwen_auth_scheme=str(qwen_cfg.get("auth_scheme", "Bearer")),
            qwen_headers=_string_dict(qwen_cfg.get("headers")),
            qwen_header_env=_string_dict(qwen_cfg.get("header_env")),
            qwen_extra_body=_json_dict(qwen_cfg.get("extra_body")),
            yolo_binary=str(yolo_cfg.get("binary", "yolo")),
            yolo_task=str(yolo_cfg.get("task", "detect")),
            yolo_detector=str(yolo_cfg.get("detector", "sidecar")),
            yolo_trainer=str(yolo_cfg.get("trainer", "mock")),
            yolo_weights=_resolve_model_reference(project_dir, yolo_cfg.get("weights"), "weights/yolo11n.pt"),
            yolo_detector_weights=_resolve_model_reference(
                project_dir,
                yolo_cfg.get("detector_weights") or yolo_cfg.get("weights"),
                "weights/yolo11n.pt",
            ),
            yolo_detector_confidence=float(yolo_cfg.get("detector_confidence", 0.25)),
            yolo_detector_iou=float(yolo_cfg.get("detector_iou", 0.45)),
            yolo_detector_image_size=int(yolo_cfg.get("detector_image_size", yolo_cfg.get("image_size", 1280))),
            yolo_image_size=int(yolo_cfg.get("image_size", 1280)),
            yolo_epochs=int(yolo_cfg.get("epochs", 50)),
            yolo_device=str(yolo_cfg.get("device", "0")),
            yolo_batch=int(yolo_cfg.get("batch", 16)),
            yolo_workers=int(yolo_cfg.get("workers", 8)),
            yolo_patience=int(yolo_cfg.get("patience", 30)),
            yolo_cache=bool(yolo_cfg.get("cache", False)),
            yolo_resume=bool(yolo_cfg.get("resume", False)),
            yolo_close_mosaic=int(yolo_cfg.get("close_mosaic", 10)),
            yolo_save_period=int(yolo_cfg.get("save_period", -1)),
            run_mode=str(raw.get("run_mode", "quick")),
            profiles=_json_dict(raw.get("profiles")),
            video=video,
            keyframe=keyframe,
            knowledge=knowledge,
            hazard=hazard,
            visualization=visualization,
            qwen_enable_thinking=bool(qwen_cfg.get("enable_thinking", True)),
            qwen_strip_thinking=bool(qwen_cfg.get("strip_thinking", True)),
            qwen_json_repair=bool(qwen_cfg.get("json_repair", True)),
            qwen_combined_call=bool(qwen_cfg.get("combined_call", True)),
            qwen_concurrency=int(qwen_cfg.get("concurrency", 1)),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if not self.labels:
            raise ValueError("labels must not be empty")
        if self.frame_step <= 0:
            raise ValueError("frame_step must be >= 1")
        if self.max_frames_per_source <= 0:
            raise ValueError("max_frames_per_source must be >= 1")
        if self.qwen_max_retries < 0:
            raise ValueError("qwen.max_retries must be >= 0")
        if self.qwen_timeout_seconds <= 0:
            raise ValueError("qwen.timeout_seconds must be >= 1")
        if self.yolo_epochs <= 0:
            raise ValueError("yolo.epochs must be >= 1")
        if self.yolo_batch == 0:
            raise ValueError("yolo.batch must not be 0")
        if self.yolo_workers < 0:
            raise ValueError("yolo.workers must be >= 0")
        # --- hazard-annotation extension validation ---
        if self.video.sample_fps <= 0:
            raise ValueError("video.sample_fps must be > 0")
        if self.video.analysis_width <= 0:
            raise ValueError("video.analysis_width must be >= 1")
        if self.keyframe.max_keyframes < 1:
            raise ValueError("keyframe.max_keyframes must be >= 1")
        if self.keyframe.phash_size < 2:
            raise ValueError("keyframe.phash_size must be >= 2")
        if self.keyframe.min_gap_seconds < 0:
            raise ValueError("keyframe.min_gap_seconds must be >= 0")
        if self.knowledge.kb_max_chars < 200:
            raise ValueError("knowledge.kb_max_chars must be >= 200")
        if self.qwen_concurrency < 1:
            raise ValueError("qwen.concurrency must be >= 1")

    def apply_profile(self, mode: str | None = None) -> None:
        """Overlay a quick/full execution profile onto the current values.

        Built-in profile defaults are applied first, then any user-provided
        ``profiles`` overrides (nested JSON) are layered on top. Values are
        coerced to the type of the existing attribute so JSON strings such as
        ``"90"`` still work.
        """
        selected = str(mode or self.run_mode or "quick").strip().lower()
        self.run_mode = selected

        settings: dict[str, Any] = {}
        settings.update(_BUILTIN_PROFILES.get(selected, {}))
        user_profile = self.profiles.get(selected) if isinstance(self.profiles, dict) else None
        if isinstance(user_profile, dict):
            settings.update(_flatten_profile(user_profile))

        for path, value in settings.items():
            self._apply_setting(path, value)

    def _apply_setting(self, path: str, value: Any) -> None:
        head, _, tail = path.partition(".")
        if not tail:
            if hasattr(self, head):
                setattr(self, head, _coerce_like(getattr(self, head), value))
            return
        if head == "qwen":
            attribute = f"qwen_{tail}"
            if hasattr(self, attribute):
                setattr(self, attribute, _coerce_like(getattr(self, attribute), value))
            return
        section = getattr(self, head, None)
        if isinstance(
            section,
            (VideoConfig, KeyframeConfig, KnowledgeConfig, HazardConfig, VisualizationConfig),
        ) and hasattr(section, tail):
            setattr(section, tail, _coerce_like(getattr(section, tail), value))

    @property
    def label_to_index(self) -> dict[str, int]:
        return {label: index for index, label in enumerate(self.labels)}

    @property
    def qwen_chat_endpoint(self) -> str:
        if self.qwen_endpoint:
            return self.qwen_endpoint
        if not self.qwen_base_url:
            return ""
        base_url = self.qwen_base_url.rstrip("/")
        request_path = self.qwen_request_path
        if not request_path.startswith("/"):
            request_path = f"/{request_path}"
        return f"{base_url}{request_path}"

    def workspace(self, *parts: str) -> Path:
        return self.workspace_dir.joinpath(*parts)

    @property
    def extracted_frames_dir(self) -> Path:
        return self.workspace("extracted", "frames")

    @property
    def candidates_path(self) -> Path:
        return self.workspace("extracted", "candidates.jsonl")

    @property
    def extraction_summary_path(self) -> Path:
        return self.workspace("extracted", "summary.json")

    @property
    def review_dir(self) -> Path:
        return self.workspace("review")

    @property
    def annotations_path(self) -> Path:
        return self.workspace("review", "annotations.jsonl")

    @property
    def review_queue_path(self) -> Path:
        return self.workspace("review", "review_queue.jsonl")

    @property
    def annotation_summary_path(self) -> Path:
        return self.workspace("review", "summary.json")

    @property
    def datasets_dir(self) -> Path:
        return self.workspace("datasets")

    @property
    def runs_dir(self) -> Path:
        return self.workspace("runs")

    @property
    def analysis_dir(self) -> Path:
        return self.workspace("analysis")

    @property
    def status_dir(self) -> Path:
        return self.workspace("status")

    @property
    def probes_dir(self) -> Path:
        return self.workspace("probes")

    @property
    def detector_runs_dir(self) -> Path:
        return self.workspace("detector_runs")

    @property
    def latest_run_path(self) -> Path:
        return self.workspace("runs", "latest.json")

    # --- hazard-annotation convenience paths ---
    @property
    def kb_path(self) -> Path:
        return Path(self.knowledge.kb_path)

    @property
    def video_path(self) -> Path | None:
        return Path(self.video.video_path) if self.video.video_path else None

    @property
    def review_labels_dir(self) -> Path:
        return self.workspace("review", "labels")

    @property
    def review_raw_responses_dir(self) -> Path:
        return self.workspace("review", "raw_responses")

    @property
    def hazards_dir(self) -> Path:
        return self.workspace("hazards")

    @property
    def hazard_report_path(self) -> Path:
        return self.workspace("hazards", "report.json")

    @property
    def visualization_dir(self) -> Path:
        return self.workspace("visualization")

    def to_dict(self) -> dict[str, Any]:
        return {
            "config_path": str(self.config_path),
            "project_dir": str(self.project_dir),
            "name": self.name,
            "scene": self.scene,
            "labels": self.labels,
            "input_frames_dir": str(self.input_frames_dir),
            "workspace_dir": str(self.workspace_dir),
            "prompt_template_path": str(self.prompt_template_path),
            "frame_step": self.frame_step,
            "max_frames_per_source": self.max_frames_per_source,
            "min_detection_confidence": self.min_detection_confidence,
            "review_confidence_threshold": self.review_confidence_threshold,
            "include_empty_frames": self.include_empty_frames,
            "include_review_queue_in_training": self.include_review_queue_in_training,
            "train_split": self.train_split,
            "val_split": self.val_split,
            "test_split": self.test_split,
            "random_seed": self.random_seed,
            "qwen_provider": self.qwen_provider,
            "qwen_base_url": self.qwen_base_url,
            "qwen_endpoint": self.qwen_endpoint,
            "qwen_chat_endpoint": self.qwen_chat_endpoint,
            "qwen_request_path": self.qwen_request_path,
            "qwen_model": self.qwen_model,
            "qwen_api_key_env": self.qwen_api_key_env,
            "qwen_require_api_key": self.qwen_require_api_key,
            "qwen_timeout_seconds": self.qwen_timeout_seconds,
            "qwen_max_retries": self.qwen_max_retries,
            "qwen_retry_backoff_seconds": self.qwen_retry_backoff_seconds,
            "qwen_temperature": self.qwen_temperature,
            "qwen_top_p": self.qwen_top_p,
            "qwen_max_tokens": self.qwen_max_tokens,
            "qwen_response_format": self.qwen_response_format,
            "qwen_auth_scheme": self.qwen_auth_scheme,
            "qwen_headers": self.qwen_headers,
            "qwen_header_env": self.qwen_header_env,
            "qwen_extra_body": self.qwen_extra_body,
            "qwen_enable_thinking": self.qwen_enable_thinking,
            "qwen_strip_thinking": self.qwen_strip_thinking,
            "qwen_json_repair": self.qwen_json_repair,
            "qwen_combined_call": self.qwen_combined_call,
            "qwen_concurrency": self.qwen_concurrency,
            "yolo_binary": self.yolo_binary,
            "yolo_task": self.yolo_task,
            "yolo_detector": self.yolo_detector,
            "yolo_trainer": self.yolo_trainer,
            "yolo_weights": self.yolo_weights,
            "yolo_detector_weights": self.yolo_detector_weights,
            "yolo_detector_confidence": self.yolo_detector_confidence,
            "yolo_detector_iou": self.yolo_detector_iou,
            "yolo_detector_image_size": self.yolo_detector_image_size,
            "yolo_image_size": self.yolo_image_size,
            "yolo_epochs": self.yolo_epochs,
            "yolo_device": self.yolo_device,
            "yolo_batch": self.yolo_batch,
            "yolo_workers": self.yolo_workers,
            "yolo_patience": self.yolo_patience,
            "yolo_cache": self.yolo_cache,
            "yolo_resume": self.yolo_resume,
            "yolo_close_mosaic": self.yolo_close_mosaic,
            "yolo_save_period": self.yolo_save_period,
            "run_mode": self.run_mode,
            "profiles": self.profiles,
            "video": self.video.to_dict(),
            "keyframe": self.keyframe.to_dict(),
            "knowledge": self.knowledge.to_dict(),
            "hazard": self.hazard.to_dict(),
            "visualization": self.visualization.to_dict(),
        }
