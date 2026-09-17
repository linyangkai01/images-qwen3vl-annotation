"""Tests for the three inference paths and the video-duration guard.

Covers the lab's private Ollama-native ``/api/generate`` adapter (base64 image
encoding, comma-joining, Bearer auth, ``response`` field parsing), the
lab-server YOLO delegation (command construction, YOLO-label → xyxy parsing),
the inference-path router, and the 1-minute duration guard that the private
multimodal model requires.

Nothing here touches the network, ffmpeg or a real model: every external call
is either absent or replaced by a stub.
"""

from __future__ import annotations

import base64
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from hazard_annotation import inference as inference_module
from hazard_annotation.inference import (
    INFERENCE_BACKENDS,
    InferenceConfigError,
    InferenceRouter,
    RemoteYoloDetector,
    RemoteYoloSettings,
    VideoDurationError,
    build_inference_router,
    build_synthetic_probe_image,
    check_video_duration,
    parse_endpoint,
    probe_inference_endpoint,
    probe_tcp,
    render_duration_notice,
    render_probe_report,
    split_duration_segments,
)
from traffic_cv_closed_loop.config import (
    ProjectConfig,
    endpoint_allowed,
    endpoint_host_of,
    endpoint_port_of,
    is_local_endpoint_host,
    normalize_allowlist_entry,
)
from traffic_cv_closed_loop.adapters.qwen import (
    MockQwenAnnotator,
    PrivateVlmAnnotator,
    build_qwen_annotator,
)
from traffic_cv_closed_loop.config import ProjectConfig
from traffic_cv_closed_loop.models import CandidateFrame

ROOT = Path(__file__).resolve().parents[1]
MOCK_CONFIG = ROOT / "examples" / "hazard_mock_project" / "config" / "project.json"
LAB_VLM_CONFIG = ROOT / "examples" / "hazard_lab_vlm_project" / "config" / "project.json"

_PNG_BYTES = b"\x89PNG\r\n\x1a\n-fake-png-payload"


def _write_config(raw: dict) -> Path:
    """Materialise a config dict as a real file so ``ProjectConfig.load`` works."""
    tmp = Path(tempfile.mkdtemp(prefix="hazard-inference-"))
    config_path = tmp / "config" / "project.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    (tmp / "input" / "frames").mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
    return config_path


def _minimal_raw(**overrides: object) -> dict:
    raw: dict = {
        "project_name": "inference-test",
        "labels": ["person", "helmet"],
        "input": {"frames_dir": "input/frames"},
        "output": {"workspace_dir": "workspace"},
        "pipeline": {"frame_step": 1, "max_frames_per_source": 4},
        "qwen": {"provider": "mock"},
        "yolo": {"detector": "sidecar", "trainer": "mock"},
    }
    raw.update(overrides)
    return raw


# --------------------------------------------------------------------------- #
# 1. Configuration surface
# --------------------------------------------------------------------------- #
class InferenceConfigTest(unittest.TestCase):
    def test_legacy_config_without_inference_section_still_loads(self) -> None:
        config = ProjectConfig.load(MOCK_CONFIG)
        self.assertEqual(config.inference_backend, "mock")
        self.assertEqual(config.qwen_images_mode, "array")
        self.assertEqual(config.inference_max_video_seconds, 60.0)
        self.assertEqual(config.inference_on_over_limit, "warn")
        self.assertEqual(config.inference_segment_seconds, 55.0)
        self.assertIsInstance(config.raw_inference, dict)

    def test_private_vlm_provider_defaults_to_lab_token_env(self) -> None:
        raw = _minimal_raw(qwen={"provider": "private_vlm"})
        config = ProjectConfig.load(_write_config(raw))
        self.assertEqual(config.qwen_api_key_env, "LAB_VLM_TOKEN")

    def test_openai_provider_keeps_qwen_api_key_env(self) -> None:
        raw = _minimal_raw(qwen={"provider": "openai_compatible"})
        config = ProjectConfig.load(_write_config(raw))
        self.assertEqual(config.qwen_api_key_env, "QWEN_API_KEY")

    def test_explicit_api_key_env_wins(self) -> None:
        raw = _minimal_raw(qwen={"provider": "private_vlm", "api_key_env": "MY_TOKEN"})
        config = ProjectConfig.load(_write_config(raw))
        self.assertEqual(config.qwen_api_key_env, "MY_TOKEN")

    def test_lab_vlm_example_config_is_wired_correctly(self) -> None:
        config = ProjectConfig.load(LAB_VLM_CONFIG)
        self.assertEqual(config.qwen_provider, "private_vlm")
        self.assertEqual(config.inference_backend, "private_vlm")
        self.assertEqual(
            config.qwen_chat_endpoint, "http://119.23.145.227:38091/api/generate"
        )
        self.assertEqual(config.qwen_model, "qwen3-vl:30b-a3b-thinking-q8_0")
        self.assertEqual(config.qwen_images_mode, "array")
        self.assertEqual(config.qwen_api_key_env, "LAB_VLM_TOKEN")
        self.assertTrue(config.qwen_require_api_key)
        self.assertEqual(config.inference_max_video_seconds, 60.0)

    def test_invalid_images_mode_rejected(self) -> None:
        raw = _minimal_raw(qwen={"provider": "private_vlm", "images_mode": "weird"})
        with self.assertRaises(ValueError):
            ProjectConfig.load(_write_config(raw))

    def test_invalid_backend_rejected(self) -> None:
        raw = _minimal_raw(inference={"backend": "not-a-path"})
        with self.assertRaises(ValueError):
            ProjectConfig.load(_write_config(raw))

    def test_invalid_over_limit_policy_rejected(self) -> None:
        raw = _minimal_raw(inference={"backend": "mock", "on_over_limit": "explode"})
        with self.assertRaises(ValueError):
            ProjectConfig.load(_write_config(raw))

    def test_segment_longer_than_limit_rejected(self) -> None:
        raw = _minimal_raw(
            inference={"backend": "mock", "max_video_seconds": 60, "segment_seconds": 90}
        )
        with self.assertRaises(ValueError):
            ProjectConfig.load(_write_config(raw))


# --------------------------------------------------------------------------- #
# 2. Private VLM adapter (Ollama-native /api/generate)
# --------------------------------------------------------------------------- #
class PrivateVlmPayloadTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = ProjectConfig.load(LAB_VLM_CONFIG)
        self.annotator = build_qwen_annotator(self.config)
        # 该配置 require_api_key=true；除「缺 token」用例外统一注入一个测试 token
        os.environ["LAB_VLM_TOKEN"] = "unit-test-token"
        self.tmp = tempfile.TemporaryDirectory()
        self.image_a = Path(self.tmp.name) / "a.png"
        self.image_b = Path(self.tmp.name) / "b.png"
        self.image_a.write_bytes(_PNG_BYTES + b"A")
        self.image_b.write_bytes(_PNG_BYTES + b"B")

    def tearDown(self) -> None:
        os.environ.pop("LAB_VLM_TOKEN", None)
        self.tmp.cleanup()

    def test_factory_returns_private_vlm_annotator(self) -> None:
        self.assertIsInstance(self.annotator, PrivateVlmAnnotator)
        self.assertEqual(self.annotator.provider_label, "private_vlm")

    def test_request_payload_matches_documented_shape(self) -> None:
        payload, preview = self.annotator._build_request_payload("描述图片的内容", self.image_a)
        self.assertEqual(payload["model"], "qwen3-vl:30b-a3b-thinking-q8_0")
        self.assertEqual(payload["prompt"], "描述图片的内容")
        self.assertIs(payload["stream"], False)
        self.assertIsInstance(payload["options"], dict)
        self.assertIn("temperature", payload["options"])
        self.assertIn("max_tokens", payload["options"])
        self.assertNotIn("messages", payload)
        self.assertIsInstance(payload["images"], list)
        self.assertEqual(preview["image_count"], 1)

    def test_images_are_raw_base64_without_data_url_prefix(self) -> None:
        images = self.annotator.encode_images([self.image_a])
        self.assertEqual(len(images), 1)
        self.assertFalse(images[0].startswith("data:"))
        self.assertEqual(base64.b64decode(images[0]), _PNG_BYTES + b"A")

    def test_array_mode_returns_one_entry_per_image(self) -> None:
        self.config.qwen_images_mode = "array"
        images = self.annotator.encode_images([self.image_a, self.image_b])
        self.assertIsInstance(images, list)
        self.assertEqual(len(images), 2)
        self.assertEqual(base64.b64decode(images[1]), _PNG_BYTES + b"B")

    def test_comma_joined_mode_joins_with_english_comma(self) -> None:
        self.config.qwen_images_mode = "comma_joined"
        joined = self.annotator.encode_images([self.image_a, self.image_b])
        self.assertIsInstance(joined, str)
        self.assertEqual(joined.count(","), 1)
        first, second = joined.split(",")
        self.assertEqual(base64.b64decode(first), _PNG_BYTES + b"A")
        self.assertEqual(base64.b64decode(second), _PNG_BYTES + b"B")

    def test_comma_joined_mode_reaches_the_payload(self) -> None:
        self.config.qwen_images_mode = "comma_joined"
        payload, _ = self.annotator._build_request_payload_multi(
            "描述图片的内容", [self.image_a, self.image_b]
        )
        self.assertIsInstance(payload["images"], str)
        self.assertEqual(payload["images"].count(","), 1)

    def test_bearer_token_is_injected_from_env(self) -> None:
        os.environ["LAB_VLM_TOKEN"] = "unit-test-token"
        try:
            headers = self.annotator._build_headers()
        finally:
            os.environ.pop("LAB_VLM_TOKEN", None)
        self.assertEqual(headers["Authorization"], "Bearer unit-test-token")
        self.assertEqual(headers["Content-Type"], "application/json")

    def test_missing_token_raises_when_required(self) -> None:
        os.environ.pop("LAB_VLM_TOKEN", None)
        with self.assertRaises(ValueError) as ctx:
            self.annotator._build_headers()
        self.assertIn("LAB_VLM_TOKEN", str(ctx.exception))

    def test_annotate_multi_parses_response_field(self) -> None:
        """The Ollama-native reply lives in ``response``, not ``choices``."""
        body = {
            "response": json.dumps(
                {
                    "objects": [
                        {"label": "person", "bbox": [0.1, 0.1, 0.5, 0.9], "confidence": 0.9}
                    ],
                    "hazard_report": {
                        "scene_type": "吊装作业",
                        "hazards": [
                            {
                                "id": "D-5-3",
                                "evidence": "人员站在吊物下方",
                                "confidence": 0.8,
                            }
                        ],
                        "overall_risk": "critical",
                    },
                }
            ),
            "done": True,
            "done_reason": "stop",
        }
        captured: dict = {}

        def fake_post(payload, headers):
            captured["payload"] = payload
            return body

        self.annotator._post_json = fake_post  # type: ignore[assignment]
        detections, hazard_report, payload = self.annotator.annotate_multi(
            [self.image_a, self.image_b], "描述图片的内容"
        )

        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].label, "person")
        self.assertEqual(hazard_report["overall_risk"], "critical")
        self.assertEqual(payload["provider"], "private_vlm")
        self.assertEqual(captured["payload"]["prompt"], "描述图片的内容")

    def test_missing_response_field_raises_clear_error(self) -> None:
        self.annotator._post_json = lambda payload, headers: {"choices": []}  # type: ignore[assignment]
        with self.assertRaises(ValueError) as ctx:
            self.annotator._request_multi("prompt", [self.image_a])
        self.assertIn("response", str(ctx.exception))

    def test_mock_provider_is_unaffected(self) -> None:
        mock_config = ProjectConfig.load(MOCK_CONFIG)
        self.assertIsInstance(build_qwen_annotator(mock_config), MockQwenAnnotator)


# --------------------------------------------------------------------------- #
# 3. Token resolution: env var first, then token file
# --------------------------------------------------------------------------- #
class TokenResolutionTest(unittest.TestCase):
    """token 的两种来源：环境变量优先，取不到时回退到 token 文件。

    为什么要有文件方式：环境变量**不跨进程/终端共享**。用户在自己的终端里
    ``export``，而探测、标注、训练各自是独立进程，看不到那个变量。文件方式
    没有这个问题，且天然落在 .gitignore 覆盖范围内。
    """

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.project = Path(self.tmp.name) / "proj"
        (self.project / "config").mkdir(parents=True, exist_ok=True)
        (self.project / "input" / "frames").mkdir(parents=True, exist_ok=True)
        os.environ.pop("LAB_VLM_TOKEN", None)

    def tearDown(self) -> None:
        os.environ.pop("LAB_VLM_TOKEN", None)
        self.tmp.cleanup()

    def _config(self, api_key_file: str | None = None) -> ProjectConfig:
        raw = json.loads(LAB_VLM_CONFIG.read_text(encoding="utf-8"))
        if api_key_file is None:
            raw["qwen"].pop("api_key_file", None)
        else:
            raw["qwen"]["api_key_file"] = api_key_file
        (self.project / "config" / "project.json").write_text(
            json.dumps(raw, ensure_ascii=False), encoding="utf-8"
        )
        return ProjectConfig.load(self.project / "config" / "project.json")

    def test_api_key_file_resolves_relative_to_project(self) -> None:
        config = self._config("token.txt")
        self.assertEqual(Path(config.qwen_api_key_file), self.project / "token.txt")

    def test_no_api_key_file_leaves_it_empty(self) -> None:
        config = self._config(None)
        self.assertEqual(config.qwen_api_key_file, "")

    def test_token_read_from_file(self) -> None:
        (self.project / "token.txt").write_text("file-token-abc\n", encoding="utf-8")
        annotator = build_qwen_annotator(self._config("token.txt"))
        headers = annotator._build_headers()
        self.assertEqual(headers["Authorization"], "Bearer file-token-abc")

    def test_environment_variable_takes_priority(self) -> None:
        (self.project / "token.txt").write_text("file-token\n", encoding="utf-8")
        os.environ["LAB_VLM_TOKEN"] = "env-token"
        try:
            annotator = build_qwen_annotator(self._config("token.txt"))
            headers = annotator._build_headers()
        finally:
            os.environ.pop("LAB_VLM_TOKEN", None)
        self.assertEqual(headers["Authorization"], "Bearer env-token")

    def test_first_non_empty_line_wins(self) -> None:
        (self.project / "token.txt").write_text(
            "\n\n   real-token   \nsecond-line\n", encoding="utf-8"
        )
        annotator = build_qwen_annotator(self._config("token.txt"))
        headers = annotator._build_headers()
        self.assertEqual(headers["Authorization"], "Bearer real-token")

    def test_missing_token_file_raises_with_path(self) -> None:
        annotator = build_qwen_annotator(self._config("nope.txt"))
        with self.assertRaises(ValueError) as ctx:
            annotator._build_headers()
        message = str(ctx.exception)
        self.assertIn("token 文件也不存在", message)
        self.assertIn("nope.txt", message)
        # 报错要同时给出两种可行的配置方式
        self.assertIn("LAB_VLM_TOKEN", message)

    def test_empty_token_file_raises(self) -> None:
        (self.project / "token.txt").write_text("\n\n   \n", encoding="utf-8")
        annotator = build_qwen_annotator(self._config("token.txt"))
        with self.assertRaises(ValueError) as ctx:
            annotator._build_headers()
        self.assertIn("token 文件是空的", str(ctx.exception))

    def test_error_mentions_both_sources(self) -> None:
        # 配了 api_key_file 但文件不存在 → 报错要点出文件路径与替代方案
        annotator = build_qwen_annotator(self._config("token.txt"))
        with self.assertRaises(ValueError) as ctx:
            annotator._build_headers()
        message = str(ctx.exception)
        self.assertIn("token 文件也不存在", message)
        self.assertIn("LAB_VLM_TOKEN", message)

        # 没配 api_key_file → 报错要同时给出环境变量与文件两种配置方式
        annotator2 = build_qwen_annotator(self._config(None))
        with self.assertRaises(ValueError) as ctx2:
            annotator2._build_headers()
        message2 = str(ctx2.exception)
        self.assertIn("LAB_VLM_TOKEN", message2)
        self.assertIn("api_key_file", message2)

    def test_describe_reports_token_source(self) -> None:
        (self.project / "token.txt").write_text("file-token\n", encoding="utf-8")
        info = InferenceRouter(self._config("token.txt")).describe()
        self.assertTrue(info["auth_token_present"])
        self.assertEqual(info["auth_token_source"], "file")

    def test_describe_reports_env_source(self) -> None:
        os.environ["LAB_VLM_TOKEN"] = "env-token"
        try:
            info = InferenceRouter(self._config(None)).describe()
        finally:
            os.environ.pop("LAB_VLM_TOKEN", None)
        self.assertTrue(info["auth_token_present"])
        self.assertEqual(info["auth_token_source"], "env")

    def test_describe_reports_none_when_unconfigured(self) -> None:
        info = InferenceRouter(self._config(None)).describe()
        self.assertFalse(info["auth_token_present"])
        self.assertEqual(info["auth_token_source"], "none")

    def test_probe_accepts_token_from_file(self) -> None:
        (self.project / "token.txt").write_text("file-token\n", encoding="utf-8")
        config = self._config("token.txt")
        with mock.patch.object(
            inference_module, "probe_tcp", return_value={"reachable": True, "elapsed_ms": 1.0}
        ), mock.patch.object(
            inference_module,
            "probe_http_status",
            return_value={"status_code": 403, "auth_enforced": True},
        ):
            report = probe_inference_endpoint(config)
        self.assertTrue(report["auth_token_present"])
        self.assertEqual(report["auth_token_source"], "file")


# --------------------------------------------------------------------------- #
# 4. Remote (lab-server) YOLO delegation
# --------------------------------------------------------------------------- #
class RemoteYoloTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = ProjectConfig.load(LAB_VLM_CONFIG)
        self.config.labels = ["person", "helmet"]
        self.settings = RemoteYoloSettings(
            host="lab-server",
            remote_dir="/data/users/lyk/.tmp/yolo-inference",
            weights="weights/hazard_yolo.pt",
            device="0",
        )
        self.commands: list[list[str]] = []

    def _runner(self, command: list[str]) -> subprocess.CompletedProcess[bytes]:
        self.commands.append(command)
        return subprocess.CompletedProcess(command, 0, b"", b"")

    def test_predict_command_carries_key_options(self) -> None:
        detector = RemoteYoloDetector(self.config, self.settings, runner=self._runner)
        command = detector.build_predict_command()
        self.assertIn("yolo predict", command)
        self.assertIn("model=weights/hazard_yolo.pt", command)
        self.assertIn("source=images", command)
        self.assertIn("save_txt=True", command)
        self.assertIn("device=0", command)
        self.assertIn("activate-hazard.sh", command)

    def test_ssh_command_wraps_the_remote_script(self) -> None:
        detector = RemoteYoloDetector(self.config, self.settings, runner=self._runner)
        command = detector.build_ssh_command("echo hi")
        self.assertEqual(command, ["ssh", "lab-server", "echo hi"])

    def test_upload_and_download_commands(self) -> None:
        detector = RemoteYoloDetector(self.config, self.settings, runner=self._runner)
        uploads = detector.build_upload_commands([Path("a.png"), Path("b.png")])
        self.assertEqual(len(uploads), 2)
        self.assertTrue(all(cmd[0] == "scp" for cmd in uploads))
        download = detector.build_download_command()
        self.assertEqual(download[0], "scp")
        self.assertIn("labels/", download[-2])

    def test_label_parsing_converts_cxcywh_to_xyxy(self) -> None:
        detector = RemoteYoloDetector(self.config, self.settings, runner=self._runner)
        with tempfile.TemporaryDirectory() as tmp:
            label = Path(tmp) / "frame.txt"
            # class 0, centre (0.5, 0.5), size (0.4, 0.2), conf 0.9
            label.write_text("0 0.5 0.5 0.4 0.2 0.9\n", encoding="utf-8")
            detections = detector._parse_label_file(label)
        self.assertEqual(len(detections), 1)
        bbox = detections[0].bbox
        self.assertAlmostEqual(bbox[0], 0.3, places=5)
        self.assertAlmostEqual(bbox[1], 0.4, places=5)
        self.assertAlmostEqual(bbox[2], 0.7, places=5)
        self.assertAlmostEqual(bbox[3], 0.6, places=5)
        self.assertEqual(detections[0].label, "person")
        self.assertEqual(detections[0].source, "yolo-lab-server")

    def test_out_of_range_class_is_skipped_and_warned(self) -> None:
        detector = RemoteYoloDetector(self.config, self.settings, runner=self._runner)
        with tempfile.TemporaryDirectory() as tmp:
            label = Path(tmp) / "frame.txt"
            label.write_text("9 0.5 0.5 0.1 0.1 0.5\n", encoding="utf-8")
            detections = detector._parse_label_file(label)
        self.assertEqual(detections, [])
        self.assertTrue(any("out_of_range" in w for w in detector.warnings))

    def test_missing_label_file_warns_without_raising(self) -> None:
        detector = RemoteYoloDetector(self.config, self.settings, runner=self._runner)
        detections = detector._parse_label_file(Path("definitely-not-here.txt"))
        self.assertEqual(detections, [])
        self.assertTrue(any("missing" in w for w in detector.warnings))

    def test_failed_remote_command_raises_with_stderr(self) -> None:
        def failing_runner(command: list[str]) -> subprocess.CompletedProcess[bytes]:
            return subprocess.CompletedProcess(command, 1, b"", b"ssh: connect failed")

        detector = RemoteYoloDetector(self.config, self.settings, runner=failing_runner)
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "a.png"
            image.write_bytes(_PNG_BYTES)
            with self.assertRaises(RuntimeError) as ctx:
                detector.detect_many([image])
        self.assertIn("ssh: connect failed", str(ctx.exception))


# --------------------------------------------------------------------------- #
# 4. Video-duration guard (1-minute private-model limit)
# --------------------------------------------------------------------------- #
class _StubProbe:
    def __init__(self, duration: float) -> None:
        self.duration = duration

    def probe_duration(self, path: object) -> float:
        return self.duration


class _FailingProbe:
    def probe_duration(self, path: object) -> float:
        raise RuntimeError("ffmpeg unavailable")


class VideoDurationGuardTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = ProjectConfig.load(MOCK_CONFIG)
        self.config.inference_max_video_seconds = 60.0

    def test_within_limit_proceeds(self) -> None:
        report = check_video_duration("x.mp4", self.config, runner=_StubProbe(45.0))
        self.assertEqual(report.status, "ok")
        self.assertEqual(report.action, "proceed")
        self.assertFalse(report.over_limit)
        self.assertIn("通过", render_duration_notice(report))

    def test_exactly_at_limit_proceeds(self) -> None:
        report = check_video_duration("x.mp4", self.config, runner=_StubProbe(60.0))
        self.assertEqual(report.status, "ok")

    def test_over_limit_warn_policy(self) -> None:
        self.config.inference_on_over_limit = "warn"
        report = check_video_duration("x.mp4", self.config, runner=_StubProbe(3600.0))
        self.assertEqual(report.status, "over_limit")
        self.assertEqual(report.action, "warn")
        self.assertIn("video_over_limit_warned", report.warnings)
        self.assertIn("告警", render_duration_notice(report))

    def test_over_limit_error_policy_raises(self) -> None:
        self.config.inference_on_over_limit = "error"
        with self.assertRaises(VideoDurationError) as ctx:
            check_video_duration("x.mp4", self.config, runner=_StubProbe(3600.0))
        self.assertIn("超过上限", str(ctx.exception))

    def test_over_limit_segment_policy_computes_segment_count(self) -> None:
        self.config.inference_on_over_limit = "segment"
        self.config.inference_segment_seconds = 55.0
        report = check_video_duration("x.mp4", self.config, runner=_StubProbe(3600.0))
        self.assertEqual(report.action, "segment")
        self.assertEqual(report.segment_count, 66)
        self.assertIn("66 段", render_duration_notice(report))

    def test_unknown_duration_is_not_treated_as_over_limit(self) -> None:
        report = check_video_duration("x.mp4", self.config, runner=_StubProbe(0.0))
        self.assertEqual(report.status, "unknown")
        self.assertEqual(report.action, "proceed")
        self.assertIn("跳过", render_duration_notice(report))

    def test_probe_failure_is_survivable(self) -> None:
        report = check_video_duration("x.mp4", self.config, runner=_FailingProbe())
        self.assertEqual(report.status, "unknown")
        self.assertEqual(report.action, "proceed")

    def test_report_serialises(self) -> None:
        report = check_video_duration("x.mp4", self.config, runner=_StubProbe(30.0))
        payload = report.to_dict()
        self.assertEqual(payload["duration_seconds"], 30.0)
        self.assertEqual(payload["limit_seconds"], 60.0)
        self.assertIn("warnings", payload)

    def test_split_duration_segments(self) -> None:
        self.assertEqual(
            split_duration_segments(125.0, 55.0),
            [(0.0, 55.0), (55.0, 110.0), (110.0, 125.0)],
        )
        self.assertEqual(split_duration_segments(55.0, 55.0), [(0.0, 55.0)])
        self.assertEqual(split_duration_segments(0.0, 55.0), [])
        with self.assertRaises(ValueError):
            split_duration_segments(10.0, 0.0)


# --------------------------------------------------------------------------- #
# 5. Inference-path router
# --------------------------------------------------------------------------- #
class InferenceRouterTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = ProjectConfig.load(LAB_VLM_CONFIG)

    def test_backend_list_is_the_documented_set(self) -> None:
        self.assertEqual(
            INFERENCE_BACKENDS, ("mock", "local_yolo", "server_yolo", "private_vlm")
        )

    def test_every_backend_can_be_selected(self) -> None:
        for backend in INFERENCE_BACKENDS:
            self.config.inference_backend = backend
            router = InferenceRouter(self.config)
            self.assertEqual(router.describe()["backend"], backend)

    def test_unknown_backend_rejected(self) -> None:
        self.config.inference_backend = "telepathy"
        with self.assertRaises(InferenceConfigError):
            InferenceRouter(self.config)

    def test_yolo_paths_do_not_offer_channel_b_annotation(self) -> None:
        for backend in ("local_yolo", "server_yolo"):
            self.config.inference_backend = backend
            router = InferenceRouter(self.config)
            with self.assertRaises(InferenceConfigError) as ctx:
                _ = router.annotator
            self.assertIn("通道 B", str(ctx.exception))

    def test_private_vlm_annotator_is_available(self) -> None:
        self.config.inference_backend = "private_vlm"
        router = InferenceRouter(self.config)
        self.assertIsInstance(router.annotator, PrivateVlmAnnotator)

    def test_server_yolo_detector_is_available(self) -> None:
        self.config.inference_backend = "server_yolo"
        router = InferenceRouter(self.config)
        self.assertIsInstance(router.detector, RemoteYoloDetector)

    def test_describe_flags_missing_token(self) -> None:
        os.environ.pop("LAB_VLM_TOKEN", None)
        self.config.inference_backend = "private_vlm"
        info = InferenceRouter(self.config).describe()
        self.assertFalse(info["auth_token_present"])
        self.assertTrue(any("auth_token_missing" in issue for issue in info["issues"]))
        self.assertTrue(any("LAB_VLM_TOKEN" in note for note in info["notes"]))

    def test_describe_reports_token_when_present(self) -> None:
        os.environ["LAB_VLM_TOKEN"] = "abc"
        try:
            self.config.inference_backend = "private_vlm"
            info = InferenceRouter(self.config).describe()
        finally:
            os.environ.pop("LAB_VLM_TOKEN", None)
        self.assertTrue(info["auth_token_present"])

    def test_describe_exposes_duration_policy(self) -> None:
        info = InferenceRouter(self.config).describe()
        self.assertEqual(info["max_video_seconds"], 60.0)
        self.assertEqual(info["on_over_limit"], "warn")

    def test_write_report_persists_describe_output(self) -> None:
        router = build_inference_router(self.config)
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "inference.json"
            router.write_report(target)
            payload = json.loads(target.read_text(encoding="utf-8"))
        self.assertEqual(payload["backend"], "private_vlm")


# --------------------------------------------------------------------------- #
# 6. Endpoint egress policy (privacy guard)
# --------------------------------------------------------------------------- #
class EndpointPolicyTest(unittest.TestCase):
    """默认只放行本机/内网；公网端点必须在配置里显式声明。

    这层护栏的作用是：让「把真实素材发到公网」从"改一个字段"变成
    "必须显式写进白名单"，从而留下可审计的痕迹。
    """

    def test_local_hosts_are_recognised(self) -> None:
        for host in (
            "localhost",
            "127.0.0.1",
            "::1",
            "10.0.0.5",
            "172.16.3.9",
            "172.31.255.254",
            "192.168.1.10",
            "169.254.1.1",
            "model.local",
            "svc.internal",
        ):
            self.assertTrue(is_local_endpoint_host(host), host)

    def test_public_hosts_are_not_local(self) -> None:
        for host in (
            "119.23.145.227",
            "8.8.8.8",
            "172.32.0.1",  # 刚好越过私有段上界
            "172.15.0.1",  # 刚好低于私有段下界
            "api.openai.com",
            "dashscope.aliyuncs.com",
        ):
            self.assertFalse(is_local_endpoint_host(host), host)

    def test_host_and_port_extraction(self) -> None:
        self.assertEqual(endpoint_host_of("http://119.23.145.227:38091/api/generate"), "119.23.145.227")
        self.assertEqual(endpoint_port_of("http://119.23.145.227:38091/api/generate"), 38091)
        self.assertEqual(endpoint_host_of("https://example.org/v1"), "example.org")
        self.assertEqual(endpoint_port_of("https://example.org/v1"), 443)
        self.assertEqual(endpoint_host_of(""), "")
        self.assertIsNone(endpoint_port_of(""))

    def test_allowlist_entry_normalisation(self) -> None:
        self.assertEqual(normalize_allowlist_entry("example.org"), ("example.org", None))
        self.assertEqual(normalize_allowlist_entry("Example.ORG:8080"), ("example.org", 8080))
        self.assertEqual(
            normalize_allowlist_entry("http://host.test:9000/x"), ("host.test", 9000)
        )
        self.assertEqual(normalize_allowlist_entry("   "), ("", None))

    def test_local_only_allows_local_and_blocks_public(self) -> None:
        ok, _ = endpoint_allowed("http://localhost:11434/v1", "local_only", [])
        self.assertTrue(ok)
        ok, reason = endpoint_allowed("http://119.23.145.227:38091/api/generate", "local_only", [])
        self.assertFalse(ok)
        self.assertIn("local_only", reason)

    def test_allowlist_mode_permits_listed_host(self) -> None:
        ok, _ = endpoint_allowed(
            "http://119.23.145.227:38091/api/generate",
            "allowlist",
            ["119.23.145.227"],
        )
        self.assertTrue(ok)

    def test_allowlist_mode_rejects_unlisted_host(self) -> None:
        ok, reason = endpoint_allowed(
            "http://119.23.145.227:38091/api/generate",
            "allowlist",
            ["other.example.com"],
        )
        self.assertFalse(ok)
        self.assertIn("allowlist", reason)

    def test_allowlist_port_must_match_when_specified(self) -> None:
        ok, _ = endpoint_allowed("http://host.test:9000/x", "allowlist", ["host.test:9000"])
        self.assertTrue(ok)
        ok, _ = endpoint_allowed("http://host.test:9001/x", "allowlist", ["host.test:9000"])
        self.assertFalse(ok)

    def test_any_mode_permits_everything(self) -> None:
        ok, _ = endpoint_allowed("http://8.8.8.8/x", "any", [])
        self.assertTrue(ok)

    def test_empty_endpoint_is_allowed(self) -> None:
        ok, _ = endpoint_allowed("", "local_only", [])
        self.assertTrue(ok)

    # -- 通过真实配置验证护栏 ------------------------------------------------ #
    def test_public_endpoint_without_allowlist_is_rejected(self) -> None:
        raw = _minimal_raw(
            qwen={"provider": "private_vlm", "base_url": "http://119.23.145.227:9000"}
        )
        with self.assertRaises(ValueError) as ctx:
            ProjectConfig.load(_write_config(raw))
        message = str(ctx.exception)
        self.assertIn("端点出网策略拒绝", message)
        self.assertIn("endpoint_policy", message)

    def test_public_endpoint_with_allowlist_loads(self) -> None:
        raw = _minimal_raw(
            qwen={"provider": "private_vlm", "base_url": "http://119.23.145.227:9000"},
            inference={
                "backend": "private_vlm",
                "endpoint_policy": {"mode": "allowlist", "allowlist": ["119.23.145.227"]},
            },
        )
        config = ProjectConfig.load(_write_config(raw))
        self.assertEqual(config.inference_endpoint_mode, "allowlist")
        self.assertEqual(config.inference_endpoint_allowlist, ["119.23.145.227"])

    def test_invalid_policy_mode_rejected(self) -> None:
        raw = _minimal_raw(inference={"backend": "mock", "endpoint_policy": {"mode": "yolo"}})
        with self.assertRaises(ValueError):
            ProjectConfig.load(_write_config(raw))

    def test_shipped_lab_vlm_config_declares_its_allowlist(self) -> None:
        config = ProjectConfig.load(LAB_VLM_CONFIG)
        self.assertEqual(config.inference_endpoint_mode, "allowlist")
        self.assertIn("119.23.145.227", config.inference_endpoint_allowlist)

    def test_shipped_local_config_needs_no_allowlist(self) -> None:
        config = ProjectConfig.load(
            ROOT / "examples" / "hazard_project" / "config" / "project.json"
        )
        self.assertEqual(config.inference_endpoint_mode, "local_only")
        self.assertEqual(config.inference_endpoint_allowlist, [])


# --------------------------------------------------------------------------- #
# 7. Endpoint probe (no real material, no live network in tests)
# --------------------------------------------------------------------------- #
class EndpointProbeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = ProjectConfig.load(LAB_VLM_CONFIG)
        os.environ.pop("LAB_VLM_TOKEN", None)

    def test_parse_endpoint_with_explicit_port(self) -> None:
        self.assertEqual(
            parse_endpoint("http://119.23.145.227:38091/api/generate"),
            ("119.23.145.227", 38091),
        )

    def test_parse_endpoint_defaults(self) -> None:
        self.assertEqual(parse_endpoint("http://localhost/v1"), ("localhost", 80))
        self.assertEqual(parse_endpoint("https://example.org/x"), ("example.org", 443))

    def test_synthetic_probe_image_is_a_real_png(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "probe" / "synthetic.png"
            result = build_synthetic_probe_image(target)
            payload = result.read_bytes()
        self.assertEqual(result, target)
        self.assertTrue(payload.startswith(b"\x89PNG\r\n\x1a\n"))
        self.assertGreater(len(payload), 500)

    def test_probe_tcp_reports_unreachable_without_raising(self) -> None:
        # port 1 on localhost should refuse immediately; no external traffic
        result = probe_tcp("127.0.0.1", 1, timeout=2.0)
        self.assertFalse(result["reachable"])
        self.assertIn("error", result)

    def test_missing_endpoint_is_reported_not_raised(self) -> None:
        self.config.qwen_base_url = ""
        self.config.qwen_endpoint = ""
        report = probe_inference_endpoint(self.config)
        self.assertIn("endpoint_missing", report["issues"])
        self.assertIn("配置缺失", report["verdict"])

    def test_probe_flags_missing_token(self) -> None:
        with mock.patch.object(inference_module, "probe_tcp", return_value={"reachable": True, "elapsed_ms": 1.0}), \
             mock.patch.object(
                 inference_module,
                 "probe_http_status",
                 return_value={"status_code": 403, "auth_enforced": True},
             ):
            report = probe_inference_endpoint(self.config)
        self.assertFalse(report["auth_token_present"])
        self.assertTrue(any("auth_token_missing" in i for i in report["issues"]))
        self.assertIn("token 未设置", report["verdict"])

    def test_probe_verdict_when_reachable_and_token_present(self) -> None:
        os.environ["LAB_VLM_TOKEN"] = "abc"
        try:
            with mock.patch.object(inference_module, "probe_tcp", return_value={"reachable": True, "elapsed_ms": 1.0}), \
                 mock.patch.object(
                     inference_module,
                     "probe_http_status",
                     return_value={"status_code": 403, "auth_enforced": True},
                 ):
                report = probe_inference_endpoint(self.config)
        finally:
            os.environ.pop("LAB_VLM_TOKEN", None)
        self.assertTrue(report["auth_token_present"])
        self.assertIn("可发真实请求", report["verdict"])

    def test_probe_verdict_when_unreachable(self) -> None:
        with mock.patch.object(
            inference_module, "probe_tcp", return_value={"reachable": False, "error": "timeout"}
        ), mock.patch.object(
            inference_module, "probe_http_status", return_value={"status_code": None}
        ):
            report = probe_inference_endpoint(self.config)
        self.assertIn("不可达", report["verdict"])

    def test_send_request_without_token_does_not_call_the_model(self) -> None:
        with mock.patch.object(inference_module, "probe_tcp", return_value={"reachable": True, "elapsed_ms": 1.0}), \
             mock.patch.object(
                 inference_module,
                 "probe_http_status",
                 return_value={"status_code": 403, "auth_enforced": True},
             ):
            report = probe_inference_endpoint(self.config, send_request=True)
        self.assertIn("cannot_send_request_without_token", report["issues"])
        self.assertNotIn("request", report["layers"])

    def test_render_probe_report_lists_all_three_layers(self) -> None:
        with mock.patch.object(inference_module, "probe_tcp", return_value={"reachable": True, "elapsed_ms": 1.0}), \
             mock.patch.object(
                 inference_module,
                 "probe_http_status",
                 return_value={"status_code": 403, "auth_enforced": True},
             ):
            report = probe_inference_endpoint(self.config)
        text = render_probe_report(report)
        self.assertIn("[1/3]", text)
        self.assertIn("[2/3]", text)
        self.assertIn("[3/3]", text)
        self.assertIn("强制鉴权", text)


if __name__ == "__main__":
    unittest.main()
