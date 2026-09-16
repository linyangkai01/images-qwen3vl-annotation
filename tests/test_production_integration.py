from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from traffic_cv_closed_loop.adapters.yolo import UltralyticsTrainer
from traffic_cv_closed_loop.config import ProjectConfig
from traffic_cv_closed_loop.diagnostics import build_doctor_report, probe_detector, probe_qwen_service
from traffic_cv_closed_loop.io_utils import ensure_dir, write_json, write_text


ROOT = Path(__file__).resolve().parents[1]

class ProductionIntegrationTest(unittest.TestCase):
    def test_qwen_chat_endpoint_can_be_built_from_base_url(self) -> None:
        config = ProjectConfig.load(ROOT / "examples" / "production_project" / "config" / "project.production.json")
        if config.qwen_endpoint:
            self.assertEqual(config.qwen_chat_endpoint, config.qwen_endpoint)
        else:
            expected = f"{config.qwen_base_url.rstrip('/')}/{config.qwen_request_path.lstrip('/')}"
            self.assertEqual(config.qwen_chat_endpoint, expected)
    
    def test_doctor_reports_missing_credentials_and_weights(self) -> None:
        config = ProjectConfig.load(ROOT / "examples" / "production_project" / "config" / "project.production.json")
        report = build_doctor_report(config)
        self.assertIn(report["status"], {"ok", "issues_found"})
        if config.qwen_require_api_key and config.qwen_api_key_env not in os.environ:
            self.assertIn("qwen_api_key_missing", report["issues"])
        for header_name, env_name in config.qwen_header_env.items():
            if env_name not in os.environ:
                self.assertIn(f"qwen_header_env_missing:{header_name}", report["issues"])
        if not Path(config.yolo_detector_weights).exists():
            self.assertIn("yolo_detector_weights_missing", report["issues"])

    def test_probe_commands_work_with_mock_config(self) -> None:
        config = ProjectConfig.load(ROOT / "examples" / "mock_project" / "config" / "project.json")
        qwen_report = probe_qwen_service(config)
        detector_report = probe_detector(config)
    
        self.assertEqual(qwen_report["status"], "ok")
        self.assertGreaterEqual(qwen_report["detection_count"], 1)
        self.assertEqual(detector_report["status"], "ok")
        self.assertGreaterEqual(detector_report["detection_count"], 1)

    def test_ultralytics_train_prepare_includes_production_params(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir)
            config_dir = project_dir / "config"
            assets_dir = project_dir / "assets" / "prompts"
            input_dir = project_dir / "input" / "frames"
            ensure_dir(config_dir)
            ensure_dir(assets_dir)
            ensure_dir(input_dir)

            write_text(assets_dir / "prompt.md", "Scene: {scene}\nLabels: {labels}\n{prior_detections}\n")
            write_text(input_dir / "frame_0001.ppm", "P3\n1 1\n255\n255 255 255\n")
            write_json(
                config_dir / "project.json",
                {
                    "project_name": "prod-test",
                    "scene": "test-scene",
                    "labels": ["person", "car"],
                    "input": {"frames_dir": "input/frames"},
                    "output": {"workspace_dir": "workspace"},
                    "qwen": {"provider": "mock", "prompt_template": "assets/prompts/prompt.md"},
                    "yolo": {
                        "trainer": "ultralytics",
                        "detector": "sidecar",
                        "binary": "yolo",
                        "weights": "yolo11n.pt",
                        "device": "0",
                        "batch": 8,
                        "workers": 4,
                        "patience": 20,
                        "close_mosaic": 5,
                        "save_period": 2
                    }
                },
            )

            config = ProjectConfig.load(config_dir / "project.json")
            trainer = UltralyticsTrainer()
            run_dir = config.runs_dir / "vtest"
            ensure_dir(run_dir)
            command = trainer._build_train_command(config, project_dir / "dataset.yaml", run_dir)

            joined = " ".join(command)
            self.assertIn("device=0", joined)
            self.assertIn("batch=8", joined)
            self.assertIn("workers=4", joined)
            self.assertIn("patience=20", joined)
            self.assertIn("close_mosaic=5", joined)
            self.assertIn("save_period=2", joined)

if __name__ == "__main__":
    unittest.main()
