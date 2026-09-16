from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from hazard_annotation.config import load_hazard_config
from hazard_annotation.diagnostics import build_doctor_report, probe_qwen_service

ROOT = Path(__file__).resolve().parents[1]


class HazardDiagnosticsTest(unittest.TestCase):
    def _prepare(self, tmp_dir: str) -> Path:
        temp_root = Path(tmp_dir)
        project = temp_root / "hazard_mock_project"
        shutil.copytree(ROOT / "examples" / "hazard_mock_project", project)
        shutil.copytree(ROOT / "hazard-detection", temp_root / "hazard-detection")
        return project / "config" / "project.json"

    def test_doctor_reports_video_keyframe_and_knowledge(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = self._prepare(tmp_dir)
            config = load_hazard_config(config_path, mode="quick")
            config.qwen_provider = "mock"
            config.yolo_detector = "sidecar"
            config.yolo_trainer = "mock"

            report = build_doctor_report(config)
            self.assertIn("video", report)
            self.assertIn("keyframe", report)
            self.assertIn("knowledge", report)

            self.assertTrue(report["knowledge"]["exists"])
            self.assertGreater(report["knowledge"]["hazard_count"], 0)
            self.assertIn("scene_ids", report["knowledge"])
            self.assertIn("ffmpeg_available", report["video"])
            self.assertIn("scene_detect_mode", report["keyframe"])
            # Mock mode must not be blocked by the (auto-generated) input frames.
            self.assertNotIn("input_images_missing", report["issues"])
            self.assertNotIn("input_frames_dir_missing", report["issues"])
            self.assertIn(report["status"], {"ready", "issues_found"})

    def test_probe_qwen_mock_reports_connectivity_and_latency(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = self._prepare(tmp_dir)
            config = load_hazard_config(config_path, mode="quick")
            config.qwen_provider = "mock"
            config.yolo_detector = "sidecar"
            config.yolo_trainer = "mock"

            report = probe_qwen_service(config)
            self.assertEqual(report["status"], "ok")
            self.assertEqual(report["connectivity"], "ok")
            self.assertIn("elapsed_ms", report)
            self.assertGreaterEqual(report["elapsed_ms"], 0.0)
            self.assertEqual(report["provider"], "mock")
            self.assertTrue(report["hazard_ids"])
            self.assertTrue((config.probes_dir / "hazard_probe.json").exists())


if __name__ == "__main__":
    unittest.main()
