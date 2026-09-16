from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from hazard_annotation.__main__ import main
from hazard_annotation.knowledge import KnowledgeBase
from traffic_cv_closed_loop.io_utils import read_json, read_jsonl

ROOT = Path(__file__).resolve().parents[1]


class HazardPipelineTest(unittest.TestCase):
    def _prepare(self, tmp_dir: str) -> tuple[Path, Path]:
        temp_root = Path(tmp_dir)
        project = temp_root / "hazard_mock_project"
        shutil.copytree(ROOT / "examples" / "hazard_mock_project", project)
        # The example ships a gitignored ``workspace/`` with leftover artifacts
        # (and timestamped dataset versions) after any demo run. Drop it so the
        # test starts from a clean slate and stays hermetic/repeatable.
        shutil.rmtree(project / "workspace", ignore_errors=True)
        # Mirror the repository layout so the knowledge-base upward search works.
        shutil.copytree(ROOT / "hazard-detection", temp_root / "hazard-detection")
        return project, project / "config" / "project.json"

    def test_full_mock_pipeline_produces_all_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            project, config_path = self._prepare(tmp_dir)
            exit_code = main(["--config", str(config_path), "all", "--mock"])
            self.assertEqual(exit_code, 0)

            workspace = project / "workspace"
            self.assertTrue((workspace / "extracted" / "candidates.jsonl").exists())
            self.assertTrue((workspace / "hazards" / "report.json").exists())
            self.assertTrue(list((workspace / "review" / "labels").rglob("*.txt")))
            self.assertTrue(list((workspace / "hazards").rglob("*.json")))
            self.assertTrue(list((workspace / "datasets").rglob("data.yaml")))
            self.assertTrue((workspace / "analysis" / "hard_examples.jsonl").exists())
            self.assertTrue((workspace / "analysis" / "next_cycle_tasks.md").exists())

            html_path = workspace / "visualization" / "report.html"
            self.assertTrue(html_path.exists())
            html_text = html_path.read_text(encoding="utf-8")
            self.assertIn("演示数据（mock）", html_text)
            self.assertIn("data:image/png;base64,", html_text)

    def test_report_structure_and_valid_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            project, config_path = self._prepare(tmp_dir)
            self.assertEqual(main(["--config", str(config_path), "all", "--mock"]), 0)
            workspace = project / "workspace"

            report = read_json(workspace / "hazards" / "report.json")
            self.assertEqual(report["overall_risk"], "critical")
            self.assertIn("D-5-3", report["hazard_counts"])
            self.assertIn("G-2-2", report["hazard_counts"])
            self.assertEqual(len(report["timeline"]), report["keyframe_count"])

            annotations = read_jsonl(workspace / "review" / "annotations.jsonl")
            self.assertTrue(annotations)
            kb_defs = {
                definition.id: definition
                for definition in KnowledgeBase.load(
                    ROOT / "hazard-detection" / "knowledge_base.json"
                ).hazards
            }
            for record in annotations:
                for hazard in record["hazard_report"]["hazards"]:
                    self.assertFalse(hazard["invalid_id"], hazard["id"])
                    self.assertTrue(hazard["action"], hazard["id"])
                    definition = kb_defs[hazard["id"]]
                    if definition.norm_ref:
                        self.assertEqual(hazard["norm_reference"], definition.norm_ref)

    def test_hard_examples_include_high_severity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            project, config_path = self._prepare(tmp_dir)
            self.assertEqual(main(["--config", str(config_path), "all", "--mock"]), 0)
            hard = read_jsonl(project / "workspace" / "analysis" / "hard_examples.jsonl")
            self.assertTrue(hard)
            severities = {
                hazard["severity"]
                for record in hard
                for hazard in record["hazard_report"]["hazards"]
            }
            self.assertTrue({"critical", "high"} & severities)

    def test_scene_specific_hazards(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            project, config_path = self._prepare(tmp_dir)
            self.assertEqual(main(["--config", str(config_path), "all", "--mock"]), 0)
            annotations = read_jsonl(project / "workspace" / "review" / "annotations.jsonl")
            by_source: dict[str, set[str]] = {}
            for record in annotations:
                source = record["keyframe"]["source_id"]
                ids = {hazard["id"] for hazard in record["hazard_report"]["hazards"]}
                by_source.setdefault(source, set()).update(ids)
            self.assertIn("D-5-3", by_source.get("hoist_scene", set()))
            self.assertIn("G-2-2", by_source.get("height_scene", set()))
            self.assertEqual(by_source.get("safe_scene", set()), set())

    def test_demo_command_regenerates_assets_and_exits_zero(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            project, config_path = self._prepare(tmp_dir)
            shutil.rmtree(project / "input" / "frames", ignore_errors=True)
            self.assertEqual(main(["--config", str(config_path), "demo"]), 0)
            self.assertTrue((project / "workspace" / "visualization" / "report.html").exists())
            self.assertTrue((project / "input" / "frames" / "hoist_scene" / "frame_0001.ppm").exists())

    def test_data_yaml_uses_hazard_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            project, config_path = self._prepare(tmp_dir)
            self.assertEqual(main(["--config", str(config_path), "all", "--mock"]), 0)
            data_yamls = list((project / "workspace" / "datasets").rglob("data.yaml"))
            self.assertEqual(len(data_yamls), 1)
            text = data_yamls[0].read_text(encoding="utf-8")
            self.assertIn("overhead_crane", text)
            self.assertIn("guardrail", text)


if __name__ == "__main__":
    unittest.main()
