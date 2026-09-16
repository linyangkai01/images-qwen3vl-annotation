from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from traffic_cv_closed_loop.cli import main
from traffic_cv_closed_loop.io_utils import read_json, read_jsonl


ROOT = Path(__file__).resolve().parents[1]


class ClosedLoopPipelineTest(unittest.TestCase):
    def test_mock_pipeline_end_to_end(self) -> None:
        source_project = ROOT / "examples" / "mock_project"
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_project = Path(tmp_dir) / "mock_project"
            shutil.copytree(source_project, temp_project)
            config_path = temp_project / "config" / "project.json"

            exit_code = main(["--config", str(config_path), "run-loop"])
            self.assertEqual(exit_code, 0)

            candidates = read_jsonl(temp_project / "workspace" / "extracted" / "candidates.jsonl")
            annotations = read_jsonl(temp_project / "workspace" / "review" / "annotations.jsonl")
            review_queue = read_jsonl(temp_project / "workspace" / "review" / "review_queue.jsonl")
            latest_report = read_json(temp_project / "workspace" / "runs" / "latest.json")

            self.assertEqual(len(candidates), 4)
            self.assertEqual(len(annotations), 4)
            self.assertEqual(len(review_queue), 2)
            self.assertEqual(latest_report["accepted_images"], 2)
            self.assertEqual(latest_report["trainer"]["status"], "completed")
            self.assertTrue((temp_project / "workspace" / "analysis" / "hard_examples.jsonl").exists())

    def test_status_command_reads_stage_outputs(self) -> None:
        source_project = ROOT / "examples" / "mock_project"
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_project = Path(tmp_dir) / "mock_project"
            shutil.copytree(source_project, temp_project)
            config_path = temp_project / "config" / "project.json"

            self.assertEqual(main(["--config", str(config_path), "extract"]), 0)
            self.assertEqual(main(["--config", str(config_path), "annotate"]), 0)
            self.assertEqual(main(["--config", str(config_path), "status"]), 0)


if __name__ == "__main__":
    unittest.main()
