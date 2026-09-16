from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class SkillsDeliveryTest(unittest.TestCase):
    def test_required_skill_directories_exist(self) -> None:
        required = [
            ROOT / "skills" / "video-frame-extraction" / "SKILL.md",
            ROOT / "skills" / "video-frame-extraction" / "scripts" / "run.ps1",
            ROOT / "skills" / "images-qwen3vl-annotation" / "SKILL.md",
            ROOT / "skills" / "images-qwen3vl-annotation" / "scripts" / "run.ps1",
            ROOT / "skills" / "images-qwen3vl-annotation" / "prompts" / "qwen3vl_traffic_zh.md",
            ROOT / "skills" / "yolo-closed-loop-training" / "SKILL.md",
            ROOT / "skills" / "yolo-closed-loop-training" / "scripts" / "run.ps1",
            ROOT / "skills" / "yolo-closed-loop-training" / "templates" / "next_cycle_tasks_template.md",
        ]

        for path in required:
            self.assertTrue(path.exists(), f"missing required artifact: {path}")

    def test_required_docs_exist(self) -> None:
        required = [
            ROOT / "README.md",
            ROOT / "docs" / "qwenvl-yolo-skills-work-order.md",
            ROOT / "docs" / "production-integration.md",
            ROOT / "docs" / "cursor-agent-skills-setup.md",
            ROOT / "docs" / "requirements-mapping.md",
        ]

        for path in required:
            self.assertTrue(path.exists(), f"missing required doc: {path}")

    def test_skill_commands_are_documented(self) -> None:
        expected = {
            ROOT / "skills" / "video-frame-extraction" / "SKILL.md": "traffic_cv_closed_loop extract",
            ROOT / "skills" / "images-qwen3vl-annotation" / "SKILL.md": "traffic_cv_closed_loop annotate",
            ROOT / "skills" / "yolo-closed-loop-training" / "SKILL.md": "traffic_cv_closed_loop train",
        }

        for path, fragment in expected.items():
            text = path.read_text(encoding="utf-8")
            self.assertIn(fragment, text)


if __name__ == "__main__":
    unittest.main()
