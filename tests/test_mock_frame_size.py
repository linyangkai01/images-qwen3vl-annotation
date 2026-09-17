"""Regression: mock frames must be large enough for Ultralytics to accept.

``hazard_annotation.demo.build_mock_assets`` used to emit **4x4 px** frames.
Ultralytics discards any image whose shorter side is under 10 px
(``ignoring corrupt image/label: image size (4, 4) <10 pixels``), which left the
train split with 0 usable images and aborted real training with
``No labels found in .../labels/train``. Frames are now ``MOCK_FRAME_SIZE``
(>= 64) px, so the shipped mock project can drive a real YOLO run.

The frames are solid colour, so the size change is semantics-preserving: the
normalised bboxes and the sidecars are untouched (asserted below).
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from hazard_annotation.demo import (
    MOCK_FRAME_SIZE,
    ULTRALYTICS_MIN_PIXELS,
    build_mock_assets,
    ensure_mock_assets,
)
from hazard_annotation.imageio import load_image
from traffic_cv_closed_loop.config import ProjectConfig

ROOT = Path(__file__).resolve().parents[1]


class MockFrameSizeTest(unittest.TestCase):
    def test_generated_frames_meet_ultralytics_minimum(self) -> None:
        self.assertGreaterEqual(MOCK_FRAME_SIZE, ULTRALYTICS_MIN_PIXELS)
        with tempfile.TemporaryDirectory() as tmp_dir:
            project = Path(tmp_dir) / "mock_project"
            summary = build_mock_assets(project)

            frames = sorted((project / "input" / "frames").rglob("*.ppm"))
            self.assertTrue(frames)
            self.assertEqual(len(frames), summary["frame_count"])
            for frame in frames:
                image = load_image(frame)
                self.assertGreaterEqual(
                    min(image.width, image.height), ULTRALYTICS_MIN_PIXELS, frame
                )
                self.assertEqual((image.width, image.height), (MOCK_FRAME_SIZE, MOCK_FRAME_SIZE))

    def test_committed_example_frames_meet_ultralytics_minimum(self) -> None:
        """The repo's shipped mock project must train as-is (server acceptance run)."""
        frames = sorted(
            (ROOT / "examples" / "hazard_mock_project" / "input" / "frames").rglob("*.ppm")
        )
        self.assertTrue(frames)
        for frame in frames:
            image = load_image(frame)
            self.assertGreaterEqual(
                min(image.width, image.height), ULTRALYTICS_MIN_PIXELS, frame
            )

    def test_resizing_preserves_normalised_bboxes_and_sidecars(self) -> None:
        """Frame size must not leak into the sidecars: bboxes stay normalised."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            project = Path(tmp_dir) / "mock_project"
            build_mock_assets(project)
            frames = sorted((project / "input" / "frames").rglob("*.ppm"))
            for frame in frames:
                detections_path = frame.with_suffix(".detections.json")
                self.assertTrue(detections_path.exists(), detections_path)
                payload = json.loads(detections_path.read_text(encoding="utf-8"))
                for obj in payload["objects"]:
                    x1, y1, x2, y2 = obj["bbox"]
                    for value in (x1, y1, x2, y2):
                        self.assertGreaterEqual(value, 0.0)
                        self.assertLessEqual(value, 1.0)

    def test_ensure_mock_assets_regenerates_stale_undersized_frames(self) -> None:
        """Existing checkouts hold git-ignored frames from an older generator.

        ``ensure_mock_assets`` must refresh them instead of leaving the project
        untrainable.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            project = Path(tmp_dir) / "mock_project"
            (project / "config").mkdir(parents=True)
            (project / "input" / "frames" / "safe_scene").mkdir(parents=True)
            # A stale 4x4 frame from the previous generator (in a scene the
            # generator will overwrite).
            stale = project / "input" / "frames" / "safe_scene" / "frame_0001.ppm"
            stale.write_text("P3\n4 4\n255\n" + " ".join(["10 20 30"] * 16) + "\n", encoding="ascii")
            self.assertLess(min(load_image(stale).width, load_image(stale).height), ULTRALYTICS_MIN_PIXELS)

            (project / "config" / "project.json").write_text(
                json.dumps(
                    {
                        "project_name": "stale-mock",
                        "labels": ["person"],
                        "input": {"frames_dir": "input/frames"},
                        "output": {"workspace_dir": "workspace"},
                        "qwen": {"provider": "mock"},
                        "yolo": {"detector": "sidecar", "trainer": "mock"},
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            config = ProjectConfig.load(project / "config" / "project.json")

            summary = ensure_mock_assets(config)

            self.assertIsNotNone(summary, "stale frames must trigger a regeneration")
            frames = sorted((project / "input" / "frames").rglob("*.ppm"))
            self.assertTrue(frames)
            for frame in frames:
                image = load_image(frame)
                self.assertGreaterEqual(min(image.width, image.height), ULTRALYTICS_MIN_PIXELS)


if __name__ == "__main__":
    unittest.main()
