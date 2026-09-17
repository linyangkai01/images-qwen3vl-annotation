"""Regression tests for dataset image materialisation (Ultralytics readability).

Covers the integration gap where image-directory / mock sources produce Netpbm
``.ppm`` frames that Ultralytics silently ignores (its ``IMG_FORMATS`` does not
include ``ppm``), making the generated dataset untrainable:

* ``.ppm`` source  -> dataset holds a ``.png`` (valid PNG signature + IHDR).
* ``.png`` source  -> copied byte-for-byte (hash + mtime preserved, no re-encode).
* label file name  -> tracks the produced image extension (image/label stay paired).
* ``manifest.json``-> ``image_path`` / ``label_path`` point at files that exist.
* end-to-end      -> hazard mock + traffic mock datasets contain only
  Ultralytics-readable extensions.
"""

from __future__ import annotations

import hashlib
import shutil
import struct
import tempfile
import unittest
from pathlib import Path

from traffic_cv_closed_loop.dataset_media import (
    ULTRALYTICS_IMAGE_SUFFIXES,
    dataset_image_relative_path,
    materialize_dataset_image,
)
from traffic_cv_closed_loop.io_utils import read_json

ROOT = Path(__file__).resolve().parents[1]


def _write_ppm(path: Path, rgb: tuple[int, int, int] = (10, 20, 30), size: int = 2) -> None:
    """Write a tiny ASCII P3 PPM (the format the mock sources emit)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    header = f"P3\n{size} {size}\n255\n"
    row = " ".join(" ".join(str(channel) for channel in rgb) for _ in range(size))
    body = "\n".join(row for _ in range(size)) + "\n"
    path.write_text(header + body, encoding="ascii")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _png_dimensions(path: Path) -> tuple[int, int]:
    """Parse the IHDR chunk of a PNG, asserting the 8-byte signature first."""
    data = path.read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise AssertionError(f"not a PNG file: {path}")
    # Layout: 8-byte signature + 4-byte length + 4-byte 'IHDR' + width(4) + height(4)
    width, height = struct.unpack(">II", data[16:24])
    return width, height


class MaterializeDatasetImageTest(unittest.TestCase):
    def test_ppm_source_is_transcoded_to_png(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            source = tmp_path / "frame_0001.ppm"
            _write_ppm(source, size=3)
            images_dir = tmp_path / "images" / "train"

            produced = materialize_dataset_image(
                source, images_dir, Path("scene/frame_0001.ppm")
            )

            self.assertEqual(produced, Path("scene/frame_0001.png"))
            target = images_dir / produced
            self.assertTrue(target.exists())
            self.assertEqual(_png_dimensions(target), (3, 3))
            # The extracted source frame must not be mutated (still Netpbm).
            self.assertTrue(source.exists())
            self.assertTrue(source.read_bytes().startswith(b"P3"))

    def test_supported_png_source_is_copied_byte_for_byte(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            source = tmp_path / "frame_0001.png"
            # Video-derived frames are already PNG; the payload need not decode
            # for this test, we only assert an exact byte-for-byte copy.
            source.write_bytes(b"\x89PNG\r\n\x1a\n" + b"opaque-payload" * 4)
            before_hash = _sha256(source)
            before_mtime = source.stat().st_mtime_ns
            images_dir = tmp_path / "images" / "val"

            produced = materialize_dataset_image(
                source, images_dir, Path("scene/frame_0001.png")
            )

            self.assertEqual(produced, Path("scene/frame_0001.png"))
            target = images_dir / produced
            self.assertEqual(_sha256(target), before_hash)
            self.assertEqual(target.stat().st_mtime_ns, before_mtime)

    def test_label_name_tracks_produced_extension(self) -> None:
        relative = Path("scene/frame_0001.ppm")
        produced = dataset_image_relative_path(relative)
        self.assertEqual(produced, Path("scene/frame_0001.png"))
        label = produced.with_suffix(".txt")
        self.assertEqual(label, Path("scene/frame_0001.txt"))
        # Image and label share the same stem -> Ultralytics can pair them.
        self.assertEqual(produced.stem, label.stem)
        # A label derived from the *produced* image is what the pipeline writes.
        self.assertTrue(str(label).endswith(".txt"))

    def test_unsupported_non_netpbm_source_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            source = tmp_path / "frame_0001.tga"
            source.write_bytes(b"not-a-netpbm-raster")
            with self.assertRaises(ValueError):
                materialize_dataset_image(source, tmp_path / "images", Path("frame_0001.tga"))

    def test_supported_extension_set_excludes_netpbm(self) -> None:
        # Guards the hard-coded Ultralytics IMG_FORMATS snapshot.
        self.assertIn(".png", ULTRALYTICS_IMAGE_SUFFIXES)
        self.assertIn(".jpg", ULTRALYTICS_IMAGE_SUFFIXES)
        self.assertNotIn(".ppm", ULTRALYTICS_IMAGE_SUFFIXES)
        self.assertNotIn(".pgm", ULTRALYTICS_IMAGE_SUFFIXES)


class HazardDatasetFormatEndToEndTest(unittest.TestCase):
    def _prepare(self, tmp_dir: str) -> tuple[Path, Path]:
        temp_root = Path(tmp_dir)
        project = temp_root / "hazard_mock_project"
        shutil.copytree(ROOT / "examples" / "hazard_mock_project", project)
        shutil.rmtree(project / "workspace", ignore_errors=True)
        shutil.copytree(ROOT / "hazard-detection", temp_root / "hazard-detection")
        return project, project / "config" / "project.json"

    def test_mock_dataset_images_are_ultralytics_readable(self) -> None:
        from hazard_annotation.__main__ import main

        with tempfile.TemporaryDirectory() as tmp_dir:
            project, config_path = self._prepare(tmp_dir)
            self.assertEqual(main(["--config", str(config_path), "all", "--mock"]), 0)

            dataset_dirs = list((project / "workspace" / "datasets").glob("v*"))
            self.assertEqual(len(dataset_dirs), 1)
            dataset_dir = dataset_dirs[0]

            images = [path for path in (dataset_dir / "images").rglob("*") if path.is_file()]
            self.assertTrue(images)
            for image in images:
                self.assertIn(image.suffix.lower(), ULTRALYTICS_IMAGE_SUFFIXES)
                # The original Netpbm extension must never reach the dataset.
                self.assertNotEqual(image.suffix.lower(), ".ppm")

            for image in images:
                relative = image.relative_to(dataset_dir / "images")
                label = (dataset_dir / "labels" / relative).with_suffix(".txt")
                self.assertTrue(label.exists(), f"missing label for {image}")

            manifest = read_json(dataset_dir / "manifest.json")
            self.assertTrue(manifest)
            for entry in manifest:
                image_path = Path(entry["image_path"])
                label_path = Path(entry["label_path"])
                self.assertTrue(image_path.exists(), entry)
                self.assertTrue(label_path.exists(), entry)
                self.assertIn(image_path.suffix.lower(), ULTRALYTICS_IMAGE_SUFFIXES)
                self.assertEqual(image_path.stem, label_path.stem)
                # image and label live under the same split directory.
                self.assertEqual(image_path.parent.name, label_path.parent.name)


class TrafficDatasetFormatEndToEndTest(unittest.TestCase):
    def test_mock_project_dataset_images_are_ultralytics_readable(self) -> None:
        from traffic_cv_closed_loop.cli import main

        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_project = Path(tmp_dir) / "mock_project"
            shutil.copytree(ROOT / "examples" / "mock_project", temp_project)
            shutil.rmtree(temp_project / "workspace", ignore_errors=True)
            config_path = temp_project / "config" / "project.json"

            self.assertEqual(main(["--config", str(config_path), "run-loop"]), 0)

            dataset_dirs = list((temp_project / "workspace" / "datasets").glob("v*"))
            self.assertEqual(len(dataset_dirs), 1)
            dataset_dir = dataset_dirs[0]

            images = [path for path in (dataset_dir / "images").rglob("*") if path.is_file()]
            self.assertTrue(images)
            for image in images:
                self.assertIn(image.suffix.lower(), ULTRALYTICS_IMAGE_SUFFIXES)
                relative = image.relative_to(dataset_dir / "images")
                label = (dataset_dir / "labels" / relative).with_suffix(".txt")
                self.assertTrue(label.exists(), f"missing label for {image}")

            manifest = read_json(dataset_dir / "manifest.json")
            self.assertTrue(manifest)
            for entry in manifest:
                self.assertTrue(Path(entry["image_path"]).exists(), entry)
                self.assertTrue(Path(entry["label_path"]).exists(), entry)
                self.assertIn(
                    Path(entry["image_path"]).suffix.lower(), ULTRALYTICS_IMAGE_SUFFIXES
                )


if __name__ == "__main__":
    unittest.main()
