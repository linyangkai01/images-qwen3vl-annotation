from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from hazard_annotation.imageio import (
    ImageStats,
    PgmImage,
    PpmImage,
    ahash,
    block_downscale,
    hamming,
    histogram16,
    histogram_delta,
    load_image,
    motion_ratio,
)


class PgmPpmParsingTest(unittest.TestCase):
    def test_binary_p5_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "gray.pgm"
            path.write_bytes(b"P5\n2 2\n255\n" + bytes([0, 128, 255, 64]))
            image = PgmImage.load(path)
            self.assertEqual(image.width, 2)
            self.assertEqual(image.height, 2)
            self.assertEqual(image.pixels, bytes([0, 128, 255, 64]))
            self.assertEqual(image.to_gray_bytes(), bytes([0, 128, 255, 64]))

    def test_binary_p5_with_comment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "gray_comment.pgm"
            path.write_bytes(b"P5\n# a comment\n1 1\n255\n" + bytes([7]))
            image = PgmImage.load(path)
            self.assertEqual(image.width, 1)
            self.assertEqual(image.pixels, bytes([7]))

    def test_ascii_p2(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "ascii.pgm"
            path.write_text("P2\n2 1\n255\n10 20\n", encoding="ascii")
            image = PgmImage.load(path)
            self.assertEqual((image.width, image.height), (2, 1))
            self.assertEqual(image.pixels, bytes([10, 20]))

    def test_binary_p6_to_gray(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "rgb.ppm"
            path.write_bytes(b"P6\n1 1\n255\n" + bytes([255, 0, 0]))
            image = PpmImage.load(path)
            self.assertEqual(image.width, 1)
            self.assertEqual(image.to_gray_bytes(), bytes([85]))

    def test_ascii_p3_to_gray(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "rgb_ascii.ppm"
            path.write_text("P3\n1 1\n255\n0 255 0\n", encoding="ascii")
            image = PpmImage.load(path)
            self.assertEqual(image.to_gray_bytes(), bytes([85]))

    def test_load_image_dispatches_by_magic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            pgm = Path(tmp_dir) / "a.pgm"
            ppm = Path(tmp_dir) / "b.ppm"
            pgm.write_bytes(b"P5\n1 1\n255\n" + bytes([1]))
            ppm.write_bytes(b"P6\n1 1\n255\n" + bytes([1, 2, 3]))
            self.assertIsInstance(load_image(pgm), PgmImage)
            self.assertIsInstance(load_image(ppm), PpmImage)

    def test_binary_p5_crlf_header_does_not_shift_raster(self) -> None:
        # Regression (F5): a CRLF-terminated header must consume exactly one
        # line ending, not leave the '\n' as the first pixel.
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "crlf.pgm"
            path.write_bytes(b"P5\r\n1 1\r\n255\r\n" + bytes([7]))
            self.assertEqual(PgmImage.load(path).pixels, bytes([7]))

    def test_raster_length_mismatch_raises(self) -> None:
        # Regression (F6): truncated / oversized rasters must raise, not pad.
        with tempfile.TemporaryDirectory() as tmp_dir:
            short = Path(tmp_dir) / "short.pgm"
            long = Path(tmp_dir) / "long.pgm"
            short.write_bytes(b"P5\n2 2\n255\n" + bytes([1, 2]))
            long.write_bytes(b"P5\n1 1\n255\n" + bytes([1, 2, 3, 4]))
            with self.assertRaises(ValueError):
                PgmImage.load(short)
            with self.assertRaises(ValueError):
                PgmImage.load(long)


class ImageStatsTest(unittest.TestCase):
    def test_ahash_self_hamming_is_zero(self) -> None:
        pixels = bytes([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160])
        digest = ahash(pixels, 4, 4, 8)
        self.assertEqual(hamming(digest, digest), 0)
        stats = ImageStats(phash_size=8)
        image = PgmImage(width=4, height=4, pixels=pixels)
        self.assertEqual(stats.hamming(stats.ahash(image), stats.ahash(image)), 0)

    def test_ahash_different_images_differ(self) -> None:
        # Two non-uniform (mirror) gradients produce different average hashes.
        ascending = bytes((index * 3) % 256 for index in range(64))
        descending = bytes(255 - (index * 3) % 256 for index in range(64))
        self.assertGreater(hamming(ahash(ascending, 8, 8, 8), ahash(descending, 8, 8, 8)), 0)

    def test_ahash_distinguishes_uniform_black_and_white(self) -> None:
        # Regression (F7): a global-brightness tie-breaker keeps two maximally
        # different uniform frames from collapsing to the same hash.
        black = ahash(bytes([0] * 256), 16, 16, 8)
        white = ahash(bytes([255] * 256), 16, 16, 8)
        self.assertGreater(hamming(black, white), 0)

    def test_hamming_known_value(self) -> None:
        self.assertEqual(hamming(0b1011, 0b0010), 2)

    def test_motion_ratio_hand_computed(self) -> None:
        first = bytes([0, 0, 0, 0])
        second = bytes([0, 0, 10, 30])
        # Only the last pixel changes by more than 25 (|0-30| = 30).
        self.assertAlmostEqual(motion_ratio(first, second, 25), 0.25)
        self.assertAlmostEqual(motion_ratio(first, second, 5), 0.5)

    def test_motion_ratio_empty(self) -> None:
        self.assertEqual(motion_ratio(b"", b"", 25), 0.0)

    def test_histogram16_normalised(self) -> None:
        histogram = histogram16(bytes([0, 16, 255]))
        self.assertAlmostEqual(sum(histogram), 1.0)
        self.assertAlmostEqual(histogram[0], 1 / 3)
        self.assertAlmostEqual(histogram[1], 1 / 3)
        self.assertAlmostEqual(histogram[15], 1 / 3)

    def test_histogram_delta(self) -> None:
        first = [1.0] + [0.0] * 15
        second = [0.0] * 15 + [1.0]
        self.assertAlmostEqual(histogram_delta(first, second), 1.0)
        self.assertAlmostEqual(histogram_delta(first, first), 0.0)

    def test_block_downscale_identity(self) -> None:
        grid = block_downscale(bytes([10, 20, 30, 40]), 2, 2, 2)
        self.assertEqual(grid, [[10, 20], [30, 40]])

    def test_block_downscale_averages(self) -> None:
        grid = block_downscale(bytes([0, 10, 20, 30]), 2, 2, 1)
        self.assertEqual(grid, [[15]])


if __name__ == "__main__":
    unittest.main()
