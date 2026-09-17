"""Pure-Python image handling shared by the pipeline infrastructure.

This module lives in the reusable ``traffic_cv_closed_loop`` base package so
that both the traffic chain and the derived ``hazard_annotation`` chain can
decode/encode images without depending on each other. Only the Python standard
library is used. Supported input formats are the Netpbm family (``P2``/``P3``
ASCII and ``P5``/``P6`` binary) because ffmpeg can emit grayscale ``P5`` PGMs
directly, avoiding any third-party decoder. The statistics helpers (average
hash, hamming distance, 16-bucket histogram, frame difference) operate on raw
``bytes`` with integer arithmetic.

The public API is re-exported unchanged from
:mod:`hazard_annotation.imageio` for backward compatibility.
"""

from __future__ import annotations

import binascii
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "PgmImage",
    "PpmImage",
    "ImageStats",
    "load_image",
    "block_downscale",
    "ahash",
    "hamming",
    "histogram16",
    "histogram_delta",
    "motion_ratio",
    "rgb_bytes",
    "downscale_rgb",
    "encode_png",
    "image_to_png_bytes",
]


def _parse_netpbm_header(data: bytes) -> tuple[str, int, int, int, int]:
    """Parse a Netpbm header, returning (magic, width, height, maxval, offset)."""

    length = len(data)
    index = 0

    def skip_whitespace_and_comments(position: int) -> int:
        while position < length:
            char = data[position : position + 1]
            if char in b" \t\r\n\v\f":
                position += 1
            elif char == b"#":
                while position < length and data[position : position + 1] not in b"\r\n":
                    position += 1
            else:
                break
        return position

    def read_token(position: int) -> tuple[str, int]:
        position = skip_whitespace_and_comments(position)
        start = position
        while position < length and data[position : position + 1] not in b" \t\r\n\v\f":
            position += 1
        token = data[start:position].decode("ascii", errors="replace")
        return token, position

    magic, index = read_token(index)
    if magic not in ("P2", "P3", "P5", "P6"):
        raise ValueError(f"Unsupported Netpbm magic: {magic!r}")
    width_token, index = read_token(index)
    height_token, index = read_token(index)
    maxval_token, index = read_token(index)
    width = int(width_token)
    height = int(height_token)
    maxval = int(maxval_token)
    if width <= 0 or height <= 0 or maxval <= 0:
        raise ValueError("Invalid Netpbm header dimensions or maxval")
    if magic in ("P5", "P6"):
        # Exactly one whitespace character separates maxval from the raster;
        # a CRLF line ending is treated as that single separator so the raster
        # does not shift by one byte on Windows-authored files.
        if data[index : index + 2] == b"\r\n":
            index += 2
        elif data[index : index + 1] in b" \t\r\n\v\f":
            index += 1
    return magic, width, height, maxval, index


def _check_raster_size(kind: str, available: bytes, expected: int, width: int, height: int) -> None:
    """Raise a clear ``ValueError`` when the raster byte count is not exact.

    A single trailing whitespace byte is tolerated (the Netpbm spec permits one
    whitespace character after the raster); any other surplus or a shortfall
    means the file is truncated or corrupt and must not be silently padded or
    truncated into fabricated content.
    """
    actual = len(available)
    if actual < expected:
        raise ValueError(
            f"{kind} raster truncated for {width}x{height}: expected {expected} bytes, got {actual}"
        )
    if actual > expected and available[expected:].strip(b" \t\r\n\v\f"):
        raise ValueError(
            f"{kind} raster oversized for {width}x{height}: expected {expected} bytes, got {actual}"
        )


def _ascii_samples(data: bytes, offset: int, count: int) -> list[int]:
    values: list[int] = []
    for token in data[offset:].split():
        values.append(int(token))
        if len(values) >= count:
            break
    return values


@dataclass(slots=True)
class PgmImage:
    """A grayscale PGM (P2/P5) image."""

    width: int
    height: int
    pixels: bytes
    maxval: int = 255

    @classmethod
    def load(cls, path: str | Path) -> "PgmImage":
        data = Path(path).read_bytes()
        magic, width, height, maxval, offset = _parse_netpbm_header(data)
        pixel_count = width * height
        if magic == "P5":
            available = data[offset:]
            _check_raster_size("PGM", available, pixel_count, width, height)
            pixels = bytes(available[:pixel_count])
        elif magic == "P2":
            values = _ascii_samples(data, offset, pixel_count)
            if len(values) != pixel_count:
                raise ValueError(
                    f"PGM raster truncated for {width}x{height}: "
                    f"expected {pixel_count} samples, got {len(values)}"
                )
            pixels = bytes(value & 0xFF for value in values)
        elif magic in ("P6", "P3"):
            rgb = _read_rgb(data, magic, width, height, offset)
            pixels = bytes((rgb[i] + rgb[i + 1] + rgb[i + 2]) // 3 for i in range(0, len(rgb), 3))
        else:  # pragma: no cover - guarded by the header parser
            raise ValueError(f"Unsupported PGM magic: {magic!r}")
        return cls(width=width, height=height, pixels=pixels, maxval=maxval)

    def to_gray_bytes(self) -> bytes:
        return self.pixels


@dataclass(slots=True)
class PpmImage:
    """An RGB PPM (P3/P6) image."""

    width: int
    height: int
    pixels: bytes
    maxval: int = 255

    @classmethod
    def load(cls, path: str | Path) -> "PpmImage":
        data = Path(path).read_bytes()
        magic, width, height, maxval, offset = _parse_netpbm_header(data)
        rgb = _read_rgb(data, magic, width, height, offset)
        return cls(width=width, height=height, pixels=rgb, maxval=maxval)

    def to_gray_bytes(self) -> bytes:
        pixels = self.pixels
        return bytes(
            (pixels[i] + pixels[i + 1] + pixels[i + 2]) // 3 for i in range(0, len(pixels), 3)
        )


def _read_rgb(data: bytes, magic: str, width: int, height: int, offset: int) -> bytes:
    pixel_count = width * height * 3
    if magic == "P6":
        available = data[offset:]
        _check_raster_size("PPM", available, pixel_count, width, height)
        return bytes(available[:pixel_count])
    values = _ascii_samples(data, offset, pixel_count)
    if len(values) != pixel_count:
        raise ValueError(
            f"PPM raster truncated for {width}x{height}: "
            f"expected {pixel_count} samples, got {len(values)}"
        )
    return bytes(value & 0xFF for value in values)


def load_image(path: str | Path) -> PgmImage | PpmImage:
    """Load a PGM or PPM image, choosing the concrete type by magic number."""
    data = Path(path).read_bytes()
    magic, *_ = _parse_netpbm_header(data)
    if magic in ("P2", "P5"):
        return PgmImage.load(path)
    return PpmImage.load(path)


def block_downscale(pixels: bytes, width: int, height: int, size: int) -> list[list[int]]:
    """Downscale to a ``size`` x ``size`` grid of integer block averages."""
    if width <= 0 or height <= 0 or size <= 0:
        return []
    totals = [[0] * size for _ in range(size)]
    counts = [[0] * size for _ in range(size)]
    for y in range(height):
        block_y = min(size - 1, y * size // height)
        row_offset = y * width
        totals_row = totals[block_y]
        counts_row = counts[block_y]
        for x in range(width):
            block_x = min(size - 1, x * size // width)
            totals_row[block_x] += pixels[row_offset + x]
            counts_row[block_x] += 1
    grid: list[list[int]] = []
    for block_y in range(size):
        row: list[int] = []
        for block_x in range(size):
            count = counts[block_y][block_x]
            row.append(totals[block_y][block_x] // count if count else 0)
        grid.append(row)
    return grid


def ahash(pixels: bytes, width: int, height: int, phash_size: int = 8) -> int:
    """Average hash as an integer bit-field (row-major, MSB not significant).

    The high bit (position ``phash_size ** 2``) folds in the global brightness.
    Without it a uniform black frame and a uniform white frame both hash to 0
    because no block exceeds the mean, and the perceptual de-duplicator would
    treat two maximally different frames as identical.
    """
    grid = block_downscale(pixels, width, height, phash_size)
    if not grid:
        return 0
    flat = [value for row in grid for value in row]
    mean = sum(flat) / len(flat)
    bits = 0
    for index, value in enumerate(flat):
        if value > mean:
            bits |= 1 << index
    if pixels and (sum(pixels) / len(pixels)) >= 128:
        bits |= 1 << len(flat)
    return bits


def hamming(first: int, second: int) -> int:
    """Population count of the XOR of two integer hashes."""
    return int(first ^ second).bit_count()


def histogram16(pixels: bytes) -> list[float]:
    """Normalised 16-bucket histogram (bucket = value // 16)."""
    buckets = [0] * 16
    for value in pixels:
        buckets[value >> 4] += 1
    total = len(pixels)
    if total == 0:
        return [0.0] * 16
    return [count / total for count in buckets]


def histogram_delta(first: list[float], second: list[float]) -> float:
    """L1 distance between two histograms divided by 2 (range 0..1)."""
    if not first or not second:
        return 0.0
    return sum(abs(a - b) for a, b in zip(first, second)) / 2.0


def motion_ratio(first: bytes, second: bytes, pixel_delta_threshold: int = 25) -> float:
    """Fraction of pixels whose absolute intensity change exceeds the threshold."""
    count = min(len(first), len(second))
    if count == 0:
        return 0.0
    changed = 0
    for index in range(count):
        delta = first[index] - second[index]
        if delta < 0:
            delta = -delta
        if delta > pixel_delta_threshold:
            changed += 1
    return changed / count


class ImageStats:
    """Convenience wrapper binding the pure functions to a hash size."""

    def __init__(self, phash_size: int = 8) -> None:
        self.phash_size = phash_size

    def ahash(self, image: PgmImage | PpmImage) -> int:
        return ahash(image.to_gray_bytes(), image.width, image.height, self.phash_size)

    def hamming(self, first: int, second: int) -> int:
        return hamming(first, second)

    def histogram16(self, image: PgmImage | PpmImage) -> list[float]:
        return histogram16(image.to_gray_bytes())

    def histogram_delta(self, first: list[float], second: list[float]) -> float:
        return histogram_delta(first, second)

    def motion_ratio(
        self,
        first: PgmImage | PpmImage,
        second: PgmImage | PpmImage,
        pixel_delta_threshold: int = 25,
    ) -> float:
        return motion_ratio(
            first.to_gray_bytes(),
            second.to_gray_bytes(),
            pixel_delta_threshold,
        )

    def block_downscale(self, image: PgmImage | PpmImage, size: int | None = None) -> list[list[int]]:
        return block_downscale(
            image.to_gray_bytes(),
            image.width,
            image.height,
            size or self.phash_size,
        )


def rgb_bytes(image: PgmImage | PpmImage) -> bytes:
    """Return packed RGB bytes for a PGM or PPM image."""
    if isinstance(image, PpmImage):
        return image.pixels
    gray = image.to_gray_bytes()
    return bytes(channel for value in gray for channel in (value, value, value))


def downscale_rgb(
    rgb: bytes,
    width: int,
    height: int,
    target_width: int,
) -> tuple[int, int, bytes]:
    """Nearest-neighbour RGB downscale (never upscales)."""
    if target_width <= 0 or width <= target_width:
        return width, height, rgb
    scale = width / target_width
    target_height = max(1, int(round(height / scale)))
    out = bytearray()
    for target_y in range(target_height):
        source_y = min(height - 1, int(target_y * scale))
        row_offset = source_y * width * 3
        for target_x in range(target_width):
            source_x = min(width - 1, int(target_x * scale))
            index = row_offset + source_x * 3
            out.extend(rgb[index : index + 3])
    return target_width, target_height, bytes(out)


def encode_png(width: int, height: int, rgb: bytes) -> bytes:
    """Encode packed RGB bytes as a PNG using only the standard library."""
    if width <= 0 or height <= 0:
        raise ValueError("PNG dimensions must be positive")
    stride = width * 3
    raw = bytearray()
    for y in range(height):
        raw.append(0)  # filter type 0 (None)
        raw.extend(rgb[y * stride : (y + 1) * stride])
    compressed = zlib.compress(bytes(raw), 9)

    def chunk(tag: bytes, data: bytes) -> bytes:
        checksum = binascii.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", checksum)

    signature = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return signature + chunk(b"IHDR", ihdr) + chunk(b"IDAT", compressed) + chunk(b"IEND", b"")


def image_to_png_bytes(path: str | Path, target_width: int | None = None) -> bytes | None:
    """Load a Netpbm image and re-encode it as PNG (optionally downscaled)."""
    try:
        image = load_image(path)
    except (ValueError, OSError):
        return None
    width, height, rgb = image.width, image.height, rgb_bytes(image)
    if target_width:
        width, height, rgb = downscale_rgb(rgb, width, height, target_width)
    return encode_png(width, height, rgb)
