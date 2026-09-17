"""Backward-compatible view of the shared Netpbm codec.

The implementation moved to :mod:`traffic_cv_closed_loop.imageio` so that the
reusable base package (dataset building, image materialisation) can decode and
encode Netpbm frames without depending on the derived ``hazard_annotation``
package. This module re-exports the exact same public API, so existing imports
such as ``from hazard_annotation.imageio import image_to_png_bytes`` keep
working unchanged.
"""

from __future__ import annotations

from traffic_cv_closed_loop.imageio import (
    ImageStats,
    PgmImage,
    PpmImage,
    ahash,
    block_downscale,
    downscale_rgb,
    encode_png,
    hamming,
    histogram16,
    histogram_delta,
    image_to_png_bytes,
    load_image,
    motion_ratio,
    rgb_bytes,
)

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
