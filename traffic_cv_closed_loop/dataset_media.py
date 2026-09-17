"""Dataset image materialisation for Ultralytics YOLO training.

The dataset builders copy each accepted frame into
``datasets/<version>/images/<split>/`` and its YOLO label into
``datasets/<version>/labels/<split>/``. Ultralytics only *enumerates* images
whose extension is in its own ``IMG_FORMATS`` allow-list, so a frame stored as
a Netpbm ``.ppm`` / ``.pgm`` (as produced by the image-directory and mock
sources) is silently invisible to the trainer, which then aborts with::

    FileNotFoundError: train: Error loading data from .../images/train

This module guarantees that every materialised image carries an Ultralytics
readable extension: already-supported extensions are copied byte-for-byte (no
needless re-encoding, so content hash and mtime are preserved), and everything
else is transcoded to PNG using the standard-library codec in
:mod:`traffic_cv_closed_loop.imageio`.
"""

from __future__ import annotations

from pathlib import Path

from traffic_cv_closed_loop.imageio import image_to_png_bytes
from traffic_cv_closed_loop.io_utils import copy_file, ensure_dir

#: Image extensions that Ultralytics will actually enumerate for a dataset.
#:
#: Source: ``ultralytics.data.utils.IMG_FORMATS`` (verified against
#: ``ultralytics==8.4.153``). It is hard-coded on purpose: ``ultralytics`` is an
#: *optional* runtime dependency and must never be imported at module load time
#: (the whole pipeline, including ``demo --mock``, must run with the standard
#: library only). Note that ``ppm`` and ``pgm`` are intentionally absent — that
#: omission is exactly the gap this module closes.
ULTRALYTICS_IMAGE_SUFFIXES: frozenset[str] = frozenset(
    {
        ".avif",
        ".bmp",
        ".dng",
        ".heic",
        ".heif",
        ".jp2",
        ".jpeg",
        ".jpg",
        ".mpo",
        ".png",
        ".tif",
        ".tiff",
        ".webp",
    }
)


def dataset_image_relative_path(relative_path: Path) -> Path:
    """Return the dataset-relative image name Ultralytics can read.

    Supported extensions are preserved so the source file can be copied
    verbatim; any other extension (e.g. ``.ppm``) is retargeted to ``.png``.
    """
    if relative_path.suffix.lower() in ULTRALYTICS_IMAGE_SUFFIXES:
        return relative_path
    return relative_path.with_suffix(".png")


def materialize_dataset_image(
    source: Path,
    images_split_dir: Path,
    relative_path: Path,
) -> Path:
    """Copy/convert ``source`` into ``images_split_dir`` for YOLO training.

    Args:
        source: The extracted frame on disk (may be ``.ppm`` / ``.pgm``).
        images_split_dir: Destination ``datasets/<version>/images/<split>`` dir.
        relative_path: The frame's dataset-relative path, e.g.
            ``hoist_scene/frame_0001.ppm``.

    Returns:
        The produced image path *relative to* ``images_split_dir``. Its suffix
        is always a member of :data:`ULTRALYTICS_IMAGE_SUFFIXES`; callers must
        derive the label file name from this value (not from ``relative_path``)
        so that image and label stay paired after an extension change.

    Raises:
        ValueError: If a source with an unsupported extension cannot be
            transcoded to PNG (missing or corrupt Netpbm raster).
    """
    target_relative = dataset_image_relative_path(relative_path)
    target = images_split_dir / target_relative
    if target_relative == relative_path:
        # Already Ultralytics-readable: byte-for-byte copy (``copy_file`` uses
        # ``shutil.copy2``, preserving content hash and mtime). Avoids a
        # pointless re-encode of e.g. video-derived PNG frames.
        copy_file(source, target)
        return target_relative

    png_bytes = image_to_png_bytes(source)
    if png_bytes is None:
        raise ValueError(
            "Cannot transcode image to PNG for YOLO training: "
            f"{source} (unsupported extension {relative_path.suffix!r})"
        )
    ensure_dir(target.parent)
    target.write_bytes(png_bytes)
    return target_relative
