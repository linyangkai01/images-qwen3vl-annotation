"""Industrial construction hazard auto-annotation package.

This package implements the dual-channel (objects + hazard events) construction
safety annotation pipeline on top of the reusable ``traffic_cv_closed_loop``
infrastructure. It intentionally depends on the Python standard library only;
the sole external binary is ``ffmpeg`` (video decoding) and the optional
Ultralytics ``yolo`` CLI for real detection/training.
"""

__all__ = ["__version__"]

__version__ = "0.1.0"
