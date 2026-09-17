"""Mock-asset builder and one-command demo.

``build_mock_assets`` synthesises the three demo scenarios (hoisting multi-hazard,
height/ladder violation, no-hazard) as small ``.ppm`` frames with the three
sidecar types. ``run_demo`` builds the assets (when in mock mode) and runs the
whole extract -> annotate -> aggregate -> train -> visualize chain with no video,
no model and no ffmpeg.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from traffic_cv_closed_loop.config import ProjectConfig
from traffic_cv_closed_loop.io_utils import iter_image_files, write_json, write_text

#: Edge length (px) of the synthetic mock frames.
#:
#: Must stay **>= 10**: Ultralytics discards any image whose shorter side is under
#: 10 px ("ignoring corrupt image/label: image size (4, 4) <10 pixels"), which
#: would empty the train split and abort with ``No labels found``. 64 px gives a
#: comfortable margin while keeping the committed ``.ppm`` fixtures small. Each
#: frame is a solid colour, so resizing does not change any semantics: the
#: normalised bboxes and the sidecars are unaffected.
MOCK_FRAME_SIZE: int = 64

#: Ultralytics' minimum image edge length (``ultralytics/data/utils.py``).
#: A mock frame below this is unusable for training and must be regenerated.
ULTRALYTICS_MIN_PIXELS: int = 10

#: Object tuple: (label, bbox, confidence)
_BBOX = {
    "crane": (0.05, 0.05, 0.95, 0.35),
    "hook": (0.45, 0.30, 0.55, 0.42),
    "rigging": (0.40, 0.40, 0.60, 0.70),
    "load": (0.35, 0.68, 0.65, 0.90),
    "person": (0.70, 0.45, 0.80, 0.95),
    "person2": (0.82, 0.50, 0.92, 0.98),
    "ladder": (0.30, 0.20, 0.45, 0.90),
    "person_top": (0.30, 0.10, 0.45, 0.45),
    "guardrail": (0.10, 0.55, 0.90, 0.62),
    "helmet": (0.70, 0.45, 0.80, 0.52),
    "vest": (0.70, 0.52, 0.80, 0.75),
}


def _render_objects(*names: str) -> list[dict[str, Any]]:
    label_map = {
        "crane": "overhead_crane",
        "person": "person",
        "person2": "person",
        "person_top": "person",
    }
    rendered = []
    for name in names:
        rendered.append(
            {
                "label": label_map.get(name, name),
                "bbox": list(_BBOX[name]),
                "confidence": 0.9,
                "notes": "",
            }
        )
    return rendered


def _ppm(path: Path, rgb: tuple[int, int, int], size: int = MOCK_FRAME_SIZE) -> None:
    """Write a solid-colour P3 PPM frame of ``size`` x ``size`` pixels.

    Solid colour is intentional: the frame's semantics live entirely in the
    sidecar (objects + hazard report), and a solid frame resizes losslessly.
    """
    red, green, blue = rgb
    header = f"P3\n{size} {size}\n255\n"
    row = " ".join(f"{red} {green} {blue}" for _ in range(size))
    body = "\n".join(row for _ in range(size)) + "\n"
    write_text(path, header + body)


def _frame_specs() -> list[dict[str, Any]]:
    hoist_hazard = {
        "scene_type": "吊装作业",
        "hazards": [
            {"id": "D-1-1", "evidence": "龙门吊吊钩处吊链呈X形交叉", "confidence": 0.95, "severity": "high"},
            {"id": "D-3-1", "evidence": "吊运钢材为散件状态，未见料斗或可靠捆绑", "confidence": 0.88, "severity": "high"},
            {"id": "D-5-1", "evidence": "吊物下方2名工人头部无安全帽", "confidence": 0.92, "severity": "high"},
            {"id": "D-5-3", "evidence": "2名工人位于吊物正下方区域", "confidence": 0.90, "severity": "critical"},
            {"id": "D-4-1", "evidence": "画面整体偏暗，阴影区域较大", "confidence": 0.75, "severity": "medium"},
        ],
        "ten_no_lift_violations": ["TNR-7: 吊下站人、物件带浮件不吊"],
        "crane_six_forbidden_violations": [],
        "overall_risk": "critical",
        "recommendation": "立即停止吊装作业，撤离吊物下方人员，整理吊链并改用料斗吊运散件，增设照明后恢复作业。",
    }
    hoist_hazard_2 = {
        "scene_type": "吊装作业",
        "hazards": [
            {"id": "D-1-1", "evidence": "吊链呈X形交叉", "confidence": 0.9, "severity": "high"},
        ],
        "ten_no_lift_violations": [],
        "crane_six_forbidden_violations": [],
        "overall_risk": "high",
        "recommendation": "停止作业，重新整理吊链。",
    }
    height_hazard = {
        "scene_type": "高处作业",
        "hazards": [
            {"id": "G-4-6", "evidence": "工人双脚位于A字梯最高一级踏面", "confidence": 0.93, "severity": "high"},
            {"id": "G-4-5", "evidence": "梯子周围未见其他人员扶持", "confidence": 0.91, "severity": "high"},
            {"id": "G-4-4", "evidence": "梯脚直接接触地面，未见防滑橡胶垫", "confidence": 0.85, "severity": "medium"},
            {"id": "G-2-2", "evidence": "工人腰间未见安全带", "confidence": 0.88, "severity": "critical"},
        ],
        "ten_no_lift_violations": [],
        "crane_six_forbidden_violations": [],
        "overall_risk": "critical",
        "recommendation": "立即停止作业，安排专人扶持梯具，不得站在最高梯面，须系挂安全带并安装防滑垫。",
    }
    height_hazard_2 = {
        "scene_type": "高处作业",
        "hazards": [
            {"id": "G-2-2", "evidence": "高处作业人员未系安全带", "confidence": 0.86, "severity": "critical"},
        ],
        "ten_no_lift_violations": [],
        "crane_six_forbidden_violations": [],
        "overall_risk": "critical",
        "recommendation": "必须系挂安全带后方可作业。",
    }
    safe_report = {
        "scene_type": "吊装作业",
        "hazards": [],
        "ten_no_lift_violations": [],
        "crane_six_forbidden_violations": [],
        "overall_risk": "none",
        "observations": ["吊链垂直受力，无缠绕", "使用专用吊具", "吊物下方无人员", "操作员佩戴安全帽和反光背心"],
        "recommendation": "当前作业状态符合安全规范，继续保持。",
    }

    return [
        {
            "source_id": "hoist_scene",
            "name": "frame_0001",
            "rgb": (120, 60, 20),
            "prior": _render_objects("crane", "hook", "load"),
            "objects": _render_objects("crane", "hook", "load", "rigging", "person", "person2"),
            "hazard_report": hoist_hazard,
            "combined": True,
        },
        {
            "source_id": "hoist_scene",
            "name": "frame_0002",
            "rgb": (90, 90, 90),
            "prior": _render_objects("crane", "hook", "rigging", "load"),
            "objects": _render_objects("crane", "hook", "rigging", "load"),
            "hazard_report": hoist_hazard_2,
            "combined": True,
        },
        {
            "source_id": "height_scene",
            "name": "frame_0001",
            "rgb": (200, 200, 210),
            "prior": _render_objects("ladder", "person_top"),
            "objects": _render_objects("ladder", "person_top", "guardrail"),
            "hazard_report": height_hazard,
            "combined": False,
        },
        {
            "source_id": "height_scene",
            "name": "frame_0002",
            "rgb": (180, 180, 190),
            "prior": _render_objects("ladder", "person_top"),
            "objects": _render_objects("ladder", "person_top"),
            "hazard_report": height_hazard_2,
            "combined": False,
        },
        {
            "source_id": "safe_scene",
            "name": "frame_0001",
            "rgb": (60, 140, 60),
            "prior": _render_objects("crane", "hook", "rigging", "load", "person", "helmet", "vest"),
            "objects": _render_objects("crane", "hook", "rigging", "load", "person", "helmet", "vest"),
            "hazard_report": safe_report,
            "combined": True,
        },
    ]


def build_mock_assets(project_dir: Path) -> dict[str, Any]:
    """Create the mock frames and sidecars (idempotent)."""
    frames_root = project_dir / "input" / "frames"
    created = 0
    for spec in _frame_specs():
        source_dir = frames_root / spec["source_id"]
        source_dir.mkdir(parents=True, exist_ok=True)
        image_path = source_dir / f"{spec['name']}.ppm"
        _ppm(image_path, spec["rgb"])

        write_json(
            image_path.with_suffix(".detections.json"),
            {"objects": spec["prior"]},
        )

        if spec["combined"]:
            write_json(
                image_path.with_suffix(".qwen.json"),
                {"objects": spec["objects"], "hazard_report": spec["hazard_report"]},
            )
        else:
            write_json(image_path.with_suffix(".qwen.json"), {"objects": spec["objects"]})
            write_json(image_path.with_suffix(".hazard.json"), spec["hazard_report"])
        created += 1
    return {"frames_root": str(frames_root), "frame_count": created}


def _mock_frames_are_trainable(frames_dir: Path) -> bool:
    """True when every existing mock frame is large enough for Ultralytics.

    The mock frames are **generated build artifacts** (``examples/*/input/`` is
    git-ignored), so an existing checkout can still hold frames produced by an
    older, undersized generator. Those are unusable for real training, hence
    this staleness check drives a regeneration.
    """
    from hazard_annotation.imageio import load_image

    images = iter_image_files(frames_dir)
    if not images:
        return False
    for path in images:
        try:
            image = load_image(path)
        except (ValueError, OSError):
            return False
        if min(image.width, image.height) < ULTRALYTICS_MIN_PIXELS:
            return False
    return True


def ensure_mock_assets(config: ProjectConfig) -> dict[str, Any] | None:
    """Build the mock frames/sidecars when running mock mode on an empty project.

    Returns the asset summary when assets were (re)created, or ``None`` when the
    project already contains **usable** frames or is not running in mock mode.
    This lets ``python -m hazard_annotation all --mock`` work end-to-end on a
    fresh project without a separate ``demo`` invocation.

    The frames are regenerated when the directory is empty **or** when the
    existing frames are too small for Ultralytics (stale artifacts from an older
    generator) — otherwise real training would silently see 0 usable images.
    """
    if config.qwen_provider.lower() != "mock":
        return None
    frames_dir = config.input_frames_dir
    has_frames = frames_dir.exists() and any(path.is_file() for path in frames_dir.rglob("*"))
    if has_frames and _mock_frames_are_trainable(frames_dir):
        return None
    return build_mock_assets(config.project_dir)


def run_demo(config: ProjectConfig) -> dict[str, Any]:
    """Run the full pipeline on mock data with no external dependencies."""
    from hazard_annotation.pipeline.aggregate import run_aggregate
    from hazard_annotation.pipeline.annotation import run_annotation
    from hazard_annotation.pipeline.extraction import run_extraction
    from hazard_annotation.pipeline.training import run_training
    from hazard_annotation.pipeline.visualization import run_visualize

    assets: dict[str, Any] = {}
    if config.qwen_provider.lower() == "mock":
        assets = build_mock_assets(config.project_dir)

    payload: dict[str, Any] = {"assets": assets}
    payload["extract"] = run_extraction(config)
    payload["annotate"] = run_annotation(config)
    payload["aggregate"] = run_aggregate(config)
    payload["train"] = run_training(config, execute_trainer=False)
    payload["visualize"] = run_visualize(config)
    return payload
