"""Stage 5 — self-contained static HTML report.

Everything (CSS, JS, thumbnails, charts) is inlined so the report opens by
double-clicking, with no server or network dependency. Netpbm keyframes are
re-encoded to PNG in pure Python so they render in any browser.
"""

from __future__ import annotations

import base64
import html
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from hazard_annotation.imageio import image_to_png_bytes
from hazard_annotation.knowledge import KnowledgeBase
from hazard_annotation.models import DualChannelRecord, severity_rank
from traffic_cv_closed_loop.config import ProjectConfig
from traffic_cv_closed_loop.io_utils import read_json, read_jsonl, write_json, write_text

_DISPLAY_SUFFIXES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".bmp": "image/bmp",
    ".webp": "image/webp",
}
_LABEL_COLORS = {
    "person": "#00B0FF",
    "helmet": "#FFD600",
    "vest": "#FF9100",
    "harness": "#00E676",
    "hook": "#FF1744",
    "rigging": "#D500F9",
    "overhead_crane": "#FF6D00",
    "load": "#8D6E63",
    "ladder": "#00BFA5",
    "scaffold": "#2979FF",
    "guardrail": "#76FF03",
    "pallet": "#A1887F",
}
_DEFAULT_COLOR = "#9E9E9E"
_SEVERITY_FALLBACK = {
    "critical": "#FF0000",
    "high": "#FF6600",
    "medium": "#FFCC00",
    "low": "#00CC00",
    "none": "#BDBDBD",
}


def _inline_image(path: Path, target_width: int | None = None) -> str:
    if not path.exists():
        return ""
    suffix = path.suffix.lower()
    if suffix in _DISPLAY_SUFFIXES:
        payload = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{_DISPLAY_SUFFIXES[suffix]};base64,{payload}"
    png = image_to_png_bytes(path, target_width=target_width)
    if png is None:
        return ""
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


def _severity_color(level: str, severity_levels: dict[str, Any]) -> str:
    entry = severity_levels.get(level)
    if isinstance(entry, dict) and entry.get("color"):
        return str(entry["color"])
    return _SEVERITY_FALLBACK.get(level, _DEFAULT_COLOR)


def _severity_label(level: str, severity_levels: dict[str, Any]) -> str:
    entry = severity_levels.get(level)
    if isinstance(entry, dict) and entry.get("label"):
        return str(entry["label"])
    return level


def _scene_group(record: DualChannelRecord) -> str:
    scene_type = record.hazard_report.scene_type or ""
    ids = [hazard.id for hazard in record.hazard_report.hazards]
    if "高处" in scene_type or any(item.startswith("G-") for item in ids):
        return "S2"
    if "吊装" in scene_type or any(item.startswith("D-") for item in ids):
        return "S1"
    return "other"


# -- SVG charts -------------------------------------------------------------
def _render_bars(
    data: list[tuple[str, int]],
    color_map: dict[str, str],
    width: int = 420,
    height: int = 200,
) -> str:
    if not data:
        return f"<svg viewBox='0 0 {width} 60'><text x='10' y='30' fill='#888'>无数据</text></svg>"
    top = max(value for _, value in data) or 1
    bar_width = width / max(1, len(data))
    parts = [f"<svg viewBox='0 0 {width} {height}' role='img' aria-label='隐患类别分布'>"]
    for index, (label, value) in enumerate(data):
        bar_height = (value / top) * (height - 50)
        x = index * bar_width + 6
        y = height - 30 - bar_height
        color = color_map.get(label, _DEFAULT_COLOR)
        parts.append(
            f"<rect x='{x:.1f}' y='{y:.1f}' width='{bar_width - 12:.1f}' "
            f"height='{bar_height:.1f}' fill='{color}' rx='2'/>"
        )
        parts.append(
            f"<text x='{x + (bar_width - 12) / 2:.1f}' y='{height - 16}' font-size='9' "
            f"text-anchor='middle' fill='#333'>{html.escape(label)}</text>"
        )
        parts.append(
            f"<text x='{x + (bar_width - 12) / 2:.1f}' y='{y - 3:.1f}' font-size='9' "
            f"text-anchor='middle' fill='#333'>{value}</text>"
        )
    parts.append("</svg>")
    return "".join(parts)


def _render_pie(
    data: list[tuple[str, int]],
    color_map: dict[str, str],
    size: int = 200,
) -> str:
    total = sum(value for _, value in data)
    if total <= 0:
        return f"<svg viewBox='0 0 {size} {size}'><text x='10' y='30' fill='#888'>无数据</text></svg>"
    radius = size / 2 - 6
    center = size / 2
    parts = [f"<svg viewBox='0 0 {size} {size}' role='img' aria-label='风险等级分布'>"]
    angle = -90.0
    for label, value in data:
        sweep = 360.0 * value / total
        start = _polar(center, center, radius, angle)
        end = _polar(center, center, radius, angle + sweep)
        large = 1 if sweep > 180 else 0
        color = color_map.get(label, _DEFAULT_COLOR)
        parts.append(
            f"<path d='M{center},{center} L{start[0]:.1f},{start[1]:.1f} "
            f"A{radius:.1f},{radius:.1f} 0 {large} 1 {end[0]:.1f},{end[1]:.1f} Z' "
            f"fill='{color}' stroke='#fff' stroke-width='1'/>"
        )
        angle += sweep
    parts.append("</svg>")
    return "".join(parts)


def _polar(cx: float, cy: float, radius: float, degrees: float) -> tuple[float, float]:
    radians = math.radians(degrees)
    return (cx + radius * math.cos(radians), cy + radius * math.sin(radians))


def _render_line(points: list[int], width: int = 460, height: int = 160) -> str:
    if not points:
        return f"<svg viewBox='0 0 {width} 60'><text x='10' y='30' fill='#888'>无数据</text></svg>"
    maximum = max(points) or 1
    step = width / max(1, len(points) - 1)
    coords = []
    for index, value in enumerate(points):
        x = index * step
        y = height - 20 - (value / maximum) * (height - 40)
        coords.append(f"{x:.1f},{y:.1f}")
    polyline = " ".join(coords)
    return (
        f"<svg viewBox='0 0 {width} {height}' role='img' aria-label='隐患随时间分布'>"
        f"<polyline fill='none' stroke='#FF6600' stroke-width='2' points='{polyline}'/>"
        "</svg>"
    )


def _render_html(
    config: ProjectConfig,
    records: list[DualChannelRecord],
    report: dict[str, Any],
    metrics: dict[str, Any],
    severity_levels: dict[str, Any],
) -> str:
    mock_mode = (
        config.qwen_provider.lower() == "mock"
        or config.yolo_detector.lower() == "sidecar"
        or config.yolo_trainer.lower() == "mock"
    )
    overall_risk = str(report.get("overall_risk", "none"))
    risk_color = _severity_color(overall_risk, severity_levels)
    risk_label = _severity_label(overall_risk, severity_levels)

    thumbnail_size = int(config.visualization.thumbnail_size)
    frames: list[dict[str, Any]] = []
    for record in records:
        image_path = Path(record.keyframe.extracted_path)
        hazards = []
        for hazard in record.hazard_report.hazards:
            hazards.append(
                {
                    "id": hazard.id,
                    "name": hazard.description or hazard.id,
                    "severity": hazard.severity,
                    "color": _severity_color(hazard.severity, severity_levels),
                    "norm_reference": hazard.norm_reference,
                    "action": hazard.action,
                    "evidence": hazard.evidence,
                    "confidence": round(float(hazard.confidence), 3),
                }
            )
        frames.append(
            {
                "relative_path": record.keyframe.relative_path,
                "source_id": record.keyframe.source_id,
                "timestamp": round(float(record.keyframe.timestamp), 2),
                "scene_group": _scene_group(record),
                "scene_type": record.hazard_report.scene_type,
                "overall_risk": record.hazard_report.overall_risk,
                "risk_color": _severity_color(record.hazard_report.overall_risk, severity_levels),
                "review_required": record.review_required,
                "review_reasons": record.review_reasons,
                "thumbnail": _inline_image(image_path, thumbnail_size),
                "image": _inline_image(image_path, 640),
                "objects": [
                    {
                        "label": item.label,
                        "bbox": list(item.bbox),
                        "confidence": round(float(item.confidence), 3),
                        "color": _LABEL_COLORS.get(item.label, _DEFAULT_COLOR),
                    }
                    for item in record.objects
                ],
                "hazards": hazards,
                "recommendation": record.hazard_report.recommendation,
            }
        )

    hazard_counter: Counter[str] = Counter()
    severity_counter: Counter[str] = Counter()
    for record in records:
        for hazard in record.hazard_report.hazards:
            hazard_counter[hazard.id] += 1
            severity_counter[hazard.severity] += 1

    bars = _render_bars(hazard_counter.most_common(10), _LABEL_COLORS)
    severity_colors = {level: _severity_color(level, severity_levels) for level in severity_counter}
    pie = _render_pie(
        sorted(severity_counter.items(), key=lambda item: -severity_rank(item[0])),
        severity_colors,
    )
    line = _render_line([len(record.hazard_report.hazards) for record in records])

    compression = report.get("compression", {}) if isinstance(report, dict) else {}
    metric_cards = [
        ("mAP50", metrics.get("mAP50", "-")),
        ("Precision", metrics.get("precision", "-")),
        ("Recall", metrics.get("recall", "-")),
        ("关键帧数", report.get("keyframe_count", len(records))),
        ("压缩比", f"{compression.get('compression_ratio', 0)}:1"),
    ]
    cards_html = "".join(
        f"<div class='metric'><div class='metric-value'>{html.escape(str(value))}</div>"
        f"<div class='metric-label'>{html.escape(label)}</div></div>"
        for label, value in metric_cards
    )

    banner_text = "演示数据（mock）" if mock_mode else "真实数据"
    banner_class = "banner mock" if mock_mode else "banner real"
    data_json = json.dumps({"frames": frames}, ensure_ascii=False).replace("</", "<\\/")

    replacements = {
        "__PROJECT_NAME__": html.escape(config.name),
        "__BANNER_TEXT__": html.escape(banner_text),
        "__BANNER_CLASS__": banner_class,
        "__RISK_COLOR__": risk_color,
        "__RISK_LABEL__": html.escape(risk_label),
        "__OVERALL_RISK__": html.escape(overall_risk),
        "__PROVIDER__": html.escape(config.qwen_provider),
        "__FRAME_COUNT__": str(len(frames)),
        "__CARDS_HTML__": cards_html,
        "__BARS_SVG__": bars,
        "__PIE_SVG__": pie,
        "__LINE_SVG__": line,
        "__DATA_JSON__": data_json,
    }
    output = _PAGE_TEMPLATE
    for token, value in replacements.items():
        output = output.replace(token, value)
    return output


_PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>施工安全隐患标注报告 — __PROJECT_NAME__</title>
<style>
  * { box-sizing: border-box; }
  body { margin: 0; font-family: "Microsoft YaHei", "PingFang SC", Arial, sans-serif; background: #f5f6f8; color: #1c1c1c; }
  .banner { padding: 8px 16px; font-weight: 700; text-align: center; }
  .banner.mock { background: #fff3cd; color: #8a6d3b; }
  .banner.real { background: #e3f2fd; color: #0d47a1; }
  .risk-band { background: __RISK_COLOR__; color: #fff; padding: 16px 20px; display: flex; justify-content: space-between; align-items: center; }
  .risk-band .title { font-size: 20px; font-weight: 700; }
  .risk-band .meta { font-size: 13px; opacity: 0.95; }
  .tabs { display: flex; gap: 8px; padding: 10px 16px; background: #fff; border-bottom: 1px solid #e0e0e0; }
  .tabs button { border: 1px solid #ccc; background: #fafafa; padding: 6px 14px; border-radius: 16px; cursor: pointer; font-size: 13px; }
  .tabs button.active { background: #1976d2; color: #fff; border-color: #1976d2; }
  .layout { display: grid; grid-template-columns: 220px 1fr 360px; gap: 12px; padding: 12px; }
  .panel { background: #fff; border: 1px solid #e6e6e6; border-radius: 8px; padding: 10px; }
  .timeline { max-height: 70vh; overflow-y: auto; display: flex; flex-direction: column; gap: 8px; }
  .thumb { position: relative; border: 2px solid transparent; border-radius: 6px; cursor: pointer; overflow: hidden; }
  .thumb.active { border-color: #1976d2; }
  .thumb img { width: 100%; display: block; background: #eee; }
  .thumb .dot { position: absolute; top: 4px; right: 4px; width: 12px; height: 12px; border-radius: 50%; border: 2px solid #fff; }
  .thumb .ts { position: absolute; bottom: 0; left: 0; right: 0; background: rgba(0,0,0,0.55); color: #fff; font-size: 11px; padding: 2px 4px; }
  .stage { position: relative; background: #111; border-radius: 8px; min-height: 320px; display: flex; align-items: center; justify-content: center; overflow: hidden; }
  .stage img { max-width: 100%; max-height: 70vh; display: block; }
  .stage .boxes { position: absolute; inset: 0; }
  .box { position: absolute; border: 2px solid #fff; border-radius: 3px; }
  .box span { position: absolute; top: -18px; left: -2px; font-size: 11px; color: #fff; background: rgba(0,0,0,0.6); padding: 1px 4px; border-radius: 3px; white-space: nowrap; }
  .cards { display: flex; flex-direction: column; gap: 10px; max-height: 70vh; overflow-y: auto; }
  .card { border: 1px solid #e0e0e0; border-left-width: 6px; border-radius: 6px; padding: 8px 10px; }
  .card .h-id { font-weight: 700; }
  .card .h-name { margin-left: 6px; }
  .card .row { font-size: 12px; color: #444; margin-top: 4px; }
  .card .chip { display: inline-block; color: #fff; border-radius: 10px; padding: 1px 8px; font-size: 11px; margin-left: 6px; }
  .stats { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; padding: 0 12px 20px; }
  .stats .panel h3 { margin: 4px 0 8px; font-size: 14px; }
  .metrics { display: flex; gap: 20px; flex-wrap: wrap; }
  .metric { text-align: center; }
  .metric-value { font-size: 22px; font-weight: 700; color: #1976d2; }
  .metric-label { font-size: 12px; color: #666; }
  .muted { color: #888; font-size: 12px; }
</style>
</head>
<body>
  <div class="__BANNER_CLASS__">__BANNER_TEXT__</div>
  <div class="risk-band">
    <div class="title">整体风险：__RISK_LABEL__（__OVERALL_RISK__）</div>
    <div class="meta">项目：__PROJECT_NAME__ · 关键帧：__FRAME_COUNT__ · provider：__PROVIDER__</div>
  </div>
  <div class="tabs">
    <button data-group="all" class="active">全部</button>
    <button data-group="S1">吊装作业</button>
    <button data-group="S2">高处作业</button>
    <button data-group="other">其他</button>
    <button id="toggle-boxes" style="margin-left:auto">显示/隐藏检出框</button>
  </div>
  <div class="layout">
    <div class="panel">
      <h3 style="margin:4px 0 8px;font-size:14px;">关键帧时间轴</h3>
      <div class="timeline" id="timeline"></div>
    </div>
    <div class="panel">
      <div class="stage" id="stage">
        <img id="stage-image" alt="关键帧" src=""/>
        <div class="boxes" id="stage-boxes"></div>
      </div>
      <div class="muted" id="stage-caption" style="margin-top:8px;"></div>
    </div>
    <div class="panel">
      <h3 style="margin:4px 0 8px;font-size:14px;">隐患列表</h3>
      <div class="cards" id="cards"></div>
    </div>
  </div>
  <div class="stats">
    <div class="panel"><h3>隐患类别分布</h3>__BARS_SVG__</div>
    <div class="panel"><h3>风险等级分布</h3>__PIE_SVG__</div>
    <div class="panel"><h3>隐患随时间分布</h3>__LINE_SVG__</div>
  </div>
  <div class="panel" style="margin:0 12px 24px;">
    <h3>通道 A 指标</h3>
    <div class="metrics">__CARDS_HTML__</div>
  </div>
  <script>
  const DATA = __DATA_JSON__;
  const frames = DATA.frames;
  let currentGroup = "all";
  let currentIndex = 0;
  let showBoxes = true;

  function visibleFrames() {
    return frames.filter(f => currentGroup === "all" || f.scene_group === currentGroup);
  }

  function renderTimeline() {
    const container = document.getElementById("timeline");
    container.innerHTML = "";
    const list = visibleFrames();
    list.forEach((frame) => {
      const globalIndex = frames.indexOf(frame);
      const div = document.createElement("div");
      div.className = "thumb" + (globalIndex === currentIndex ? " active" : "");
      div.innerHTML = '<img src="' + frame.thumbnail + '" alt=""/>' +
        '<span class="dot" style="background:' + frame.risk_color + '"></span>' +
        '<span class="ts">#' + globalIndex + ' · ' + frame.timestamp + 's</span>';
      div.onclick = () => { currentIndex = globalIndex; renderTimeline(); renderStage(); };
      container.appendChild(div);
    });
    if (list.length && !list.some(f => frames.indexOf(f) === currentIndex)) {
      currentIndex = frames.indexOf(list[0]);
    }
  }

  function renderStage() {
    const frame = frames[currentIndex];
    if (!frame) return;
    document.getElementById("stage-image").src = frame.image;
    const boxes = document.getElementById("stage-boxes");
    boxes.innerHTML = "";
    boxes.style.display = showBoxes ? "block" : "none";
    if (showBoxes) {
      frame.objects.forEach((obj) => {
        const bbox = obj.bbox;
        const box = document.createElement("div");
        box.className = "box";
        box.style.left = (bbox[0] * 100) + "%";
        box.style.top = (bbox[1] * 100) + "%";
        box.style.width = ((bbox[2] - bbox[0]) * 100) + "%";
        box.style.height = ((bbox[3] - bbox[1]) * 100) + "%";
        box.style.borderColor = obj.color;
        box.innerHTML = '<span style="background:' + obj.color + '">' + obj.label + ' ' + obj.confidence + '</span>';
        boxes.appendChild(box);
      });
    }
    document.getElementById("stage-caption").textContent =
      frame.relative_path + " · 场景：" + (frame.scene_type || "未知") +
      " · 风险：" + frame.overall_risk + (frame.review_required ? " · 需复核(" + frame.review_reasons.join(",") + ")" : "");
    renderCards(frame);
  }

  function renderCards(frame) {
    const container = document.getElementById("cards");
    container.innerHTML = "";
    if (!frame.hazards.length) {
      container.innerHTML = '<div class="muted">未检出隐患（overall_risk=' + frame.overall_risk + '）。</div>';
      if (frame.recommendation) {
        container.innerHTML += '<div class="row">建议：' + frame.recommendation + '</div>';
      }
      return;
    }
    frame.hazards.forEach((h) => {
      const card = document.createElement("div");
      card.className = "card";
      card.style.borderLeftColor = h.color;
      card.innerHTML =
        '<div><span class="h-id">' + h.id + '</span><span class="h-name">' + h.name + '</span>' +
        '<span class="chip" style="background:' + h.color + '">' + h.severity + '</span></div>' +
        '<div class="row">规范依据：' + (h.norm_reference || "—") + '</div>' +
        '<div class="row">整改建议：' + (h.action || "—") + '</div>' +
        '<div class="row">证据：' + (h.evidence || "—") + ' · 置信度 ' + h.confidence + '</div>';
      container.appendChild(card);
    });
    if (frame.recommendation) {
      const note = document.createElement("div");
      note.className = "row";
      note.textContent = "整体建议：" + frame.recommendation;
      container.appendChild(note);
    }
  }

  document.querySelectorAll(".tabs button[data-group]").forEach((button) => {
    button.onclick = () => {
      currentGroup = button.dataset.group;
      document.querySelectorAll(".tabs button[data-group]").forEach((b) => b.classList.remove("active"));
      button.classList.add("active");
      renderTimeline();
      renderStage();
    };
  });
  document.getElementById("toggle-boxes").onclick = () => { showBoxes = !showBoxes; renderStage(); };

  renderTimeline();
  renderStage();
  </script>
</body>
</html>
"""


def run_visualize(
    config: ProjectConfig,
    records: list[DualChannelRecord] | None = None,
    report: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if records is None:
        records = [DualChannelRecord.from_dict(item) for item in read_jsonl(config.annotations_path)]
    if report is None:
        report = read_json(config.hazard_report_path, default={}) or {}
    if metrics is None:
        latest = read_json(config.latest_run_path, default={}) or {}
        metrics = latest.get("trainer", {}).get("metrics", {}) if isinstance(latest, dict) else {}

    severity_levels: dict[str, Any] = {}
    if config.kb_path.exists():
        severity_levels = KnowledgeBase.load(config.kb_path).severity_levels
    if config.hazard.severity_levels:
        severity_levels = {**severity_levels, **config.hazard.severity_levels}

    html_text = _render_html(config, records, report, metrics, severity_levels)
    config.visualization_dir.mkdir(parents=True, exist_ok=True)
    output_path = config.visualization_dir / "report.html"
    write_text(output_path, html_text)

    summary = {
        "project_name": config.name,
        "mode": config.run_mode,
        "frame_count": len(records),
        "overall_risk": report.get("overall_risk", "none"),
        "mock_mode": (
            config.qwen_provider.lower() == "mock"
            or config.yolo_detector.lower() == "sidecar"
            or config.yolo_trainer.lower() == "mock"
        ),
        "report_path": str(output_path.resolve()),
    }
    write_json(config.status_dir / "visualize.json", summary)
    return summary
