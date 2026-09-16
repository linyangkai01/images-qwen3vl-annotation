"""Hazard knowledge base: normalisation, compressed injection and scene routing.

The knowledge base JSON mixes two shapes: S1/S2 use
``scenes[].hazard_categories[].hazards[]`` (hazards inherit the category), while
S3 hangs ``hazards[]`` directly off the scene. :class:`KnowledgeBase` normalises
both into a flat ``list[HazardDef]`` and builds an ``id_index`` used for local
back-fill of ``category`` / ``description`` / ``norm_reference`` / ``action``.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable

from hazard_annotation.models import HazardDef, HazardReport, severity_rank
from traffic_cv_closed_loop.io_utils import read_json

#: Channel-A label -> scene rules (see ARCH decision 3).
SCENE_LABEL_RULES: dict[str, set[str]] = {
    "S1": {"hook", "rigging", "overhead_crane", "load"},
    "S2": {"ladder", "scaffold", "guardrail", "harness"},
}


class KnowledgeBase:
    """Normalised hazard knowledge base."""

    def __init__(
        self,
        hazards: list[HazardDef],
        severity_levels: dict[str, Any],
        ten_no_lift: list[dict[str, Any]],
        crane_six_forbidden: list[dict[str, Any]],
        scenes: list[dict[str, Any]],
        meta: dict[str, Any],
    ) -> None:
        self.hazards = hazards
        self.severity_levels = severity_levels
        self.ten_no_lift = ten_no_lift
        self.crane_six_forbidden = crane_six_forbidden
        self.scenes = scenes
        self.meta = meta

        self.id_index: dict[str, HazardDef] = {}
        for hazard in hazards:
            if hazard.id in self.id_index:
                raise ValueError(f"Duplicate hazard id in knowledge base: {hazard.id}")
            self.id_index[hazard.id] = hazard

    # -- construction -------------------------------------------------------
    @classmethod
    def load(cls, path: str | Path) -> "KnowledgeBase":
        raw = read_json(Path(path))
        if raw is None:
            raise FileNotFoundError(f"Knowledge base not found: {path}")

        hazards: list[HazardDef] = []
        scenes = list(raw.get("scenes", []))
        for scene in scenes:
            scene_id = str(scene.get("scene_id", ""))
            categories = scene.get("hazard_categories") or []
            for category in categories:
                category_name = str(category.get("category_name", ""))
                for item in category.get("hazards", []):
                    hazards.append(cls._build_hazard(item, category_name, scene_id))
            for item in scene.get("hazards", []):
                hazards.append(
                    cls._build_hazard(item, str(scene.get("scene_name", "通用隐患")), scene_id)
                )

        severity_levels = dict(raw.get("severity_levels", {}))
        ten_no_lift = list(raw.get("ten_no_lift", {}).get("rules", []))
        crane_six_forbidden = list(raw.get("crane_six_forbidden", {}).get("rules", []))
        return cls(
            hazards=hazards,
            severity_levels=severity_levels,
            ten_no_lift=ten_no_lift,
            crane_six_forbidden=crane_six_forbidden,
            scenes=scenes,
            meta=dict(raw.get("meta", {})),
        )

    @staticmethod
    def _build_hazard(item: dict[str, Any], category: str, scene_id: str) -> HazardDef:
        return HazardDef(
            id=str(item.get("id", "")),
            name=str(item.get("name", "")),
            category=category,
            severity=str(item.get("severity", "medium")),
            visual_cues=[str(cue) for cue in item.get("visual_cues", [])],
            description=str(item.get("description", "")),
            norm_ref=str(item.get("norm_ref", "")),
            action=str(item.get("action", "")),
            scene_id=scene_id,
        )

    # -- accessors ----------------------------------------------------------
    def scene_ids(self) -> list[str]:
        return [str(scene.get("scene_id", "")) for scene in self.scenes]

    def scene_name(self, scene_id: str) -> str:
        for scene in self.scenes:
            if str(scene.get("scene_id", "")) == scene_id:
                return str(scene.get("scene_name", scene_id))
        return scene_id

    def get(self, hazard_id: str) -> HazardDef | None:
        return self.id_index.get(hazard_id)

    def hazards_for_scenes(self, scene_ids: Iterable[str]) -> list[HazardDef]:
        wanted = {str(scene_id) for scene_id in scene_ids}
        if not wanted:
            return list(self.hazards)
        return [hazard for hazard in self.hazards if hazard.scene_id in wanted]

    @property
    def total(self) -> int:
        return len(self.hazards)


class KnowledgeCompressor:
    """Build the compressed injection brief and back-fill hazard fields."""

    def __init__(
        self,
        kb: KnowledgeBase,
        include_ten_no_lift: bool = True,
        include_crane_six_forbidden: bool = True,
    ) -> None:
        self.kb = kb
        self.include_ten_no_lift = include_ten_no_lift
        self.include_crane_six_forbidden = include_crane_six_forbidden
        self.cache: dict[tuple[tuple[str, ...], int], str] = {}
        self.warnings: list[str] = []

    def build_brief(self, scene_ids: list[str], max_chars: int) -> str:
        """Return the injection text limited to ``max_chars`` characters."""
        max_chars = max(200, int(max_chars))
        scenes = list(scene_ids) if scene_ids else self.kb.scene_ids()
        cache_key = (tuple(sorted(scenes)), max_chars)
        if cache_key in self.cache:
            return self.cache[cache_key]

        hazards = sorted(
            self.kb.hazards_for_scenes(scenes),
            key=lambda item: (-severity_rank(item.severity), item.id),
        )

        lines = ["【隐患编号清单（仅可返回以下 id）】"]
        used = len(lines[0])
        truncated = False
        for hazard in hazards:
            cues = "、".join(hazard.visual_cues[:4])
            line = f"- {hazard.id} | {hazard.name} | 线索: {cues} | 等级: {hazard.severity}"
            if used + 1 + len(line) > max_chars:
                truncated = True
                break
            lines.append(line)
            used += 1 + len(line)

        if "S1" in scenes and self.include_ten_no_lift and self.kb.ten_no_lift:
            block = self._rules_block("【吊装红线·十不吊】", self.kb.ten_no_lift)
            if used + 1 + len(block) <= max_chars:
                lines.append(block)
                used += 1 + len(block)
        if "S1" in scenes and self.include_crane_six_forbidden and self.kb.crane_six_forbidden:
            block = self._rules_block("【行车六严禁】", self.kb.crane_six_forbidden)
            if used + 1 + len(block) <= max_chars:
                lines.append(block)
                used += 1 + len(block)

        if truncated:
            note = "（已按严重度截断，仅保留高等级隐患）"
            if used + 1 + len(note) <= max_chars:
                lines.append(note)
            self.warnings.append(
                f"kb_brief truncated for scenes={scenes} at max_chars={max_chars}"
            )

        text = "\n".join(lines)
        if len(text) > max_chars:
            text = text[:max_chars]
        self.cache[cache_key] = text
        return text

    @staticmethod
    def _rules_block(header: str, rules: list[dict[str, Any]]) -> str:
        body = [header]
        for rule in rules:
            body.append(f"- {rule.get('id', '')} {rule.get('rule', '')}")
        return "\n".join(body)

    def backfill(self, report: HazardReport) -> HazardReport:
        """Fill authoritative fields from the knowledge base by hazard id."""
        for hazard in report.hazards:
            definition = self.kb.get(hazard.id)
            if definition is None:
                hazard.invalid_id = True
                if not hazard.category:
                    hazard.category = "未知编号"
                continue
            hazard.category = definition.category or hazard.category
            if not hazard.description:
                hazard.description = definition.name or definition.description
            hazard.severity = definition.severity or hazard.severity
            if definition.norm_ref:
                hazard.norm_reference = definition.norm_ref
            if definition.action:
                hazard.action = definition.action
        return report


class SceneRouter:
    """Rule-first scene routing with optional model fallback."""

    def __init__(
        self,
        mode: str = "rules",
        classifier: Callable[[list[Any]], list[str]] | None = None,
    ) -> None:
        self.mode = (mode or "rules").lower()
        self.classifier = classifier

    def route(self, prior_detections: list[Any]) -> list[str]:
        labels = {getattr(item, "label", "") for item in prior_detections}
        scenes = self._route_labels(labels)
        if scenes:
            return scenes
        if self.mode in ("auto", "model") and self.classifier is not None:
            return list(self.classifier(prior_detections))
        return []

    def route_by_video(self, prior_frames: list[list[Any]]) -> list[str]:
        """Vote the scene across the first N frames of a (fixed-camera) video."""
        votes: Counter[str] = Counter()
        for frame in prior_frames:
            for scene_id in self.route(frame):
                votes[scene_id] += 1
        if not votes:
            return []
        top = max(votes.values())
        return sorted(scene_id for scene_id, count in votes.items() if count == top)

    @staticmethod
    def _route_labels(labels: set[str]) -> list[str]:
        has_s1 = bool(labels & SCENE_LABEL_RULES["S1"])
        has_s2 = bool(labels & SCENE_LABEL_RULES["S2"])
        if has_s1 and has_s2:
            return ["S1", "S2"]
        if has_s1:
            return ["S1"]
        if has_s2:
            return ["S2"]
        return []
