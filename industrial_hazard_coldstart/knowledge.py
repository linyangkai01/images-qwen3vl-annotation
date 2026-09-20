"""隐患知识库加载与**本地回填**（功能点 F5 的语义层，T03.3）。

设计立场（与需求 §1.3.10 一致）：隐患语义标注**已降级为内部手段**——它的价值不是
产出交付物，而是：

1. 用知识库把「该关注什么」显式告诉 VLM，避免漏掉甲方真正关心的目标；
2. 要求模型**只输出受控词表内的隐患 id + evidence**，等价于给输出加约束，显著降低幻觉；
3. ``category / norm_reference / action`` 一律由**本地知识库**回填，模型**无从伪造规范号**。

⚠️ **母项目没有 KB 解析能力**：``traffic_cv_closed_loop/config.py`` 只解析 ``kb_path``
（路径），从不读取 ``knowledge_base.json`` 的内容。故本模块从零补齐「加载 + 回填」，
只依赖标准库与母项目 ``io_utils``。

回填铁律：

- 模型输出的 ``hazard_id`` **必须在知识库中**，否则**拒绝**（记入复核），
  **绝不允许 KB 之外的规范号凭空出现在报告里**；
- 模型输出的 ``hazard_id`` 即使存在于 KB，若**不属于当前场景**（如吊装场景里出现高处作业
  的 ``G-x-x``），同样拒绝——受控词表按场景收窄；
- 输出里的 ``category / norm_reference / action`` **只来自 KB**，模型返回的同名字段被丢弃。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from traffic_cv_closed_loop.io_utils import read_json

__all__ = [
    # 场景映射
    "SCENE_TO_KB_SCENE_ID",
    "KB_SCENE_TO_CONFIG_SCENE",
    "GENERIC_KB_SCENE_ID",
    # 严重度
    "SEVERITY_ORDER",
    "VALID_OVERALL_RISK",
    "severity_rank",
    "overall_risk_of",
    "recommendation_of",
    # 数据结构
    "HazardEntry",
    "KnowledgeBase",
    # 加载
    "load_knowledge_base",
    # 回填
    "extract_model_hazards",
    "resolve_hazard",
    "backfill_hazards",
]

#: 配置里的 ``scene`` → 知识库 ``scene_id``（吊装 = S1，高处 = S2）。
SCENE_TO_KB_SCENE_ID: dict[str, str] = {
    "hoist": "S1",
    "height": "S2",
}

#: 知识库 ``scene_id`` → 配置里的 ``scene``（反向表，便于文档与追溯）。
KB_SCENE_TO_CONFIG_SCENE: dict[str, str] = {
    "S1": "hoist",
    "S2": "height",
    "S3": "generic",
}

#: 通用隐患场景（S3，如「场地积水」）—— 不属于任何具体作业场景。
GENERIC_KB_SCENE_ID = "S3"

#: 严重度从低到高（用于取「最高风险」）；不在表内的严重度视为最低。
SEVERITY_ORDER: tuple[str, ...] = ("low", "medium", "high", "critical")

#: ``overall_risk`` 的合法取值（含 ``none``）。
VALID_OVERALL_RISK: tuple[str, ...] = ("none",) + SEVERITY_ORDER


def severity_rank(severity: str) -> int:
    """返回严重度的排序权重；未知严重度返回 ``-1``（最低）。"""
    try:
        return SEVERITY_ORDER.index(str(severity or "").strip().lower())
    except ValueError:
        return -1


def overall_risk_of(
    hazards: list[dict[str, Any]],
    model_overall: str = "",
) -> str:
    """取「已回填隐患」中的最高严重度作为整体风险。

    若一条隐患都没有，则回退到模型给出的 ``overall_risk``（仅当其取值合法），
    否则返回 ``"none"``。**绝不接受模型在无隐患时凭空抬高整体风险**之外的值——
    这里只做白名单校验，不做自由文本透传。
    """
    best_rank = -1
    for item in hazards:
        best_rank = max(best_rank, severity_rank(item.get("severity", "")))
    if best_rank >= 0:
        return SEVERITY_ORDER[best_rank]
    candidate = str(model_overall or "").strip().lower()
    if candidate in VALID_OVERALL_RISK:
        return candidate
    return "none"


def recommendation_of(hazards: list[dict[str, Any]]) -> str:
    """把已回填隐患的 ``action`` 去重后拼成整改建议（中文分号分隔）。"""
    actions: list[str] = []
    seen: set[str] = set()
    for item in hazards:
        action = str(item.get("action", "") or "").strip()
        if action and action not in seen:
            seen.add(action)
            actions.append(action)
    return "；".join(actions)


# --------------------------------------------------------------------------- #
# 数据结构
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class HazardEntry:
    """知识库中的一条隐患（含所属类别与回填字段）。"""

    hazard_id: str
    name: str = ""
    category_id: str = ""
    category_name: str = ""
    severity: str = ""
    risk_type: str = ""
    #: 规范号——**回填的唯一来源**（KB 字段名为 ``norm_ref``，输出字段名为 ``norm_reference``）。
    norm_reference: str = ""
    action: str = ""
    description: str = ""
    visual_cues: list[str] = field(default_factory=list)
    scene_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "hazard_id": self.hazard_id,
            "name": self.name,
            "category_id": self.category_id,
            "category_name": self.category_name,
            "severity": self.severity,
            "risk_type": self.risk_type,
            "norm_reference": self.norm_reference,
            "action": self.action,
            "description": self.description,
            "visual_cues": list(self.visual_cues),
            "scene_id": self.scene_id,
        }

    def resolved(self, evidence: str, source_frame: str = "") -> dict[str, Any]:
        """把本条目回填成报告里的一条隐患（``category/norm_reference/action`` 全来自 KB）。"""
        return {
            "hazard_id": self.hazard_id,
            "name": self.name,
            "evidence": str(evidence or ""),
            "category": self.category_name,
            "category_id": self.category_id,
            "norm_reference": self.norm_reference,
            "action": self.action,
            "severity": self.severity,
            "risk_type": self.risk_type,
            "scene_id": self.scene_id,
            "source_frame": source_frame,
        }


@dataclass(slots=True)
class KnowledgeBase:
    """已加载的知识库：隐患索引 + 场景索引 + 每场景的受控 id 顺序。"""

    entries: dict[str, HazardEntry] = field(default_factory=dict)
    scenes: dict[str, dict[str, Any]] = field(default_factory=dict)
    scene_ids: dict[str, list[str]] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)
    source_path: str = ""

    def get(self, hazard_id: str) -> HazardEntry | None:
        """按 id 取隐患；不存在返回 ``None``。"""
        return self.entries.get(str(hazard_id or "").strip())

    def scene_hazard_ids(self, scene_id: str) -> list[str]:
        """返回某知识库场景（如 ``S1``）下的隐患 id 列表（保持 KB 原顺序）。"""
        return list(self.scene_ids.get(str(scene_id or "").strip(), []))

    def scene_id_for(self, scene: str) -> str:
        """把配置场景（如 ``hoist``）映射为知识库 ``scene_id``（如 ``S1``）。

        映射不到时尝试把 ``scene`` 直接当作 ``scene_id``（大小写不敏感）；
        仍匹配不到任何已知场景时返回空串。
        """
        key = str(scene or "").strip().lower()
        scene_id = SCENE_TO_KB_SCENE_ID.get(key)
        if scene_id is None:
            scene_id = str(scene or "").strip().upper()
        return scene_id if scene_id in self.scenes else ""

    def allowed_hazard_ids(self, scene: str) -> list[str]:
        """返回配置场景（如 ``hoist``）允许引用的受控隐患 id 列表。

        配置场景经 :data:`SCENE_TO_KB_SCENE_ID` 映射到知识库 ``scene_id``；
        映射不到时尝试把 ``scene`` 直接当作 ``scene_id``（大小写不敏感）。
        未知场景返回空列表——**空即全拒**（宁缺毋滥）。
        """
        return self.scene_hazard_ids(self.scene_id_for(scene))


# --------------------------------------------------------------------------- #
# 加载
# --------------------------------------------------------------------------- #
def _build_entry(
    raw: dict[str, Any],
    scene_id: str,
    category_id: str,
    category_name: str,
) -> HazardEntry | None:
    """把一条 KB 隐患条目转成 :class:`HazardEntry`；缺 ``id`` 时返回 ``None``。"""
    hazard_id = str(raw.get("id", "") or "").strip()
    if not hazard_id:
        return None
    return HazardEntry(
        hazard_id=hazard_id,
        name=str(raw.get("name", "") or ""),
        category_id=str(category_id or ""),
        category_name=str(category_name or ""),
        severity=str(raw.get("severity", "") or ""),
        risk_type=str(raw.get("risk_type", "") or ""),
        norm_reference=str(raw.get("norm_ref", "") or ""),
        action=str(raw.get("action", "") or ""),
        description=str(raw.get("description", "") or ""),
        visual_cues=[str(item) for item in (raw.get("visual_cues") or [])],
        scene_id=scene_id,
    )


def load_knowledge_base(path: str | Path) -> KnowledgeBase:
    """加载 ``knowledge_base.json``，建立隐患索引与场景索引。

    Args:
        path: 知识库文件路径（可含中文）。

    Returns:
        :class:`KnowledgeBase`（``entries`` / ``scenes`` / ``scene_ids`` / ``meta``）。

    Raises:
        FileNotFoundError: 文件不存在（**fail-loud**，绝不静默返回空库——
            否则下游会把「KB 没加载到」误判为「模型没报隐患」）。
        ValueError: 文件不是合法 JSON 对象。
    """
    kb_path = Path(path)
    if not kb_path.exists():
        raise FileNotFoundError(
            f"知识库文件不存在：{kb_path}（配置项 knowledge.kb_path）。"
            "回填阶段需要它来校验受控隐患 id 并回填 category/norm_reference/action。"
        )
    raw = read_json(kb_path)
    if not isinstance(raw, dict):
        raise ValueError(f"知识库格式非法（应为 JSON 对象）：{kb_path}")

    entries: dict[str, HazardEntry] = {}
    scenes: dict[str, dict[str, Any]] = {}
    scene_ids: dict[str, list[str]] = {}

    for scene in raw.get("scenes", []) or []:
        if not isinstance(scene, dict):
            continue
        scene_id = str(scene.get("scene_id", "") or "").strip()
        scenes[scene_id] = scene
        ordered: list[str] = []

        for category in scene.get("hazard_categories", []) or []:
            if not isinstance(category, dict):
                continue
            category_id = str(category.get("category_id", "") or "")
            category_name = str(category.get("category_name", "") or "")
            for item in category.get("hazards", []) or []:
                if not isinstance(item, dict):
                    continue
                entry = _build_entry(item, scene_id, category_id, category_name)
                if entry is None:
                    continue
                entries[entry.hazard_id] = entry
                ordered.append(entry.hazard_id)

        # 通用场景（S3）直接在 scene 下挂 hazards，没有 hazard_categories。
        for item in scene.get("hazards", []) or []:
            if not isinstance(item, dict):
                continue
            entry = _build_entry(item, scene_id, "", "")
            if entry is None:
                continue
            entries[entry.hazard_id] = entry
            ordered.append(entry.hazard_id)

        scene_ids[scene_id] = ordered

    return KnowledgeBase(
        entries=entries,
        scenes=scenes,
        scene_ids=scene_ids,
        meta=dict(raw.get("meta", {}) or {}),
        source_path=str(kb_path.resolve()),
    )


# --------------------------------------------------------------------------- #
# 回填
# --------------------------------------------------------------------------- #
def extract_model_hazards(hazard_report: Any) -> list[dict[str, str]]:
    """从模型返回的 ``hazard_report`` 里取出 ``[{id, evidence}, ...]``。

    兼容三种形态：``{"hazards": [...]}``、裸列表、以及缺字段/类型异常（返回空列表）。
    每条同时兼容 ``id`` 与 ``hazard_id`` 两种键名。**不做任何信任假设**——
    这里只做结构规整，真正的白名单校验在 :func:`resolve_hazard`。
    """
    if isinstance(hazard_report, dict):
        raw_items = hazard_report.get("hazards", [])
    elif isinstance(hazard_report, list):
        raw_items = hazard_report
    else:
        return []
    if not isinstance(raw_items, list):
        return []

    results: list[dict[str, str]] = []
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        hazard_id = item.get("id", item.get("hazard_id", ""))
        results.append(
            {
                "hazard_id": str(hazard_id or "").strip(),
                "evidence": str(item.get("evidence", "") or ""),
            }
        )
    return results


def resolve_hazard(
    hazard_id: str,
    evidence: str,
    kb: KnowledgeBase,
    allowed_ids: set[str] | frozenset[str],
    source_frame: str = "",
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """校验并回填**单条**隐患。

    Returns:
        ``(resolved, rejection)``：命中受控词表时返回回填后的隐患（``rejection=None``）；
        否则 ``resolved=None``，并给出带 ``reason`` 的拒绝记录。二者恰有一个非 ``None``。
    """
    hid = str(hazard_id or "").strip()
    if not hid:
        return None, {
            "hazard_id": hid,
            "evidence": str(evidence or ""),
            "reason": "missing_hazard_id",
            "source_frame": source_frame,
        }

    entry = kb.get(hid)
    if entry is None:
        # 关键护栏：KB 之外的 id 一律拒绝，绝不放行（防止凭空出现的隐患 / 规范号）。
        return None, {
            "hazard_id": hid,
            "evidence": str(evidence or ""),
            "reason": "not_in_knowledge_base",
            "source_frame": source_frame,
        }

    if hid not in allowed_ids:
        # 存在于 KB 但不属当前场景的受控词表（如吊装场景里出现 G-x-x）→ 同样拒绝。
        return None, {
            "hazard_id": hid,
            "evidence": str(evidence or ""),
            "reason": "out_of_scene_vocabulary",
            "source_frame": source_frame,
        }

    return entry.resolved(evidence, source_frame), None


def backfill_hazards(
    hazard_report: Any,
    kb: KnowledgeBase,
    allowed_ids: set[str] | frozenset[str],
    source_frame: str = "",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """把一帧的 ``hazard_report`` 回填为 ``(resolved, rejected)`` 两个列表。

    ``resolved`` 里每条都含来自 KB 的 ``category / norm_reference / action``；
    ``rejected`` 里每条都带拒绝原因（KB 外 / 场景外 / 缺 id）。
    """
    resolved: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for item in extract_model_hazards(hazard_report):
        good, bad = resolve_hazard(
            item.get("hazard_id", ""),
            item.get("evidence", ""),
            kb,
            allowed_ids,
            source_frame,
        )
        if good is not None:
            resolved.append(good)
        if bad is not None:
            rejected.append(bad)
    return resolved, rejected
