"""吊装场景的类别体系与共享契约（**单一真源**）。

本模块把跨文件约定**写死**在这里，任何其他文件都不得另行定义这些常量：

1. **12 类 class_id 顺序**：数组下标即 ``class_id``，不可重排。
2. **明确排除的高处作业类别**：``ladder`` / ``scaffold`` / ``guardrail`` / ``harness``。
3. **场景枚举**：``hoist``（本期）/ ``height``（留位不实现）；本期 ``scene`` 恒为 ``hoist``。
4. **S1 的 27 条隐患 → 能力层** 映射（一条不漏），并给出目标检测层能覆盖的 18 条。
5. **A/B/C 尺寸分档**与各档验收阈值（供 T05 评估判定）。

依赖：仅 ``models``（用于构造 ``TaxonomyClass`` / ``HazardCapability``）与标准库。
"""

from __future__ import annotations

from industrial_hazard_coldstart.models import HazardCapability, TaxonomyClass

__all__ = [
    # 场景
    "SCENE_HOIST",
    "SCENE_HEIGHT",
    "SUPPORTED_SCENES",
    "ACTIVE_SCENES",
    "DEFAULT_SCENE",
    # 类别
    "CLASS_NAMES",
    "CLASS_COUNT",
    "EXCLUDED_CLASS_NAMES",
    "CLASS_ID_TO_NAME",
    "CLASS_NAME_TO_ID",
    "CLASS_SERVED_HAZARDS",
    "class_id_of",
    "class_name_of",
    # 尺寸分档
    "TIER_A",
    "TIER_B",
    "TIER_C",
    "TIER_ORDER",
    "TIER_CLASSES",
    "TIER_SHORT_SIDE_BOUNDS",
    "TIER_THRESHOLDS",
    "TIER_RISK_NOTES",
    "tier_of_class",
    "tier_threshold",
    # 能力层
    "LAYER_DETECTION",
    "LAYER_GEOMETRY",
    "LAYER_TEMPORAL",
    "LAYER_OCR",
    "LAYER_SUPER_RESOLUTION",
    "LAYER_IMAGE_QUALITY",
    "LAYER_ORDER",
    "LAYER_LABELS_ZH",
    # S1 隐患
    "S1_HAZARD_IDS",
    "S1_HAZARD_NAMES",
    "S1_DETECTION_HAZARD_IDS",
    "S1_GEOMETRY_HAZARD_IDS",
    "S1_TEMPORAL_HAZARD_IDS",
    "S1_OCR_HAZARD_IDS",
    "S1_SUPER_RESOLUTION_HAZARD_IDS",
    "S1_IMAGE_QUALITY_HAZARD_IDS",
    "S1_HAZARD_LAYERS",
    "layer_of_hazard",
    # 构造与自检
    "build_taxonomy_classes",
    "build_hazard_capabilities",
    "validate_taxonomy",
]


# --------------------------------------------------------------------------- #
# 1. 场景枚举
# --------------------------------------------------------------------------- #
SCENE_HOIST = "hoist"
SCENE_HEIGHT = "height"
#: 已知场景（``height`` 留位不实现）。
SUPPORTED_SCENES: tuple[str, ...] = (SCENE_HOIST, SCENE_HEIGHT)
#: 本期实际启用的场景（仅吊装）。
ACTIVE_SCENES: tuple[str, ...] = (SCENE_HOIST,)
DEFAULT_SCENE = SCENE_HOIST


# --------------------------------------------------------------------------- #
# 2. 12 类体系（class_id 顺序写死，不可重排）
# --------------------------------------------------------------------------- #
CLASS_NAMES: tuple[str, ...] = (
    "person",          # 0
    "helmet",          # 1
    "vest",            # 2
    "hook",            # 3
    "rigging",         # 4
    "overhead_crane",  # 5
    "spreader",        # 6
    "balance_beam",    # 7
    "load",            # 8
    "pallet",          # 9
    "hopper",          # 10
    "ebike",           # 11
)
CLASS_COUNT: int = len(CLASS_NAMES)

#: 属高处作业，本期明确排除，**不得出现在任何配置里**。
EXCLUDED_CLASS_NAMES: tuple[str, ...] = ("ladder", "scaffold", "guardrail", "harness")

#: ``class_id`` ↔ ``name`` 双向表（下标即 class_id）。
CLASS_ID_TO_NAME: dict[int, str] = {index: name for index, name in enumerate(CLASS_NAMES)}
CLASS_NAME_TO_ID: dict[str, int] = {name: index for index, name in enumerate(CLASS_NAMES)}

#: 每个检测类「服务」的隐患 id（源自需求 §8.2 的逐条推导，仅供追溯/文档用）。
CLASS_SERVED_HAZARDS: dict[str, tuple[str, ...]] = {
    "person": ("D-1-7", "D-3-5", "D-5-1", "D-5-2", "D-5-3", "D-5-4", "D-5-5"),
    "helmet": ("D-5-1",),
    "vest": ("D-5-2",),
    "hook": ("D-1-5", "D-1-6"),
    "rigging": ("D-1-1", "D-1-2", "D-1-3", "D-1-4", "D-2-2", "D-3-4"),
    "overhead_crane": ("D-3-7",),
    "spreader": ("D-2-1", "D-2-2", "D-2-4"),
    "balance_beam": ("D-3-3",),
    "load": ("D-3-1", "D-3-2", "D-3-3", "D-3-4", "D-3-5", "D-3-6", "D-3-7"),
    "pallet": ("D-2-3", "D-2-5"),
    "hopper": ("D-3-1",),
    "ebike": ("D-4-2",),
}


def class_id_of(name: str) -> int:
    """返回类别名的 ``class_id``；未知类别抛 ``KeyError``。"""
    return CLASS_NAME_TO_ID[name]


def class_name_of(class_id: int) -> str:
    """返回 ``class_id`` 对应的类别名；越界抛 ``KeyError``。"""
    return CLASS_ID_TO_NAME[class_id]


# --------------------------------------------------------------------------- #
# 3. A/B/C 尺寸分档（按源分辨率下目标短边像素归并）
# --------------------------------------------------------------------------- #
TIER_A = "A"  # 大目标
TIER_B = "B"  # 中目标
TIER_C = "C"  # 小目标
TIER_ORDER: tuple[str, ...] = (TIER_A, TIER_B, TIER_C)

TIER_CLASSES: dict[str, tuple[str, ...]] = {
    TIER_A: ("overhead_crane", "load", "person"),
    TIER_B: ("hook", "rigging", "spreader", "balance_beam", "pallet", "hopper", "ebike"),
    TIER_C: ("helmet", "vest"),
}

#: 各档「短边像素」区间（``None`` 表示无上界）。
TIER_SHORT_SIDE_BOUNDS: dict[str, tuple[int, int | None]] = {
    TIER_A: (100, None),
    TIER_B: (32, 100),
    TIER_C: (0, 32),
}

#: 各档验收阈值（首轮锚定值，允许首次实测后校准一次）。
TIER_THRESHOLDS: dict[str, dict[str, float | bool]] = {
    TIER_A: {"map50": 0.70, "recall": 0.80, "pixel_risk": False},
    TIER_B: {"map50": 0.50, "recall": 0.65, "pixel_risk": False},
    TIER_C: {"map50": 0.35, "recall": 0.55, "pixel_risk": True},
}

TIER_RISK_NOTES: dict[str, str] = {
    TIER_C: (
        "960x540 上 helmet 约 38.4x21.6 像素，短边低于行业常用的 32 像素下限，"
        "该档指标天然偏低，不得归因于「模型不行」。"
    ),
}

_TIER_OF_CLASS: dict[str, str] = {
    name: tier for tier, names in TIER_CLASSES.items() for name in names
}


def tier_of_class(name: str) -> str:
    """返回检测类所属的尺寸档（``A`` / ``B`` / ``C``）；未知类别抛 ``KeyError``。"""
    return _TIER_OF_CLASS[name]


def tier_threshold(tier: str) -> dict[str, float | bool]:
    """返回某档的验收阈值副本。"""
    return dict(TIER_THRESHOLDS[tier])


# --------------------------------------------------------------------------- #
# 4. 能力层
# --------------------------------------------------------------------------- #
LAYER_DETECTION = "detection"                # 目标检测层
LAYER_GEOMETRY = "geometry"                  # 几何关系层
LAYER_TEMPORAL = "temporal"                  # 时序层
LAYER_OCR = "ocr"                            # 文字识别（OCR）
LAYER_SUPER_RESOLUTION = "super_resolution"  # 超分辨 / 细节识别
LAYER_IMAGE_QUALITY = "image_quality"        # 图像质量层

LAYER_ORDER: tuple[str, ...] = (
    LAYER_DETECTION,
    LAYER_GEOMETRY,
    LAYER_TEMPORAL,
    LAYER_OCR,
    LAYER_SUPER_RESOLUTION,
    LAYER_IMAGE_QUALITY,
)

LAYER_LABELS_ZH: dict[str, str] = {
    LAYER_DETECTION: "目标检测层",
    LAYER_GEOMETRY: "几何关系层",
    LAYER_TEMPORAL: "时序层",
    LAYER_OCR: "文字识别（OCR）",
    LAYER_SUPER_RESOLUTION: "超分辨 / 细节识别",
    LAYER_IMAGE_QUALITY: "图像质量层",
}

#: 各层在本项目的实现状态（本批 T01 只做基础设施，实现落在 T02–T05）。
LAYER_SUPPORTED: dict[str, bool] = {
    LAYER_DETECTION: True,           # 本期实现（交付物层）
    LAYER_GEOMETRY: False,           # 留位不实现
    LAYER_TEMPORAL: False,           # 留位不实现
    LAYER_OCR: False,                # 留位不实现（许可待定）
    LAYER_SUPER_RESOLUTION: False,   # 留位不实现（源分辨率不足）
    LAYER_IMAGE_QUALITY: True,       # 部分实现（仅亮度/清晰度，无 OCR）
}

LAYER_NOTES: dict[str, str] = {
    LAYER_DETECTION: "回答「物体在哪里」，覆盖 27 条中的 18 条。",
    LAYER_GEOMETRY: "框间空间关系（包含/夹角/平衡），本期只留位 + 单测，不接入交付指标。",
    LAYER_TEMPORAL: "多帧运动/静止判定，本期只留位 + 单测。",
    LAYER_OCR: "铭牌文字识别，需额外模型且许可待确认，本期留位不实现。",
    LAYER_SUPER_RESOLUTION: "吊带割口等细节，源分辨率不足，本期留位不实现。",
    LAYER_IMAGE_QUALITY: "画面整体亮度/清晰度指标（本期部分实现）。",
}


# --------------------------------------------------------------------------- #
# 5. S1 的 27 条隐患 → 能力层
# --------------------------------------------------------------------------- #
#: S1（吊装作业）27 条隐患 id，顺序与知识库一致。
S1_HAZARD_IDS: tuple[str, ...] = (
    "D-1-1", "D-1-2", "D-1-3", "D-1-4", "D-1-5", "D-1-6", "D-1-7",
    "D-2-1", "D-2-2", "D-2-3", "D-2-4", "D-2-5",
    "D-3-1", "D-3-2", "D-3-3", "D-3-4", "D-3-5", "D-3-6", "D-3-7",
    "D-4-1", "D-4-2", "D-4-3",
    "D-5-1", "D-5-2", "D-5-3", "D-5-4", "D-5-5",
)

#: S1 隐患中文名（与知识库逐条对应，便于文档与提示词生成）。
S1_HAZARD_NAMES: dict[str, str] = {
    "D-1-1": "绳索交叉缠绕",
    "D-1-2": "绳索缠绕",
    "D-1-3": "吊链/吊带与硬质表面接触无防割措施",
    "D-1-4": "吊带存在割口损伤",
    "D-1-5": "吊钩与被吊物品连接不牢固",
    "D-1-6": "吊具与索具未标识承重重量",
    "D-1-7": "吊装无牵引绳",
    "D-2-1": "吊架不平衡",
    "D-2-2": "吊架下吊链斜拉",
    "D-2-3": "使用卡板吊重物",
    "D-2-4": "托架捆绑不牢靠",
    "D-2-5": "违规使用托盘作为吊具或承重平台",
    "D-3-1": "散件未用料斗吊运/未可靠捆绑固定",
    "D-3-2": "小构件吊装未采取防脱落措施",
    "D-3-3": "大件物品未使用平衡梁",
    "D-3-4": "重物直接压在吊装带上",
    "D-3-5": "吊物悬停在通道上方",
    "D-3-6": "吊物悬停在空中",
    "D-3-7": "相邻龙门吊同时运行，重物长时间悬停",
    "D-4-1": "吊装作业照明不足",
    "D-4-2": "吊装作业区域违规停放电瓶车",
    "D-4-3": "歪拉斜吊行为",
    "D-5-1": "人员未佩戴安全帽",
    "D-5-2": "人员未穿着反光背心或安全警示服",
    "D-5-3": "人员站在吊物下方/上方",
    "D-5-4": "吊物未放稳人员即上车操作",
    "D-5-5": "人员离开现场未收回吊链",
}

#: 目标检测层能覆盖的 18 条（27 − 9 条「框做不了」）。
S1_DETECTION_HAZARD_IDS: tuple[str, ...] = (
    "D-1-1", "D-1-2", "D-1-3", "D-1-5", "D-1-7",
    "D-2-2", "D-2-3", "D-2-5",
    "D-3-1", "D-3-2", "D-3-3", "D-3-4", "D-3-5",
    "D-4-2",
    "D-5-1", "D-5-2", "D-5-4", "D-5-5",
)

#: 几何关系层：需比较框间空间关系（框是输入不是结论）。
S1_GEOMETRY_HAZARD_IDS: tuple[str, ...] = ("D-2-1", "D-2-4", "D-4-3", "D-5-3")

#: 时序层：单帧做不了，需多帧运动/静止判定。
S1_TEMPORAL_HAZARD_IDS: tuple[str, ...] = ("D-3-6", "D-3-7")

#: OCR：需读铭牌文字。
S1_OCR_HAZARD_IDS: tuple[str, ...] = ("D-1-6",)

#: 超分辨 / 细节识别：源分辨率不足。
S1_SUPER_RESOLUTION_HAZARD_IDS: tuple[str, ...] = ("D-1-4",)

#: 图像质量层：画面整体属性（不是目标）。
S1_IMAGE_QUALITY_HAZARD_IDS: tuple[str, ...] = ("D-4-1",)

#: 隐患 id → 层（由上面的分桶拼装，构造后由 ``validate_taxonomy`` 校验完整性）。
S1_HAZARD_LAYERS: dict[str, str] = {
    **{hazard_id: LAYER_DETECTION for hazard_id in S1_DETECTION_HAZARD_IDS},
    **{hazard_id: LAYER_GEOMETRY for hazard_id in S1_GEOMETRY_HAZARD_IDS},
    **{hazard_id: LAYER_TEMPORAL for hazard_id in S1_TEMPORAL_HAZARD_IDS},
    **{hazard_id: LAYER_OCR for hazard_id in S1_OCR_HAZARD_IDS},
    **{hazard_id: LAYER_SUPER_RESOLUTION for hazard_id in S1_SUPER_RESOLUTION_HAZARD_IDS},
    **{hazard_id: LAYER_IMAGE_QUALITY for hazard_id in S1_IMAGE_QUALITY_HAZARD_IDS},
}


def layer_of_hazard(hazard_id: str) -> str:
    """返回某条 S1 隐患所属的能力层；未知 id 抛 ``KeyError``。"""
    return S1_HAZARD_LAYERS[hazard_id]


# --------------------------------------------------------------------------- #
# 6. 构造器与自检
# --------------------------------------------------------------------------- #
def build_taxonomy_classes() -> list[TaxonomyClass]:
    """按 ``class_id`` 顺序构造 12 个 :class:`TaxonomyClass`。"""
    return [
        TaxonomyClass(
            class_id=index,
            name=name,
            tier=tier_of_class(name),
            served_hazards=list(CLASS_SERVED_HAZARDS.get(name, ())),
        )
        for index, name in enumerate(CLASS_NAMES)
    ]


def build_hazard_capabilities() -> list[HazardCapability]:
    """按 S1 顺序构造 27 个 :class:`HazardCapability`（一条不漏）。"""
    return [
        HazardCapability(
            hazard_id=hazard_id,
            layer=S1_HAZARD_LAYERS[hazard_id],
            supported=LAYER_SUPPORTED[S1_HAZARD_LAYERS[hazard_id]],
            notes=LAYER_NOTES[S1_HAZARD_LAYERS[hazard_id]],
        )
        for hazard_id in S1_HAZARD_IDS
    ]


def validate_taxonomy() -> None:
    """自检共享契约的自洽性；不一致即抛 ``ValueError``。

    检查项：
    - 12 类 ``class_id`` 唯一且顺序固定；
    - 排除类不出现在 12 类中；
    - A/B/C 三档覆盖全部 12 类、无重叠；
    - S1 的 27 条隐患一条不漏、无重复；
    - 6 个能力层分桶互斥且并集恰为 27 条。
    """
    if len(CLASS_NAMES) != 12:
        raise ValueError(f"类别数应为 12，实际 {len(CLASS_NAMES)}")
    if len(set(CLASS_NAMES)) != len(CLASS_NAMES):
        raise ValueError("12 类中存在重复类别名")
    overlap = set(CLASS_NAMES) & set(EXCLUDED_CLASS_NAMES)
    if overlap:
        raise ValueError(f"排除类别混入了 12 类体系：{sorted(overlap)}")

    tiered = [name for tier in TIER_ORDER for name in TIER_CLASSES[tier]]
    if sorted(tiered) != sorted(CLASS_NAMES):
        raise ValueError("A/B/C 三档未恰好覆盖 12 类")
    if len(tiered) != len(set(tiered)):
        raise ValueError("A/B/C 三档之间存在重叠")

    if len(S1_HAZARD_IDS) != 27:
        raise ValueError(f"S1 隐患数应为 27，实际 {len(S1_HAZARD_IDS)}")
    if len(set(S1_HAZARD_IDS)) != len(S1_HAZARD_IDS):
        raise ValueError("S1 隐患 id 存在重复")

    buckets = (
        S1_DETECTION_HAZARD_IDS,
        S1_GEOMETRY_HAZARD_IDS,
        S1_TEMPORAL_HAZARD_IDS,
        S1_OCR_HAZARD_IDS,
        S1_SUPER_RESOLUTION_HAZARD_IDS,
        S1_IMAGE_QUALITY_HAZARD_IDS,
    )
    seen: set[str] = set()
    for bucket in buckets:
        for hazard_id in bucket:
            if hazard_id in seen:
                raise ValueError(f"隐患 {hazard_id} 被归入多个能力层")
            seen.add(hazard_id)
    if seen != set(S1_HAZARD_IDS):
        missing = sorted(set(S1_HAZARD_IDS) - seen)
        extra = sorted(seen - set(S1_HAZARD_IDS))
        raise ValueError(f"能力分层表与 S1 不一致：缺 {missing}，多 {extra}")
    if len(seen) != 27:
        raise ValueError(f"能力分层表应覆盖 27 条，实际 {len(seen)}")
