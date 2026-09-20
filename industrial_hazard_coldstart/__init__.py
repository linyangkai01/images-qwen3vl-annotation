"""工业隐患单场景（吊装）YOLO 冷启动标注与训练。

本包是增量新增的顶层包，与母项目 ``traffic_cv_closed_loop`` 平级。
设计立场：**把母项目当库用**（``import`` 复用其公开 API），母项目一行不改。

本批（T01）只交付「项目基础设施与共享契约」：
- :mod:`industrial_hazard_coldstart.taxonomy` —— 12 类体系 / class_id 顺序 /
  场景枚举 / S1 能力分层表 / 尺寸分档（共享契约的单一真源）。
- :mod:`industrial_hazard_coldstart.models` —— 9 个新数据结构，
  并复用母项目的 ``Detection`` / ``CandidateFrame`` / ``AnnotationRecord``。
- :mod:`industrial_hazard_coldstart.cli` —— 子命令骨架（本批仅 ``doctor`` 落地）。
- :mod:`industrial_hazard_coldstart.doctor` —— 探活扩展。

T02–T05（视频解码 / 标注编排 / 数据集 / holdout / 评估 / layers 子包）
不在本批范围，对应的子命令会明确返回「尚未实现」并给出非 0 退出码。
"""

from __future__ import annotations

from industrial_hazard_coldstart import models, taxonomy

__all__ = [
    "__version__",
    "models",
    "taxonomy",
]

__version__ = "0.1.0"
