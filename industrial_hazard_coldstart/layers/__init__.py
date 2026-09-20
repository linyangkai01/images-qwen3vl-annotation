"""T05.3–T05.5 能力层子包（几何关系层 / 时序层 / 图像质量层 + OCR 占位）。

**重要性质**：本子包三层**本期只留位、不接入交付指标**（需求已决策：本期交付物
只有检测权重）。理由逐层见各模块 docstring：

- :mod:`~industrial_hazard_coldstart.layers.geometry` —— 框间空间关系（``D-5-3`` /
  ``D-4-3`` / ``D-2-1`` / ``D-2-4``），框做不了；
- :mod:`~industrial_hazard_coldstart.layers.temporal` —— 多帧运动 / 静止（``D-3-6`` /
  ``D-3-7``），单帧做不了；
- :mod:`~industrial_hazard_coldstart.layers.semantic` —— 画面整体亮度（``D-4-1``）与
  铭牌 OCR（``D-1-6``，许可待定）。

三层的结论**不得**混进 :class:`~industrial_hazard_coldstart.models.EvalReport` 的
验收判定；此处只导出结构化的能力（函数 + 数据结构），供将来接入。
"""

from __future__ import annotations

from industrial_hazard_coldstart.layers import geometry, semantic, temporal

__all__ = [
    "geometry",
    "temporal",
    "semantic",
]
