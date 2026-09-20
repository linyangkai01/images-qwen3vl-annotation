"""T05.1 固定 holdout 逐类评估（F10）+ T05.2 困难样本回流与消费率（需求 7.5.5）。

**为什么必须自己实现评估**：母项目 ``UltralyticsTrainer`` **没有 val / 评估能力**
（``adapters/yolo.py`` 只有 ``train`` 与 ``UltralyticsDetector._build_predict_command``），
因此本模块从零实现评估，但**只复用**母项目已有的底层件：

- 命令构造复用 ``config`` 里的 YOLO 参数（与训练命令同口径）；
- ``execute=True`` 时复用 ``UltralyticsDetector`` 做推理，并复用其
  ``_read_results_csv`` 解析（若本次运行顺带产出 ``results.csv``）；
- holdout 完整性复用 :mod:`holdout` 的 ``verify_holdout`` / ``holdout_hash_of``。

**判据（需求 §7 定的验收口径，逐条落地）**：

1. 在**固定 holdout** 上计算，且**必须校验 holdout 哈希与登记值一致**——不一致
   立刻 ``ValueError``，**绝不静默继续**（:func:`evaluate` 第一步）；
2. **逐类**报告 ``precision`` / ``recall`` / ``mAP50``，**不只有平均值**
   （``per_class`` 每个类一条）；
3. 报告**误检率**（每帧误检数，``false_detection_rate``）；
4. 按 **A/B/C 三档**判定（:func:`tier_verdict`，用 ``taxonomy.tier_threshold()``）；
5. **C 档显式标注像素下限风险**（``taxonomy.TIER_RISK_NOTES``）；
6. holdout 中**样本数为 0 的类判定为「无效」**（``invalid``），**不计为达标**；
7. 产出 ``workspace/eval/<version>/report.json`` + ``metrics.json``。

**数据喂法**：``data.yaml`` 的 ``val`` 指向的是 train/val 里的 val，**不是 holdout**。
本模块把**冻结副本**（逐字节不变）拷进评估自带的 ``dataset/images/val`` +
``labels/val``，并为 holdout 生成**独立 data.yaml**，因此 ``yolo val`` 读到的就是
holdout 本身。

**指标怎么算**：无论 mock 还是 ultralytics 路径，**逐类指标都由本模块的纯 Python
引擎**（:func:`compute_detection_metrics`）在「预测框 vs 真值框」上计算——
IoU 贪心匹配得 TP/FP/FN，VOC 全点插值得 AP50。这样 mock 路径与服务器路径**同一口径**，
且天然给出逐类结果（``yolo val`` 的 stdout 逐类格式不稳定，不可依赖）。

仅使用 Python 标准库 + 母项目；零新增第三方依赖；中文路径安全。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from traffic_cv_closed_loop.adapters.yolo import (
    _read_results_csv,
    build_detector,
)
from traffic_cv_closed_loop.config import ProjectConfig
from traffic_cv_closed_loop.dataset_media import dataset_image_relative_path
from traffic_cv_closed_loop.io_utils import (
    copy_file,
    ensure_dir,
    read_json,
    stem_sidecar,
    write_json,
    write_text,
)
from traffic_cv_closed_loop.models import Detection

from industrial_hazard_coldstart import taxonomy
from industrial_hazard_coldstart.holdout import (
    frozen_frame_path,
    frozen_label_path,
    holdout_hash_of,
    holdout_index_path,
    load_holdout_entries,
    verify_holdout,
)
from industrial_hazard_coldstart.models import EvalReport

__all__ = [
    # 常量
    "DEFAULT_IOU_THRESHOLD",
    "DEFAULT_CONSUMPTION_TARGET",
    "EVAL_DIRNAME",
    "REPORT_FILENAME",
    "METRICS_FILENAME",
    "OVERALL_KEY",
    "HARD_EXAMPLES_FILENAME",
    "NEXT_CYCLE_TASKS_FILENAME",
    # T05.1
    "compute_detection_metrics",
    "average_precision",
    "tier_verdict",
    "evaluate",
    "build_holdout_data_yaml",
    "build_val_command",
    # T05.2
    "consumption_rate",
    "build_hard_examples",
    "render_next_cycle_tasks",
]

#: IoU 匹配阈值（AP50 口径）。
DEFAULT_IOU_THRESHOLD: float = 0.5

#: 困难样本消费率判据（需求 §7.5.5：只写不读不算闭环）。
DEFAULT_CONSUMPTION_TARGET: float = 0.9

#: 评估产物目录名（相对 workspace）。
EVAL_DIRNAME: str = "eval"

#: 报告 / 指标文件名。
REPORT_FILENAME: str = "report.json"
METRICS_FILENAME: str = "metrics.json"

#: ``EvalReport.per_class`` 里承载「整体统计」的保留键（真实类名不含下划线前缀，
#: 故与 12 类**不会冲突**；误检率等整体量放在这里，使 report.json 自包含）。
OVERALL_KEY: str = "_overall"

#: 回流产物文件名。
HARD_EXAMPLES_FILENAME: str = "hard_examples.jsonl"
NEXT_CYCLE_TASKS_FILENAME: str = "next_cycle_tasks.md"

#: 困难样本判定阈值（低置信度 / 通道差异）。
LOW_CONFIDENCE_THRESHOLD: float = 0.35
CHANNEL_DISAGREEMENT_THRESHOLD: int = 3

#: 困难样本信号权重（用于排序，不参与验收判定）。
_HARD_EXAMPLE_WEIGHTS: dict[str, float] = {
    "eval_false_negative": 3.0,
    "eval_false_positive": 2.0,
    "low_confidence": 1.0,
    "channel_disagreement": 1.5,
    "review_required": 1.0,
}


# --------------------------------------------------------------------------- #
# 纯几何工具（不依赖母项目 Detection 之外的任何东西）
# --------------------------------------------------------------------------- #
def _area(bbox: Sequence[float]) -> float:
    """矩形面积（宽 × 高）。"""
    x1, y1, x2, y2 = bbox
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _intersection_area(a: Sequence[float], b: Sequence[float]) -> float:
    """两矩形交集面积。"""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    return inter_w * inter_h


def _iou(a: Sequence[float], b: Sequence[float]) -> float:
    """两框 IoU（与 ``Detection.iou`` 同口径，避免构造中间对象）。"""
    inter = _intersection_area(a, b)
    union = _area(a) + _area(b) - inter
    if union <= 0.0:
        return 0.0
    return inter / union


# --------------------------------------------------------------------------- #
# T05.1 指标引擎
# --------------------------------------------------------------------------- #
def average_precision(scored: Sequence[tuple[float, bool]], n_gt: int) -> float:
    """按 VOC 全点插值计算单类平均精度（AP）。

    Args:
        scored: ``(confidence, is_true_positive)`` 列表（跨全部帧汇总）。
        n_gt: 该类真值框总数。

    Returns:
        ``[0, 1]`` 的 AP；``n_gt <= 0`` 或无非空预测时返回 ``0.0``。
    """
    if n_gt <= 0 or not scored:
        return 0.0
    ordered = sorted(scored, key=lambda item: item[0], reverse=True)
    recalls: list[float] = []
    precisions: list[float] = []
    tp = 0
    fp = 0
    for _conf, is_tp in ordered:
        if is_tp:
            tp += 1
        else:
            fp += 1
        precisions.append(tp / (tp + fp))
        recalls.append(tp / n_gt)
    # 全点插值：precision 单调包络。
    mrec = [0.0, *recalls, 1.0]
    mpre = [0.0, *precisions, 0.0]
    for index in range(len(mpre) - 2, -1, -1):
        mpre[index] = max(mpre[index], mpre[index + 1])
    ap = 0.0
    for index in range(len(mrec) - 1):
        ap += (mrec[index + 1] - mrec[index]) * mpre[index + 1]
    return ap


def _normalize_predictions(
    raw: Mapping[Any, Any] | None,
) -> dict[str, list[Detection]]:
    """把「预测」规整为 ``{relative_path: [Detection, ...]}``。

    接受三种形态：``Detection`` 列表、``Detection.to_dict()`` 列表、``{"objects": [...]}``。
    """
    result: dict[str, list[Detection]] = {}
    if not raw:
        return result
    for key, value in raw.items():
        detections: list[Detection] = []
        items = value
        if isinstance(value, dict):
            items = value.get("objects", [])
        for item in items or []:
            if isinstance(item, Detection):
                detections.append(item)
            elif isinstance(item, dict):
                try:
                    detections.append(Detection.from_dict(item))
                except (KeyError, TypeError, ValueError):
                    continue
        result[str(key)] = detections
    return result


def _match_image(
    gt: list[Detection],
    preds: list[Detection],
    iou_threshold: float,
) -> tuple[list[tuple[float, bool]], int, int, int]:
    """对**单帧**做贪心 IoU 匹配。

    Returns:
        ``(scored, tp, fp, fn)``：``scored`` 为 ``(conf, is_tp)`` 列表（供 AP 汇总），
        ``tp``/``fp``/``fn`` 为该帧计数。
    """
    scored: list[tuple[float, bool]] = []
    matched_gt: set[int] = set()
    tp = fp = 0
    for pred in sorted(preds, key=lambda item: item.confidence, reverse=True):
        best_index = -1
        best_iou = 0.0
        for index, target in enumerate(gt):
            if index in matched_gt:
                continue
            overlap = _iou(pred.bbox, target.bbox)
            if overlap > best_iou:
                best_iou = overlap
                best_index = index
        if best_index >= 0 and best_iou >= iou_threshold:
            matched_gt.add(best_index)
            tp += 1
            scored.append((pred.confidence, True))
        else:
            fp += 1
            scored.append((pred.confidence, False))
    fn = len(gt) - len(matched_gt)
    return scored, tp, fp, fn


def compute_detection_metrics(
    ground_truth: Mapping[str, Sequence[Detection]],
    predictions: Mapping[str, Sequence[Detection]],
    labels: Sequence[str],
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
) -> dict[str, Any]:
    """在给定真值 / 预测上计算**逐类** P/R/mAP50 与误检率。

    Args:
        ground_truth: ``{relative_path: [Detection, ...]}``（真值）。
        predictions: ``{relative_path: [Detection, ...]}``（预测）。
        labels: 12 类类别名（决定报告顺序；不在其中的类会被忽略）。
        iou_threshold: IoU 匹配阈值（默认 0.5 = AP50 口径）。

    Returns:
        ``{"per_class": {...}, "per_frame": {...}, "frames": int,
        "false_detection_rate": float, "overall": {...}}``。

    ``per_class[name]`` 含 ``precision / recall / map50 / tp / fp / fn /
    gt_count / pred_count / tier``；**0 样本类**（``gt_count == 0``）额外带
    ``status="invalid"``，且 ``recall=map50=0.0``（其预测若存在则全记 FP）。
    """
    label_set = list(labels)
    frames = sorted(set(ground_truth) | set(predictions))

    per_frame: dict[str, dict[str, int]] = {}
    # 逐类累计：scored 供 AP，计数供 P/R。
    class_scored: dict[str, list[tuple[float, bool]]] = {name: [] for name in label_set}
    class_tp: dict[str, int] = {name: 0 for name in label_set}
    class_fp: dict[str, int] = {name: 0 for name in label_set}
    class_fn: dict[str, int] = {name: 0 for name in label_set}
    class_gt: dict[str, int] = {name: 0 for name in label_set}
    class_pred: dict[str, int] = {name: 0 for name in label_set}
    total_fp = 0

    for frame in frames:
        gt_all = [item for item in ground_truth.get(frame, []) if item.label in class_scored]
        pred_all = [item for item in predictions.get(frame, []) if item.label in class_scored]
        frame_tp = frame_fp = frame_fn = 0
        for name in label_set:
            gt = [item for item in gt_all if item.label == name]
            preds = [item for item in pred_all if item.label == name]
            class_gt[name] += len(gt)
            class_pred[name] += len(preds)
            if not gt and not preds:
                continue
            scored, tp, fp, fn = _match_image(gt, preds, iou_threshold)
            class_scored[name].extend(scored)
            class_tp[name] += tp
            class_fp[name] += fp
            class_fn[name] += fn
            frame_tp += tp
            frame_fp += fp
            frame_fn += fn
        total_fp += frame_fp
        per_frame[frame] = {
            "gt": len(gt_all),
            "pred": len(pred_all),
            "tp": frame_tp,
            "fp": frame_fp,
            "fn": frame_fn,
        }

    per_class: dict[str, Any] = {}
    for name in label_set:
        gt_count = class_gt[name]
        pred_count = class_pred[name]
        tp = class_tp[name]
        fp = class_fp[name]
        fn = class_fn[name]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        map50 = average_precision(class_scored[name], gt_count)
        entry: dict[str, Any] = {
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "map50": round(map50, 6),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "gt_count": gt_count,
            "pred_count": pred_count,
            "tier": taxonomy.tier_of_class(name),
        }
        if gt_count <= 0:
            # 判据 6：holdout 中样本数为 0 的类 → 「无效」，不得计为达标。
            entry["status"] = "invalid"
        per_class[name] = entry

    frame_count = len(frames)
    false_detection_rate = (total_fp / frame_count) if frame_count else 0.0
    overall = {
        "frames": frame_count,
        "false_detection_rate": round(false_detection_rate, 6),
        "total_tp": sum(class_tp.values()),
        "total_fp": sum(class_fp.values()),
        "total_fn": sum(class_fn.values()),
        "total_gt": sum(class_gt.values()),
        "total_pred": sum(class_pred.values()),
        "iou_threshold": iou_threshold,
    }
    return {
        "per_class": per_class,
        "per_frame": per_frame,
        "frames": frame_count,
        "false_detection_rate": round(false_detection_rate, 6),
        "overall": overall,
    }


# --------------------------------------------------------------------------- #
# T05.1 三档判定
# --------------------------------------------------------------------------- #
def tier_verdict(
    per_class: Mapping[str, Any],
    taxonomy_module: Any = taxonomy,
) -> dict[str, Any]:
    """按 A/B/C 三档给判定（判据 4/5/6）。

    每类状态：

    - ``gt_count == 0`` → ``invalid``（**无效**，不计为达标）；
    - 否则 ``map50 >= 阈值`` **且** ``recall >= 阈值`` → ``pass``，否则 ``fail``。

    每档状态：任一类 ``fail`` → ``fail``；否则任一类 ``invalid`` → ``invalid``；
    全 ``pass`` → ``pass``。**C 档**始终附 ``pixel_risk`` 与 ``risk_note``（判据 5）。

    Args:
        per_class: :func:`compute_detection_metrics` 的 ``per_class``（可含保留键
            :data:`OVERALL_KEY`，本函数忽略之）。
        taxonomy_module: 提供 ``TIER_ORDER / TIER_CLASSES / tier_threshold /
            TIER_RISK_NOTES`` 的模块（默认 :mod:`taxonomy`）。

    Returns:
        ``{tier: {"status", "classes", "threshold", ...}}``。
    """
    result: dict[str, Any] = {}
    for tier in taxonomy_module.TIER_ORDER:
        threshold = taxonomy_module.tier_threshold(tier)
        class_names = list(taxonomy_module.TIER_CLASSES[tier])
        class_verdicts: dict[str, str] = {}
        for name in class_names:
            metrics = per_class.get(name, {}) if isinstance(per_class, Mapping) else {}
            gt_count = int(metrics.get("gt_count", 0) or 0)
            if gt_count <= 0:
                class_verdicts[name] = "invalid"
                continue
            map50 = float(metrics.get("map50", 0.0) or 0.0)
            recall = float(metrics.get("recall", 0.0) or 0.0)
            ok = map50 >= float(threshold["map50"]) and recall >= float(threshold["recall"])
            class_verdicts[name] = "pass" if ok else "fail"

        values = list(class_verdicts.values())
        if any(value == "fail" for value in values):
            status = "fail"
        elif any(value == "invalid" for value in values):
            status = "invalid"
        else:
            status = "pass"

        entry: dict[str, Any] = {
            "status": status,
            "classes": class_verdicts,
            "threshold": threshold,
        }
        if tier == taxonomy_module.TIER_C:
            # 判据 5：C 档必须显式标注像素下限风险。
            entry["pixel_risk"] = True
            note = taxonomy_module.TIER_RISK_NOTES.get(tier, "")
            if note:
                entry["risk_note"] = note
        result[tier] = entry
    return result


# --------------------------------------------------------------------------- #
# T05.1 评估数据集 / 命令构造
# --------------------------------------------------------------------------- #
def build_holdout_data_yaml(dataset_dir: Path, labels: Sequence[str]) -> str:
    """为 holdout 生成**独立** data.yaml（``train``/``val`` 均指向 ``images/val``）。

    与母项目 ``scripts/curate_video_batch.build_dataset_yaml`` 同格式，但**不**经过
    训练集切分——评估只读 holdout 副本，绝不碰 ``data.yaml`` 里 train/val 的 val。
    """
    name_lines = [f"  {index}: {label}" for index, label in enumerate(labels)]
    posix = str(dataset_dir).replace("\\", "/")
    return "\n".join(
        [
            f'path: "{posix}"',
            "train: images/val",
            "val: images/val",
            "names:",
            *name_lines,
            "",
        ]
    )


def build_val_command(
    config: ProjectConfig,
    weights: Path,
    data_yaml: Path,
    eval_dir: Path,
) -> list[str]:
    """构造 ``yolo val`` 命令（**execute=False 时也必须正确**，供服务器直接复用）。"""
    return [
        config.yolo_binary,
        config.yolo_task,
        "val",
        f"model={weights}",
        f"data={data_yaml}",
        f"imgsz={config.yolo_image_size}",
        f"conf={config.yolo_detector_confidence}",
        f"iou={config.yolo_detector_iou}",
        f"device={config.yolo_device}",
        f"project={eval_dir.parent}",
        f"name={eval_dir.name}",
        "exist_ok=True",
        "verbose=False",
        "plots=False",
    ]


def _read_label_file(label_path: Path, labels: Sequence[str]) -> list[Detection]:
    """把 YOLO 标签文件（``cls cx cy w h``）解析为 ``Detection`` 列表。"""
    if not label_path.exists():
        return []
    detections: list[Detection] = []
    for raw_line in label_path.read_text(encoding="utf-8").splitlines():
        parts = raw_line.strip().split()
        if len(parts) < 5:
            continue
        try:
            class_index = int(float(parts[0]))
        except ValueError:
            continue
        if class_index < 0 or class_index >= len(labels):
            continue
        x_center, y_center, width, height = map(float, parts[1:5])
        detections.append(
            Detection(
                label=labels[class_index],
                bbox=(
                    x_center - width / 2,
                    y_center - height / 2,
                    x_center + width / 2,
                    y_center + height / 2,
                ),
                confidence=1.0,
                source="holdout-gt",
            )
        )
    return detections


def _materialize_eval_dataset(
    entries: Iterable[Any],
    holdout_dir: Path,
    dataset_dir: Path,
    labels: Sequence[str],
) -> dict[str, list[Detection]]:
    """把**冻结副本**拷进评估数据集，并返回逐帧真值。

    图片 → ``dataset/images/val``、标签 → ``dataset/labels/val``（Ultralytics 可读布局）。
    若冻结帧旁存在 ``*.detections.json`` 侧车（sidecar），一并拷入（供 sidecar 预测路径）。
    """
    images_dir = dataset_dir / "images" / "val"
    labels_dir = dataset_dir / "labels" / "val"
    ensure_dir(images_dir)
    ensure_dir(labels_dir)
    ground_truth: dict[str, list[Detection]] = {}
    for entry in entries:
        produced = dataset_image_relative_path(Path(entry.relative_path))
        frozen_frame = frozen_frame_path(holdout_dir, entry.relative_path)
        frozen_label = frozen_label_path(holdout_dir, entry.relative_path)
        image_target = images_dir / produced
        label_target = labels_dir / produced.with_suffix(".txt")
        copy_file(frozen_frame, image_target)
        if frozen_label.exists():
            copy_file(frozen_label, label_target)
        else:
            ensure_dir(label_target.parent)
            label_target.write_text("", encoding="utf-8")
        sidecar = stem_sidecar(frozen_frame, "detections")
        if sidecar.exists():
            copy_file(sidecar, stem_sidecar(image_target, "detections"))
        ground_truth[entry.relative_path] = _read_label_file(label_target, labels)
    return ground_truth


def _resolve_version(holdout_dir: Path, version: str | None) -> str:
    """确定数据集版本号：显式参数优先，其次读 holdout 清单，最后回落默认值。"""
    if version is not None and str(version).strip():
        return str(version).strip()
    payload = read_json(holdout_index_path(holdout_dir), default=None)
    if isinstance(payload, dict):
        recorded = str(payload.get("version", "") or "").strip()
        if recorded:
            return recorded
    return "coldstart-v1"


def _predict_with_detector(
    config: ProjectConfig,
    images: list[Path],
) -> dict[Path, list[Detection]]:
    """用母项目检测器对评估图片推理（sidecar 路径离线、ultralytics 路径需执行权限）。

    Returns:
        ``{resolve(image_path): [Detection, ...]}``。
    """
    detector = build_detector(config)
    resolved = [path.resolve() for path in images]
    return detector.detect_many(resolved)


# --------------------------------------------------------------------------- #
# T05.1 主入口
# --------------------------------------------------------------------------- #
def evaluate(
    weights: Path,
    holdout_dir: Path,
    config: ProjectConfig,
    *,
    version: str | None = None,
    eval_dir: Path | None = None,
    execute: bool = False,
    predictions: Mapping[Any, Any] | None = None,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    consumption: float = 0.0,
) -> EvalReport:
    """在**固定 holdout** 上逐类评估并按 A/B/C 三档判定（T05.1）。

    流程：

    1. **校验 holdout**：读冻结清单并重算每帧 sha256；不一致 / 缺失 → ``ValueError``
       （判据 1，绝不静默继续）；聚合 ``holdout_hash`` 写入报告；
    2. 把冻结副本拷进评估数据集（``images/val`` + ``labels/val``），写 holdout 专用
       data.yaml，并构造 / 落盘 ``yolo val`` 命令（判据：``execute=False`` 也正确）；
    3. 取预测：``predictions`` 显式给出时直接用（mock 路径）；否则 ``execute`` 时用
       ``UltralyticsDetector`` 推理、或检测器为 ``sidecar`` 时读离线侧车；二者皆非
       （``execute=False`` 且非 sidecar）则**只准备命令、不推理**（``prepare_only``）；
    4. 纯 Python 引擎算**逐类** P/R/mAP50 与误检率（判据 2/3）；
    5. :func:`tier_verdict` 给三档判定（判据 4/5/6）；
    6. 写 ``report.json``（``EvalReport``）与 ``metrics.json``（判据 7）。

    Args:
        weights: 待评估权重路径（``best.pt``；mock 路径可不存在）。
        holdout_dir: 固定 holdout 根目录（含 ``index.json`` / ``frames`` / ``labels``）。
        config: 项目配置。
        version: 数据集 / 评估版本号；省略时读 holdout 清单的 ``version``。
        eval_dir: 评估产物目录；省略时用 ``workspace/eval/<version>``。
        execute: 是否真正执行 ``yolo val`` / ``yolo predict``（默认否）。
        predictions: 显式预测（``{relative_path: [Detection] | {"objects": [...]}}``）。
        iou_threshold: IoU 匹配阈值。
        consumption: 困难样本消费率（由编排层回填；默认 ``0.0``）。

    Returns:
        :class:`EvalReport`（``per_class`` 含逐类指标 + 保留键 :data:`OVERALL_KEY`）。

    Raises:
        FileNotFoundError: holdout 尚未冻结（无 ``index.json`` 或清单为空）。
        ValueError: holdout 完整性校验失败（篡改 / 缺失）。
    """
    holdout_dir = Path(holdout_dir)
    entries = load_holdout_entries(holdout_dir)
    if not entries:
        raise FileNotFoundError(
            f"holdout 尚未冻结（{holdout_index_path(holdout_dir)} 不存在或为空）；"
            "请先运行 `freeze-holdout` 阶段。"
        )

    # 判据 1：哈希校验，不一致立刻报错，绝不静默继续。
    verification = verify_holdout(entries, holdout_dir)
    if not verification["hash_ok"]:
        raise ValueError(
            "holdout 完整性校验失败："
            f"tampered={verification['tampered']} missing={verification['missing']}。"
            "固定评估集被改动，跨轮指标不可比；评估已中止（绝不静默继续）。"
        )
    holdout_hash = holdout_hash_of(entries)

    dataset_version = _resolve_version(holdout_dir, version)
    target_dir = Path(eval_dir) if eval_dir is not None else config.workspace(
        EVAL_DIRNAME, dataset_version
    )
    ensure_dir(target_dir)
    dataset_dir = target_dir / "dataset"
    ground_truth = _materialize_eval_dataset(
        entries, holdout_dir, dataset_dir, config.labels
    )

    data_yaml_path = target_dir / "holdout_data.yaml"
    write_text(data_yaml_path, build_holdout_data_yaml(dataset_dir, config.labels))

    # 命令构造：execute=False 也产出正确命令（服务器可直接复用）。
    command = build_val_command(config, Path(weights), data_yaml_path, target_dir)
    write_text(target_dir / "command.txt", " ".join(command))

    # --- 取预测 ---
    images_dir = dataset_dir / "images" / "val"
    images = sorted(path for path in images_dir.rglob("*") if path.is_file())
    produced_to_relative = {
        dataset_image_relative_path(Path(entry.relative_path)).as_posix(): entry.relative_path
        for entry in entries
    }
    if predictions is not None:
        raw_predictions = _normalize_predictions(predictions)
        predictions_source = "explicit"
    else:
        detector_mode = str(getattr(config, "yolo_detector", "")).lower()
        # 仅在「真正执行」或「离线 sidecar」时推理；``execute=False`` 且 ultralytics
        # 时**只准备命令、绝不跑二进制**（符合「execute=False 只产出正确命令」的约定）。
        if execute or detector_mode == "sidecar":
            by_path = _predict_with_detector(config, images)
            raw_predictions = {}
            for image in images:
                relative = image.relative_to(images_dir).as_posix()
                key = produced_to_relative.get(relative, relative)
                raw_predictions[key] = by_path.get(image.resolve(), [])
            predictions_source = "detector" if execute else "sidecar"
        else:
            raw_predictions = {}
            predictions_source = "prepare_only"

    metrics = compute_detection_metrics(
        ground_truth, raw_predictions, config.labels, iou_threshold
    )
    verdict = tier_verdict(metrics["per_class"])

    per_class: dict[str, Any] = dict(metrics["per_class"])
    per_class[OVERALL_KEY] = dict(metrics["overall"])
    report = EvalReport(
        dataset_version=dataset_version,
        holdout_hash=holdout_hash,
        per_class=per_class,
        tier_verdict=verdict,
        consumption_rate=float(consumption),
    )

    # 若 ultralytics 路径顺带产出 results.csv，则额外并入其聚合指标（best-effort）。
    aggregate = _read_results_csv(target_dir / "results.csv")
    metrics_payload: dict[str, Any] = {
        "dataset_version": dataset_version,
        "holdout_hash": holdout_hash,
        "weights": str(weights),
        "command": " ".join(command),
        "predictions_source": predictions_source,
        "frames": metrics["frames"],
        "false_detection_rate": metrics["false_detection_rate"],
        "overall": metrics["overall"],
        "per_class": metrics["per_class"],
        "per_frame": metrics["per_frame"],
        "aggregate_metrics": aggregate,
    }
    write_json(target_dir / REPORT_FILENAME, report.to_dict())
    write_json(target_dir / METRICS_FILENAME, metrics_payload)
    return report


# --------------------------------------------------------------------------- #
# T05.2 困难样本回流与消费率
# --------------------------------------------------------------------------- #
def _key_of(item: Any) -> str:
    """从困难样本 / 记录中取出相对路径键（兼容 dict / 字符串 / 带 ``relative_path`` 的对象）。"""
    if item is None:
        return ""
    if isinstance(item, str):
        return item
    if isinstance(item, Mapping):
        value = item.get("relative_path")
        return "" if value is None else str(value)
    candidate = getattr(item, "candidate", None)
    if candidate is not None and getattr(candidate, "relative_path", None) is not None:
        return str(candidate.relative_path)
    value = getattr(item, "relative_path", None)
    return "" if value is None else str(value)


def _keys_of(items: Iterable[Any]) -> set[str]:
    """取出非空键集合。"""
    keys = {_key_of(item) for item in items or []}
    keys.discard("")
    return keys


def consumption_rate(prev_hard: Iterable[Any], this_round: Iterable[Any]) -> float:
    """计算困难样本消费率（需求 §7.5.5：只写不读不算闭环）。

    口径：**上一轮**标记的困难样本中，有多大比例**本轮真的被消费**（进入本轮数据集 /
    标注池）。``prev_hard`` 与 ``this_round`` 可为 dict（含 ``relative_path``）、字符串
    或带 ``relative_path`` / ``candidate.relative_path`` 的对象。

    Args:
        prev_hard: 上一轮的困难样本（通常读自 ``analysis/hard_examples.jsonl``）。
        this_round: 本轮实际消费的相对路径（通常为本轮 ``dataset_ready`` 的标注路径）。

    Returns:
        ``[0, 1]`` 的消费率；``prev_hard`` 为空时返回 ``1.0``（无可消费项，视为闭环成立）。
    """
    prev_keys = _keys_of(prev_hard)
    if not prev_keys:
        return 1.0
    this_keys = _keys_of(this_round)
    consumed = prev_keys & this_keys
    return round(len(consumed) / len(prev_keys), 6)


def build_hard_examples(
    config: ProjectConfig,
    report: EvalReport,
    annotations: Iterable[Any] | None = None,
    frame_stats: Mapping[str, Mapping[str, int]] | None = None,
    *,
    low_confidence: float = LOW_CONFIDENCE_THRESHOLD,
    disagreement: int = CHANNEL_DISAGREEMENT_THRESHOLD,
) -> list[dict[str, Any]]:
    """汇总本轮的困难样本（供下一轮优先标注 / 消费）。

    **困难样本的定义**（依据写进每条 ``reasons``，绝不只给一个黑盒分数）：

    - ``low_confidence``：该帧最低检测置信度低于 ``low_confidence``（模型不确定）；
    - ``channel_disagreement``：通道 A（初检）与通道 B（VLM）差异 ``added+removed+adjusted``
      不低于 ``disagreement``（两通道看法不一致）；
    - ``review_required``：该帧在复核队列里（``review_required``）；
    - ``eval_false_negative`` / ``eval_false_positive``：评估该帧有漏检 / 误检。

    Args:
        config: 项目配置（用于读阈值相关的默认值；本函数不产生副作用）。
        report: :func:`evaluate` 的产物（用于确认本轮有评估）。
        annotations: 本轮标注记录（``AnnotationRecord`` 列表；可为 ``None``）。
        frame_stats: 逐帧评估统计（``metrics.json["per_frame"]``；可为 ``None``）。
        low_confidence: 低置信度阈值。
        disagreement: 通道差异阈值。

    Returns:
        困难样本列表（按 ``score`` 降序），每项含 ``relative_path / reasons / signals /
        score``。
    """
    records = list(annotations or [])
    stats = dict(frame_stats or {})
    by_path: dict[str, dict[str, Any]] = {}

    def _entry(path: str) -> dict[str, Any]:
        if path not in by_path:
            by_path[path] = {
                "relative_path": path,
                "reasons": [],
                "signals": {},
                "score": 0.0,
            }
        return by_path[path]

    for record in records:
        path = _key_of(record)
        if not path:
            continue
        detections = getattr(record, "qwen_detections", None) or []
        candidate = getattr(record, "candidate", None)
        if not detections and candidate is not None:
            detections = getattr(candidate, "detections", None) or []
        confidences = [float(item.confidence) for item in detections]
        min_conf = min(confidences) if confidences else 0.0
        diff = getattr(record, "diff", None) or {}
        diff_total = sum(int(value) for value in diff.values()) if diff else 0
        review_required = bool(getattr(record, "review_required", False))

        reasons: list[str] = []
        if confidences and min_conf < low_confidence:
            reasons.append("low_confidence")
        if diff_total >= disagreement:
            reasons.append("channel_disagreement")
        if review_required:
            reasons.append("review_required")
        if not reasons:
            continue
        entry = _entry(path)
        entry["reasons"].extend(reasons)
        entry["signals"].update(
            {
                "min_confidence": round(min_conf, 4),
                "diff_total": diff_total,
                "review_required": review_required,
            }
        )

    for path, frame in stats.items():
        fp = int(frame.get("fp", 0) or 0)
        fn = int(frame.get("fn", 0) or 0)
        if fp <= 0 and fn <= 0:
            continue
        entry = _entry(path)
        if fn > 0:
            entry["reasons"].append("eval_false_negative")
        if fp > 0:
            entry["reasons"].append("eval_false_positive")
        entry["signals"].update({"eval_fp": fp, "eval_fn": fn})

    for entry in by_path.values():
        entry["reasons"] = sorted(set(entry["reasons"]))
        entry["score"] = round(
            sum(_HARD_EXAMPLE_WEIGHTS.get(reason, 0.5) for reason in entry["reasons"]),
            4,
        )
    return sorted(by_path.values(), key=lambda item: (-item["score"], item["relative_path"]))


def render_next_cycle_tasks(
    report: EvalReport,
    hard_examples: Sequence[Mapping[str, Any]],
    rate: float,
    *,
    target: float = DEFAULT_CONSUMPTION_TARGET,
) -> str:
    """渲染 ``next_cycle_tasks.md``（下一轮待办：困难样本清单 + 消费率判据）。"""
    met = rate >= target
    lines: list[str] = [
        "# 下一轮待办（困难样本回流）",
        "",
        f"- 数据集版本：`{report.dataset_version}`",
        f"- holdout 聚合哈希：`{report.holdout_hash}`",
        f"- 困难样本数：`{len(hard_examples)}`",
        f"- 上一轮困难样本消费率：`{rate:.4f}`（判据 ≥ {target:.2f}："
        f"{'达标' if met else '未达标'}）",
        "",
        "> 判据来源：需求 §7.5.5 —— **只写不读不算闭环**。上一轮标记的困难样本，"
        "本轮必须有 ≥ 90% 真正进入数据集 / 标注池，否则视为回流未闭合。",
        "",
    ]
    if not hard_examples:
        lines.append("本轮无新增困难样本。")
        return "\n".join(lines) + "\n"

    lines.extend(
        [
            "## 待优先处理（按分数降序）",
            "",
            "| # | relative_path | 分数 | 原因 |",
            "|---|---------------|------|------|",
        ]
    )
    for index, item in enumerate(hard_examples[:100], start=1):
        reasons = "、".join(item.get("reasons", []))
        lines.append(
            f"| {index} | `{item.get('relative_path', '')}` | "
            f"{item.get('score', 0.0)} | {reasons} |"
        )
    return "\n".join(lines) + "\n"
