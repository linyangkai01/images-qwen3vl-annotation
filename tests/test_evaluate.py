"""T05.1 + T05.2 单测：固定 holdout 逐类评估、三档判定、困难样本回流与消费率。

覆盖点：

1. **指标引擎**：完美预测 / 误检漏检 / AP 插值 / 0 样本类判「无效」；
2. **三档判定**：A/B/C 各自 pass/fail/invalid；C 档显式像素下限风险；
3. **holdout 铁律**：哈希不一致 → ``ValueError``（绝不静默继续）；未冻结 → ``FileNotFoundError``；
4. **命令构造**：``execute=False`` 也产出正确 ``yolo val`` 命令（服务器可直接复用）；
5. **mock 端到端**：``MockTrainer`` 训练 → ``evaluate`` 产出 ``EvalReport``（逐类指标 + 三档）；
6. **sidecar 预测路径**：冻结帧旁的 ``*.detections.json`` 被评估读取；
7. **消费率**：满/部分/空三种情形；困难样本产物落盘。

约束：不联网、不调用真实 VLM / ultralytics、不读真实素材；临时文件用 ``tempfile``。
"""

from __future__ import annotations

import contextlib
import io
import json
import struct
import tempfile
import unittest
import zlib
from pathlib import Path

from traffic_cv_closed_loop.config import ProjectConfig
from traffic_cv_closed_loop.io_utils import read_json, stem_sidecar
from traffic_cv_closed_loop.models import AnnotationRecord, CandidateFrame, Detection

from industrial_hazard_coldstart import cli, taxonomy
from industrial_hazard_coldstart.cli import IMPLEMENTED_COMMANDS
from industrial_hazard_coldstart.evaluate import (
    DEFAULT_CONSUMPTION_TARGET,
    OVERALL_KEY,
    average_precision,
    build_hard_examples,
    build_val_command,
    compute_detection_metrics,
    consumption_rate,
    evaluate,
    tier_verdict,
)
from industrial_hazard_coldstart.holdout import (
    frozen_frame_path,
    holdout_dir_of,
    load_holdout_entries,
)
from industrial_hazard_coldstart.orchestrator import run_evaluate, run_train

ROOT = Path(__file__).resolve().parents[1]
HOIST_CONFIG = ROOT / "own" / "hoist_project" / "config" / "project.hoist.json"
KB_PATH = ROOT / "hazard-detection" / "knowledge_base.json"


# --------------------------------------------------------------------------- #
# 测试夹具
# --------------------------------------------------------------------------- #
def _write_png(path: Path, color: tuple[int, int, int] = (90, 140, 200)) -> Path:
    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    width = height = 8
    raw = (b"\x00" + bytes(color) * width) * height
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )
    return path


def _make_config(tmp: Path, *, trainer: str = "mock", detector: str = "sidecar") -> ProjectConfig:
    raw = json.loads(HOIST_CONFIG.read_text(encoding="utf-8"))
    raw["qwen"]["provider"] = "mock"
    raw["knowledge"]["kb_path"] = str(KB_PATH)
    raw["yolo"]["trainer"] = trainer
    raw["yolo"]["detector"] = detector
    config_dir = tmp / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "project.json"
    config_path.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
    return ProjectConfig.load(config_path)


def _make_record(config: ProjectConfig, relative_path: str) -> AnnotationRecord:
    extracted = config.input_frames_dir / relative_path
    _write_png(extracted)
    label_path = config.workspace("review", "labels") / Path(relative_path).with_suffix(".txt")
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    detections = [Detection(label="person", bbox=(0.1, 0.1, 0.3, 0.5), confidence=0.9)]
    return AnnotationRecord(
        candidate=CandidateFrame(
            source_id="clip",
            relative_path=relative_path,
            original_path=str(extracted),
            extracted_path=str(extracted),
            detections=detections,
            scene="hoist",
        ),
        qwen_detections=detections,
        diff={"added": 0, "removed": 0, "adjusted": 0},
        review_required=False,
        review_reasons=[],
        prompt_path="",
        raw_response_path="",
        label_path=str(label_path),
        dataset_ready=True,
    )


def _seed_annotations(config: ProjectConfig, count: int = 10) -> None:
    records = [_make_record(config, f"clip/frame_{i:06d}.png") for i in range(count)]
    config.annotations_path.parent.mkdir(parents=True, exist_ok=True)
    with config.annotations_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")


def _perfect_predictions(config: ProjectConfig, holdout_dir: Path) -> dict[str, list[Detection]]:
    """从冻结标签构造「完美预测」（与真值逐框一致）。"""
    labels = config.labels
    predictions: dict[str, list[Detection]] = {}
    for entry in load_holdout_entries(holdout_dir):
        label_path = (
            holdout_dir / "labels" / Path(entry.relative_path).with_suffix(".txt")
        )
        detections: list[Detection] = []
        if label_path.exists():
            for line in label_path.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                class_index = int(float(parts[0]))
                cx, cy, width, height = map(float, parts[1:5])
                detections.append(
                    Detection(
                        label=labels[class_index],
                        bbox=(cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2),
                        confidence=0.95,
                    )
                )
        predictions[entry.relative_path] = detections
    return predictions


# --------------------------------------------------------------------------- #
# 指标引擎
# --------------------------------------------------------------------------- #
class DetectionMetricsTest(unittest.TestCase):
    def test_perfect_predictions(self) -> None:
        gt = {"f1": [Detection(label="person", bbox=(0.1, 0.1, 0.3, 0.3), confidence=1.0)]}
        preds = {"f1": [Detection(label="person", bbox=(0.1, 0.1, 0.3, 0.3), confidence=0.9)]}
        result = compute_detection_metrics(gt, preds, list(taxonomy.CLASS_NAMES))
        person = result["per_class"]["person"]
        self.assertEqual(person["precision"], 1.0)
        self.assertEqual(person["recall"], 1.0)
        self.assertEqual(person["map50"], 1.0)
        self.assertEqual(result["false_detection_rate"], 0.0)

    def test_false_positive_and_miss(self) -> None:
        gt = {"f1": [Detection(label="person", bbox=(0.1, 0.1, 0.3, 0.3), confidence=1.0)]}
        # 预测框与真值无重叠（IoU=0）→ 全误检 + 全漏检。
        preds = {"f1": [Detection(label="person", bbox=(0.5, 0.5, 0.7, 0.7), confidence=0.9)]}
        result = compute_detection_metrics(gt, preds, list(taxonomy.CLASS_NAMES))
        person = result["per_class"]["person"]
        self.assertEqual(person["precision"], 0.0)
        self.assertEqual(person["recall"], 0.0)
        self.assertEqual(person["fp"], 1)
        self.assertEqual(person["fn"], 1)
        self.assertEqual(result["false_detection_rate"], 1.0)

    def test_zero_sample_class_is_invalid(self) -> None:
        gt = {"f1": [Detection(label="person", bbox=(0.1, 0.1, 0.3, 0.3), confidence=1.0)]}
        preds = {"f1": []}
        result = compute_detection_metrics(gt, preds, list(taxonomy.CLASS_NAMES))
        # helmet 在 holdout 中样本数为 0 → 判「无效」，不得计为达标。
        self.assertEqual(result["per_class"]["helmet"]["gt_count"], 0)
        self.assertEqual(result["per_class"]["helmet"]["status"], "invalid")

    def test_average_precision_known_value(self) -> None:
        # 两个预测：一个 TP（高置信）、一个 FP（低置信）→ AP = 1.0（TP 在前）。
        self.assertAlmostEqual(average_precision([(0.9, True), (0.1, False)], 1), 1.0, places=6)
        # FP 在前、TP 在后：recall 在 FP 处为 0 → AP ≈ 0.5。
        self.assertAlmostEqual(average_precision([(0.9, False), (0.1, True)], 1), 0.5, places=6)
        self.assertEqual(average_precision([], 0), 0.0)

    def test_false_detection_rate_per_frame(self) -> None:
        gt = {"f1": [], "f2": []}
        preds = {
            "f1": [Detection(label="person", bbox=(0.1, 0.1, 0.2, 0.2), confidence=0.5)],
            "f2": [
                Detection(label="person", bbox=(0.1, 0.1, 0.2, 0.2), confidence=0.5),
                Detection(label="helmet", bbox=(0.3, 0.3, 0.4, 0.4), confidence=0.5),
            ],
        }
        result = compute_detection_metrics(gt, preds, list(taxonomy.CLASS_NAMES))
        # 3 个误检 / 2 帧 = 1.5
        self.assertAlmostEqual(result["false_detection_rate"], 1.5, places=6)


# --------------------------------------------------------------------------- #
# 三档判定
# --------------------------------------------------------------------------- #
class TierVerdictTest(unittest.TestCase):
    def _class_metrics(self, gt: int, map50: float, recall: float) -> dict[str, object]:
        return {"gt_count": gt, "map50": map50, "recall": recall, "precision": 1.0}

    def test_all_pass(self) -> None:
        per_class = {name: self._class_metrics(5, 0.95, 0.95) for name in taxonomy.CLASS_NAMES}
        verdict = tier_verdict(per_class)
        for tier in taxonomy.TIER_ORDER:
            self.assertEqual(verdict[tier]["status"], "pass", tier)

    def test_fail_below_threshold(self) -> None:
        per_class = {name: self._class_metrics(5, 0.95, 0.95) for name in taxonomy.CLASS_NAMES}
        per_class["person"] = self._class_metrics(5, 0.10, 0.10)  # A 档不达标
        verdict = tier_verdict(per_class)
        self.assertEqual(verdict["A"]["status"], "fail")

    def test_invalid_when_zero_sample(self) -> None:
        per_class = {name: self._class_metrics(5, 0.95, 0.95) for name in taxonomy.CLASS_NAMES}
        per_class["helmet"] = self._class_metrics(0, 0.0, 0.0)  # C 档 0 样本
        verdict = tier_verdict(per_class)
        self.assertEqual(verdict["C"]["classes"]["helmet"], "invalid")
        # 有 invalid → 该档不得判 pass。
        self.assertEqual(verdict["C"]["status"], "invalid")

    def test_tier_c_pixel_risk(self) -> None:
        per_class = {name: self._class_metrics(5, 0.95, 0.95) for name in taxonomy.CLASS_NAMES}
        verdict = tier_verdict(per_class)
        self.assertTrue(verdict["C"]["pixel_risk"])
        self.assertIn("32 像素", verdict["C"]["risk_note"])

    def test_ignores_overall_key(self) -> None:
        per_class = {name: self._class_metrics(5, 0.95, 0.95) for name in taxonomy.CLASS_NAMES}
        per_class[OVERALL_KEY] = {"false_detection_rate": 0.0}
        verdict = tier_verdict(per_class)  # 不应抛错 / 不应把 _overall 当类
        self.assertEqual(verdict["A"]["status"], "pass")


# --------------------------------------------------------------------------- #
# 消费率
# --------------------------------------------------------------------------- #
class ConsumptionRateTest(unittest.TestCase):
    def test_full(self) -> None:
        prev = [{"relative_path": "a"}, {"relative_path": "b"}]
        this = [{"relative_path": "a"}, {"relative_path": "b"}, {"relative_path": "c"}]
        self.assertEqual(consumption_rate(prev, this), 1.0)

    def test_partial(self) -> None:
        prev = [{"relative_path": "a"}, {"relative_path": "b"}]
        this = [{"relative_path": "a"}]
        self.assertEqual(consumption_rate(prev, this), 0.5)

    def test_empty_prev_is_vacuous(self) -> None:
        self.assertEqual(consumption_rate([], [{"relative_path": "a"}]), 1.0)

    def test_target_is_ninety_percent(self) -> None:
        self.assertEqual(DEFAULT_CONSUMPTION_TARGET, 0.9)

    def test_accepts_strings_and_objects(self) -> None:
        record = _dummy_record("x")
        self.assertEqual(consumption_rate(["x"], [record]), 1.0)


def _dummy_record(relative_path: str) -> AnnotationRecord:
    candidate = CandidateFrame(
        source_id="s",
        relative_path=relative_path,
        original_path="",
        extracted_path="",
    )
    return AnnotationRecord(
        candidate=candidate,
        qwen_detections=[],
        diff={},
        review_required=False,
        review_reasons=[],
        prompt_path="",
        raw_response_path="",
        label_path="",
        dataset_ready=True,
    )


# --------------------------------------------------------------------------- #
# holdout 铁律
# --------------------------------------------------------------------------- #
class HoldoutIntegrityTest(unittest.TestCase):
    def test_mismatch_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = _make_config(Path(tmp))
            _seed_annotations(config, 10)
            run_train(config, version="t1", execute=False)
            holdout_dir = holdout_dir_of(config)
            entries = load_holdout_entries(holdout_dir)
            # 篡改一个冻结帧 → 哈希不一致。
            victim = frozen_frame_path(holdout_dir, entries[0].relative_path)
            victim.write_bytes(victim.read_bytes() + b"tamper")
            with self.assertRaises(ValueError) as ctx:
                evaluate(
                    config.runs_dir / "t1" / "weights" / "best.pt",
                    holdout_dir,
                    config,
                    version="t1",
                    predictions={},
                )
            self.assertIn("完整性校验失败", str(ctx.exception))

    def test_missing_holdout_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = _make_config(Path(tmp))
            with self.assertRaises(FileNotFoundError):
                evaluate(
                    config.runs_dir / "nope" / "weights" / "best.pt",
                    config.workspace("holdout"),
                    config,
                    version="t1",
                )


# --------------------------------------------------------------------------- #
# 端到端（mock）
# --------------------------------------------------------------------------- #
class EvaluateEndToEndTest(unittest.TestCase):
    def _prepared(self, tmp: Path, *, detector: str = "sidecar") -> tuple[ProjectConfig, Path]:
        config = _make_config(tmp, detector=detector)
        _seed_annotations(config, 10)
        run_train(config, version="t1", execute=False)
        return config, holdout_dir_of(config)

    def test_mock_end_to_end_perfect(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config, holdout_dir = self._prepared(Path(tmp))
            predictions = _perfect_predictions(config, holdout_dir)
            report = evaluate(
                config.runs_dir / "t1" / "weights" / "best.pt",
                holdout_dir,
                config,
                version="t1",
                predictions=predictions,
                consumption=1.0,
            )
            # 逐类指标（不是只有平均值）。
            self.assertIn("person", report.per_class)
            self.assertEqual(report.per_class["person"]["precision"], 1.0)
            self.assertEqual(report.per_class["person"]["recall"], 1.0)
            self.assertEqual(report.per_class["person"]["map50"], 1.0)
            # 整体统计（含误检率）在保留键里。
            self.assertIn(OVERALL_KEY, report.per_class)
            self.assertEqual(report.per_class[OVERALL_KEY]["false_detection_rate"], 0.0)
            # 三档判定存在。
            self.assertEqual(set(report.tier_verdict), set(taxonomy.TIER_ORDER))
            # holdout 哈希已记录。
            self.assertTrue(report.holdout_hash)
            self.assertEqual(report.dataset_version, "t1")
            self.assertEqual(report.consumption_rate, 1.0)

    def test_report_and_metrics_written(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config, holdout_dir = self._prepared(Path(tmp))
            evaluate(
                config.runs_dir / "t1" / "weights" / "best.pt",
                holdout_dir,
                config,
                version="t1",
                predictions=_perfect_predictions(config, holdout_dir),
            )
            eval_dir = config.workspace("eval", "t1")
            self.assertTrue((eval_dir / "report.json").exists())
            self.assertTrue((eval_dir / "metrics.json").exists())
            metrics = read_json(eval_dir / "metrics.json")
            self.assertIn("per_frame", metrics)
            self.assertIn("false_detection_rate", metrics)
            self.assertIn("per_class", metrics)

    def test_val_command_prepared(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config, holdout_dir = self._prepared(Path(tmp), detector="ultralytics")
            weights = config.runs_dir / "t1" / "weights" / "best.pt"
            evaluate(
                weights,
                holdout_dir,
                config,
                version="t1",
                predictions=_perfect_predictions(config, holdout_dir),
                execute=False,
            )
            command = (config.workspace("eval", "t1") / "command.txt").read_text(encoding="utf-8")
            self.assertIn("val", command)
            self.assertIn(f"model={weights}", command)
            self.assertIn("holdout_data.yaml", command)
            # 独立 data.yaml 的 val 指向 holdout 副本，不是 train/val 的 val。
            data_yaml = (config.workspace("eval", "t1") / "holdout_data.yaml").read_text(
                encoding="utf-8"
            )
            self.assertIn("val: images/val", data_yaml)
            self.assertIn("images/val", data_yaml)

    def test_build_val_command_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = _make_config(Path(tmp))
            command = build_val_command(
                config, Path("w.pt"), Path("d.yaml"), config.workspace("eval", "v")
            )
            self.assertEqual(command[2], "val")
            self.assertIn("model=w.pt", command)
            self.assertIn("data=d.yaml", command)

    def test_prepare_only_does_not_run_binary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config, holdout_dir = self._prepared(Path(tmp), detector="ultralytics")
            # execute=False 且无显式预测：只准备命令，绝不跑 ultralytics 二进制。
            report = evaluate(
                config.runs_dir / "t1" / "weights" / "best.pt",
                holdout_dir,
                config,
                version="t1",
                execute=False,
            )
            metrics = read_json(config.workspace("eval", "t1") / "metrics.json")
            self.assertEqual(metrics["predictions_source"], "prepare_only")
            # 无预测 → 全漏检，但不报错。
            self.assertGreaterEqual(report.per_class["person"]["fn"], 1)

    def test_sidecar_predictions_used(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config, holdout_dir = self._prepared(Path(tmp), detector="sidecar")
            entries = load_holdout_entries(holdout_dir)
            target = entries[0]
            sidecar = stem_sidecar(
                frozen_frame_path(holdout_dir, target.relative_path), "detections"
            )
            sidecar.write_text(
                json.dumps(
                    {
                        "objects": [
                            {
                                "label": "person",
                                "bbox": [0.1, 0.1, 0.3, 0.5],
                                "confidence": 0.88,
                                "source": "test",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            report = evaluate(
                config.runs_dir / "t1" / "weights" / "best.pt",
                holdout_dir,
                config,
                version="t1",
            )
            # sidecar 被读入 → person 至少有一个预测。
            self.assertGreaterEqual(report.per_class["person"]["pred_count"], 1)

    def test_holdout_hash_matches_registered(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config, holdout_dir = self._prepared(Path(tmp))
            train_report = read_json(config.runs_dir / "t1" / "report.json")
            report = evaluate(
                config.runs_dir / "t1" / "weights" / "best.pt",
                holdout_dir,
                config,
                version="t1",
                predictions=_perfect_predictions(config, holdout_dir),
            )
            self.assertEqual(report.holdout_hash, train_report["holdout_hash"])


# --------------------------------------------------------------------------- #
# 编排阶段 + 困难样本
# --------------------------------------------------------------------------- #
class RunEvaluateStageTest(unittest.TestCase):
    def test_writes_analysis_products(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = _make_config(Path(tmp))
            _seed_annotations(config, 10)
            run_train(config, version="t1", execute=False)
            summary = run_evaluate(config, version="t1")
            self.assertEqual(summary["stage"], "evaluate")
            self.assertTrue((config.workspace("analysis", "hard_examples.jsonl")).exists())
            self.assertTrue((config.workspace("analysis", "next_cycle_tasks.md")).exists())
            self.assertTrue((config.status_dir / "evaluate.json").exists())
            self.assertIn("consumption_ok", summary)

    def test_consumption_uses_previous_hard_examples(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = _make_config(Path(tmp))
            _seed_annotations(config, 10)
            run_train(config, version="t1", execute=False)
            # 上一轮困难样本：一条命中本轮标注、一条未命中 → 0.5。
            records = [
                AnnotationRecord.from_dict(item)
                for item in _read_jsonl(config.annotations_path)
            ]
            hit = records[0].candidate.relative_path
            prev = [{"relative_path": hit}, {"relative_path": "clip/missing.png"}]
            analysis = config.workspace("analysis")
            analysis.mkdir(parents=True, exist_ok=True)
            with (analysis / "hard_examples.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
                for item in prev:
                    fh.write(json.dumps(item, ensure_ascii=False) + "\n")
            summary = run_evaluate(config, version="t1")
            self.assertEqual(summary["consumption_rate"], 0.5)
            self.assertFalse(summary["consumption_ok"])

    def test_build_hard_examples_reasons(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = _make_config(Path(tmp))
            _seed_annotations(config, 4)
            run_train(config, version="t1", execute=False)
            report = evaluate(
                config.runs_dir / "t1" / "weights" / "best.pt",
                holdout_dir_of(config),
                config,
                version="t1",
                predictions={},
            )
            annotations = [
                AnnotationRecord.from_dict(item)
                for item in _read_jsonl(config.annotations_path)
            ]
            # 逐帧统计：制造一帧漏检。
            path = annotations[0].candidate.relative_path
            frame_stats = {path: {"gt": 1, "pred": 0, "tp": 0, "fp": 0, "fn": 1}}
            hard = build_hard_examples(config, report, annotations, frame_stats)
            paths = {item["relative_path"] for item in hard}
            self.assertIn(path, paths)
            entry = next(item for item in hard if item["relative_path"] == path)
            self.assertIn("eval_false_negative", entry["reasons"])
            self.assertGreater(entry["score"], 0.0)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
class EvaluateCliTest(unittest.TestCase):
    def test_evaluate_is_implemented(self) -> None:
        self.assertIn("evaluate", IMPLEMENTED_COMMANDS)
        self.assertEqual(cli.PENDING_COMMANDS, frozenset())

    def test_cli_evaluate_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = _make_config(Path(tmp))
            _seed_annotations(config, 10)
            run_train(config, version="cli-v1", execute=False)
            out, err = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                code = cli.main(
                    [
                        "evaluate",
                        "--config",
                        str(config.config_path),
                        "--version",
                        "cli-v1",
                    ]
                )
            self.assertEqual(code, cli.EXIT_READY, err.getvalue())
            self.assertTrue((config.workspace("eval", "cli-v1") / "report.json").exists())

    def test_cli_evaluate_without_holdout_is_readable_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = _make_config(Path(tmp))
            _seed_annotations(config, 4)
            stderr = io.StringIO()
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(stderr):
                code = cli.main(
                    ["evaluate", "--config", str(config.config_path), "--version", "x"]
                )
            self.assertEqual(code, cli.EXIT_STAGE_FAILED)
            self.assertIn("执行失败", stderr.getvalue())


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


if __name__ == "__main__":
    unittest.main()
