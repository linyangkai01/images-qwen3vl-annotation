"""T04.3 单测：训练编排（复用 build_trainer，**不**用 run_training）+ CLI 转正。

覆盖点：

1. **mock 端到端** —— ``run_train(execute=False)`` 走 ``MockTrainer``，产出
   ``runs/<ver>/report.json`` + ``metrics.json``；
2. **未复用母项目 run_training** —— 断言 ``analysis/hard_examples.jsonl`` **未**产生
   （那是 ``run_training`` 的副作用），证明我们只复用底层训练器；
3. **execute 路径命令构造正确** —— ``UltralyticsTrainer`` 在 ``execute=False`` 时产出
   含 ``data=<我们的 data.yaml>`` 的 ``command``；
4. **CLI 转正** —— ``train`` 在 ``IMPLEMENTED_COMMANDS`` 中，且可端到端跑通。

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
from traffic_cv_closed_loop.io_utils import read_json
from traffic_cv_closed_loop.models import AnnotationRecord, CandidateFrame, Detection

from industrial_hazard_coldstart import cli
from industrial_hazard_coldstart.cli import IMPLEMENTED_COMMANDS
from industrial_hazard_coldstart.orchestrator import (
    DEFAULT_DATASET_VERSION,
    run_cycle,
    run_train,
)

ROOT = Path(__file__).resolve().parents[1]
HOIST_CONFIG = ROOT / "own" / "hoist_project" / "config" / "project.hoist.json"
KB_PATH = ROOT / "hazard-detection" / "knowledge_base.json"


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


def _make_config(tmp: Path, *, trainer: str = "mock") -> ProjectConfig:
    raw = json.loads(HOIST_CONFIG.read_text(encoding="utf-8"))
    raw["qwen"]["provider"] = "mock"
    raw["knowledge"]["kb_path"] = str(KB_PATH)
    raw["yolo"]["trainer"] = trainer
    raw["yolo"]["detector"] = "sidecar"
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


class TrainStageTest(unittest.TestCase):
    def test_mock_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = _make_config(Path(tmp))
            _seed_annotations(config, 10)

            report = run_train(config, version="t1", execute=False)

            run_dir = config.runs_dir / "t1"
            self.assertTrue((run_dir / "report.json").exists())
            self.assertTrue((run_dir / "metrics.json").exists())
            self.assertEqual(report["dataset_version"], "t1")
            self.assertEqual(report["trainer"]["mode"], "mock")
            self.assertEqual(report["trainer"]["status"], "completed")
            self.assertGreaterEqual(report["holdout_count"], 1)
            self.assertTrue(report["holdout_hash"])
            self.assertEqual(report["trainer_mode"], "mock")
            # 训练器拿到的是**我们**的 data.yaml。
            self.assertTrue(Path(report["data_yaml_path"]).exists())

    def test_does_not_reuse_run_training(self) -> None:
        """母项目 run_training 会写 analysis/hard_examples.jsonl；我们不该有该副作用。"""
        with tempfile.TemporaryDirectory() as tmp:
            config = _make_config(Path(tmp))
            _seed_annotations(config, 8)
            run_train(config, version="t1", execute=False)
            self.assertFalse(
                (config.analysis_dir / "hard_examples.jsonl").exists(),
                "run_train 不应触发 run_training 的 hard_examples 副作用",
            )

    def test_ultralytics_command_prepared(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = _make_config(Path(tmp), trainer="ultralytics")
            _seed_annotations(config, 8)

            report = run_train(config, version="t2", execute=False)
            self.assertEqual(report["trainer"]["mode"], "ultralytics")
            self.assertEqual(report["trainer"]["status"], "prepared")
            command = report["trainer"]["command"]
            self.assertIn("train", command)
            self.assertIn(f"data={report['data_yaml_path']}", command)
            self.assertTrue((config.runs_dir / "t2" / "command.txt").exists())

    def test_default_version_is_not_timestamp(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = _make_config(Path(tmp))
            _seed_annotations(config, 6)
            report = run_train(config, execute=False)
            self.assertEqual(report["dataset_version"], DEFAULT_DATASET_VERSION)
            self.assertNotIn("_", DEFAULT_DATASET_VERSION.replace("coldstart-", ""))

    def test_run_cycle_train_branch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = _make_config(Path(tmp))
            _seed_annotations(config, 8)
            summary = run_cycle(config, "train", version="cycle-1")
            self.assertEqual(summary["stage"], "train")
            self.assertEqual(summary["dataset_version"], "cycle-1")
            self.assertTrue((config.runs_dir / "cycle-1" / "report.json").exists())


class TrainCliTest(unittest.TestCase):
    def test_train_is_implemented(self) -> None:
        self.assertIn("train", IMPLEMENTED_COMMANDS)

    def test_cli_train_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = _make_config(Path(tmp))
            _seed_annotations(config, 8)
            out, err = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                code = cli.main(
                    [
                        "train",
                        "--config",
                        str(config.config_path),
                        "--version",
                        "cli-v1",
                    ]
                )
            self.assertEqual(code, cli.EXIT_READY, err.getvalue())
            report = read_json(config.runs_dir / "cli-v1" / "report.json")
            self.assertEqual(report["dataset_version"], "cli-v1")

    def test_cli_train_without_annotations_is_readable_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = _make_config(Path(tmp))
            stderr = io.StringIO()
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(stderr):
                code = cli.main(["train", "--config", str(config.config_path)])
            self.assertEqual(code, cli.EXIT_STAGE_FAILED)
            self.assertIn("执行失败", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
