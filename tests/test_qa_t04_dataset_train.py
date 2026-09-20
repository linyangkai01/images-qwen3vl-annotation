"""T04 独立验证（QA 严过关）第 2 部分：质量过滤 / 版本与路径 / 训练编排 / doctor。

与工程师自测 ``tests/test_dataset_train.py`` **独立**，只攻自测未覆盖的边界：

5. **质量过滤** —— 逐类构造「会被剔除」的记录（``annotation_error`` / 低置信 / 0 标签），
   核实是否真的不进 manifest、是否写进 ``quality_report.json``；阈值覆盖是否生效；
   **未知阈值键被静默忽略**（D5，典型静默失败）；
6. **版本与路径** —— 同名版本重建是否清空、**中文版本号**、**中文路径帧**、
   以及 **版本号未校验 → 路径穿越 + 越界 ``shutil.rmtree``**（D3，已实测删掉既有文件）；
7. **训练编排** —— ``run_train`` 在 mock 路径下端到端产出 ``runs/<version>/report.json``；
   ``execute=False`` 的命令构造；**确认没有走 ``run_training``**（无
   ``analysis/hard_examples.jsonl``）；以及 ``doctor`` 的 holdout 完整性检查仍是
   T01 占位（D6，T04.2 已实现却仍报「尚未实现」）。

已知缺陷（D3/D5/D6）中，D3 / D6 断言**期望行为**（当前为红）；D5 与
「冻结副本缺失时的原始 OSError」按本项目 T01 既定约定做**行为刻画 + 绊线**。

约束：不修改任何生产代码；不联网、不调用真实 VLM、不跑真实训练；不读真实素材；
临时文件用 ``tempfile``。
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from traffic_cv_closed_loop.config import ProjectConfig
from traffic_cv_closed_loop.io_utils import read_json

import industrial_hazard_coldstart.orchestrator as orchestrator
from industrial_hazard_coldstart.dataset import build_dataset
from industrial_hazard_coldstart.doctor import build_hoist_doctor_report
from industrial_hazard_coldstart.holdout import (
    frozen_frame_path,
    holdout_dir_of,
    load_holdout_entries,
)
from industrial_hazard_coldstart.orchestrator import run_train

# ``tests/`` 不是包（无 ``__init__.py``），两种跑法（``-m unittest tests.x`` 与
# ``discover -s tests``）下模块可见性不同；显式把 tests 目录放进 ``sys.path``，
# 保证两种跑法都能复用第 1 部分的合成素材工具（不重复实现，避免两处漂移）。
_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

from test_qa_t04_holdout_boundary import (  # noqa: E402
    _make_config,
    _make_record,
    _TempProject,
    _uniq_records,
    _write_annotations,
)

ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- #
# 5. 质量过滤
# --------------------------------------------------------------------------- #
class QualityFilterTest(unittest.TestCase):
    def _seed(self) -> _TempProject:
        env = _TempProject(0)
        good = _uniq_records(env.config, 6, prefix="ok")
        bad_error = _make_record(
            env.config, "bad/error.png", color=(200, 1, 1),
            review_reasons=["annotation_error"],
        )
        low_conf = _make_record(env.config, "bad/low.png", color=(200, 2, 2), confidence=0.05)
        no_det = _make_record(env.config, "bad/none.png", color=(200, 3, 3), det_count=0)
        _write_annotations(env.config, [bad_error, low_conf, no_det] + good)
        return env

    def test_rejected_records_absent_from_manifest_and_present_in_quality_report(self) -> None:
        with self._seed() as env:
            summary = build_dataset(env.config, "v1", holdout_ratio=0.25, seed=7)
            self.assertEqual(summary["accepted"], 6)
            self.assertEqual(summary["rejected"], 3)

            manifest = read_json(Path(summary["manifest_path"]))
            manifest_paths = {item["relative_path"] for item in manifest}
            for rejected in ("bad/error.png", "bad/low.png", "bad/none.png"):
                with self.subTest(rejected=rejected):
                    self.assertNotIn(rejected, manifest_paths, f"{rejected} 泄漏进 manifest")

            report = read_json(Path(summary["quality_report_path"]))
            self.assertEqual(len(report), 9, "quality_report 必须覆盖全部记录（含被剔除的）")
            reasons = {
                row["frame"]: row["filter_reasons"]
                for row in report
                if not row["accepted"]
            }
            self.assertEqual(
                sorted(reasons), ["bad/error.png", "bad/low.png", "bad/none.png"]
            )
            self.assertIn("annotation_error", reasons["bad/error.png"])
            self.assertTrue(
                any(r.startswith("min_conf_lt_") for r in reasons["bad/low.png"]),
                f"低置信原因缺失：{reasons['bad/low.png']}",
            )
            self.assertTrue(
                any(r.startswith("label_count_lt_") for r in reasons["bad/none.png"]),
                f"0 标签原因缺失：{reasons['bad/none.png']}",
            )
            # 每行都要带 accepted 标记，便于下游区分。
            for row in report:
                self.assertIn("accepted", row)
                self.assertIn("filter_reasons", row)

    def test_threshold_override_takes_effect(self) -> None:
        with _TempProject(0) as env:
            high = [
                _make_record(env.config, f"ok/hi_{i}.png", color=(i * 10, 1, 1), confidence=0.9)
                for i in range(4)
            ]
            mid = [
                _make_record(env.config, f"ok/mid_{i}.png", color=(i * 10, 2, 2), confidence=0.5)
                for i in range(4)
            ]
            _write_annotations(env.config, high + mid)

            base = build_dataset(env.config, "v1", holdout_ratio=0.25, seed=7)
            strict = build_dataset(
                env.config, "v2", holdout_ratio=0.25, seed=7,
                quality_thresholds={"min_confidence": 0.7},
            )
            self.assertEqual(base["accepted"], 8)
            self.assertEqual(strict["accepted"], 4)
            self.assertEqual(strict["rejected"], 4)

    def test_unknown_threshold_key_is_rejected(self) -> None:
        """D5 修复后：拼错的阈值键必须 **fail-loud**，不再被静默丢弃。

        原缺陷（P2）：``quality_thresholds={"min_conf": 0.7}``（正确键是
        ``min_confidence``）与默认阈值的接受数完全一致 —— 使用者以为收紧了过滤，
        实际没生效，且无任何告警。工程师已修：``_quality_namespace`` 对未知键抛
        ``ValueError`` 并列出合法键（``dataset.py`` 的 ``_quality_namespace``）。

        本用例断言**期望行为**：未知键 → ``ValueError``，且错误消息必须列出合法键名，
        让使用者一眼看出拼错在哪。
        """
        with _TempProject(0) as env:
            mid = [
                _make_record(env.config, f"ok/mid_{i}.png", color=(i * 10, 2, 2), confidence=0.5)
                for i in range(4)
            ]
            _write_annotations(env.config, mid)

            with self.assertRaises(ValueError) as ctx:
                build_dataset(
                    env.config, "v1", holdout_ratio=0.25, seed=7,
                    quality_thresholds={"min_conf": 0.95},  # 键名拼错
                )
            message = str(ctx.exception)
            self.assertIn("min_conf", message, "错误消息必须点名非法键")
            for legal_key in ("min_confidence", "max_added", "max_removed",
                              "max_adjusted", "min_labels"):
                self.assertIn(legal_key, message, f"错误消息应列出合法键 {legal_key}")

            # 未产出任何数据集目录（fail-loud 不留下半成品）。
            self.assertFalse((env.config.datasets_dir / "v1").exists())

    def test_none_threshold_value_falls_back_to_default(self) -> None:
        """键值为 ``None`` 时回落默认（``_quality_namespace`` 的明文约定）。"""
        with _TempProject(0) as env:
            mid = [
                _make_record(env.config, f"ok/mid_{i}.png", color=(i * 10, 2, 2), confidence=0.5)
                for i in range(4)
            ]
            _write_annotations(env.config, mid)
            summary = build_dataset(
                env.config, "v1", holdout_ratio=0.25, seed=7,
                quality_thresholds={"min_confidence": None},
            )
            self.assertEqual(summary["accepted"], 4)

    def test_all_rejected_raises_readable_runtime_error(self) -> None:
        with _TempProject(0) as env:
            bad = _make_record(
                env.config, "bad/only.png", color=(9, 9, 9),
                review_reasons=["annotation_error"],
            )
            _write_annotations(env.config, [bad])
            with self.assertRaises(RuntimeError) as ctx:
                build_dataset(env.config, "v1", holdout_ratio=0.25, seed=7)
            self.assertIn("质量过滤", str(ctx.exception))


# --------------------------------------------------------------------------- #
# 6. 版本与路径
# --------------------------------------------------------------------------- #
class VersionAndPathTest(unittest.TestCase):
    def test_rebuild_same_version_clears_stale_files_and_keeps_holdout(self) -> None:
        with _TempProject(12) as env:
            first = build_dataset(env.config, "same", holdout_ratio=0.25, seed=7)
            stale = env.config.datasets_dir / "same" / "STALE.txt"
            stale.write_text("stale", encoding="utf-8")
            second = build_dataset(env.config, "same", holdout_ratio=0.25, seed=7)
            self.assertFalse(stale.exists(), "同名版本重建未清空上一轮残留文件")
            self.assertEqual(first["holdout_hash"], second["holdout_hash"])

    def test_chinese_version_and_chinese_frame_path(self) -> None:
        with _TempProject(0) as env:
            records = _uniq_records(env.config, 6) + [
                _make_record(env.config, "吊装现场/帧_000001.png", color=(7, 7, 7))
            ]
            _write_annotations(env.config, records)
            summary = build_dataset(env.config, "吊装-v1", holdout_ratio=0.25, seed=7)
            self.assertTrue(Path(summary["dataset_dir"]).exists())
            self.assertEqual(Path(summary["dataset_dir"]).name, "吊装-v1")
            self.assertEqual(summary["accepted"], 7)
            manifest = read_json(Path(summary["manifest_path"]))
            self.assertTrue(
                any("吊装现场" in item["relative_path"] for item in manifest),
                "中文路径的帧未进 manifest",
            )

    def test_version_is_rejected_when_it_escapes_datasets_dir(self) -> None:
        """**已知缺陷 D3（P1）**：版本号**未做任何校验**就被拼进路径并 ``shutil.rmtree``。

        实测（``industrial_hazard_coldstart/dataset.py:258-261``）：
        ``build_dataset(config, "../../victim")`` → ``dataset_dir`` 解析到
        ``<project>/victim``（**越出** ``workspace/datasets``），随后
        ``shutil.rmtree(dataset_dir, ignore_errors=True)`` 把该既有目录**连同其中文件
        一起删除**，再往里写数据集 —— 既不报错也不告警，静默造成工作区外数据丢失。
        绝对路径版本号（``D:\\...\\outside``）同样生效。

        期望：版本号含路径分隔符 / ``..`` / 为绝对路径时**显式抛 ValueError**。
        """
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            config = _make_config(root / "proj")
            _write_annotations(config, _uniq_records(config, 6))

            victim = root / "proj" / "victim"
            victim.mkdir(parents=True)
            (victim / "important.txt").write_text("do not delete", encoding="utf-8")

            for bad_version in ("../../victim", "../victim", str(root / "outside")):
                with self.subTest(version=bad_version):
                    with self.assertRaises(ValueError):
                        build_dataset(config, bad_version, holdout_ratio=0.25, seed=7)

            self.assertTrue(
                (victim / "important.txt").exists(),
                "工作区外的既有文件被删除了（版本号路径穿越 + 越界 rmtree）",
            )

    def test_blank_version_is_rejected(self) -> None:
        for blank in ("", "   ", "\t"):
            with self.subTest(version=repr(blank)):
                with _TempProject(6) as env:
                    with self.assertRaises(ValueError):
                        build_dataset(env.config, blank, holdout_ratio=0.25, seed=7)


# --------------------------------------------------------------------------- #
# 7. 训练编排
# --------------------------------------------------------------------------- #
class TrainOrchestrationTest(unittest.TestCase):
    def test_mock_trainer_end_to_end_produces_report(self) -> None:
        with _TempProject(10) as env:
            report = run_train(env.config, version="coldstart-v1", holdout_ratio=0.25,
                               execute=False)
            run_dir = Path(report["run_dir"])
            self.assertTrue((run_dir / "report.json").exists())
            self.assertTrue((run_dir / "metrics.json").exists())
            self.assertTrue((run_dir / "trainer_summary.md").exists())
            self.assertTrue((env.config.status_dir / "train.json").exists())

            self.assertEqual(report["dataset_version"], "coldstart-v1")
            self.assertEqual(report["trainer"]["mode"], "mock")
            self.assertEqual(report["trainer"]["status"], "completed")
            self.assertEqual(report["trainer"]["executed"], False)
            self.assertTrue(report["holdout_count"] > 0)
            self.assertTrue(report["holdout_hash"])
            self.assertEqual(
                report["split_counts"]["train"]
                + report["split_counts"]["val"]
                + report["split_counts"]["holdout"],
                report["accepted_images"],
            )
            # report.json 落盘内容与返回值一致（便于下游读取）。
            self.assertEqual(read_json(run_dir / "report.json")["holdout_hash"],
                             report["holdout_hash"])

    def test_does_not_use_mother_project_run_training(self) -> None:
        """不得复用母项目 ``run_training``（时间戳版本号 + 每轮重切分会破坏 holdout）。"""
        with _TempProject(10) as env:
            run_train(env.config, version="v1", holdout_ratio=0.25, execute=False)
            # 1) 模块里根本没有 run_training 这个名字。
            self.assertFalse(hasattr(orchestrator, "run_training"))
            # 2) 母项目 run_training 的副作用产物不应出现。
            self.assertFalse(
                (env.config.analysis_dir / "hard_examples.jsonl").exists(),
                "出现了 analysis/hard_examples.jsonl —— 疑似走了母项目 run_training",
            )
            self.assertFalse(env.config.analysis_dir.exists())

    def test_ultralytics_command_is_prepared_with_our_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            config = _make_config(root / "proj")
            raw_cfg = json.loads(config.config_path.read_text(encoding="utf-8"))
            raw_cfg["yolo"]["trainer"] = "ultralytics"
            config.config_path.write_text(json.dumps(raw_cfg, ensure_ascii=False), encoding="utf-8")
            config = ProjectConfig.load(config.config_path)
            _write_annotations(config, _uniq_records(config, 10))

            report = run_train(config, version="ultra-v1", holdout_ratio=0.25, execute=False)
            self.assertEqual(report["trainer"]["mode"], "ultralytics")
            self.assertEqual(report["trainer"]["status"], "prepared")
            command = report["trainer"]["command"]
            self.assertIn(" train ", command)
            self.assertIn(f"data={report['data_yaml_path']}", command,
                          "训练命令必须指向我们自己的 data.yaml")
            self.assertIn("ultra-v1", command)
            # 命令也落盘，供 lab-server 执行。
            self.assertIn("data=", (Path(report["run_dir"]) / "command.txt").read_text(
                encoding="utf-8"))

    def test_missing_frozen_copy_fails_loud_but_with_raw_oserror(self) -> None:
        """行为刻画 + 绊线：冻结副本被删后重建**确实报错**（不静默 ✅）。

        但错误是 ``copy_file`` 抛出的原始 ``FileNotFoundError``（Windows 上表现为
        ``[WinError 2] 系统找不到指定的文件``），完全没有提到 holdout —— 使用者无法
        据此判断「holdout 完整性坏了」。建议：物化 holdout 前先跑
        :func:`verify_holdout`，失败时给出「holdout 完整性校验失败：<帧>」的可读错误。
        """
        with _TempProject(12) as env:
            build_dataset(env.config, "v1", holdout_ratio=0.25, seed=7)
            entries = load_holdout_entries(holdout_dir_of(env.config))
            self.assertTrue(entries)
            frozen_frame_path(holdout_dir_of(env.config), entries[0].relative_path).unlink()
            with self.assertRaises(OSError):
                build_dataset(env.config, "v2", holdout_ratio=0.25, seed=7)


class DatasetObservabilityTest(unittest.TestCase):
    """``manifest.json`` / ``quality_report.json`` 的可观测性（下游 T05 要消费它们）。"""

    def test_manifest_entries_share_a_common_key_set(self) -> None:
        """**已知缺陷 D7（P2）**：manifest 条目**不同构**，按 split 键集不一致。

        实测（``industrial_hazard_coldstart/dataset.py:286-321``）：
        - ``train`` / ``val`` 条目：``split/relative_path/image_path/label_path/
          review_required/review_reasons/diff``（**无** ``sha256``）；
        - ``holdout`` 条目：``split/relative_path/image_path/label_path/sha256``
          （**无** ``review_required`` / ``review_reasons`` / ``diff``）。

        下游（T05 逐类评估）若统一按 ``item["review_required"]`` 过滤就会 ``KeyError``，
        而这种缺键在构建期完全无感。期望：所有条目至少带同一组公共键。
        """
        with _TempProject(10) as env:
            summary = build_dataset(env.config, "v1", holdout_ratio=0.25, seed=7)
            manifest = read_json(Path(summary["manifest_path"]))
            self.assertTrue({item["split"] for item in manifest} >= {"train", "holdout"})
            required = {"split", "relative_path", "image_path", "label_path", "review_required"}
            for item in manifest:
                with self.subTest(split=item["split"], frame=item["relative_path"]):
                    self.assertTrue(
                        required.issubset(item),
                        f"manifest 条目缺公共键 {sorted(required - set(item))}（split={item['split']}）",
                    )

    def test_missing_label_file_is_not_silent(self) -> None:
        """**已知缺陷 D8（P2）**：标签文件丢失 → 静默写空标签当**背景**，全程无痕。

        实测：``_copy_label``（``dataset.py:158-165``）在源标签不存在时写空文件。
        构造 6 条标注记录、删掉它们的 ``.txt`` 后重建 → 产出 6 个空标签文件，而
        ``summary.json`` 与 ``quality_report.json`` **都没有任何字段**记录此事
        （实测 ``summary`` 键中不含任何 ``label`` 相关项，``quality_report`` 行也没有
        ``label_missing`` 之类标记）。

        后果：这些帧被 YOLO 当作「无目标背景」参与训练 → 教会模型「这里没东西」，
        直接制造漏检；而使用者从任何产物上都看不出来。
        期望：构建必须在 ``summary`` 或 ``quality_report`` 里留下可见记录。
        """
        with _TempProject(0) as env:
            records = _uniq_records(env.config, 6, prefix="nolabel")
            for record in records:
                label = Path(record.label_path)
                if label.exists():
                    label.unlink()
            _write_annotations(env.config, records)

            summary = build_dataset(env.config, "v1", holdout_ratio=0.25, seed=7)
            manifest = read_json(Path(summary["manifest_path"]))
            empty_labels = [
                item["relative_path"]
                for item in manifest
                if Path(item["label_path"]).exists()
                and not Path(item["label_path"]).read_text(encoding="utf-8").strip()
            ]
            self.assertEqual(len(empty_labels), 6, "前置条件：确应产出 6 个空标签")

            quality_rows = read_json(Path(summary["quality_report_path"]))
            visible = any("label" in key.lower() for key in summary) or any(
                "label_missing" in row or "label_path_exists" in row for row in quality_rows
            )
            self.assertTrue(
                visible,
                "标签文件缺失被静默当成背景图：summary / quality_report 都没有任何记录",
            )


class DoctorHoldoutCheckTest(unittest.TestCase):
    """``doctor`` 的 holdout 完整性检查（需求 7.4 / T04.2）。"""

    def test_doctor_reports_real_holdout_status_after_freeze(self) -> None:
        """**已知缺陷 D6（P1）**：T04.2 已实现冻结，``doctor`` 仍是 T01 占位。

        ``industrial_hazard_coldstart/doctor.py:127-138`` 无条件写死
        ``"status": "not_implemented"`` 并把 ``"holdout_not_implemented"`` 塞进
        ``pending``，且**从不校验 sha256**。实测：``index.json`` 已存在（
        ``index_exists: true``）时，doctor 依旧输出「固定 holdout 冻结与 sha256
        登记属 T04.2，本批（T01）尚未实现」——对使用者是**误导**（实际已实现），
        也让「holdout 被篡改 / 被删」这类真正的完整性问题无处暴露。

        期望：``index.json`` 存在时，``status`` 不得为 ``not_implemented``，
        ``pending`` 不得含 ``holdout_not_implemented``，并给出完整性结论。
        """
        with _TempProject(12) as env:
            build_dataset(env.config, "v1", holdout_ratio=0.25, seed=7)
            self.assertTrue((holdout_dir_of(env.config) / "index.json").exists())

            report = build_hoist_doctor_report(env.config)
            holdout = report["checks"]["holdout"]
            self.assertTrue(holdout["index_exists"], "doctor 未感知到已存在的冻结清单")
            self.assertNotEqual(
                holdout["status"], "not_implemented",
                "T04.2 已实现，doctor 仍把 holdout 完整性标记为「尚未实现」",
            )
            self.assertNotIn(
                "holdout_not_implemented", report["pending"],
                "T04.2 已实现，doctor 仍在 pending 里挂 holdout_not_implemented",
            )


if __name__ == "__main__":
    unittest.main()
