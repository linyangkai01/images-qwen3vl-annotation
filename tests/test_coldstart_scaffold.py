"""T01 脚手架测试：CLI 骨架 / doctor 探活 / 提示词模板格式安全。

要点：
- ``--help`` 可用；
- 未实现子命令返回**非 0** 退出码且打印明确提示（不得静默成功）；
- ``doctor`` 报告含四项检查（ffmpeg / VLM 端点 / 端点护栏 / holdout），
  且 holdout 如实标注「尚未实现」；
- 吊装提示词能被母项目 ``render_annotation_prompt`` 成功渲染（花括号已转义）。
"""

from __future__ import annotations

import contextlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from traffic_cv_closed_loop.config import ProjectConfig
from traffic_cv_closed_loop.models import CandidateFrame
from traffic_cv_closed_loop.prompting import render_annotation_prompt

from industrial_hazard_coldstart import __version__
from industrial_hazard_coldstart.cli import (
    EXIT_CONFIG_UNREADABLE,
    EXIT_NOT_IMPLEMENTED,
    EXIT_READY,
    _PENDING_COMMANDS,
    main,
)
from industrial_hazard_coldstart.doctor import (
    LOCAL_ONLY_ALLOWLIST_WARNING,
    build_hoist_doctor_report,
)

ROOT = Path(__file__).resolve().parents[1]
HOIST_CONFIG = ROOT / "own" / "hoist_project" / "config" / "project.hoist.json"


class CliSkeletonTest(unittest.TestCase):
    def test_help_is_available(self) -> None:
        with self.assertRaises(SystemExit) as ctx, contextlib.redirect_stdout(io.StringIO()):
            main(["--help"])
        self.assertEqual(ctx.exception.code, 0)

    def test_subcommand_help_is_available(self) -> None:
        with self.assertRaises(SystemExit) as ctx, contextlib.redirect_stdout(io.StringIO()):
            main(["doctor", "--help"])
        self.assertEqual(ctx.exception.code, 0)

    def test_pending_commands_are_not_silently_successful(self) -> None:
        # 除 doctor 外的每个子命令都必须返回非 0，并打印「尚未实现」。
        for command in _PENDING_COMMANDS:
            out, err = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                code = main([command, "--config", str(HOIST_CONFIG)])
            self.assertNotEqual(code, 0, f"子命令 {command} 不应返回 0（静默成功）")
            self.assertEqual(code, EXIT_NOT_IMPLEMENTED, f"子命令 {command} 退出码应为 {EXIT_NOT_IMPLEMENTED}")
            self.assertIn("尚未实现", out.getvalue() + err.getvalue())

    def test_doctor_runs_and_returns_code(self) -> None:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = main(["doctor", "--config", str(HOIST_CONFIG)])
        self.assertIn(code, (0, 1))
        self.assertIn("checks", out.getvalue())


class DoctorReportTest(unittest.TestCase):
    def test_report_contains_four_checks(self) -> None:
        config = ProjectConfig.load(HOIST_CONFIG)
        report = build_hoist_doctor_report(config)
        self.assertIn(report["status"], {"ready", "issues_found"})
        checks = report["checks"]
        self.assertEqual(
            set(checks),
            {"ffmpeg", "vlm_endpoint", "endpoint_policy", "holdout"},
        )

    def test_holdout_is_marked_not_implemented(self) -> None:
        config = ProjectConfig.load(HOIST_CONFIG)
        report = build_hoist_doctor_report(config)
        # holdout 属 T04.2，本批未实现 → 必须如实标注，不得假装通过。
        self.assertEqual(report["checks"]["holdout"]["status"], "not_implemented")
        self.assertIn("holdout_not_implemented", report["pending"])

    def test_endpoint_policy_guard_is_allowlisted(self) -> None:
        config = ProjectConfig.load(HOIST_CONFIG)
        report = build_hoist_doctor_report(config)
        policy = report["checks"]["endpoint_policy"]
        self.assertEqual(policy["mode"], "allowlist")
        self.assertTrue(policy["allowed"])
        # VLM 端点探活不在本机执行。
        self.assertFalse(report["checks"]["vlm_endpoint"]["probed"])


class PromptTemplateTest(unittest.TestCase):
    def test_prompt_has_required_placeholders(self) -> None:
        template = (ROOT / "own" / "hoist_project" / "assets" / "prompts" / "hoist_zh.md").read_text(
            encoding="utf-8"
        )
        for placeholder in ("{scene}", "{labels}", "{candidate_path}", "{prior_detections}"):
            self.assertIn(placeholder, template)
        # 受控词表要求与 KB S1 注入痕迹。
        self.assertIn("受控词表", template)
        self.assertIn("D-5-1", template)

    def test_prompt_renders_with_mother_project(self) -> None:
        config = ProjectConfig.load(HOIST_CONFIG)
        template = config.prompt_template_path.read_text(encoding="utf-8")
        candidate = CandidateFrame(
            source_id="probe",
            relative_path="s1/frame_0001.png",
            original_path="",
            extracted_path="",
            detections=[],
            scene="hoist",
        )
        rendered = render_annotation_prompt(config, candidate, template)
        # 花括号已转义 → format 后应还原为 JSON 示意。
        self.assertIn('"objects"', rendered)
        self.assertIn("s1/frame_0001.png", rendered)
        self.assertNotIn("{scene}", rendered)


class PackageTest(unittest.TestCase):
    def test_version_is_exposed(self) -> None:
        self.assertEqual(__version__, "0.1.0")


class ConfigErrorTest(unittest.TestCase):
    """缺陷 A：配置读不到时给出可读错误 + 专用退出码，且不打印栈。"""

    def _run_cli(self, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-m", "industrial_hazard_coldstart", *args],
            cwd=str(ROOT), capture_output=True, text=True,
        )

    def test_missing_config_returns_dedicated_code_without_traceback(self) -> None:
        result = self._run_cli("doctor", "--config", "does/not/exist.json")
        combined = result.stdout + result.stderr
        self.assertNotIn("Traceback (most recent call last)", combined)
        self.assertEqual(result.returncode, EXIT_CONFIG_UNREADABLE)
        self.assertIn("无法读取配置文件", combined)
        self.assertIn("Config file not found", combined)

    def test_corrupt_config_returns_dedicated_code_without_traceback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            broken = Path(tmp) / "broken.json"
            broken.write_text("{ this is not valid json", encoding="utf-8")
            result = self._run_cli("doctor", "--config", str(broken))
        combined = result.stdout + result.stderr
        self.assertNotIn("Traceback (most recent call last)", combined)
        self.assertEqual(result.returncode, EXIT_CONFIG_UNREADABLE)
        self.assertIn("无法读取配置文件", combined)

    def test_config_error_code_differs_from_doctor_issues_code(self) -> None:
        # 退出码必须能区分「配置读不到」(3) 与「读到了但发现问题」(1)。
        self.assertNotEqual(EXIT_CONFIG_UNREADABLE, 1)
        self.assertNotEqual(EXIT_CONFIG_UNREADABLE, EXIT_READY)

    def test_main_returns_config_code_in_process(self) -> None:
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = main(["doctor", "--config", "does/not/exist.json"])
        self.assertEqual(code, EXIT_CONFIG_UNREADABLE)
        self.assertNotIn("Traceback", err.getvalue())


class EndpointPolicyWarningTest(unittest.TestCase):
    """新增：doctor 检测「local_only + 白名单」危险组合并告警（不混入 issues）。"""

    def _config_with_local_only_and_allowlist(self, tmp_dir: str) -> ProjectConfig:
        raw = json.loads(HOIST_CONFIG.read_text(encoding="utf-8"))
        raw["inference"]["endpoint_policy"]["mode"] = "local_only"
        raw["inference"]["endpoint_policy"]["allowlist"] = ["119.23.145.227:38091"]
        cfg_path = Path(tmp_dir) / "cfg.json"
        cfg_path.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
        return ProjectConfig.load(cfg_path)

    def test_local_only_with_allowlist_emits_warning(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = self._config_with_local_only_and_allowlist(tmp)
            report = build_hoist_doctor_report(config)

        policy = report["checks"]["endpoint_policy"]
        self.assertTrue(policy["local_only_with_allowlist"])
        self.assertEqual(policy["warning"], LOCAL_ONLY_ALLOWLIST_WARNING)
        self.assertIn(LOCAL_ONLY_ALLOWLIST_WARNING, report["warnings"])
        # 告警不得混进阻塞项 issues。
        self.assertNotIn(LOCAL_ONLY_ALLOWLIST_WARNING, report["issues"])
        self.assertIn("local_only", LOCAL_ONLY_ALLOWLIST_WARNING)

    def test_allowlist_mode_does_not_warn(self) -> None:
        config = ProjectConfig.load(HOIST_CONFIG)
        report = build_hoist_doctor_report(config)
        self.assertFalse(report["checks"]["endpoint_policy"]["local_only_with_allowlist"])
        self.assertEqual(report["checks"]["endpoint_policy"]["warning"], "")
        self.assertEqual(report["warnings"], [])

    def test_local_only_without_allowlist_does_not_warn(self) -> None:
        # local_only + 空白名单：用本机端点使配置可加载，此时不应告警。
        with tempfile.TemporaryDirectory() as tmp:
            raw = json.loads(HOIST_CONFIG.read_text(encoding="utf-8"))
            raw["qwen"]["endpoint"] = "http://127.0.0.1:11434/api/generate"
            raw["inference"]["endpoint_policy"]["mode"] = "local_only"
            raw["inference"]["endpoint_policy"]["allowlist"] = []
            cfg_path = Path(tmp) / "cfg.json"
            cfg_path.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
            config = ProjectConfig.load(cfg_path)
            report = build_hoist_doctor_report(config)
        self.assertFalse(report["checks"]["endpoint_policy"]["local_only_with_allowlist"])
        self.assertEqual(report["warnings"], [])


if __name__ == "__main__":
    unittest.main()
