"""把「待补规范依据清单」里的 norm_ref 合并回 knowledge_base.json。

纯标准库实现，不引入任何第三方依赖，不做任何联网操作。

用法::

    # 1) 先干跑（默认），只看会改什么，不写文件
    python scripts/fill_norm_ref.py

    # 2) 确认无误后真正写入（自动备份）
    python scripts/fill_norm_ref.py --apply

    # 3) 顺便修正 meta.total_hazards 陈旧字段
    python scripts/fill_norm_ref.py --apply --fix-meta

默认路径：
    --checklist deliverables/software-company/norm_ref-待补清单.md
    --kb        hazard-detection/knowledge_base.json

清单填写格式（每条隐患下面那行）::

    - **D-2-1** 吊架不平衡
      - 视觉特征：……
      - 规范依据：JGJ 276-2012
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CHECKLIST = REPO_ROOT / "deliverables" / "software-company" / "norm_ref-待补清单.md"
DEFAULT_KB = REPO_ROOT / "hazard-detection" / "knowledge_base.json"

# - **D-2-1** 吊架不平衡
_ENTRY_RE = re.compile(r"^\s*-\s*\*\*(?P<id>[A-Za-z0-9][A-Za-z0-9\-]*)\*\*\s*(?P<name>.*?)\s*$")
#   - 规范依据：JGJ 276-2012
_NORM_RE = re.compile(r"^\s*[-*]?\s*规范依据\s*[:：]\s*(?P<value>.*?)\s*$")

PLACEHOLDERS = {"", "-", "—", "无", "待查", "待定", "TODO", "?"}


def parse_checklist(path: Path) -> list[dict[str, str]]:
    """从 Markdown 清单中解析出 (id, name, norm_ref) 三元组。"""
    if not path.exists():
        raise FileNotFoundError(f"清单文件不存在：{path}")

    entries: list[dict[str, str]] = []
    current: dict[str, str] | None = None

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        entry_match = _ENTRY_RE.match(raw_line)
        if entry_match:
            if current is not None:
                entries.append(current)
            current = {
                "id": entry_match.group("id").strip(),
                "name": entry_match.group("name").strip(),
                "norm_ref": "",
            }
            continue

        if current is None:
            continue

        norm_match = _NORM_RE.match(raw_line)
        if norm_match:
            value = norm_match.group("value").strip()
            # 去掉常见的 Markdown 包裹符
            value = value.strip("`*_ ")
            current["norm_ref"] = value

    if current is not None:
        entries.append(current)

    return entries


def index_hazards(node: object, index: dict[str, dict]) -> None:
    """递归建立 id -> 隐患节点 的索引。

    只收录位于 ``hazards`` 列表下的节点，避免把 ``ten_no_lift.rules`` /
    ``crane_six_forbidden.rules`` 这类同样带 ``id`` 的规则误当成隐患。
    """
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "hazards" and isinstance(value, list):
                for item in value:
                    if not isinstance(item, dict):
                        continue
                    node_id = item.get("id")
                    if isinstance(node_id, str) and node_id.strip():
                        index.setdefault(node_id.strip(), item)
                    index_hazards(item, index)
            else:
                index_hazards(value, index)
    elif isinstance(node, list):
        for item in node:
            index_hazards(item, index)


def build_plan(entries: list[dict[str, str]], index: dict[str, dict]) -> dict[str, list]:
    to_fill: list[tuple[str, str, str]] = []
    already: list[tuple[str, str]] = []
    unknown: list[str] = []
    blank: list[str] = []

    for entry in entries:
        hazard_id = entry["id"]
        value = entry["norm_ref"]

        if hazard_id not in index:
            unknown.append(hazard_id)
            continue

        node = index[hazard_id]
        existing = str(node.get("norm_ref") or "").strip()

        if value in PLACEHOLDERS:
            blank.append(hazard_id)
            continue

        if existing and existing == value:
            already.append((hazard_id, existing))
            continue

        if existing:
            to_fill.append((hazard_id, existing, value))
        else:
            to_fill.append((hazard_id, "", value))

    return {
        "to_fill": to_fill,
        "already": already,
        "unknown": unknown,
        "blank": blank,
    }


def coverage(index: dict[str, dict]) -> tuple[int, int]:
    total = len(index)
    filled = sum(1 for node in index.values() if str(node.get("norm_ref") or "").strip())
    return filled, total


def main() -> int:
    parser = argparse.ArgumentParser(description="合并 norm_ref 到 knowledge_base.json")
    parser.add_argument("--checklist", type=Path, default=DEFAULT_CHECKLIST, help="待补清单 Markdown 路径")
    parser.add_argument("--kb", type=Path, default=DEFAULT_KB, help="knowledge_base.json 路径")
    parser.add_argument("--apply", action="store_true", help="真正写入（默认只干跑）")
    parser.add_argument("--fix-meta", action="store_true", help="同时把 meta.total_hazards 修正为实际条数")
    args = parser.parse_args()

    print(f"清单文件：{args.checklist}")
    print(f"知识库  ：{args.kb}")
    print()

    entries = parse_checklist(args.checklist)
    print(f"清单解析到 {len(entries)} 条隐患条目")

    raw_text = args.kb.read_text(encoding="utf-8")
    data = json.loads(raw_text)

    index: dict[str, dict] = {}
    index_hazards(data, index)

    filled_before, total_before = coverage(index)
    print(f"知识库识别到 {total_before} 条隐患；当前已有 norm_ref：{filled_before} 条"
          f"（覆盖率 {100 * filled_before / max(total_before, 1):.2f}%）")
    print()

    plan = build_plan(entries, index)

    if plan["unknown"]:
        print(f"[警告] 清单里有 {len(plan['unknown'])} 个 id 在知识库中找不到：")
        for hazard_id in plan["unknown"]:
            print(f"    - {hazard_id}")
        print()

    if plan["blank"]:
        print(f"[提示] 清单里有 {len(plan['blank'])} 条尚未填写（视为跳过）：")
        print("    " + "、".join(plan["blank"]))
        print()

    if plan["already"]:
        print(f"[提示] {len(plan['already'])} 条与知识库现值一致，无需改动")

    if not plan["to_fill"]:
        print()
        print("没有需要写入的条目。")
        print("→ 请在清单里把「规范依据：」后面填上国标/行标号，再重新运行本脚本。")
        return 0

    print()
    print(f"将要写入 {len(plan['to_fill'])} 条：")
    print()
    for hazard_id, old, new in plan["to_fill"]:
        name = index[hazard_id].get("name", "")
        old_display = old if old else "（空）"
        print(f"  {hazard_id:8s} {name[:24]:26s} {old_display:24s} -> {new}")

    if not args.apply:
        print()
        print("以上为干跑结果，未写入任何文件。")
        print("确认无误后加 --apply 真正写入。")
        return 0

    backup_path = args.kb.with_suffix(args.kb.suffix + f".bak-{datetime.now():%Y%m%d-%H%M%S}")
    shutil.copy2(args.kb, backup_path)
    print()
    print(f"已备份原文件 -> {backup_path.name}")

    for hazard_id, _old, new in plan["to_fill"]:
        index[hazard_id]["norm_ref"] = new

    if args.fix_meta:
        meta = data.get("meta")
        if isinstance(meta, dict):
            old_meta = meta.get("total_hazards")
            meta["total_hazards"] = total_before
            print(f"meta.total_hazards: {old_meta} -> {total_before}")

    args.kb.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    filled_after, total_after = coverage(index)
    print()
    print(f"写入完成。norm_ref 覆盖：{filled_before}/{total_before} -> {filled_after}/{total_after}"
          f"（覆盖率 {100 * filled_after / max(total_after, 1):.2f}%）")

    if filled_after == total_after:
        print("全部条目已具备规范依据。")
    else:
        print(f"仍有 {total_after - filled_after} 条缺 norm_ref。")
    print()
    print("下一步建议：")
    print("  python -m unittest tests.test_hazard_adversarial_qa -v   # 复测 F1 相关断言")
    print("  python -m hazard_annotation demo --mock                  # 确认链路仍可跑通")

    return 0


if __name__ == "__main__":
    sys.exit(main())
