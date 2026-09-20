"""``python -m industrial_hazard_coldstart`` 入口。

把命令行参数原样交给 :func:`industrial_hazard_coldstart.cli.main`，
并以它的返回值作为进程退出码（非 0 表示失败或「尚未实现」）。
"""

from __future__ import annotations

from industrial_hazard_coldstart.cli import main

if __name__ == "__main__":  # pragma: no cover - 进程入口
    raise SystemExit(main())
