#!/usr/bin/env bash
# run_hoist_cycle.sh —— lab-server（Linux）全量入口
#
# 用法：
#   bash scripts/run_hoist_cycle.sh                 # 默认只跑 doctor
#   bash scripts/run_hoist_cycle.sh doctor
#   bash scripts/run_hoist_cycle.sh inventory       # T02–T05 未实现，会明确报错（退出码 2）
#   bash scripts/run_hoist_cycle.sh all             # 顺序跑完整链路（当前会在首个未实现阶段停下）
#
# 说明：全量标注 / VLM 调用 / 训练一律在 lab-server 上跑，工作区仅限 /data/users/lyk。

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CONFIG="${CONFIG:-own/hoist_project/config/project.hoist.json}"
STAGE="${1:-doctor}"

# 保证从仓库根导入新包与母项目。
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# 完整链路的阶段顺序（doctor 可随时单独跑）。
CYCLE_STAGES=(
  inventory
  extract-video
  extract
  annotate
  prioritize
  curate
  freeze-holdout
  train
  evaluate
)

run_stage() {
  local stage="$1"
  echo "[run_hoist_cycle] stage=$stage config=$CONFIG"
  python -m industrial_hazard_coldstart "$stage" --config "$CONFIG"
}

if [[ "$STAGE" == "all" ]]; then
  run_stage doctor
  for stage in "${CYCLE_STAGES[@]}"; do
    # 未实现阶段会返回非 0；set -e 会让脚本在此停下，避免误以为整条链路跑通。
    run_stage "$stage"
  done
else
  run_stage "$STAGE"
fi
