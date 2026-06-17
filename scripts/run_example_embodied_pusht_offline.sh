#!/usr/bin/env bash
set -euo pipefail

# 用法：
#   bash scripts/run_example_embodied_pusht_offline.sh
#   bash scripts/run_example_embodied_pusht_offline.sh /绝对/或/相对/数据集目录
#
# 作用：
#   1. 默认运行仓库自带的 sample_demos/0
#   2. 如果传了第一个参数，就把它当作离线数据集目录
#   3. 不再需要手改 example_embodied_pusht_offline.py 源码

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_DATASET="$ROOT_DIR/examples/embodied_environments/pusht_embodied/sample_demos/0"
DATASET_PATH="${1:-$DEFAULT_DATASET}"

echo "[run_example_embodied_pusht_offline] 数据集目录: $DATASET_PATH"

cd "$ROOT_DIR"
env EMBODIED_GAUSSIANS_DATASET="$DATASET_PATH" python examples/example_embodied_pusht_offline.py
