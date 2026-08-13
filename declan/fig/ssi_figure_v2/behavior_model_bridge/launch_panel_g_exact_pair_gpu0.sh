#!/usr/bin/env bash
set -euo pipefail

cd /home/declan/VisionCore

output_root="outputs/fig/ssi_figure_v2/behavior_model_bridge/panel_g_exact_pair_fig4_trace_bank_n1000_v1"
log_dir="${output_root}/background_logs"
mkdir -p "${log_dir}"
echo "$$" > "${log_dir}/production_gpu0.pid"

exec env MPLCONFIGDIR=/tmp/matplotlib-cache \
  .venv/bin/python -u -m \
  declan.fig.ssi_figure_v2.behavior_model_bridge.run_panel_g_exact_pair_production \
  --out-dir "${output_root}" \
  --pair-start 0 \
  --pair-stop 1000 \
  --device cuda:0 \
  --frame-batch-size 16 \
  --trace-batch-size 8
