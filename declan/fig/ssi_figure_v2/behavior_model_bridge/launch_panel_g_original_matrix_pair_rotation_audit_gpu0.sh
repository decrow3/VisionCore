#!/usr/bin/env bash
set -euo pipefail

cd /home/declan/VisionCore

output_root="outputs/fig/ssi_figure_v2/behavior_model_bridge/panel_g_original_matrix_pair_rotation_audit_v1/fresh_direct_rotation_n32_gpu0"
log_dir="${output_root}/background_logs"
mkdir -p "${log_dir}"
echo "$$" > "${log_dir}/audit_gpu0.pid"

set +e
env CUDA_VISIBLE_DEVICES=0 MPLCONFIGDIR=/tmp/matplotlib-cache \
  .venv/bin/python -u -m \
  declan.fig.ssi_figure_v2.behavior_model_bridge.run_panel_g_original_matrix_pair_rotation_audit \
  --out-dir "${output_root}" \
  --n-rotations 32 \
  --device cuda:0 \
  --frame-batch-size 4 \
  --trace-batch-size 2 \
  --map-rotation-examples 2
status=$?
echo "[launcher] exit_status=${status} completed_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
exit "${status}"
