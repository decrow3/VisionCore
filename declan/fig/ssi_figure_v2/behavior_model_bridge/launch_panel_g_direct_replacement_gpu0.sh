#!/usr/bin/env bash
set -euo pipefail

cd /home/declan/VisionCore

run_root="outputs/fig/ssi_figure_v2/behavior_model_bridge/panel_g_direct_replacement_strong_contour_v1"
log_root="${run_root}/background_logs"
smoke_root="${run_root}/smoke_n32_gpu0"
full_root="${run_root}/direct_n32_gpu0"
mkdir -p "${log_root}"

echo "$$" > "${log_root}/direct_n32_gpu0_guard.pid"
echo "queued_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "contract=wait for physical GPU 0 to have no compute process, run two-pair smoke, then run all 104 pairs"

while nvidia-smi --id=0 --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | grep -Eq '[0-9]'; do
    gpu_state=$(nvidia-smi --id=0 --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits 2>/dev/null || true)
    echo "waiting_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) gpu0=${gpu_state}"
    sleep 60
done

echo "smoke_started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
env CUDA_VISIBLE_DEVICES=0 MPLCONFIGDIR=/tmp/matplotlib-cache \
    .venv/bin/python -u -m \
    declan.fig.ssi_figure_v2.behavior_model_bridge.run_panel_g_direct_replacement_production \
    --device cuda:0 \
    --n-rotations 32 \
    --pair-start 0 \
    --pair-stop 2 \
    --frame-batch-size 16 \
    --trace-batch-size 8 \
    --out-dir "${smoke_root}"

test -s "${smoke_root}/shards/frozen_000000_000002/direct_pair_rotation_contrasts.csv"
test -s "${smoke_root}/shards/frozen_000000_000002/direct_pair_unit_metrics.npz"
echo "smoke_passed_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"

echo "full_started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
env CUDA_VISIBLE_DEVICES=0 MPLCONFIGDIR=/tmp/matplotlib-cache \
    .venv/bin/python -u -m \
    declan.fig.ssi_figure_v2.behavior_model_bridge.run_panel_g_direct_replacement_production \
    --device cuda:0 \
    --n-rotations 32 \
    --pair-start 0 \
    --pair-stop 104 \
    --frame-batch-size 16 \
    --trace-batch-size 8 \
    --out-dir "${full_root}"

test -s "${full_root}/shards/frozen_000000_000104/direct_pair_rotation_contrasts.csv"
test -s "${full_root}/shards/frozen_000000_000104/direct_pair_unit_metrics.npz"
echo "full_passed_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
