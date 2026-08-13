#!/usr/bin/env bash
set -euo pipefail

repo_dir="/home/declan/VisionCore"
output_dir="$repo_dir/outputs/redundancy_resolved_v1_twin/rr100_zero_gaze_separable_sf_tf_native_production_v1"
log_dir="$output_dir/background_logs"
wait_pid="2615062"
wait_command_fragment="/home/jake/repos/VisionCore/paper/fig4/mechanism_audit_v1/correction/run_corrected_core.py"
gpu_uuid="GPU-62ed0907-857c-dba9-643b-513fcdf71d16"

mkdir -p "$log_dir"
echo "$$" > "$log_dir/queue_and_run.pid"
date --iso-8601=seconds
echo "Waiting behind Jake PID $wait_pid on GPU 0 ($gpu_uuid)"

while kill -0 "$wait_pid" 2>/dev/null; do
    current_command="$(tr '\0' ' ' < "/proc/$wait_pid/cmdline" 2>/dev/null || true)"
    if [[ "$current_command" != *"$wait_command_fragment"* ]]; then
        echo "PID $wait_pid no longer matches Jake's expected command; leaving PID wait"
        break
    fi
    date --iso-8601=seconds
    echo "Jake PID $wait_pid is still running"
    sleep 60
done

echo "Jake's process is gone; waiting for GPU 0 to have no compute process"
while nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader,nounits 2>/dev/null | grep -Fqx "$gpu_uuid"; do
    date --iso-8601=seconds
    echo "GPU 0 still has a compute process"
    sleep 60
done

date --iso-8601=seconds
echo "GPU 0 is clear; launching resumable RR100 zero-gaze SF/TF production sweep"
cd "$repo_dir"
exec env \
    MPLCONFIGDIR=/tmp/matplotlib-cache \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    .venv/bin/python -m declan.run_rr100_zero_gaze_separable_sf_tf_native_production \
    --device cuda:0 \
    --batch-size 128 \
    --checkpoint-every 8 \
    --out-dir "$output_dir"
