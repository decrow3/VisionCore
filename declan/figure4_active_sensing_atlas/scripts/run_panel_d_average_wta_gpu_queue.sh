#!/usr/bin/env bash
set -euo pipefail

cd /home/declan/VisionCore

export MPLCONFIGDIR=/tmp/matplotlib-cache
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

BASE="outputs/fixation_statistics_by_stimulus_all_sessions_after_review"
MANIFEST="${BASE}/backimage_axis_conditioned_matched_static_percandidate_gpu1_n64_c4_k16_v1/selected_windows.csv"
AVERAGE_INPUT="${BASE}/backimage_image_structure_reviewed_v2_screenfiltered_yfix/backimage_image_fem_windows.csv"
WTA_INPUT="${BASE}/backimage_wta_orientation_axis_input_v1/backimage_image_fem_windows_wta_axis.csv"
FEATURE_NPZ="${BASE}/backimage_axis_conditioned_matched_static_feature_posterior_gabor_pyramid_k4_8_v1/feature_latent_arrays.npz"

AVERAGE_RUN="${BASE}/backimage_axis_conditioned_matched_static_percandidate_gpu1_n64_c4_k16_average_axis_wta_comparison_v1"
WTA_RUN="${BASE}/backimage_axis_conditioned_matched_static_percandidate_gpu1_n64_c4_k16_wta_axis_wta_comparison_v1"
AVERAGE_FEATURE="${BASE}/backimage_axis_conditioned_matched_static_feature_posterior_gabor_pyramid_k4_8_average_axis_wta_comparison_v1"
WTA_FEATURE="${BASE}/backimage_axis_conditioned_matched_static_feature_posterior_gabor_pyramid_k4_8_wta_axis_wta_comparison_v1"

echo "started $(date -Is)"
echo "GPU target: cuda:1"
echo "manifest: ${MANIFEST}"
echo "average input: ${AVERAGE_INPUT}"
echo "WTA input: ${WTA_INPUT}"

for path in "${AVERAGE_RUN}" "${WTA_RUN}" "${AVERAGE_FEATURE}" "${WTA_FEATURE}"; do
  if [[ -e "${path}" ]]; then
    echo "Refusing to overwrite existing output: ${path}" >&2
    exit 2
  fi
done

run_observer() {
  local input_csv="$1"
  local out_dir="$2"
  local axis_col="$3"
  echo "observer start axis=${axis_col} $(date -Is)"
  .venv/bin/python -m declan.fixation_statistics_by_stimulus.run_backimage_trajectory_table_observer \
    --input "${input_csv}" \
    --out-dir "${out_dir}" \
    --window-manifest "${MANIFEST}" \
    --max-images 64 \
    --n-candidates 4 \
    --candidate-set-modes matched_static_response \
    --observation-family empirical \
    --prior-families axis_edge_parallel,axis_edge_orthogonal \
    --observed-rms-scales 0.5 \
    --trajectory-prior-mode leave_one_out \
    --n-prior-trajectories 16 \
    --axis-source-column "${axis_col}" \
    --axis-template-mode same_dominant_projection \
    --axis-match-policy strict \
    --axis-catalog-mode per_candidate \
    --likelihood-scales 1.0 \
    --patch-size-px 540 \
    --n-timepoints 40 \
    --reliable-image-coherence-min 0.20 \
    --reliable-drift-anisotropy-min 0.20 \
    --min-duration-s 0.10 \
    --max-rms-deg 0.12 \
    --twin-batch-size 8 \
    --twin-trace-batch-size 8 \
    --device cuda:1 \
    --seed 23 \
    --progress-every 4
  echo "observer done axis=${axis_col} $(date -Is)"
}

run_feature_posterior() {
  local run_dir="$1"
  local out_dir="$2"
  echo "feature posterior start run=${run_dir} $(date -Is)"
  .venv/bin/python -m declan.backimage_trajectory_observer.analyze_feature_posterior \
    --run-dir "${run_dir}" \
    --out-dir "${out_dir}" \
    --feature-npz "${FEATURE_NPZ}" \
    --latent-names gabor_local_field,pyramid_local_field \
    --pca-k-list 4,8 \
    --likelihood-scales 1.0 \
    --posterior-temperature 1.0 \
    --candidate-set-modes matched_static_response \
    --priors axis_edge_parallel,axis_edge_orthogonal \
    --motion-scales 0.5 \
    --patch-size-px 540 \
    --latent-crop-px 151 \
    --center-crop-px 41 \
    --local-field-grid 8 \
    --progress-every 32 \
    --n-bootstrap 10000 \
    --n-permutations 10000 \
    --uncertainty-confidence 0.95 \
    --uncertainty-seed 17
  echo "feature posterior done run=${run_dir} $(date -Is)"
}

run_observer "${AVERAGE_INPUT}" "${AVERAGE_RUN}" "image_edge_axis_deg"
run_feature_posterior "${AVERAGE_RUN}" "${AVERAGE_FEATURE}"

run_observer "${WTA_INPUT}" "${WTA_RUN}" "wta_edge_axis_deg"
run_feature_posterior "${WTA_RUN}" "${WTA_FEATURE}"

echo "finished $(date -Is)"
