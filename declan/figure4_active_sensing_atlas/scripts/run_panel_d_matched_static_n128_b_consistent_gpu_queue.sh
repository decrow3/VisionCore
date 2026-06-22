#!/usr/bin/env bash
set -euo pipefail

cd /home/declan/VisionCore

export MPLCONFIGDIR=/tmp/matplotlib-cache
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

BASE="outputs/fixation_statistics_by_stimulus_all_sessions_after_review"
INPUT="${BASE}/backimage_image_structure_reviewed_v2_screenfiltered_yfix/backimage_image_fem_windows.csv"

DEVICE="${DEVICE:-cuda:1}"
SCALES="${SCALES:-0.5,1.0,2.0}"
MAX_IMAGES="${MAX_IMAGES:-128}"
N_PRIOR_TRAJECTORIES="${N_PRIOR_TRAJECTORIES:-16}"

OBSERVER_RUN="${BASE}/backimage_axis_conditioned_matched_static_percandidate_gpu1_n128_c4_k16_scales_0p5_1_2_bconsistent_v1"
FEATURE_RUN="${BASE}/backimage_axis_conditioned_matched_static_feature_posterior_pyramid_k8_16_n128_scales_0p5_1_2_bconsistent_v1"

echo "started $(date -Is)"
echo "device: ${DEVICE}"
echo "input: ${INPUT}"
echo "scales: ${SCALES}"
echo "observer run: ${OBSERVER_RUN}"
echo "feature run: ${FEATURE_RUN}"

for path in "${OBSERVER_RUN}" "${FEATURE_RUN}"; do
  if [[ -e "${path}" ]]; then
    echo "Refusing to overwrite existing output: ${path}" >&2
    exit 2
  fi
done

echo "observer start $(date -Is)"
.venv/bin/python -m declan.fixation_statistics_by_stimulus.run_backimage_trajectory_table_observer \
  --input "${INPUT}" \
  --out-dir "${OBSERVER_RUN}" \
  --max-images "${MAX_IMAGES}" \
  --n-candidates 4 \
  --candidate-set-modes matched_static_response \
  --observation-family empirical \
  --prior-families axis_edge_parallel,axis_edge_orthogonal \
  --observed-rms-scales "${SCALES}" \
  --trajectory-prior-mode leave_one_out \
  --n-prior-trajectories "${N_PRIOR_TRAJECTORIES}" \
  --axis-source-column image_edge_axis_deg \
  --axis-template-mode same_dominant_projection \
  --axis-match-policy strict \
  --axis-catalog-mode per_candidate \
  --likelihood-scales 1.0 \
  --patch-size-px 540 \
  --min-patch-image-margin-px 270 \
  --n-timepoints 40 \
  --reliable-image-coherence-min 0.20 \
  --reliable-drift-anisotropy-min 0.20 \
  --min-duration-s 0.10 \
  --max-rms-deg 0.12 \
  --max-trace-source-rms-deg 0.06 \
  --max-trace-source-radius-deg 0.20 \
  --max-trace-source-speed-p95-deg-s 20.0 \
  --max-trace-source-microsaccade-events 0 \
  --max-rendered-trace-path-length-deg 1.5 \
  --twin-batch-size 8 \
  --twin-trace-batch-size 8 \
  --device "${DEVICE}" \
  --seed 23 \
  --progress-every 4
echo "observer done $(date -Is)"

echo "feature posterior start $(date -Is)"
.venv/bin/python -m declan.backimage_trajectory_observer.analyze_feature_posterior \
  --run-dir "${OBSERVER_RUN}" \
  --out-dir "${FEATURE_RUN}" \
  --latent-names pyramid_local_field \
  --pca-k-list 8,16 \
  --likelihood-scales 1.0 \
  --posterior-temperature 1.0 \
  --candidate-set-modes matched_static_response \
  --priors axis_edge_parallel,axis_edge_orthogonal \
  --motion-scales "${SCALES}" \
  --patch-size-px 540 \
  --latent-crop-px 151 \
  --center-crop-px 41 \
  --local-field-grid 8 \
  --progress-every 16 \
  --n-bootstrap 10000 \
  --n-permutations 10000 \
  --uncertainty-confidence 0.95 \
  --uncertainty-seed 17
echo "feature posterior done $(date -Is)"

echo "finished $(date -Is)"
