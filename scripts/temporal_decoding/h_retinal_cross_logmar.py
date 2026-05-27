"""
H: Cross-LogMAR retinal input audit at model-native PPD.
Generates stimuli at -0.35/-0.40/-0.45/-0.50/-0.55 and computes pairwise
pixelwise differences to check whether the model-native retinal tensors
are distinguishable in the plateau region.

Run from VisionCore root:
  uv run python scripts/temporal_decoding/h_retinal_cross_logmar.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from stimulus_hires import hires_counterfactual_stim
from rate_computation import OUT_SIZE, N_LAGS

lms = [-0.35, -0.40, -0.45, -0.50, -0.55]
conditions = ["real", "stabilized"]
T_max = 60  # first 60 frames per trace

eye_data = np.load("scripts/temporal_decoding/data/eye_traces.npz", allow_pickle=True)
traces = eye_data['traces']  # (N, T, 2)
eyepos = traces[0][:T_max + N_LAGS]  # (T, 2)

print("H: Model-native retinal input cross-LogMAR audit")
print(f"  trace shape: {eyepos.shape}, T_max={T_max}")
print(f"  OUT_SIZE={OUT_SIZE}, N_LAGS={N_LAGS}")
print()

movies = {}
for cond in conditions:
    for lm in lms:
        print(f"  generating lm={lm:+.2f} {cond}...", flush=True)
        stim = hires_counterfactual_stim(
            0, lm, eyepos,
            condition=cond,
            n_lags=N_LAGS,
            retina_size=OUT_SIZE,
            device='cpu',
        )
        # stim: (T_valid, 1, n_lags, H, W) — take lag=0 frame
        movies[(cond, lm)] = stim[:, 0, 0].numpy()  # (T_valid, H, W)

print()
print("Cross-LogMAR pairwise differences (adjacent LogMARs):")
print(f"  {'Pair':<35} {'max|diff|':>10} {'mean|diff|':>12} {'norm':>10} {'r':>8}")
print("  " + "-" * 80)

for cond in conditions:
    for i in range(len(lms) - 1):
        lm1, lm2 = lms[i], lms[i+1]
        m1 = movies[(cond, lm1)].ravel()
        m2 = movies[(cond, lm2)].ravel()
        T = min(len(m1), len(m2))
        m1, m2 = m1[:T], m2[:T]
        diff = m1 - m2
        maxd = float(np.max(np.abs(diff)))
        meand = float(np.mean(np.abs(diff)))
        normd = float(np.linalg.norm(diff))
        r = float(np.corrcoef(m1, m2)[0, 1])
        label = f"lm{lm1:+.2f}→lm{lm2:+.2f} {cond}"
        print(f"  {label:<35} {maxd:>10.6f} {meand:>12.6f} {normd:>10.4f} {r:>8.6f}")

print()
print("Real vs Stabilized differences per LogMAR:")
print(f"  {'LogMAR':<10} {'max|diff|':>10} {'mean|diff|':>12} {'norm':>10}")
print("  " + "-" * 50)
for lm in lms:
    mr = movies[("real", lm)].ravel()
    ms = movies[("stabilized", lm)].ravel()
    T = min(len(mr), len(ms))
    mr, ms = mr[:T], ms[:T]
    diff = mr - ms
    maxd = float(np.max(np.abs(diff)))
    meand = float(np.mean(np.abs(diff)))
    normd = float(np.linalg.norm(diff))
    print(f"  {lm:>+.2f}         {maxd:>10.6f} {meand:>12.6f} {normd:>10.4f}")

print("\nDone.")
