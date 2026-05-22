# VisionCore Temporal Analysis: Current Approach, Issues, and Alternatives

## 1. Current Analysis Pipeline

### Data Flow
- **Input:** Raw movie (T frames) + eye traces
- **embed_time_lags:** Converts to windowed stimulus (T_out, 1, n_lags, H, W), where each sample is a 32-frame window
- **Model:**
  - **Frontend → ConvNet (3D ResNet) → ConvGRU**
  - **Core output:** (B, C_hidden, T_gru, H_feat, W_feat)
  - **Readout:** Only the last GRU timestep x[:, :, -1] is used
  - **Prediction:** (B, N_neurons)

### Key Points
- **GRU hidden state is reset for each window** (no memory across windows)
- **Each window is processed independently**
- **Only the last GRU output is read out**
- **Training matches this: model is optimized to predict a single spike rate from a 25-frame (or 32-frame) window**

### What the Model Can and Cannot Do
- **Can:** Integrate temporal information within a 32-frame window
- **Cannot:** Maintain memory or dynamics across windows (no continuous temporal state)
- **All temporal analyses are limited to the windowed context**

## 2. Issues Identified

### Double Counting Problem
- If you try to chain GRU hidden states across overlapping windows, the same frames are included both in the input and in the hidden state, leading to double counting of history.
- The frontend and convnet are also temporal, so their outputs are not streamable frame-by-frame.

### Interpretation Limitations
- The model is a **windowed spatial encoder** with finite temporal context, not a continuous-time dynamical system.
- Analyses that assume continuous temporal memory (e.g., velocity tracking, transformation channel) are not valid with this architecture.
- Null results for long-timescale temporal coding may reflect architectural limits, not biological absence.

### Diagnostic Limitations
- Test 1 (dynamic vs static window) shows history sensitivity, but does not isolate the GRU's contribution vs. the convnet/frontend.
- Feature RSA (pre-GRU vs post-GRU) can help, but only within the window.

## 3. Alternative Solutions

### A. Non-Overlapping Windows
- Use stride = window length (e.g., 32), so each window is independent and no double counting occurs.
- Downside: only get predictions every 32 frames.

### B. Chained Hidden State with Careful Input
- Pass the final GRU hidden state from one window as the initial state for the next, but:
  - Only valid if windows are non-overlapping, or
  - If overlapping, ensure new input frames do not duplicate history already in hidden state (requires redesign of input pipeline).
- Not currently supported due to frontend/convnet temporal structure.

### C. Streaming Model Redesign
- Redesign frontend and convnet to be streamable (process one frame at a time, maintain their own state).
- Allows true continuous-time recurrence and memory.
- Major architectural change; requires retraining.

### D. Diagnostic Stress Test
- As a diagnostic, try passing hidden state across windows and neutralizing history frames in the input (e.g., zeros or repeats), to see if the GRU can maintain memory.
- Out-of-distribution for the trained model, but can reveal if the GRU weights support longer-range integration.

### E. Retrain with Chained State
- Retrain the model with hidden state carried across windows, so it learns to use continuous temporal memory.
- Requires changes to data loader and training loop.

## 4. Recommendations
- **Be explicit in all analyses and writing about the model’s windowed nature and its implications.**
- **Do not claim results about continuous temporal coding or velocity tracking unless the model is modified to support it.**
- **If continuous memory is needed, consider redesigning the model and retraining.**
- **Use feature-level diagnostics (e.g., pre/post-GRU RSA) to localize where temporal integration occurs.**
- **For current architecture, focus claims on spatial coding and short-timescale temporal context.**

### Practical note: rerunning temporal-decoding analyses

The E-optotype temporal-decoding pipeline lives under `scripts/temporal_decoding/`. For integration-time analyses, be explicit about the window featurization method:
- `time_mean`: mean over the last W frames (accumulation-aligned)
- `flat_pca`: flatten last W frames + PCA (legacy; can wash out signal in hyperacuity)

If external data-package imports block the full simulation stack, Phase 2 can be rerun from cached rate `.npz` files using `scripts/temporal_decoding/run_analysis.py --decode_only`.

---

