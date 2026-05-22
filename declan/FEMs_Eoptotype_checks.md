# FEM E-Optotype Diagnostic Pipeline

## Goal

Generate a dataset of retinal movies, digital twin spatial activations, and simple summary statistics across a range of E-optotype sizes to visually and quantitatively inspect:

- Whether subpixel "shimmer" exists in retinal input under FEM
- Whether that shimmer propagates into spatial neural activations
- How this behavior changes across size (resolved → subthreshold)

This is a **diagnostic pipeline**, not a modeling or decoding pipeline.

---

# 1. Stimulus Specification

## Grid
- Retinal sampling grid: **0.25 arcmin per pixel**
- Field of view: choose minimal size that fits the largest E (e.g., ~64×64 or 96×96 pixels)

## Optotype Sizes (10 total)
Define either:
- LogMAR values:

[0.8, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0, -0.1, -0.2]

OR
- Explicit gap sizes in pixels (preferred):

[12, 9, 8, 6.5, 5, 4, 3.2, 2.5, 2.0, 1.6]


Each size defines a full E-optotype image.

---

# 2. Conditions

For each size, generate:

- `stabilized`: no motion
- `fem`: real FEM trajectory
- (optional) `matched_null`: shuffled FEM

---

# 3. Output Directory Structure

Organize by size:


E_diagnostics/
size_00_logmar_0.8/
retina_stabilized.mp4
retina_fem.mp4
retina_matched_null.mp4 (optional)

retina_mean.npy
retina_std.npy
retina_delta_energy.npy

retina_xt_slice.npy

neural_maps_fem.npy
neural_maps_stabilized.npy

neural_mean_map.npy
neural_std_map.npy

com_traj.npy
width_traj.npy

size_01_logmar_0.6/
...


---

# 4. Retinal Movie Generation

For each size and condition:

## Generate movie:

I(x, y, t)


- Apply FEM trajectory (translation in retinal coordinates)
- Stabilized = static image repeated over time

## Save:
- MP4 video (grayscale)
- Raw array (optional `.npy`)

---

# 5. Retinal Statistics

Compute and save:

## 5.1 Mean Image

mean_img = mean(I, axis=time)


## 5.2 Temporal Std Image (SHIMMER MAP)

std_img = std(I, axis=time)


## 5.3 Frame-to-frame Energy

delta_energy[t] = || I[t+1] - I[t] ||_2


Save as:

retina_mean.npy
retina_std.npy
retina_delta_energy.npy


---

# 6. X–T Slice Diagnostic

Take horizontal slice through center:


xt_slice = I[y_center, :, :]


Transpose to shape:

(time, x)


Save as:

retina_xt_slice.npy


---

# 7. Neural Activations (Digital Twin)

For each movie:

## Run digital twin:

rate_maps = model(I)


Expected shape:

(T, N, H, W)


Save:

neural_maps_fem.npy
neural_maps_stabilized.npy


---

# 8. Neural Summary Statistics

For each neuron:

## 8.1 Mean Map

mean_map = mean(rate_maps, axis=time)


## 8.2 Temporal Std Map

std_map = std(rate_maps, axis=time)


---

## 8.3 Center of Mass (CoM)

For each frame:


x_com = sum(x * map) / sum(map)
y_com = sum(y * map) / sum(map)


Store:

com_traj: (T, N, 2)


---

## 8.4 Width / Second Moment

Compute variance:


var_x = sum((x - x_com)^2 * map) / sum(map)
var_y = sum((y - y_com)^2 * map) / sum(map)


Store:

width_traj: (T, N, 2)


---

# 9. Minimal Visualization (Optional but Recommended)

Generate quick PNGs per size:

- retinal_mean.png
- retinal_std.png
- neural_mean_map (avg across neurons)
- neural_std_map (avg across neurons)

---

# 10. Execution Plan

Loop over sizes:


for size in sizes:
generate_E(size)
for condition in [stabilized, fem]:
render_movie()
compute_retina_stats()
run_digital_twin()
compute_neural_stats()
save_all()


---

# 11. Important Constraints

- DO NOT downsample or collapse neural maps before saving
- DO NOT threshold retinal images unless explicitly for a separate diagnostic
- Keep all outputs in float precision (.npy)
- Ensure consistent coordinate system across all outputs

---

# 12. Success Criteria (Qualitative)

Across sizes, expect:

## Large sizes:
- stabilized: clear E
- FEM: moving E

## Near threshold:
- stabilized: blurred E
- FEM: structured flicker

## Subthreshold:
- stabilized: blob
- FEM: structured shimmer
- retina_std should preserve E-like structure

If retina_std becomes featureless → no hyperacuity signal in stimulus

---

# End