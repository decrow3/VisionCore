# Figure 5 control spec: random_amp_cloud_matched

## Purpose

The current `random_amp` control can exceed real FEMs in bits per expected spike. This is informative, but not yet interpretable because it may preserve step amplitude/path length while changing the retinal occupancy cloud or the local image structure sampled by the trajectory.

`random_amp_cloud_matched` should test whether the random-motion advantage survives a stricter match to the real trace's spatial occupancy.

## Control Definition

For each real trace, generate random trajectories that:

1. preserve or closely match the empirical step-amplitude distribution;
2. match total path length within tolerance;
3. match mean retinal position;
4. match RMS displacement from the trace mean;
5. match the 2D position covariance matrix or its eigenvalues within tolerance;
6. use the same movie duration and valid-frame mask;
7. are paired to the same image and trace seed.

## Suggested Algorithm

For each real trajectory:

1. Compute real summary statistics:

```text
mean_x, mean_y
rms_radius
cov_xx, cov_xy, cov_yy
path_length
step_amplitude_distribution
n_valid_frames
```

2. Generate candidate random paths:

- draw step amplitudes by resampling real step amplitudes;
- draw step directions uniformly or from a fitted direction distribution;
- integrate steps into a path;
- recenter to real mean position;
- optionally affine-transform the candidate cloud to match the real covariance;
- clip/reject paths that leave the renderable image region.

3. Accept if tolerances pass:

```text
abs(path_length_candidate / path_length_real - 1) < tol_path
abs(rms_radius_candidate / rms_radius_real - 1) < tol_rms
eig_cov_candidate close to eig_cov_real
mean_position_error < tol_mean
valid_frame_count == real_valid_frame_count
```

4. If no path passes after `max_attempts`, write a failure row and do not silently fall back to a looser control.

## Validation Table

Write one row per real/control pair:

```text
image_id
trace_id
kind
control_seed
condition = random_amp_cloud_matched
n_valid_frames_real
n_valid_frames_control
path_length_real
path_length_control
rms_radius_real
rms_radius_control
cov_xx_real
cov_xx_control
cov_xy_real
cov_xy_control
cov_yy_real
cov_yy_control
step_amp_mean_real
step_amp_mean_control
step_amp_p95_real
step_amp_p95_control
local_gradient_mean_real
local_gradient_mean_control
local_highpass_energy_mean_real
local_highpass_energy_mean_control
accepted
reject_reason
```

## Figure 5 Use

Add to Panel D:

```text
stabilized
real
random_amp
random_amp_cloud_matched
random_cov
trajectory_order_shuffle
```

Interpretation:

- If `random_amp_cloud_matched` remains above real, real trajectory optimality is not supported.
- If `random_amp` is above real but `random_amp_cloud_matched` is near real or below real, the original `random_amp` advantage likely reflected occupancy or sampled-image-structure mismatch.
- Either outcome is useful; it tells us what the matched-motion controls actually mean.

