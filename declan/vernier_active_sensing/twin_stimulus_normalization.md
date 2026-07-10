# Digital Twin Stimulus Normalization

This note records the model-facing intensity convention for stimuli passed into
the marmoset V1 digital twin. It applies to Vernier, E-optotype, RSVP,
natural-image, and any future synthetic stimulus path.

The central repository note is `docs/digital_twin_stimulus_normalization.md`.
If these notes ever diverge, the central note and the dataset `pixelnorm`
definition in `models/data/transforms.py` are authoritative.

## Required Model Input Convention

The trained/evaluated twin expects the dataset `pixelnorm` transform:

```text
x_model = (x_raw - 127.0) / 255.0
```

where `x_raw` is in the same raw 8-bit image units used by the dataset. This is
defined in `models/data/transforms.py`:

```python
def pixelnorm(x):
    return (x.float() - 127) / 255
```

Useful anchor values:

```text
raw 0.0    -> -0.498
raw 63.5   -> -0.249
raw 127.0  ->  0.000
raw 255.0  ->  0.502
```

So a gray background should enter the twin near `0.0`, not `0.5`.

## What Not To Do

Do not feed model-bound stimuli as display-normalized `[0, 1]` images. In
particular, do not use:

```text
x_model = x_raw / 127.0
x_model = x_raw / 255.0
```

unless the output is only for visualization or an explicitly documented
non-model diagnostic. A `[0, 1]` tensor makes a gray background look like a
bright stimulus relative to the twin's training convention.

## Renderer-Specific Conventions

Some high-resolution stimulus renderers historically store retinal movies in a
local renderer range or return a display-normalized tensor. Before calling the
twin, map those local values into the dataset raw 8-bit convention and then
apply pixelnorm.

If a renderer returns dataset raw 8-bit values:

```python
stim_model = (stim_raw_u8 - 127.0) / 255.0
```

The Vernier renderer's local audited range is `0..max_raw` with neutral near
`max_raw / 2`, so model-bound Vernier tensors use:

```python
stim_raw_u8 = stim_renderer_raw * (255.0 / renderer_max_raw)
stim_model = (stim_raw_u8 - 127.0) / 255.0
```

If a renderer returns `stim_display = stim_raw_u8 / 127.0`:

```python
stim_raw_u8 = stim_display * 127.0
stim_model = (stim_raw_u8 - 127.0) / 255.0
```

This is the convention already used in
`scripts/fem_eoptotype_diagnostics.py::to_model_stim`.

## Vernier Audit Targets

The Vernier active-sensing scripts should use the same model input convention.
Any path that calls the digital twin should be checked for `/ 127.0`,
`/ geom.max_raw`, or `/ retina.geometry.max_raw` before inference. These are
display normalizations unless followed by the pixelnorm conversion above.

Vernier paths currently intended to use the shared normalization helper:

- `declan/vernier_active_sensing/forward.py::build_vernier_movie`
- `declan/vernier_active_sensing/forward.py::compute_vernier_rates_continuous`
- `declan/vernier_active_sensing/run_lag_geometry_diagnostic.py`
- `declan/vernier_active_sensing/run_end_to_end_pose_profile.py`

Downstream scripts that call `build_vernier_movie` inherit its model input
normalization. Once that function is corrected and caches are invalidated, rerun
the RR100 and full756 SSI/polarity diagnostics before interpreting suppression
rates.

## Reporting Rule

When reporting any Vernier twin result, record the stimulus normalization in the
output metadata. Preferred text:

```text
stimulus_normalization = pixelnorm_renderer_raw_scaled_to_u8_minus_127_div_255
```

If a result was produced with legacy display normalization, mark it explicitly:

```text
stimulus_normalization = legacy_display_div_max_raw_do_not_interpret
```
