# Digital Twin Stimulus Normalization

All stimuli passed into the digital twin must use the same normalization as the
training and evaluation datasets:

```python
stim_model = (stim_raw_u8.float() - 127.0) / 255.0
```

This is the repo's `pixelnorm` transform in `models/data/transforms.py`. It is a
model-input contract, not a display convention. A valid model-bound stimulus is
therefore approximately in `[-0.5, +0.5]`, with the intended neutral gray or
background near `0.0`.

| Intended raw value | Model value |
| --- | --- |
| `0` | `-0.498` |
| `127` | `0.000` |
| `255` | `0.502` |

Do not pass renderer display values, `[0, 1]` images, or `raw / max_raw`
tensors directly to the twin. In particular, a neutral background entering the
model near `0.5` is already in the model's high-luminance regime and is not a
valid gray baseline.

## Renderer-Specific Conversion

Some high-resolution renderers use local display/provenance units before model
inference. That is fine for pixel audits and PNGs, but model-bound tensors must
be converted to raw-image units first and then pixel-normalized.

If the renderer returns dataset raw 8-bit values:

```python
stim_model = (stim_raw_u8 - 127.0) / 255.0
```

If a renderer uses a local audited range that is not 8-bit, first map that
local range onto `0..255`. The Vernier renderer is one of these paths: its
`RenderGeometry` local range is `0..max_raw` with neutral near `max_raw / 2`.
The model-bound conversion is therefore:

```python
stim_raw_u8 = stim_renderer_raw * (255.0 / renderer_max_raw)
stim_model = (stim_raw_u8 - 127.0) / 255.0
```

This maps the current Vernier neutral background (`63.5 / 127`) to approximately
zero in model units, while preserving the dataset's raw-8-bit pixelnorm range.

If the renderer returns a display tensor produced from dataset raw values as
`stim_display = stim_raw_u8 / 127.0`, reconstruct raw values first:

```python
stim_raw_u8 = stim_display * 127.0
stim_model = (stim_raw_u8 - 127.0) / 255.0
```

This conversion is already used in
`scripts/fem_eoptotype_diagnostics.py::to_model_stim`.

Renderers may use their own local units for pixel audits, movies, or PNGs. Before
calling `compute_trial_rates`, `model.model.core_forward`, the recurrent
frontend, or any readout path, first map those renderer units into the intended
raw 8-bit convention and then apply `pixelnorm`. For synthetic stimuli, record
both mappings explicitly:

- renderer/provenance units, such as local raw luminance or display `[0, 1]`;
- model units, always the pixelnorm result `(raw_u8 - 127.0) / 255.0`.

When auditing old analyses, search for `/ 127.0`, `/ max_raw`, `/ geom.max_raw`,
or `/ retina.geometry.max_raw` immediately upstream of model calls. Those are
display normalizations unless the surrounding code reconstructs `raw_u8` and
then applies `pixelnorm`.

## Vernier Audit Targets

The Vernier active-sensing path should follow the same model-input contract as
natural images and E-optotypes. Before interpreting or rerunning Vernier twin
outputs, audit any path that creates a tensor and immediately calls
`compute_trial_rates`, `model.model.core_forward`, the recurrent frontend, or a
readout. Current high-priority call sites include:

- `declan/vernier_active_sensing/forward.py::build_vernier_movie`
- `declan/vernier_active_sensing/forward.py::compute_vernier_rates_continuous`
- `declan/vernier_active_sensing/run_lag_geometry_diagnostic.py`
- `declan/vernier_active_sensing/run_end_to_end_pose_profile.py`

Downstream Vernier scripts that call `build_vernier_movie` inherit that
function's normalization. Once the model-bound normalization is corrected, stale
Vernier caches should be invalidated and the RR100/full756 SSI, polarity, and
activation-map diagnostics rerun.

## Output Metadata

Every new twin-facing analysis should write a normalization field into its
manifest or summary metadata.

Preferred value for Vernier model inputs:

```text
stimulus_normalization = pixelnorm_renderer_raw_scaled_to_u8_minus_127_div_255
```

Preferred value for stimuli that are already dataset raw 8-bit:

```text
stimulus_normalization = pixelnorm_raw_u8_minus_127_div_255
```

Use an explicit warning value for legacy or audit-only results:

```text
stimulus_normalization = legacy_display_div_max_raw_do_not_interpret
```
