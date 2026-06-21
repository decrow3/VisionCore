# Canonical Active Sensing

This package contains stable, config-driven entry points for the BackImage
active-sensing analyses. The wrappers call the existing analysis scripts; they
do not reimplement the science code.

Current candidate feature spec:

```text
Primary aggregate readout: pyramid_local_field k16 temporal_pca
Local mechanistic sensitivity: pyramid_local_field k16 delta_mean
```

Use `--print-command` before launching heavy jobs. By default, executable
wrappers refuse to write into an existing non-empty `out_dir`; pass
`--allow-existing-output` only for an intentional refresh of a known cache.

Examples:

```bash
.venv/bin/python -m declan.canonical_active_sensing.validate_configs
.venv/bin/python -m declan.canonical_active_sensing.validate_configs --check-output-freshness
.venv/bin/python -m declan.canonical_active_sensing.run_aggregate_fem --print-command
.venv/bin/python -m declan.canonical_active_sensing.run_incremental_posthoc --print-command
.venv/bin/python -m declan.canonical_active_sensing.run_local_pairing --print-command
.venv/bin/python -m declan.canonical_active_sensing.run_local_pairing --section local_pairing_sentinel --print-command
.venv/bin/python -m declan.canonical_active_sensing.run_incremental_posthoc --config declan/canonical_active_sensing/configs/local_pairing_k16_v1.json --section local_incremental_primary --print-command
.venv/bin/python -m declan.canonical_active_sensing.run_incremental_posthoc --config declan/canonical_active_sensing/configs/local_pairing_k16_v1.json --section local_incremental_sentinel --print-command
.venv/bin/python -m declan.canonical_active_sensing.run_joint_observer --print-command
.venv/bin/python -m declan.canonical_active_sensing.analyze_joint_posterior --print-command
.venv/bin/python -m declan.canonical_active_sensing.adjudicate_feature_spec --print-command
.venv/bin/python -m declan.canonical_active_sensing.make_active_sensing_figure_pack --print-command
```

Production order:

1. Complete `run_joint_observer` only if the rel0.25 observer output is not
   already live or complete.
2. Run `analyze_joint_posterior`, then `adjudicate_feature_spec`, to finish the
   model-selection evidence cache.
3. Run the aggregate canonical forward pass and aggregate incremental posthoc.
4. Run local primary and sentinel forward passes, then both local incremental
   posthocs.
5. Build the active-sensing figure pack from the completed aggregate canonical
   run and its incremental posthoc.

Production configs live in `configs/`. The current v4 closure run must land
before the canonical aggregate run is treated as final.

Current output provenance lives in `provenance/current_outputs.md`.

## Figure 4 Power Rerun

`configs/figure4_power_rerun_v1.json` is the higher-power rerun surface for the
current Figure 4 model analyses. It intentionally does not reopen feature
selection: it locks `pyramid_local_field k16` and carries the two reviewed
readouts, `temporal_pca` for aggregate/ensemble utility and `delta_mean` for
local mechanistic sensitivity.

Recommended launch order:

1. `aggregate_power_primary`: n384, K8 aggregate rerun; this is the main
   candidate for Figure 4 panel B.
2. `aggregate_incremental_power_primary`: cache-only posthoc for the aggregate
   run.
3. `local_pairing_power_seed7` and `local_pairing_power_seed11`: n128 local
   pairing reruns with K64 matched-unpaired controls.
4. `local_incremental_power_seed7` and `local_incremental_power_seed11`.
5. `aggregate_power_replicate_seed11` and its incremental posthoc only if we
   need a second aggregate seed after the primary run.
6. `joint_observer_rel0p25_power_prior32` and
   `joint_posterior_rel0p25_power_prior32` only if the existing rel0.25
   completion still leaves the joint-axis result underpowered.

Examples:

```bash
.venv/bin/python -m declan.canonical_active_sensing.run_aggregate_fem --config declan/canonical_active_sensing/configs/figure4_power_rerun_v1.json --section aggregate_power_primary --print-command
.venv/bin/python -m declan.canonical_active_sensing.run_incremental_posthoc --config declan/canonical_active_sensing/configs/figure4_power_rerun_v1.json --section aggregate_incremental_power_primary --print-command
.venv/bin/python -m declan.canonical_active_sensing.run_local_pairing --config declan/canonical_active_sensing/configs/figure4_power_rerun_v1.json --section local_pairing_power_seed7 --print-command
.venv/bin/python -m declan.canonical_active_sensing.run_joint_observer --config declan/canonical_active_sensing/configs/figure4_power_rerun_v1.json --section joint_observer_rel0p25_power_prior32 --print-command
```

## Cache-Bank Optimization Notes

The lower-level cache-bank path is:

```bash
.venv/bin/python -m declan.fixation_statistics_by_stimulus.run_backimage_aggregate_trace_catalog ...
.venv/bin/python -m declan.fixation_statistics_by_stimulus.run_backimage_response_cache_bank ...
.venv/bin/python -m declan.fixation_statistics_by_stimulus.assemble_backimage_response_cache_bank ...
```

`run_backimage_response_cache_bank` now reuses identical trace contents within
each image and can validate trace batching with
`--check-trace-batch-equivalence`. For production, do a one-shard smoke with
that flag before raising `--twin-trace-batch-size` above the conservative value
in older handoffs. Larger trace batches should be scientifically identical when
the equivalence check passes, but may need to be backed down if GPU memory is
tight.

Important boundary: the cache-bank generator can write `delta_mean`, `mean`, and
DCT summaries directly. `temporal_pca` needs a prefit temporal basis supplied via
`--temporal-basis-npz`, so the cache-bank route is not yet a drop-in replacement
for the monolithic aggregate temporal-PCA run until the basis-generation step is
made canonical.
