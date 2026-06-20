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
