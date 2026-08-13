# Panel G exact-pair production runbook

## Cohort and estimand

The production cohort reuses the 1,000 real fixation traces from the earlier
Figure 4 movie bank. Each trace is restored to its own reviewed BackImage
window rather than crossed with an unrelated image. Each native pair receives
nine fresh RR100 evaluations: the recorded trajectory and eight deterministic
full-circle midpoint rotations.

The primary pair-level estimand is direct real SSI minus the mean of the eight
directly evaluated rotations. Dose-curve interpolation and a stabilized
baseline are both excluded.

## Preflight

```bash
.venv/bin/python -u -m \
  declan.fig.ssi_figure_v2.behavior_model_bridge.run_panel_g_exact_pair_production \
  --dry-run
```

The committed preflight target is 1,000 exact pairs and 9,000 fresh movies.

## Recommended four-shard production run

Run each command in its own persistent shell. Use `cuda:0` sequentially while
GPU 1 is occupied; if GPU 1 becomes free, disjoint shards can run concurrently.

```bash
.venv/bin/python -u -m declan.fig.ssi_figure_v2.behavior_model_bridge.run_panel_g_exact_pair_production --pair-start 0 --pair-stop 250 --device cuda:0 --frame-batch-size 16 --trace-batch-size 8
.venv/bin/python -u -m declan.fig.ssi_figure_v2.behavior_model_bridge.run_panel_g_exact_pair_production --pair-start 250 --pair-stop 500 --device cuda:0 --frame-batch-size 16 --trace-batch-size 8
.venv/bin/python -u -m declan.fig.ssi_figure_v2.behavior_model_bridge.run_panel_g_exact_pair_production --pair-start 500 --pair-stop 750 --device cuda:0 --frame-batch-size 16 --trace-batch-size 8
.venv/bin/python -u -m declan.fig.ssi_figure_v2.behavior_model_bridge.run_panel_g_exact_pair_production --pair-start 750 --pair-stop 1000 --device cuda:0 --frame-batch-size 16 --trace-batch-size 8
```

The four-pair timing smoke took 1.2 minutes after model initialization. The
current operational estimate is approximately 75 minutes per 250-pair shard,
or five GPU-hours total on one GPU. Per-pair caches are identity checked and
reused on restart.

## Merge after all shards finish

```bash
.venv/bin/python -u -m \
  declan.fig.ssi_figure_v2.behavior_model_bridge.merge_panel_g_exact_pair_production
```

The merge refuses incomplete or overlapping shards by default. It creates a
raw merged table and array store but deliberately performs no population
inference; that remains the next map-first checkpoint.
