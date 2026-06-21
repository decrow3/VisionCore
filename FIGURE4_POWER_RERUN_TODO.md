# Figure 4 Power Rerun Todo

Date: 2026-06-20

Goal: rerun the Figure 4 active-sensing analyses with the reviewed feature target,
more statistical power, and enough seed/control coverage to support a production
figure decision.

Locked model target:

- Aggregate/ensemble readout: `pyramid_local_field`, `k=16`, `temporal_pca`
- Local mechanistic readout: `pyramid_local_field`, `k=16`, `delta_mean`

Primary config:

- `declan/canonical_active_sensing/configs/figure4_power_rerun_v1.json`

## Current GPU State

- GPU0: free at planning time; use for `aggregate_power_primary`.
- GPU1: occupied by existing rel0.25 joint observer at planning time; expected to
  free soon. Use it for local pairing seeds once available.

## Launch Status

- 2026-06-20 16:37 PDT: started `aggregate_power_primary` on GPU0.
  - Host PID: `2556842`
  - Log:
    `outputs/fixation_statistics_by_stimulus_all_sessions_after_review/background_logs/figure4_power_aggregate_primary_seed0_gpu0.log`
  - PID file:
    `outputs/fixation_statistics_by_stimulus_all_sessions_after_review/background_logs/figure4_power_aggregate_primary_seed0_gpu0.pid`
  - Startup check: model loaded and GPU0 reached full utilization.
- 2026-06-20 16:45 PDT: queued local GPU1 seed chain.
  - Queue process PID: `2557972`
  - Queue log:
    `outputs/fixation_statistics_by_stimulus_all_sessions_after_review/background_logs/figure4_power_local_gpu1_queue.log`
  - Queue script:
    `outputs/fixation_statistics_by_stimulus_all_sessions_after_review/background_logs/figure4_power_local_gpu1_queue.sh`
  - Behavior: waits for the existing rel0.25 observer PID `2507704`,
    waits for physical GPU1 memory to clear, then runs
    `local_pairing_power_seed7` followed by `local_pairing_power_seed11`.
  - GPU mapping: uses `CUDA_VISIBLE_DEVICES=1`, so the config's `cuda:0`
    means physical GPU1 inside the local-pairing process.

## Wall-Clock Budget

Assuming one clean GPU per heavy job:

- Core pipeline, sequential: about 32-38 GPU-hours.
- Core pipeline, two GPUs kept busy: about 18-22 hours wall-clock.
- Full optional pipeline, sequential: about 50-60 GPU-hours.
- Full optional pipeline, two GPUs kept busy: about 28-36 hours wall-clock.

Largest cost centers:

- `aggregate_power_primary`: about 12-18 hours.
- Each local K64 seed: about 8-9 hours.
- Optional aggregate seed11 replicate: about 8-12 hours.
- Optional joint rel0.25 prior32 observer: about 9-10 hours.

## Launch Plan

1. Started `aggregate_power_primary` on GPU0.
   - Output:
     `outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_aggregate_fem_information_n384_pyramid_k16_tworeadout_rel025-2_power_seed0_k8_v1`
   - Expected wall time: 12-18 hours.
   - Priority: required.

2. When GPU1 frees, start `local_pairing_power_seed7`.
   - Output:
     `outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_local_pairing_Iz_power_pyramid_k16_rel025_0p5_1_seed7_k64_v1`
   - Expected wall time: 8-9 hours.
   - Priority: required.

3. Start `local_pairing_power_seed11` on the next free GPU.
   - Output:
     `outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_local_pairing_Iz_power_pyramid_k16_rel025_0p5_1_seed11_k64_v1`
   - Expected wall time: 8-9 hours.
   - Priority: required seed replicate.

4. Run `aggregate_incremental_power_primary` after the aggregate run completes.
   - Expected wall time: minutes to less than 30 minutes.
   - Priority: required.

5. Run `local_incremental_power_seed7` and `local_incremental_power_seed11`
   after their corresponding local runs complete.
   - Expected wall time: minutes to less than 30 minutes each.
   - Priority: required.

6. Decide whether to run `aggregate_power_replicate_seed11`.
   - Run if the primary aggregate result is promising but needs a seed
     stability check before figure lock.
   - Skip if primary aggregate is already stable enough and local seed
     replication is decisive.

7. Decide whether to run `joint_observer_rel0p25_power_prior32` and
   `joint_posterior_rel0p25_power_prior32`.
   - Run only if the existing rel0.25 joint completion leaves the joint-axis
     result underpowered or ambiguous.
   - Skip if joint remains a supporting sensitivity analysis rather than a core
     claim-critical result.

8. Re-run feature adjudication and figure-pack generation after required runs
   land.
   - Confirm whether the two-readout candidate becomes the production Figure 4
     contract.
   - Update `declan/ANALYSIS_NARRATIVE.md`, `declan/MANIFEST.md`, and
     `declan/fem_v1_maximal_story_priority_checklist.md` with final provenance.

## Monitoring Commands

```bash
nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits
ps -o pid,etime,stat,pcpu,pmem,args -p <pid>
tail -n 40 <background-log>
```

## Completion Criteria

- Required heavy runs complete without partial-output reuse or stale-output
  ambiguity.
- Incremental posthocs complete for aggregate and both local seeds.
- Aggregate `temporal_pca` and local `delta_mean` conclusions are checked
  against controls, scales, and seed stability.
- Optional runs are either completed or explicitly marked unnecessary with a
  reason in the final figure/provenance notes.
- Figure-pack panels and source tables point to the final audited run folders.
