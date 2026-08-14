# VisionCore
multidataset training for digital twin models of visual cortex

For coding-agent navigation, start with [`AGENT_CONTEXT.md`](AGENT_CONTEXT.md).

## BackImage analysis provenance notice

The legacy reconstructed BackImage/Figure-4 retinal-movie caches used an incorrect sampling and crop-geometry contract and are superseded. Do not use their downstream power, SSI, spectral-overlap, population-ordering, or gain results as current evidence without rerunning them from the corrected movie inputs. The controlled fixed-retina grating SF and SF×TF tuning products are not affected.

Before continuing or reusing an eye-movement analysis, read the [impact assessment and recovery notes](outputs/fig4_active_sensing/rr100_eye_trace_conditioning_nyquist_audit_checkpoint_19_v1/IMPACT_NOTES.md). The associated [audit summary](outputs/fig4_active_sensing/rr100_eye_trace_conditioning_nyquist_audit_checkpoint_19_v1/audit_summary.json) records the validated 240-to-120-Hz visual sampling contract, corrected RF crop and retinal-motion sign, and revised 32–60 Hz power estimates.

The corrected outcome-independent 100-image × 1,000-trace production cohort
and exact frozen inputs are now prepared under
`outputs/fig4_active_sensing/rr100_corrected100x1000_production_cohort_v1/` and
`outputs/fig4_active_sensing/rr100_corrected100x1000_response_cache_v1/`.
The response cache contains one bounded ten-movie systems preflight, but no
complete balanced round. Do not mistake it or the frozen input contract for a
completed neural result.

## Installation

Requires [uv](https://docs.astral.sh/uv/getting-started/installation/) and Python 3.12.

```bash
# Clone the repository
git clone https://github.com/Yates-Lab/VisionCore.git
cd VisionCore

# Install dependencies
uv sync
```

### Optional: Data packages

To include the data loading packages, first clone the data repositories as siblings to VisionCore:

```bash
cd ..
git clone https://github.com/Yates-Lab/DataYatesV1.git
git clone https://github.com/Yates-Lab/DataRowleyV1V2.git
cd VisionCore
```

Then install with the `data` extra:

```bash
uv sync --extra data
```
