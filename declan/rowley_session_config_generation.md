# Adapting `multi_dataset_build_session_configs.py` for Rowley Data

This document describes, with full specificity, how
`DataYatesV1/examples/multi_dataset_build_session_configs.py` must be adapted to
produce equivalent per-session YAML files for Rowley sessions.

---

## What the Yates script does

The script loops over all complete Yates sessions and, for each one:

1. Loads a base YAML config (transforms, datafilters, key-lags, etc.)
2. Computes three per-unit quality metrics:
   - **Visual SNR** from gaborium STA/STE
   - **Median missing spike %** from truncation QC
   - **Minimum contamination %** from refractory-period QC
3. Applies thresholds to select `good_units` (visually responsive + not contaminated)
4. Writes `{sess.name}.yaml` containing the base config plus session-specific fields:
   `cids`, `session`, and the per-metric lists for reference

The output YAMLs live in `VisionCore/experiments/dataset_configs/sessions/`.

---

## Summary of structural differences

| Concern | Yates | Rowley |
|---|---|---|
| Session class | `YatesV1Session` (from DataYatesV1) | `RowleySession` (from DataRowleyV1V2) |
| Session list | `get_complete_sessions()` → all sorted sessions | `V1_SESSIONS` list in `registry.py` (curated) |
| `cids` meaning | **0-based column indices** into `robs` | **Global cluster IDs** (mapped via `all_cids`) |
| Dataset path | `sess.sess_dir / 'datasets' / 'gaborium.dset'` | `sess.processed_path / 'datasets' / '{eye}_eye' / 'gaborium.dset'` |
| Eye dimension | Single eye per session | Per-eye datasets; `eye` must be chosen explicitly |
| STA/STE cache | `sess_dir / 'datasets' / 'gaborium_sta_ste.npy'` | Must be created; recommend `sess.processed_path / 'datasets' / '{eye}_eye' / 'gaborium_sta_ste.npy'` |
| Missing % QC | `truncation.npz` with 0-based local unit IDs | `sess.get_missing_pct_qc()` returns global cluster IDs + time-windowed values |
| Contamination QC | `refractory.npz['min_contam_props']` pre-computed per unit | `refractory_qc.npz['rvl_tensor']` shape `(n_units, n_contam_props, n_refract_periods)` — requires derivation |
| QC file location | `sess.sess_dir / 'qc' / ...` | `shank_dir / 'qc' / ...` (per shank, accessed via `sess._parse_available_shanks()`) |
| Output YAML extra fields | none (base config + cids + session) | Must add `lab: Rowley` and `directory: <path>` |
| Region filtering | Not applicable (all units are V1) | May want separate V1 / V2 passes using `region` from `.dset` metadata |

---

## Section-by-section adaptation

### 1. Imports and base config

**Yates:**
```python
from DataYatesV1 import get_session, get_complete_sessions, get_gaborium_sta_ste, calc_sta

base_config_path = Path("/mnt/ssd/YatesMarmoV1/conv_model_fits/data_configs/.../ray_base.yaml")
```

**Rowley:**
```python
from DataRowleyV1V2.data.registry import RowleySession, V1_SESSIONS
from DataRowleyV1V2.utils.datasets import DictDataset
from DataRowleyV1V2.utils.rf import calc_sta
from DataRowleyV1V2.shifter.preprocess import normalize_stimulus, create_valid_eyepos_mask

# The base config should contain everything EXCEPT cids, session, lab, directory.
# A good starting point is the existing parent YAML, stripped of its sessions list.
base_config_path = Path("experiments/dataset_configs/rowley_base.yaml")
```

The Rowley base YAML needs two extra fields that the Yates base does not need:
```yaml
lab: Rowley
# 'directory' will be added per-session-per-eye in the script
```

---

### 2. Session iteration

**Yates:**
```python
sessions = get_complete_sessions()   # all sessions in processed dir
for sess in tqdm(sessions):
    ...
```

**Rowley:**
```python
for session_config in V1_SESSIONS:
    session_name = session_config['session_name']
    sess = RowleySession(session_name)
    shanks_map = session_config['shanks']   # e.g. {0: 'V2', 1: 'V1'}
    eyes = session_config.get('eyes', ['right'])

    for eye in eyes:
        # Derive the dataset directory for this session + eye
        datasets_dir = sess.processed_path / 'datasets' / f'{eye}_eye'
        if not (datasets_dir / 'gaborium.dset').exists():
            print(f'No gaborium dataset for {session_name} {eye} eye, skipping.')
            continue
        # ... process this (session, eye) pair ...
```

Note: `V1_SESSIONS` is a curated list. It can be extended or replaced with
`get_complete_sessions()` if all processed sessions should be included.

---

### 3. Loading the dataset and extracting `cluster_ids`

**Yates** implicitly uses the gaborium dataset inside `get_gaborium_sta_ste()`, which
loads `sess.sess_dir / 'datasets' / 'gaborium.dset'`. The unit ordering in that dataset
is 0-based (column 0 = unit 0, etc.) and these indices are used directly as `cids`.

**Rowley** must load the dataset explicitly to read `cluster_ids` — the global cluster
IDs that identify each `robs` column. These are what go into the YAML as `cids`.

```python
dset_path = datasets_dir / 'gaborium.dset'
dset = DictDataset.load(dset_path)

# cluster_ids is stored at generation time; it maps column i → global cluster ID
cluster_ids = np.asarray(dset.metadata['cluster_ids'])   # shape (N_units,)
region      = np.asarray(dset.metadata['region'])         # 'V1' or 'V2' per unit
n_units     = len(cluster_ids)
```

For V1-only YAMLs:
```python
v1_mask    = region == 'V1'
v1_cids    = cluster_ids[v1_mask]    # global IDs of V1 units only
v1_indices = np.where(v1_mask)[0]    # column indices within the full dataset
# All subsequent quality arrays should be indexed by v1_indices before thresholding
```

---

### 4. Visual SNR (gaborium STA/STE)

**Yates** calls `get_gaborium_sta_ste(sess, n_lags)` which loads a cached `.npy` if
available, otherwise computes and caches. It normalises the stimulus as
`(stim - stim.mean()) / 255` and uses `get_valid_dfs(dset, n_lags)` for the frame mask.

**Rowley** equivalent — the Rowley package has `DataRowleyV1V2.utils.rf.calc_sta`
(same interface) and `normalize_stimulus` / `create_valid_eyepos_mask` for preprocessing.
A cache should be used for the same reason (computation takes ~minutes):

```python
from DataRowleyV1V2.utils.rf import calc_sta
from DataRowleyV1V2.shifter.preprocess import normalize_stimulus, create_valid_eyepos_mask

cache_path = datasets_dir / 'gaborium_sta_ste.npy'
n_lags = 20   # match n_lags used in VisionCore keys_lags for stim

if cache_path.exists():
    stas, stes = np.load(cache_path, allow_pickle=True)
else:
    dset_gauss = DictDataset.load(dset_path)
    stim = normalize_stimulus(dset_gauss['stim'].float())   # (T, H, W), float
    robs = dset_gauss['robs'].float()                       # (T, N_units)

    # Frame validity mask: eye position in bounds + dpi_valid
    dfs = create_valid_eyepos_mask(
        dset_gauss['eyepos'], dset_gauss['dpi_valid'],
        valid_eyepos_radius=dset_gauss.metadata['valid_eyepos_radius'],
    ).squeeze()   # (T,)

    stas = calc_sta(stim, robs, lags=range(n_lags),
                    dfs=dfs, progress=True).numpy()    # (N_units, n_lags, H, W)
    stes = calc_sta(stim, robs, lags=range(n_lags),
                    dfs=dfs, stim_modifier=lambda x: x**2,
                    progress=True).numpy()             # (N_units, n_lags, H, W)

    np.save(cache_path, [stas, stes])
```

SNR per unit is then computed identically to the Yates script:
```python
from scipy.ndimage import gaussian_filter

signal = np.abs(stes - np.median(stes, axis=(2, 3), keepdims=True))
signal = gaussian_filter(signal, sigma=[0, 2, 2, 2])
noise  = np.median(signal[:, 0], axis=(1, 2))   # baseline noise from lag 0
snr_per_lag = np.max(signal, axis=(2, 3)) / noise[:, None]   # (N_units, n_lags)
visual_snr  = snr_per_lag.max(axis=1)            # (N_units,) — peak SNR over lags
```

---

### 5. Missing spike percentage

**Yates** loads pre-computed truncation QC from a flat `.npz`:
```python
truncation = np.load(sess.sess_dir / 'qc' / 'amp_truncation' / 'truncation.npz')
# truncation['cid']   — 0-based unit index (one entry per time window)
# truncation['mpcts'] — missing % for that unit in that window
med_missing_pct = np.array([
    np.median(truncation['mpcts'][truncation['cid'] == iU])
    for iU in range(len(cids))
])
```

**Rowley** stores the equivalent in a per-shank `truncation_qc.npz` with local cluster
IDs (not 0-based column indices). The session method `get_missing_pct_qc(shanks)` already
aggregates across shanks and remaps to global cluster IDs:

```python
# Returns: (all_cids, all_time_windows, all_mpcts)
# all_cids     — global cluster ID for each (unit, window) row
# all_mpcts    — missing % for that unit in that window
qc_cids, _, qc_mpcts = sess.get_missing_pct_qc(shanks=list(shanks_map.keys()))

# Compute median missing% per unit (indexed by position in cluster_ids)
med_missing_pct = np.full(n_units, fill_value=100.0)
for i, cid in enumerate(cluster_ids):
    mask = qc_cids == cid
    if mask.any():
        med_missing_pct[i] = np.median(qc_mpcts[mask])
```

Note: units with no QC windows (e.g. because they didn't exist in the sorted data at
QC-run time) get `fill_value=100.0` and will fail the threshold. This is conservative
and correct — if QC data is missing the unit should not be included.

---

### 6. Contamination percentage

**Yates** gets pre-computed `min_contam_props` per unit (one float per unit, in
`[0, 1]`), already the minimum contamination proportion consistent with the observed
refractory violations:
```python
refractory = np.load(sess.sess_dir / 'qc' / 'refractory' / 'refractory.npz')
min_contam_proportions = refractory['min_contam_props']   # (N_units,)
contam_pct = np.array([np.min(min_contam_proportions[iU]) * 100
                        for iU in range(len(cids))])
```

**Rowley** stores `refractory_qc.npz` per shank (under
`patched_pipeline_results_*_imecX/qc/refractory/refractory_qc.npz`) with shape
`(n_local_units, n_contam_props, n_refract_periods)`.

The file structure:
```
rvl_tensor                  float64  (n_units, n_contam_props, n_refract_periods)
refractory_periods          float64  (n_refract_periods,)   — in seconds
contamination_test_proportions float64 (n_contam_props,)   — in [0, 1]
```

`rvl_tensor[u, c, r]` is the ratio of observed ISI violations to the expected count
given contamination proportion `c` and refractory period `r`. A value ≤ 1 means the
observed violations are *consistent with* having contamination ≤ `c`.

**The minimum contamination proportion per unit** is the smallest `c` for which there
exists some refractory period `r` where `rvl ≤ 1`:
```python
# rvl_min_over_r[u, c] = min over all refractory periods of rvl_tensor[u, c, r]
rvl_min_over_r = d['rvl_tensor'].min(axis=2)              # (n_units, n_contam_props)
consistent = rvl_min_over_r <= 1.0                        # bool (n_units, n_contam_props)
test_props  = d['contamination_test_proportions']         # (n_contam_props,)

min_contam_prop = np.array([
    test_props[consistent[u].argmax()] if consistent[u].any() else 1.0
    for u in range(len(rvl_min_over_r))
])   # (n_local_units,)  — proportion in [0, 1]
```

This is computed **per shank** (indexed by local unit position within that shank).
To align with global `cluster_ids`, iterate per shank and remap using the same
`_get_shank_cluster_offsets` logic used elsewhere:

```python
shank_info    = sess._get_shank_cluster_offsets(shanks=list(shanks_map.keys()))
contam_pct    = np.full(n_units, fill_value=100.0)
available_shanks = sess._parse_available_shanks()

for shank_num, info in shank_info.items():
    shank_dir = available_shanks[shank_num]
    refract_path = shank_dir / 'qc' / 'refractory' / 'refractory_qc.npz'
    if not refract_path.exists():
        continue

    d = np.load(refract_path, allow_pickle=True)
    rvl     = d['rvl_tensor']                         # (n_local, n_contam, n_refract)
    props   = d['contamination_test_proportions']
    local_cids = info['local_cids']                   # sorted local cluster IDs for this shank
    offset     = info['offset']
    global_cids_shank = local_cids + offset           # global IDs for units on this shank

    rvl_min  = rvl.min(axis=2)                       # (n_local, n_contam)
    ok       = rvl_min <= 1.0

    for i_local, gcid in enumerate(global_cids_shank):
        col = np.searchsorted(cluster_ids, gcid)
        if col < len(cluster_ids) and cluster_ids[col] == gcid:
            prop = props[ok[i_local].argmax()] if ok[i_local].any() else 1.0
            contam_pct[col] = prop * 100.0
```

> **Caution**: The semantic equivalence between `min_contam_props` (Yates) and this
> derivation from `rvl_tensor` (Rowley) should be validated on a session where both
> pipelines can be run, before finalising the threshold.

---

### 7. Applying thresholds and building `cids`

**Yates:**
```python
visually_responsive = np.where(visual_snr > snr_thresh)[0]
not_contaminated    = np.where(contam_pct < contam_thresh)[0]
good_units          = np.intersect1d(visually_responsive, not_contaminated)
session_config['cids'] = good_units.tolist()   # 0-based column indices
```

**Rowley** — same logic but the output `cids` must be **global cluster IDs**, not
column positions. Apply to V1 units only if a V1-specific YAML is desired:

```python
snr_thresh    = 5
contam_thresh = 50    # %
missing_thresh = 45   # % (matching the runtime datafilter threshold)

# Work on V1 units only
vis_responsive_v1 = v1_indices[visual_snr[v1_indices]  > snr_thresh]
not_contam_v1     = v1_indices[contam_pct[v1_indices]  < contam_thresh]
not_missing_v1    = v1_indices[med_missing_pct[v1_indices] < missing_thresh]

# good = visually responsive + not contaminated (missing handled at runtime by datafilter)
good_col_indices  = np.intersect1d(vis_responsive_v1, not_contam_v1)
good_cids         = cluster_ids[good_col_indices]   # global cluster IDs

session_config['cids'] = sorted(good_cids.tolist())  # GLOBAL IDs, not column indices
```

The `missing_pct` datafilter in VisionCore handles per-frame missing-spike masking at
runtime, so it is NOT necessary to require `med_missing_pct < threshold` to enter
`cids`. However, if a unit is *chronically* missing (>threshold throughout the session)
it makes sense to exclude it here too — this matches the Yates approach of storing
`qcmissing` for reference even if it is not applied to `cids`.

---

### 8. Output YAML structure

**Yates session YAML** (e.g. `Allen_2022-02-16.yaml`):
```yaml
session: Allen_2022-02-16
cids: [3, 7, 12, ...]          # 0-based column indices
visual: [3, 5, 7, ...]         # all visually responsive (reference only)
qcmissing: [3, 7, ...]         # all units passing missing threshold (reference)
qccontam: [3, 7, ...]          # all units passing contamination threshold (reference)
snr: 5
missingth: 25
contamth: 50
# (all other keys inherited from base config — no lab or directory needed)
```

**Rowley session YAML** (e.g. `Luke_2025-08-04.yaml`):
```yaml
lab: Rowley
session: Luke_2025-08-04
directory: /mnt/ssd2/RowleyMarmoV1V2/processed/Luke_2025-08-04/datasets/right_eye
cids: [775, 786, 803, ...]     # GLOBAL cluster IDs (not column indices)
region: V1                     # which region this YAML covers
eye: right
visual_v1: [775, 786, ...]     # all V1 units passing SNR (reference only)
qcmissing_v1: [775, ...]       # all V1 units passing missing threshold (reference)
qccontam_v1: [775, ...]        # all V1 units passing contamination threshold (reference)
snr: 5
missingth: 45
contamth: 50
# (all other keys inherited from base config)
```

Notes:
- `lab: Rowley` is required so `prepare_data` routes to `DataRowleyV1V2.data.registry.get_session`
- `directory` is required because Rowley processed datasets live outside of VisionCore
- `cids` are global cluster IDs; they are mapped to column indices at load time via
  `dset.metadata['all_cids']` in `registry.get_dataset()` and `models/data/loading.py`
- If producing both V1 and V2 YAMLs for a session, use e.g. `Luke_2025-08-04_V1.yaml`
  / `Luke_2025-08-04_V2.yaml` and reference them explicitly in the parent config's
  `sessions` list

---

### 9. Output file path and naming

**Yates** writes to the same directory as the base config (under
`/mnt/ssd/YatesMarmoV1/conv_model_fits/data_configs/.../`). The files later need to be
copied into `VisionCore/experiments/dataset_configs/sessions/`.

**Rowley** should write directly to:
```python
output_dir = Path("experiments/dataset_configs/sessions/")
output_file = output_dir / f"{session_name}.yaml"  # or f"{session_name}_{eye}.yaml"
```

---

### 10. Putting it all together — skeleton

```python
"""
Build per-session YAML configs for Rowley sessions.

Run from VisionCore root:
    python DataRowleyV1V2/examples/build_rowley_session_configs.py
"""

import yaml
import numpy as np
from pathlib import Path
from tqdm import tqdm
from scipy.ndimage import gaussian_filter

from DataRowleyV1V2.data.registry import RowleySession, V1_SESSIONS
from DataRowleyV1V2.utils.datasets import DictDataset
from DataRowleyV1V2.utils.rf import calc_sta
from DataRowleyV1V2.shifter.preprocess import normalize_stimulus, create_valid_eyepos_mask

# ── Config ────────────────────────────────────────────────────────────────────
base_config_path = Path("experiments/dataset_configs/rowley_base.yaml")
output_dir       = Path("experiments/dataset_configs/sessions/")
n_lags           = 20
snr_thresh       = 5
contam_thresh    = 50    # %
missing_thresh   = 45    # %

with open(base_config_path) as f:
    base_config = yaml.safe_load(f)

def represent_list(dumper, data):
    return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)
yaml.add_representer(list, represent_list)

# ── Main loop ─────────────────────────────────────────────────────────────────
for session_config in tqdm(V1_SESSIONS, desc="Sessions"):
    session_name = session_config['session_name']
    shanks_map   = session_config['shanks']
    eyes         = session_config.get('eyes', ['right'])

    sess = RowleySession(session_name)

    for eye in eyes:
        datasets_dir = sess.processed_path / 'datasets' / f'{eye}_eye'
        dset_path    = datasets_dir / 'gaborium.dset'
        if not dset_path.exists():
            print(f"  {session_name} {eye}: no gaborium.dset, skipping")
            continue

        # 1. Load dataset and cluster metadata
        dset         = DictDataset.load(dset_path)
        cluster_ids  = np.asarray(dset.metadata['cluster_ids'])
        region       = np.asarray(dset.metadata['region'])
        n_units      = len(cluster_ids)
        v1_mask      = region == 'V1'
        v1_indices   = np.where(v1_mask)[0]

        # 2. Compute visual SNR (with caching)
        cache_path = datasets_dir / 'gaborium_sta_ste.npy'
        if cache_path.exists():
            stas, stes = np.load(cache_path, allow_pickle=True)
        else:
            stim = normalize_stimulus(dset['stim'].float())
            robs = dset['robs'].float()
            dfs  = create_valid_eyepos_mask(
                dset['eyepos'], dset['dpi_valid'],
                valid_eyepos_radius=dset.metadata['valid_eyepos_radius'],
            ).squeeze()
            stas = calc_sta(stim, robs, range(n_lags), dfs=dfs, progress=True).numpy()
            stes = calc_sta(stim, robs, range(n_lags), dfs=dfs,
                            stim_modifier=lambda x: x**2, progress=True).numpy()
            np.save(cache_path, [stas, stes])

        signal      = np.abs(stes - np.median(stes, axis=(2,3), keepdims=True))
        signal      = gaussian_filter(signal, [0, 2, 2, 2])
        noise       = np.median(signal[:, 0], axis=(1, 2))
        visual_snr  = (np.max(signal, axis=(2,3)) / noise[:, None]).max(axis=1)

        # 3. Missing spike % (per unit, median over time windows)
        qc_cids, _, qc_mpcts = sess.get_missing_pct_qc(shanks=list(shanks_map.keys()))
        med_missing_pct = np.full(n_units, 100.0)
        for i, cid in enumerate(cluster_ids):
            mask = qc_cids == cid
            if mask.any():
                med_missing_pct[i] = np.median(qc_mpcts[mask])

        # 4. Contamination % (per unit, from refractory QC)
        shank_info       = sess._get_shank_cluster_offsets(list(shanks_map.keys()))
        available_shanks = sess._parse_available_shanks()
        contam_pct_arr   = np.full(n_units, 100.0)
        for shank_num, info in shank_info.items():
            shank_dir    = available_shanks[shank_num]
            refract_path = shank_dir / 'qc' / 'refractory' / 'refractory_qc.npz'
            if not refract_path.exists():
                continue
            d          = np.load(refract_path, allow_pickle=True)
            rvl        = d['rvl_tensor']            # (n_local, n_contam, n_refract)
            props      = d['contamination_test_proportions']
            ok         = rvl.min(axis=2) <= 1.0     # (n_local, n_contam)
            local_cids = info['local_cids']
            global_cids_shank = local_cids + info['offset']
            for i_local, gcid in enumerate(global_cids_shank):
                col = np.searchsorted(cluster_ids, gcid)
                if col < n_units and cluster_ids[col] == gcid:
                    prop = props[ok[i_local].argmax()] if ok[i_local].any() else 1.0
                    contam_pct_arr[col] = prop * 100.0

        # 5. Apply thresholds (V1 units only)
        vis_v1      = v1_indices[visual_snr[v1_indices]     > snr_thresh]
        ok_contam   = v1_indices[contam_pct_arr[v1_indices] < contam_thresh]
        ok_missing  = v1_indices[med_missing_pct[v1_indices] < missing_thresh]

        good_cols   = np.intersect1d(vis_v1, ok_contam)
        good_cids   = sorted(cluster_ids[good_cols].tolist())

        # 6. Write YAML
        cfg = dict(base_config)
        cfg['lab']       = 'Rowley'
        cfg['session']   = session_name
        cfg['directory'] = str(datasets_dir)
        cfg['eye']       = eye
        cfg['region']    = 'V1'
        cfg['cids']      = [int(c) for c in good_cids]
        cfg['visual_v1']    = [int(c) for c in cluster_ids[vis_v1]]
        cfg['qcmissing_v1'] = [int(c) for c in cluster_ids[ok_missing]]
        cfg['qccontam_v1']  = [int(c) for c in cluster_ids[ok_contam]]
        cfg['snr']          = snr_thresh
        cfg['missingth']    = missing_thresh
        cfg['contamth']     = contam_thresh

        out_path = output_dir / f"{session_name}.yaml"
        with open(out_path, 'w') as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
        print(f"  Wrote {out_path}: {len(good_cids)} V1 good units")
```

---

## Key invariant to maintain

> **Rowley `cids` in YAML are always global cluster IDs, never column indices.**

This is enforced at load time: `registry.get_dataset()` stores `dset.metadata['all_cids']`
from the `.dset`'s `cluster_ids` field, and `models/data/loading.py` uses `searchsorted`
on that array to translate YAML cids into column indices. If a YAML cid is not found in
`all_cids`, `prepare_data` raises a `ValueError` with the missing IDs, making mistakes
immediately visible.

For Yates, `all_cids` is never set (Yates sessions don't have this in metadata), so the
fallback `col_indices = np.asarray(cids)` path is taken — preserving backward
compatibility.
