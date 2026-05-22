# Clean Branch Runtime Fixes

This note documents the runtime fixes applied after reviewing `clean/from-upstream-main`.

## Previous behavior by branch

- `main`: the local branch already contained regenerated Rowley datasets whose metadata stored `cluster_ids`.
- `clean/from-upstream-main` at commit `dc69952`: `models/data/loading.py` and `models/data/datafilters.py` looked for `metadata['all_cids']` when trying to align YAML `cids` with dataset columns.
- `clean/from-upstream-main` at commit `dc69952`: if a Rowley dataset only stored `cluster_ids`, `loading.py` fell back to treating YAML cluster IDs as direct column indices.
- `clean/from-upstream-main` at commit `dc69952`: `datafilters.py` assumed `get_missing_pct_interp()` returned a torch tensor and called `.median(dim=0)`, which is not valid for NumPy outputs.

## Behavior after this fix

- `models/data/loading.py` now accepts either `cluster_ids` or legacy `all_cids` metadata when mapping YAML `cids` to `robs` columns.
- `models/data/datafilters.py` now accepts either NumPy or torch outputs from `get_missing_pct_interp()` and still returns a torch boolean mask.
- Added `tests/test_runtime_data_paths.py` to lock in both behaviors.

## Scope

- These changes are runtime correctness fixes only.
- They do not alter the new covariance APIs in `VisionCore/covariance.py`.