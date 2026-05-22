"""
Regenerate Luke_2025-08-04 fixrsvp / gaborium / backimage .dset files using
the current spike sorting.

The original datasets were generated from a Kilosort run that has since been
updated, so cluster IDs in the .dset files no longer match the current
spike_clusters.npy.  This script regenerates them in-place, preserving the
existing ROI and DPI-shifter correction (dpi_shifted.csv), and now storing
'cluster_ids' in metadata so prepare_data() can map YAML cids → column indices.

Run from the VisionCore root:
    python scripts/regen_luke0804_datasets.py

Outputs overwrite:
    /mnt/ssd2/RowleyMarmoV1V2/processed/Luke_2025-08-04/datasets/right_eye/
After running, update Luke_2025-08-04.yaml cids to match the printed V1 cluster IDs.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.interpolate import interp1d

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / 'DataRowleyV1V2'))

from DataRowleyV1V2.data.registry import RowleySession
from DataRowleyV1V2.utils.mat import loadmat
from DataRowleyV1V2.utils.datasets import DictDataset
from DataRowleyV1V2.exp.fix_rsvp import generate_fixrsvp_dataset
from DataRowleyV1V2.exp.gaborium_pregen import generate_gaborium_pregen_dataset
from DataRowleyV1V2.exp.backimage import generate_backimage_dataset

# ──────────────────────────────────────────────────────────────────────────────
# Config — preserved from original generation
# ──────────────────────────────────────────────────────────────────────────────
SESSION_NAME    = 'Luke_2025-08-04'
EYE             = 'right'
SHANKS_MAP      = {0: 'V2', 1: 'V1'}   # both shanks, same as original
DT              = 1 / 240
DEPTH_BAND_UM   = 1500.0

# Metadata preserved from original .dset (run once, copy-paste here)
FINAL_ROI            = np.array([[-50, 23], [-60, 13]])
ROI_SRC_ORIGINAL     = np.array([[-141, 110], [-151, 100]])
RF_CENTER_OFFSET     = np.array([2, 2])
VALID_EYEPOS_RADIUS  = 7
PPD                  = 53.735069137666954
GAMMA                = 2.166

OUTPUT_DIR = Path('/mnt/ssd2/RowleyMarmoV1V2/processed/Luke_2025-08-04/datasets/right_eye')


def main():
    session = RowleySession(SESSION_NAME)
    shanks  = list(SHANKS_MAP.keys())

    # ── Experiment file ──────────────────────────────────────────────────────
    print('Loading experiment file...')
    exp = session.load_exp()

    screen_resolution = (exp['S']['screenRect'][2:] - exp['S']['screenRect'][:2]).astype(int)
    screen_width      = float(exp['S']['screenWidth'])
    screen_distance   = float(exp['S']['screenDistance'])
    screen_height     = screen_width * screen_resolution[1] / screen_resolution[0]

    # ── Spikes ───────────────────────────────────────────────────────────────
    print('Loading spikes (this may take a while)...')
    st, clu, cids_all = session.load_spikes(shanks=shanks)
    print(f'  {len(cids_all)} units, {len(st)/1e6:.2f}M spikes')

    # ── Depth band ───────────────────────────────────────────────────────────
    print('Applying depth band filter...')
    depth_mask, depth_um_full, depth_bounds = session.get_depth_band_mask(
        cids_all, shanks=shanks, band_um=DEPTH_BAND_UM,
    )
    cids_band   = cids_all[depth_mask]
    depth_um    = depth_um_full[depth_mask]
    spk_mask    = np.isin(clu, cids_band)
    st_band     = st[spk_mask]
    clu_band    = clu[spk_mask]
    print(f'  Kept {len(cids_band)} units in depth band [{depth_bounds[0]:.0f}, {depth_bounds[1]:.0f}] µm')

    # ── Shank/region labels per unit ─────────────────────────────────────────
    shank_info  = session._get_shank_cluster_offsets(shanks)
    shank_ids   = np.zeros(len(cids_band), dtype=int)
    region_ids  = np.empty(len(cids_band), dtype='U4')
    for shank_num, info in shank_info.items():
        global_cids = info['local_cids'] + info['offset']
        mask = np.isin(cids_band, global_cids)
        shank_ids[mask]  = shank_num
        region_ids[mask] = SHANKS_MAP[shank_num]

    # ── Calibrated + shifted DPI ─────────────────────────────────────────────
    dpi_shifted_csv = session.processed_path / 'shifter' / f'{EYE}_eye' / 'dpi_shifted.csv'
    print(f'Loading shifted DPI from: {dpi_shifted_csv}')
    dpi_df = pd.read_csv(dpi_shifted_csv)

    t_dpi            = dpi_df['t_ephys'].to_numpy()
    dpi_pix_shifted  = dpi_df[['i_shifted', 'j_shifted']].to_numpy()
    dpi_deg          = dpi_df[['az', 'el']].to_numpy()
    dpi_valid        = dpi_df['valid'].to_numpy().astype(bool)

    pix_interp         = interp1d(t_dpi, dpi_pix_shifted, kind='linear',
                                  fill_value='extrapolate', axis=0)
    eyepos_deg_interp  = interp1d(t_dpi, dpi_deg, kind='linear',
                                  fill_value='extrapolate', axis=0)
    valid_interp       = interp1d(t_dpi, dpi_valid.astype(float), kind='nearest',
                                  fill_value='extrapolate')

    # ── Clock alignment ──────────────────────────────────────────────────────
    print('Loading clock alignment...')
    _, ptb2ephys, _, _ = session.load_clocks(plot=False)

    # ── Gaborium stimulus ────────────────────────────────────────────────────
    stim_files = list(session.raw_path.glob('ForagePregenRepeatingNoise_*.mat'))
    pregen_stim = None
    if stim_files:
        print(f'Loading stimulus: {stim_files[0].name}')
        pregen_stim = loadmat(stim_files[0])['StimFramesu8']
        print(f'  shape: {pregen_stim.shape}')
    else:
        print('No gaborium stimulus file found — skipping gaborium/backimage.')

    # ── Shared metadata ──────────────────────────────────────────────────────
    final_metadata = {
        'screen_resolution': screen_resolution,
        'screen_width':      screen_width,
        'screen_height':     screen_height,
        'screen_distance':   screen_distance,
        'ppd':               PPD,
        'gamma':             GAMMA,
        'roi_src':           FINAL_ROI,
        'roi_src_original':  ROI_SRC_ORIGINAL,
        'rf_center_offset':  RF_CENTER_OFFSET,
        'valid_eyepos_radius': VALID_EYEPOS_RADIUS,
        'eye':               EYE,
        'shank_ids':         shank_ids,
        'region':            region_ids,
        'depth_um':          depth_um,
        'use_foveal_depth_band': True,
        'foveal_depth_band_um':  DEPTH_BAND_UM,
        'depth_bounds_um':       depth_bounds,
    }

    interps = {
        'eyepos':    eyepos_deg_interp,
        'dpi_valid': valid_interp,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── FixRSVP ──────────────────────────────────────────────────────────────
    print('\nGenerating fixrsvp...')
    fixrsvp_dset = generate_fixrsvp_dataset(
        exp=exp, ptb2ephys=ptb2ephys,
        st=st_band, clu=clu_band,
        roi_src=FINAL_ROI, pix_interp=pix_interp,
        interps=interps, dt=DT,
        metadata=final_metadata,
        min_duration=0.5,
    )
    if fixrsvp_dset is not None:
        fixrsvp_dset.save(OUTPUT_DIR / 'fixrsvp.dset')
        cids_stored = fixrsvp_dset.metadata['cluster_ids']
        print(f'  robs: {fixrsvp_dset["robs"].shape}, cluster_ids stored: {len(cids_stored)}')
    else:
        print('  No fixrsvp trials found.')
        cids_stored = np.array([])

    # ── Gaborium ─────────────────────────────────────────────────────────────
    if pregen_stim is not None:
        print('\nGenerating gaborium...')
        gaborium_dset = generate_gaborium_pregen_dataset(
            exp=exp, ptb2ephys=ptb2ephys, pregen_stim=pregen_stim,
            st=st_band, clu=clu_band,
            roi_src=FINAL_ROI, pix_interp=pix_interp,
            interps=interps, dt=DT,
            metadata=final_metadata,
        )
        if gaborium_dset is not None:
            gaborium_dset.save(OUTPUT_DIR / 'gaborium.dset')
            print(f'  robs: {gaborium_dset["robs"].shape}')

        # ── Backimage ─────────────────────────────────────────────────────────
        print('\nGenerating backimage...')
        backimage_dset = generate_backimage_dataset(
            exp=exp, ptb2ephys=ptb2ephys,
            st=st_band, clu=clu_band,
            roi_src=FINAL_ROI, pix_interp=pix_interp,
            interps=interps, dt=DT,
            metadata=final_metadata,
        )
        if backimage_dset is not None:
            backimage_dset.save(OUTPUT_DIR / 'backimage.dset')
            print(f'  robs: {backimage_dset["robs"].shape}')

    # ── Print new YAML cids ──────────────────────────────────────────────────
    print('\n' + '='*70)
    print('DONE. Update Luke_2025-08-04.yaml with the new V1 cluster IDs:')
    if len(cids_stored):
        v1_mask = region_ids == 'V1'
        v1_cids = cids_stored[v1_mask]
        print(f'cids: {list(int(c) for c in v1_cids)}')
        print(f'({len(v1_cids)} V1 units)')
    print('='*70)


if __name__ == '__main__':
    main()
