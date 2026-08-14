#!/usr/bin/env python3
"""CPU equivalence and throughput check for the lag-zero input renderer."""
from __future__ import annotations
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from declan.fig4_active_sensing.input_only_retinal_renderer import render_retinal_frames_lag_zero
from declan.fig4_active_sensing.make_rr100_explicit_history_input_checkpoint import explicit_segments
from declan.fig4_active_sensing.run_rr100_corrected_ssi_map_first_smoke import corrected_patch
from declan.fig4_active_sensing.audit_rr100_eye_trace_conditioning_and_nyquist_power import load_dset
from declan.active_sensing_movie_information.run_backimage_real_trace_ssi_matrix_pilot import load_source_rows
from declan.active_sensing_movie_information.plot_temporal_power_shift_map_first import source_row_by_id
from declan.fixation_statistics_by_stimulus.run_backimage_twin_drift_geometry import _load_twin_common,_standardize_uint_like,_trace_xy_to_twin_helper_order
ROOT=Path(__file__).resolve().parents[2]
def main():
 cohort=ROOT/'outputs/fig4_active_sensing/rr100_interim49x973_bridge_cohort_checkpoint_28_v1'
 images=pd.read_csv(cohort/'interim49_images.csv'); traces=pd.read_csv(cohort/'interim973_traces.csv')
 # Keep one large BackImage session resident during CPU validation.
 image=images.loc[images.session.eq('Allen_2022-04-01')].iloc[0]
 trace_row=traces.loc[traces.session.eq(str(image.session))].iloc[0]
 source=load_source_rows(ROOT/'outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_image_structure_reviewed_v2_screenfiltered_yfix/backimage_image_fem_windows.csv')
 dc={};cc={}; isrc=source_row_by_id(source,int(image.source_row));tsrc=source_row_by_id(source,int(trace_row.source_row))
 patch,meta,_=corrected_patch(isrc,load_dset(str(isrc.session),dc),cc)
 history,score,_=explicit_segments(tsrc,load_dset(str(tsrc.session),dc)); trace=np.concatenate([history,score]);common=_load_twin_common();standard=_standardize_uint_like(patch)
 direct=render_retinal_frames_lag_zero(common,standard,-trace,ppd=float(meta['patch_ppd'])).cpu()
 full=np.broadcast_to(standard[None],(105,*patch.shape)).copy(); eye=torch.from_numpy(_trace_xy_to_twin_helper_order(-trace))
 embedded=common.make_counterfactual_stim(full,eye,ppd=float(meta['patch_ppd']),n_lags=32,out_size=(51,51));reference=embedded[1:73,0,0]
 err=float((direct-reference).abs().max())
 if err>1e-5: raise AssertionError(f'lag-zero mismatch {err}')
 t=time.perf_counter()
 for _ in range(10):
  movie=render_retinal_frames_lag_zero(common,standard,-trace,ppd=float(meta['patch_ppd']))
  power=torch.fft.rfft(movie[32:]-movie[32:].mean(0),dim=0).abs().square().mean((1,2));del movie,power
 print({'max_abs_error':err,'cpu_seconds_per_movie':(time.perf_counter()-t)/10,'frame_shape':list(direct.shape)})
if __name__=='__main__':main()
