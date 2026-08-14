#!/usr/bin/env python3
"""Structural-only convergence, influence, and selection audit of legacy cache."""
from __future__ import annotations
import argparse,json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[2];CACHE=ROOT/'outputs/active_sensing_movie_information/backimage_real_trace_ssi_matrix_large_contour_no_driftgate_ms200_n100x1000_v1/merged';TRACE=ROOT/'outputs/fig4_active_sensing/rr100_legacy1000_trace_agreement_checkpoint_23_v1/corrected_trace_crosswalk.csv';IMAGE=ROOT/'outputs/fig4_active_sensing/rr100_legacy100_corrected_image_audit_checkpoint_24_v1/corrected_image_crosswalk.csv';OUT=ROOT/'outputs/fig4_active_sensing/rr100_legacy1000_structural_pilot_checkpoint_27_v1'
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=OUT);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);rng=np.random.default_rng(20260813)
 movie=pd.read_csv(CACHE/'movie_feature_table.csv').sort_values('matrix_row_index');tr=pd.read_csv(TRACE).sort_values('trace_index');im=pd.read_csv(IMAGE).sort_values('image_index');rate=np.load(CACHE/'mean_rate_matrix.npy');ssi=np.load(CACHE/'ssi_matrix.npy');pop=np.load(CACHE/'population_ssi.npy')
 # aggregate each trace across images, each image across traces, then calculate influence.
 r3=rate.reshape(100,1000,100);s3=ssi.reshape(100,1000,100);pt=pop.reshape(100,1000)
 trace_table=pd.DataFrame({'trace_index':np.arange(1000),'legacy_population_ssi_mean':pt.mean(0),'legacy_rate_mean':r3.mean((0,2)),'legacy_ssi_mean':s3.mean((0,2))}).merge(tr[['trace_index','explicit_history_valid','corrected_dpi_crop120_path_length_arcmin','corrected_dpi_crop120_position_power_fraction_32plus_hz','inclusion_recommendation']],on='trace_index')
 base=float(pt.mean());trace_table['population_ssi_leave_trace_out']= (pt.sum()-pt.sum(0))/(100000-100);trace_table['abs_trace_influence']=abs(base-trace_table.population_ssi_leave_trace_out)
 image_table=pd.DataFrame({'image_index':np.arange(100),'legacy_population_ssi_mean':pt.mean(1),'legacy_rate_mean':r3.mean((1,2)),'legacy_ssi_mean':s3.mean((1,2))}).merge(im[['image_index','corrected_crop_valid','reconstruction_exact_pixel_r']],on='image_index')
 image_table['population_ssi_leave_image_out']=(pt.sum()-pt.sum(1))/(100000-1000);image_table['abs_image_influence']=abs(base-image_table.population_ssi_leave_image_out)
 # repeated disjoint subset convergence for population SSI only
 rows=[]
 for ni,nt in [(10,100),(25,250),(50,500),(100,1000)]:
  vals=[]
  for _ in range(100): vals.append(float(pt[np.ix_(rng.choice(100,ni,False),rng.choice(1000,nt,False))].mean()))
  rows.append({'n_images':ni,'n_traces':nt,'mean_population_ssi':float(np.mean(vals)),'sd_across_disjoint_style_subsamples':float(np.std(vals))})
 conv=pd.DataFrame(rows);trace_table.to_csv(a.out/'trace_structural_influence.csv',index=False);image_table.to_csv(a.out/'image_structural_influence.csv',index=False);conv.to_csv(a.out/'convergence_summary.csv',index=False)
 # roles for bridge bank/design; never based on validity-invalid rows except control.
 valid=trace_table[trace_table.explicit_history_valid].copy();sel=pd.concat([valid.nlargest(1,'legacy_population_ssi_mean').assign(selection_role='cached_positive'),valid.nsmallest(1,'legacy_population_ssi_mean').assign(selection_role='cached_negative_control'),valid.nlargest(1,'corrected_dpi_crop120_path_length_arcmin').assign(selection_role='long_corrected_path'),valid.nlargest(1,'corrected_dpi_crop120_position_power_fraction_32plus_hz').assign(selection_role='high_corrected_tf'),trace_table[~trace_table.explicit_history_valid].head(1).assign(selection_role='history_invalid_control')]);sel.to_csv(a.out/'bridge_trace_selection.csv',index=False)
 manifest={'created_utc':datetime.now(timezone.utc).isoformat(),'status':'phase_4_structural_pilot_complete','n_movies':100000,'n_valid_history_traces':int(tr.explicit_history_valid.sum()),'guardrail':'All outputs are legacy reconstructed-motion structural-only. They do not establish corrected FEM effects or determine the corrected production cohort.'};(a.out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest,indent=2))
if __name__=='__main__':main()
