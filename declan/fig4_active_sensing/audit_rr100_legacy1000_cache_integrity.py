#!/usr/bin/env python3
"""Phase-4A integrity audit of the legacy 100x1000 RR100 response cache."""
from __future__ import annotations
import argparse,json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[2]
CACHE=ROOT/'outputs/active_sensing_movie_information/backimage_real_trace_ssi_matrix_large_contour_no_driftgate_ms200_n100x1000_v1/merged'
OUT=ROOT/'outputs/fig4_active_sensing/rr100_legacy1000_cache_integrity_checkpoint_26_v1'
def main():
 p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,default=CACHE);p.add_argument('--out',type=Path,default=OUT);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
 image=pd.read_csv(a.cache/'image_feature_table.csv').sort_values('image_index');trace=pd.read_csv(a.cache/'trace_feature_table.csv').sort_values('trace_bank_index');movie=pd.read_csv(a.cache/'movie_feature_table.csv');unit=pd.read_csv(a.cache/'unit_feature_table.csv');rate=np.load(a.cache/'mean_rate_matrix.npy');spike=np.load(a.cache/'expected_spikes_matrix.npy');ssi=np.load(a.cache/'ssi_matrix.npy');pop=np.load(a.cache/'population_ssi.npy')
 checks={'n_images':len(image),'n_traces':len(trace),'n_movies_table':len(movie),'n_units':len(unit),'matrix_shapes':{'rate':list(rate.shape),'expected_spikes':list(spike.shape),'ssi':list(ssi.shape),'population_ssi':list(pop.shape)},'movie_key_duplicates':int(movie.duplicated(['image_index','trace_index']).sum()),'movie_grid_unique':int(movie[['image_index','trace_index']].drop_duplicates().shape[0]),'finite':{'rate':bool(np.isfinite(rate).all()),'spike':bool(np.isfinite(spike).all()),'ssi':bool(np.isfinite(ssi).all()),'population_ssi':bool(np.isfinite(pop).all())},'nonnegative':{'rate':bool((rate>=0).all()),'spike':bool((spike>=0).all())}}
 # inferred matrix axes: [image, trace, unit]; expected spikes equal rate*40/120.
 checks['expected_spike_identity_max_abs_error']=float(np.max(abs(spike-rate*(40/120))))
 checks['expected_spike_identity_pass']=checks['expected_spike_identity_max_abs_error']<1e-5
 checks['contract']='legacy reconstructed-motion cache; structural-only pilot. Values are not corrected 240Hz/dpi_pix/explicit-history responses.'
 pd.DataFrame([checks]).to_json(a.out/'integrity_summary.json',orient='records',indent=2);(a.out/'manifest.json').write_text(json.dumps({'created_utc':datetime.now(timezone.utc).isoformat(),'status':'phase_4a_cache_integrity_complete','checks':checks,'guardrail':'Do not use this cache for calibrated FEM rate, SSI, event, temporal-frequency, gain, pairing, or orientation claims.'},indent=2)+'\n');print(json.dumps(checks,indent=2))
if __name__=='__main__':main()
