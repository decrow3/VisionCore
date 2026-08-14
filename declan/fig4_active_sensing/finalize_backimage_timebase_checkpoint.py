#!/usr/bin/env python3
"""Finalize Phase-3 BackImage 240-Hz tables and old/new interval crosswalk."""
from __future__ import annotations
import argparse, json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
OLD=ROOT/'outputs/fixation_statistics_by_stimulus_all_sessions_after_review/window_features.csv'
NEW=ROOT/'outputs/fig4_active_sensing/backimage_240hz_timebase_checkpoint_25_v1/raw/window_features.csv'
OUT=ROOT/'outputs/fig4_active_sensing/backimage_240hz_timebase_checkpoint_25_v1'
KEY=['session','trial_idx','global_start','global_stop']
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--old',type=Path,default=OLD);p.add_argument('--new',type=Path,default=NEW);p.add_argument('--out-dir',type=Path,default=OUT);a=p.parse_args()
 old=pd.read_csv(a.old);old=old[old.stimulus.eq('backimage')].copy();new=pd.read_csv(a.new);new=new[new.stimulus.eq('backimage')].copy()
 if not new.sample_rate_hz.eq(240).all():raise ValueError('New BackImage table is not uniformly 240 Hz')
 start=new.global_start.to_numpy(int); stop=new.global_stop.to_numpy(int); even_start=start+(start%2); n_even=((stop-even_start)+1)//2
 new['model_visual_rate_hz']=120.0;new['model_global_even_start']=even_start;new['model_global_even_stop_exclusive']=even_start+2*n_even;new['model_n_frames']=n_even;new['model_duration_s']=(n_even-1)/120.0
 new.to_csv(a.out_dir/'backimage_window_features_240hz.csv',index=False)
 events=pd.read_csv(a.new.parent/'saccade_event_features.csv');events=events[events.stimulus.eq('backimage')].copy();events.to_csv(a.out_dir/'backimage_event_features_240hz.csv',index=False)
 joined=old.merge(new,on=KEY,how='outer',suffixes=('_old120','_new240'),indicator=True,validate='one_to_one')
 joined.to_csv(a.out_dir/'backimage_old_new_window_crosswalk.csv',index=False)
 common=joined[joined._merge.eq('both')]
 metrics=['duration_s','epoch_duration_s','speed_mean_deg_s','speed_median_deg_s','speed_p95_deg_s','path_length_deg_s','diffusion_constant_deg2_s','position_high_freq_power_fraction_15_60hz']
 rows=[]
 for key in metrics:
  x=common[f'{key}_old120'];y=common[f'{key}_new240'];rows.append({'metric':key,'n':len(common),'median_old':float(x.median()),'median_new':float(y.median()),'median_ratio_new_over_old':float((y/x.replace(0,np.nan)).median()),'pearson_r':float(x.corr(y))})
 pd.DataFrame(rows).to_csv(a.out_dir/'old_new_timebase_metric_summary.csv',index=False)
 summary={'created_utc':datetime.now(timezone.utc).isoformat(),'status':'phase_3_backimage_timebase_tables_complete','counts':{'n_old_backimage_windows':len(old),'n_new_backimage_windows':len(new),'n_exact_interval_matches':int(len(common)),'n_old_only':int((joined._merge=='left_only').sum()),'n_new_only':int((joined._merge=='right_only').sum()),'n_new_events':len(events)},'behavior_contract':'raw BackImage eyepos at 240 Hz','visual_contract':'global-even raw indices at 120 Hz retained separately per window','guardrail':'Old event, speed, phase, PSD, duration and diffusion labels remain superseded; use this regenerated table for behavior labels.'}
 (a.out_dir/'manifest.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
