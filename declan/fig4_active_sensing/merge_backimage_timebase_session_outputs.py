#!/usr/bin/env python3
"""Merge complete per-session Phase-3 BackImage extraction outputs."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
def main():
 p=argparse.ArgumentParser();p.add_argument('--partials',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 dirs=sorted(x for x in a.partials.iterdir() if x.is_dir())
 windows=[pd.read_csv(x/'window_features.csv') for x in dirs if (x/'window_features.csv').exists()]
 events=[pd.read_csv(x/'saccade_event_features.csv') for x in dirs if (x/'saccade_event_features.csv').exists()]
 if len(windows)!=30: raise RuntimeError(f'Expected 30 complete session outputs, found {len(windows)}')
 a.out.mkdir(parents=True,exist_ok=True);pd.concat(windows,ignore_index=True).to_csv(a.out/'window_features.csv',index=False);pd.concat(events,ignore_index=True).to_csv(a.out/'saccade_event_features.csv',index=False)
 print(f'Merged {len(windows)} sessions, {sum(len(x) for x in windows)} windows, {sum(len(x) for x in events)} events')
if __name__=='__main__':main()
