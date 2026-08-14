#!/usr/bin/env python3
"""Phase-2 image audit with session-isolated, atomic input-only workers.

Each worker processes every selected legacy image in one session, compares the
corrected large-field reconstruction with the exact saved 51x51 model frames,
then exits.  Assembly refuses a partial crosswalk.
"""
from __future__ import annotations
import argparse, json, os, tempfile
from datetime import datetime, timezone
from pathlib import Path
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from declan.active_sensing_movie_information.plot_temporal_power_shift_map_first import source_row_by_id
from declan.active_sensing_movie_information.run_backimage_real_trace_ssi_matrix_pilot import load_source_rows
from declan.fig4_active_sensing.analyze_rr100_corrected_figure4_cache import render_with_common
from declan.fig4_active_sensing.audit_rr100_eye_trace_conditioning_and_nyquist_power import centered, corrected_crop_xy_deg, load_dset, model_aligned_indices
from declan.fixation_statistics_by_stimulus.image_features import backimage_trial_geometry, gaze_deg_to_screen_px
from declan.fixation_statistics_by_stimulus.run_backimage_trajectory_table_observer import _extract_patch
from declan.fixation_statistics_by_stimulus.run_backimage_twin_drift_geometry import _load_twin_common

ROOT=Path(__file__).resolve().parents[2]
LEGACY=ROOT/"outputs/active_sensing_movie_information/backimage_real_trace_ssi_matrix_large_contour_no_driftgate_ms200_n100x1000_v1/merged/image_feature_table.csv"
SOURCE=ROOT/"outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_image_structure_reviewed_v2_screenfiltered_yfix/backimage_image_fem_windows.csv"
OUT=ROOT/"outputs/fig4_active_sensing/rr100_legacy100_corrected_image_audit_checkpoint_24_v1"
METRICS=("rms_contrast","gradient_energy","orientation_coherence","contour_axis_deg","sf_centroid_cpd","high_sf_fraction")
PATCH_SIZE=540

def args():
 p=argparse.ArgumentParser(description=__doc__); p.add_argument("--legacy-table",type=Path,default=LEGACY);p.add_argument("--source-csv",type=Path,default=SOURCE);p.add_argument("--out-dir",type=Path,default=OUT);p.add_argument("--partials-dir",type=Path);p.add_argument("--session");p.add_argument("--assemble",action="store_true");p.add_argument("--dpi",type=int,default=190);return p.parse_args()
def _jsonable(value):
 if isinstance(value,np.generic): return value.item()
 if isinstance(value,dict): return {str(k):_jsonable(v) for k,v in value.items()}
 if isinstance(value,(list,tuple)): return [_jsonable(v) for v in value]
 return value
def atomic_json(path,payload):
 path.parent.mkdir(parents=True,exist_ok=True)
 with tempfile.NamedTemporaryFile("w",dir=path.parent,delete=False,encoding="utf-8") as h: json.dump(_jsonable(payload),h);h.write("\n");tmp=Path(h.name)
 os.replace(tmp,path)
def atomic_npz(path,**items):
 path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(".tmp.npz");np.savez_compressed(tmp,**items);os.replace(tmp,path)
def metrics(image,ppd):
 x=np.asarray(image,float); x=(x-x.mean())/max(x.std(),1e-12);gy,gx=np.gradient(x);mag=np.hypot(gx,gy);z=np.sum(mag**2*np.exp(2j*np.arctan2(gy,gx)))/max(np.sum(mag**2),1e-12)
 fft=np.fft.fftshift(np.fft.fft2(x*np.outer(np.hanning(x.shape[0]),np.hanning(x.shape[1]))));power=np.abs(fft)**2;fy=np.fft.fftshift(np.fft.fftfreq(x.shape[0],d=1/ppd));fx=np.fft.fftshift(np.fft.fftfreq(x.shape[1],d=1/ppd));sf=np.hypot(fx[None,:],fy[:,None]);ok=sf>0;den=max(float(power[ok].sum()),1e-12)
 return {"rms_contrast":float(np.std(image)/max(np.mean(image),1e-12)),"gradient_energy":float(np.mean(mag**2)),"orientation_coherence":float(abs(z)),"contour_axis_deg":float(np.degrees(np.angle(z)/2)%180),"sf_centroid_cpd":float((power[ok]*sf[ok]).sum()/den),"high_sf_fraction":float(power[(sf>=8)&ok].sum()/den)}
def axial(a,b): return ((np.asarray(b)-np.asarray(a)+90)%180)-90
def worker(a):
 if not a.session: raise ValueError("--session is required for a worker")
 partials=a.partials_dir or a.out_dir/"partials";legacy=pd.read_csv(a.legacy_table);source=load_source_rows(a.source_csv); work=legacy[legacy.session.eq(a.session)].sort_values("image_index")
 if work.empty: raise ValueError(f"No selected images in {a.session}")
 dset=load_dset(a.session,{});common=_load_twin_common();canvas={}
 for image in work.itertuples():
  src=source_row_by_id(source,int(image.source_row));indices=model_aligned_indices(int(src.global_start),int(src.global_stop));crop=corrected_crop_xy_deg(dset)[indices];trial=np.asarray(dset.covariates["trial_inds"]).reshape(-1);valid=np.asarray(dset.covariates["dpi_valid"]).reshape(-1).astype(bool)
  roi=np.asarray(dset.metadata["roi_src"],float);cyx=(roi[:,0]+roi[:,1]-1)/2;offset=np.array([cyx[1],-cyx[0]])/float(dset.metadata["ppd"]);center=crop.mean(0)+offset; corrected=src.copy();corrected["mean_x_deg"],corrected["mean_y_deg"]=map(float,center)
  old,oldmeta=_extract_patch(src,canvas_cache=canvas,patch_size_px=PATCH_SIZE);new,newmeta=_extract_patch(corrected,canvas_cache=canvas,patch_size_px=PATCH_SIZE);recon=render_with_common(np.asarray(new,np.float32),-centered(crop),ppd=float(newmeta["patch_ppd"]),common=common);exact_available=False;exact=np.full_like(recon,np.nan)
  try: exact=np.asarray(dset["stim"][indices],dtype=np.float32);exact_available=exact.shape==recon.shape and np.isfinite(exact).all()
  except (KeyError,IndexError,TypeError): pass
  geometry=backimage_trial_geometry(a.session,int(src.trial_idx));px=gaze_deg_to_screen_px(center,ppd=float(geometry["ppd"]),screen_shape=geometry["screen_shape"]);border=min(px[0],px[1],geometry["screen_width_px"]-px[0],geometry["screen_height_px"]-px[1]); crop_valid=bool(np.isfinite(crop).all() and np.all(valid[indices]) and np.all(trial[indices]==int(src.trial_idx)) and border>=PATCH_SIZE/2 and np.isfinite(recon).all())
  oldm,newm,exactm=metrics(old,float(oldmeta["patch_ppd"])),metrics(new,float(newmeta["patch_ppd"])),metrics(exact.mean(0),float(dset.metadata["ppd"])) if exact_available else {k:np.nan for k in METRICS}
  rec={"image_index":int(image.image_index),"source_row":int(image.source_row),"session":a.session,"trial_idx":int(image.trial_idx),"exact_saved_stim_available":exact_available,"corrected_crop_valid":crop_valid,"corrected_crop_border_px":float(border),"reconstruction_exact_pixel_r":float(np.corrcoef(recon.ravel(),exact.ravel())[0,1]) if exact_available else np.nan,"reconstruction_exact_mae":float(np.mean(abs(recon-exact))) if exact_available else np.nan,"legacy_corrected_patch_pixel_r":float(np.corrcoef(old.ravel(),new.ravel())[0,1]),"center_shift_deg":float(np.linalg.norm(center-np.array([src.mean_x_deg,src.mean_y_deg])))}
  for prefix,data in (("legacy",oldm),("corrected_reconstruction",newm),("exact_saved",exactm)): rec.update({f"{prefix}_{k}":v for k,v in data.items()})
  rec["reconstruction_exact_abs_contour_axis_delta_deg"]=float(abs(axial(newm["contour_axis_deg"],exactm["contour_axis_deg"]))) if exact_available else np.nan
  atomic_json(partials/f"image_{int(image.image_index):03d}.json",rec);atomic_npz(partials/f"image_{int(image.image_index):03d}.npz",legacy_patch=old,corrected_patch=new,reconstruction=recon,exact_saved=exact)
  print(f"{a.session}: image {image.image_index}",flush=True)
def assemble(a):
 partials=a.partials_dir or a.out_dir/"partials"; legacy=pd.read_csv(a.legacy_table).sort_values("image_index"); files=sorted(partials.glob("image_*.json")); records=[json.loads(x.read_text()) for x in files];frame=pd.DataFrame(records)
 if len(frame)!=100 or frame.image_index.nunique()!=100 or set(frame.image_index)!=set(legacy.image_index): raise RuntimeError(f"Refusing incomplete assembly: found {len(frame)} records for {frame.image_index.nunique() if len(frame) else 0} unique identities")
 if not frame.exact_saved_stim_available.all(): raise RuntimeError("Refusing assembly: not every identity has exact saved input")
 rows=[]
 for key in METRICS:
  x,y=frame[f"corrected_reconstruction_{key}"],frame[f"exact_saved_{key}"]
  if key=="contour_axis_deg": rows.append({"metric":key,"pixel_space":"reconstruction_vs_exact_51x51","pearson_r_axial_cos2":float(stats.pearsonr(np.cos(np.deg2rad(2*x)),np.cos(np.deg2rad(2*y))).statistic),"spearman_rho":np.nan,"median_abs_difference":float(np.median(abs(axial(x,y))) )})
  else: rows.append({"metric":key,"pixel_space":"reconstruction_vs_exact_51x51","pearson_r_axial_cos2":float(stats.pearsonr(x,y).statistic),"spearman_rho":float(stats.spearmanr(x,y).statistic),"median_abs_difference":float(np.median(abs(x-y)))})
 agreement=pd.DataFrame(rows); frame.to_csv(a.out_dir/"corrected_image_crosswalk.csv",index=False);agreement.to_csv(a.out_dir/"reconstruction_exact_descriptor_agreement.csv",index=False)
 chosen=pd.concat([frame.nlargest(1,"reconstruction_exact_pixel_r"),frame.nsmallest(1,"reconstruction_exact_pixel_r"),frame.nlargest(1,"reconstruction_exact_mae")]).drop_duplicates("image_index");chosen.to_csv(a.out_dir/"selected_image_examples.csv",index=False)
 fig,axes=plt.subplots(len(chosen),3,figsize=(9,2.5*len(chosen)),constrained_layout=True)
 if len(chosen)==1: axes=np.array([axes])
 for axrow,row in zip(axes,chosen.itertuples()):
  data=np.load(partials/f"image_{row.image_index:03d}.npz");
  for ax,img,title in zip(axrow,(data["corrected_patch"],data["reconstruction"].mean(0),data["exact_saved"].mean(0)),("corrected RF-centred source patch","corrected reconstruction (51x51)","exact saved input (51x51)")): ax.imshow(img,cmap="gray",vmin=0,vmax=255);ax.set_axis_off();ax.set_title(title,fontsize=8)
  axrow[0].set_ylabel(f"image {row.image_index}\npixel r={row.reconstruction_exact_pixel_r:.3f}",fontsize=8)
 fig.suptitle("Phase 2: corrected reconstruction validated against exact saved model input",fontweight="bold");fig.savefig(a.out_dir/"corrected_image_examples.png",dpi=a.dpi);plt.close(fig)
 manifest={"created_utc":datetime.now(timezone.utc).isoformat(),"status":"phase_2_image_identity_crop_audit_complete","counts":{"n_images":100,"n_unique_successful_identities":100,"n_exact_saved_stim_available":100,"n_corrected_crop_valid":int(frame.corrected_crop_valid.sum())},"worker_contract":"one process per session; atomic per-image JSON/NPZ records","validation_contract":"corrected reconstruction versus exact saved 51x51 BackImage input","guardrail":"No legacy neural response is recalibrated; only descriptors passing this exact-input validation may be considered for structural stratification."};atomic_json(a.out_dir/"manifest.json",manifest);print(json.dumps(manifest,indent=2))
def main():
 a=args();
 if a.assemble: assemble(a)
 else: worker(a)
if __name__=="__main__": main()
