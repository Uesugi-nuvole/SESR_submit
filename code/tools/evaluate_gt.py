"""Fixed-crop application evaluation; not an official KITTI benchmark submission."""
from pathlib import Path
import hashlib,json,csv
import numpy as np
root=Path(__file__).resolve().parents[1]
pkg=root/'deployment/v7'
z=np.load(pkg/'sample/input.npz')
gt=z['ground_truth'].astype(np.float64)
mask=z['mask']
assert set(np.unique(mask)).issubset({0,1})
assert np.array_equal(mask>0,z['input_disp']>0), 'Check input mask semantics'
valid=np.isfinite(gt)&(gt>0)
regions={'all_gt_valid':valid,'original_holes_gt_valid':valid&(mask==0),'original_known_gt_valid':valid&(mask>0)}
files={'sgm':pkg/'sample/input.npz','software_pytorch':pkg/'reference/original_pytorch_final_le_f32.bin','software_deployment':pkg/'reference/deployment_postscale_final_le_f32.bin','board_run1':root/'evidence/run1/board_output_run1.bin'}
preds={name:(z['input_disp'] if name=='sgm' else np.fromfile(p,dtype='<f4').reshape(gt.shape)).astype(np.float64) for name,p in files.items()}
rows=[]
for region,sel in regions.items():
 for name,pred in preds.items():
  p=pred[sel]; g=gt[sel]; e=np.abs(p-g); finite=np.isfinite(p); positive=finite&(p>0)
  rows.append(dict(region=region,method=name,pixels=int(sel.sum()),mae_px=float(e.mean()) if finite.all() else None,rmse_px=float(np.sqrt(np.mean(e**2))) if finite.all() else None,d1_style_pct=float(100*np.mean((~finite)|((e>3)&(e>0.05*g)))),bad3_pct=float(100*np.mean((~finite)|(e>3))),positive_prediction_pct=float(100*positive.mean())))
out=root/'evaluation_output'
out.mkdir(exist_ok=True)
report={'sample':'000082_10','crop_xyxy':[448,128,576,256],'scope':'One 128x128 crop, not official KITTI score. Same GT-valid pixels for all methods; no per-method masking. SGM missing disparity remains zero and is scored as missing/zero, not excluded. D1-style uses >3px AND >5%; no official interpolation or leaderboard aggregation.', 'mask_verified_equals_input_positive':True,'gt_valid_pixels':int(valid.sum()),'total_pixels':int(gt.size),'input_holes_total':int((mask==0).sum()),'sha256':{n:hashlib.sha256(p.read_bytes()).hexdigest() for n,p in files.items()},'rows':rows}
(out/'run1_gt_metrics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
with (out/'run1_gt_metrics.csv').open('w',newline='',encoding='utf-8-sig') as f:
 w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
for row in rows: print(json.dumps(row))
