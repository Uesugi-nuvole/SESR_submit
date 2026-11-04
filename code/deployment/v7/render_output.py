"""Render a board dump; visual inspection does not replace numeric comparison."""
from pathlib import Path
import argparse,numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('dump',type=Path);p.add_argument('--dump-endian',required=True,choices=['little','big']);p.add_argument('--out',type=Path,default=Path('board_comparison.png'));p.add_argument('--label',default='Board output');a=p.parse_args()
r=Path(__file__).resolve().parent
if a.dump.stat().st_size!=65536:raise ValueError('Expected exactly 65536 bytes')
b=np.fromfile(a.dump,dtype='<f4' if a.dump_endian=='little' else '>f4').reshape(128,128)
if not np.isfinite(b).all():raise ValueError('Nonfinite output; inspect numeric failure before plotting')
z=np.load(r/'sample/input.npz');ref=np.fromfile(r/'reference/deployment_postscale_final_le_f32.bin',dtype='<f4').reshape(128,128)
vmax=float(np.percentile(z['ground_truth'][z['ground_truth']>0],99))
fig,axs=plt.subplots(1,4,figsize=(12,3.5));axs[0].imshow(z['rgb'].astype(float));axs[0].set_title('RGB 000082 crop')
for ax,data,title in zip(axs[1:],[z['input_disp'],b,ref],['Original SGM input',a.label,'Integer reference']):
 ax.imshow(np.ma.masked_where(data<=0,data),vmin=0,vmax=vmax,cmap='turbo');ax.set_title(title)
for ax in axs:ax.axis('off')
fig.suptitle('Shared disparity color scale; use numeric report for acceptance')
fig.tight_layout();fig.savefig(a.out,dpi=150);plt.close(fig);print('Saved',a.out)
