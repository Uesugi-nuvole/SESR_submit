"""Compare a board dump against an explicitly selected reference."""
import argparse,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser()
p.add_argument('dump',type=Path)
p.add_argument('reference',type=Path)
p.add_argument('--dump-endian',required=True,choices=['little','big'])
p.add_argument('--report',type=Path,default=Path('board_comparison.json'))
a=p.parse_args()
x=np.fromfile(str(a.dump),dtype='<f4' if a.dump_endian=='little' else '>f4')
y=np.fromfile(str(a.reference),dtype='<f4')
if x.size!=16384 or y.size!=16384:raise ValueError('Expected 128x128 float32, 65536 bytes per file')
if a.dump.stat().st_size!=65536 or a.reference.stat().st_size!=65536:raise ValueError('Unexpected byte length')
diff=np.abs(x.astype(np.float64)-y.astype(np.float64))
finite=bool(np.isfinite(x).all() and np.isfinite(y).all())
report={'dump':str(a.dump),'reference':str(a.reference),'finite':finite,
 'mae':float(diff.mean()) if finite else None,'max_abs_diff':float(diff.max()) if finite else None,
 'different_values':int(np.count_nonzero(x!=y)),
 'outside_tolerance':int(np.count_nonzero(~np.isfinite(x)|~np.isfinite(y)|(diff>1e-5+1e-4*np.abs(y))))}
a.report.write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
raise SystemExit(0 if finite and report['outside_tolerance']==0 else 1)
