from pathlib import Path
import subprocess,sys
import argparse
parser=argparse.ArgumentParser()
parser.add_argument('--compiler',required=True,help='mb-g++ executable from the selected Vitis installation')
parser.add_argument('--bsp',required=True,help='Directory containing include/xparameters.h and lib/libxil.a')
parser.add_argument('--out',default='rebuild_output')
args=parser.parse_args()
root=Path(__file__).resolve().parent
src=root/'source/sesr_accel/src'
out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=True)
bsp=Path(args.bsp).resolve();cc=str(Path(args.compiler).resolve())
for required in [Path(cc),src/'main.cpp',bsp/'include/xparameters.h',bsp/'lib/libxil.a']:
 if not required.is_file():raise SystemExit('Missing: '+str(required))
subprocess.run([sys.executable,str(root/'check_bsp.py'),str(bsp)],check=True)
(out/'toolchain.txt').write_text(subprocess.check_output([cc,'--version']).decode(errors='replace'))
flags=['-std=c++11','-O2','-ffp-contract=off','-fno-fast-math','-mlittle-endian','-mcpu=v11.0','-mhard-float','-ffunction-sections','-fdata-sections','-I'+str(src),'-I'+str(bsp/'include')]
objects=[]
with (out/'build.log').open('w') as log:
 for f in src.rglob('*.cpp'):
  obj=out/(str(f.relative_to(src)).replace('\\','_').replace('/','_')+'.o')
  print('COMPILE',f.relative_to(src),flush=True)
  r=subprocess.run([cc,*flags,'-c',str(f),'-o',str(obj)],stdout=log,stderr=log)
  if r.returncode:raise SystemExit('Compile failed; see '+str(out/'build.log'))
  objects.append(str(obj))
 r=subprocess.run([cc,*flags,*objects,'-Wl,--gc-sections','-Wl,-Map='+str(out/'completion.map'),'-T'+str(src/'lscript.ld'),'-L'+str(bsp/'lib'),'-Wl,--start-group','-lxil','-lc','-lm','-lgcc','-Wl,--end-group','-o',str(out/'completion.elf')],stdout=log,stderr=log)
 print('LINK status',r.returncode)
 if r.returncode:raise SystemExit(r.returncode)
