from pathlib import Path
import re,json,sys
p=Path(sys.argv[1])/'include/xparameters.h'
defs=dict(re.findall(r'^#define\s+(\w+)\s+([^\r\n]+)',p.read_text(errors='replace'),re.M))
def value(name,seen=()):
 if name in seen:raise ValueError('Macro cycle')
 v=defs[name].split('//')[0].strip().strip('()')
 if v in defs:return value(v,seen+(name,))
 return int(v.rstrip('uUlL'),0)
expected={'XPAR_CPU_CORE_CLOCK_FREQ_HZ':100000000,'XPAR_MICROBLAZE_0_ENDIANNESS':1,'XPAR_MICROBLAZE_0_USE_FPU':2,'XPAR_TCP_0_BASEADDR':0x10000,'XPAR_AXI_INTC_0_BASEADDR':0x41200000,'XPAR_DDR4_0_C0_DDR4_MEMORY_MAP_BASEADDR':0x80000000,'STDOUT_BASEADDRESS':0x40600000}
errors=[]
for n,want in expected.items():
 try:
  got=value(n)
  if got!=want:errors.append(n+': expected '+hex(want)+' got '+hex(got))
 except (KeyError,ValueError):errors.append(n+': missing or unsupported macro form; inspect BSP, do not bypass silently')
if errors:raise SystemExit('BSP MISMATCH\n'+'\n'.join(errors))
print('PASS expected BSP CPU configuration and peripheral bases; does not prove physical hardware match')
