from pathlib import Path
import hashlib,json
r=Path(__file__).resolve().parents[1]
m=json.loads((r/'provenance/package_hashes.json').read_text(encoding='utf-8'))
bad=[n for n,h in m.items() if not (r/n).is_file() or hashlib.sha256((r/n).read_bytes()).hexdigest()!=h]
print('PASS' if not bad else 'FAIL',len(m),'files',bad)
raise SystemExit(bool(bad))
