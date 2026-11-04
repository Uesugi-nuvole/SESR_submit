"""Compare two dense raw tensors, extending the v7 board readback comparator."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def read_tensor(path, dtype, shape):
    expected = math.prod(shape) * dtype.itemsize
    if path.stat().st_size != expected:
        raise ValueError(f"{path}: expected {expected} bytes, found {path.stat().st_size}")
    data = path.read_bytes()
    return np.frombuffer(data, dtype), hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("actual", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("--shape", required=True, type=int, nargs="+")
    parser.add_argument("--dtype", required=True, choices="float16 float32 int8 uint8 int16 uint16 int32 uint32".split())
    for side in ("actual", "reference"):
        parser.add_argument(f"--{side}-endian", required=True, choices=("little", "big"))
    parser.add_argument("--atol", type=float, default=0)
    parser.add_argument("--rtol", type=float, default=0)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    if min(args.shape) < 1 or not all(math.isfinite(t) and t >= 0 for t in (args.atol, args.rtol)):
        parser.error("dimensions must be positive and tolerances finite and non-negative")
    if args.report.resolve() in (args.actual.resolve(), args.reference.resolve()):
        parser.error("report path must differ from input paths")
    report = dict(shape=args.shape, dtype=args.dtype, atol=args.atol, rtol=args.rtol)
    try:
        arrays = []
        for side in ("actual", "reference"):
            path = getattr(args, side)
            endian = getattr(args, side + "_endian")
            dtype = np.dtype(args.dtype).newbyteorder("<" if endian == "little" else ">")
            values, digest = read_tensor(path, dtype, args.shape)
            report[side] = dict(file=path.name, endian=endian, sha256=digest,
                                nonfinite=int(np.count_nonzero(~np.isfinite(values))))
            arrays.append(values.astype(np.float64))
        actual, reference = arrays
        valid = np.isfinite(actual) & np.isfinite(reference)
        error = np.abs(actual[valid] - reference[valid])
        bad = ~valid
        bad[valid] = error > args.atol + args.rtol * np.abs(reference[valid])
        count = int(bad.sum())
        first = int(np.argmax(bad)) if count else None
        report.update(status="mismatch" if count else "pass", elements=actual.size,
                      outside_tolerance=count, mae=float(error.mean()) if valid.all() else None,
                      max_abs_diff=float(error.max()) if valid.all() else None,
                      first_bad_index=list(map(int,np.unravel_index(first,args.shape))) if count else None)
        code = int(bool(count))
    except (OSError, ValueError) as exc:
        report.update(status="invalid_input", error=str(exc))
        code = 2
    result = json.dumps(report, indent=2, allow_nan=False)
    args.report.write_text(result + "\n", encoding="utf-8")
    print(result)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
