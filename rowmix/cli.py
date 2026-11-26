import argparse
import json
import os
from pathlib import Path
import tempfile

import numpy as np
from .compiler import Calibration, diagnose
from .graph import Graph


def atomic_write(path, payload):
    target = Path(path)
    with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as handle:
        tmp = Path(handle.name)
    try:
        tmp.write_bytes(payload)
        os.replace(tmp, target)
    finally:
        tmp.unlink(missing_ok=True)


def main():
    p = argparse.ArgumentParser(description="Compile a RowMix graph for the standalone CPU runtime")
    p.add_argument("graph", help="float graph .npz (neutral interchange, no pickle)")
    p.add_argument("calibration", help="representative .npy input batch")
    p.add_argument("output", help="destination .rmx file")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--bytes", type=int, help="hard limit for the complete deployed file")
    mode.add_argument("--fixed", type=int, choices=(4, 8))
    p.add_argument("--audit", help="independent .npy batch for per-layer error measurements")
    p.add_argument("--calibration-batch", type=int, help="maximum samples per float forward pass (streaming calibration)")
    args = p.parse_args()
    try:
        g = Graph.load(args.graph)
        calibration = Calibration(g, np.load(args.calibration, mmap_mode="r", allow_pickle=False),
                                  batch_size=args.calibration_batch)
        m = calibration.fit(args.bytes) if args.bytes is not None else calibration.fixed(args.fixed)
        blob = m.to_bytes()
        info = dict(file_bytes=len(blob), arena_bytes=m.arena, live_lower_bound=m.lower,
                    min_file_bytes=calibration.minimum, max_file_bytes=calibration.maximum)
        if args.audit:
            info["layers"] = diagnose(m, np.load(args.audit, allow_pickle=False))
        atomic_write(args.output, blob)
        print(json.dumps(info, indent=2, allow_nan=False))
    except (ValueError, OSError, KeyError, OverflowError) as exc:
        p.error(str(exc))


if __name__ == "__main__":
    main()
