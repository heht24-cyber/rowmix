"""Measure peak Linux RSS of separate calibrator processes on identical Digits inputs."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from time import perf_counter

import numpy as np

from rowmix.compiler import Calibration
from rowmix.graph import Graph

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "examples" / "digits"


def worker(mode, samples_path, output):
    import resource
    graph = Graph.load(MODEL / "float.npz")
    samples = np.load(samples_path, mmap_mode="r", allow_pickle=False)
    start = perf_counter()
    calibration = Calibration(graph, samples, batch_size=8 if mode == "stream" else None)
    blob = calibration.fit(8726).to_bytes()
    output.write_bytes(blob)
    return dict(seconds=round(perf_counter() - start, 4),
                peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                model_bytes=len(blob))


def experiment():
    if sys.platform != "linux":
        raise SystemExit("this experiment records Linux ru_maxrss in KiB")
    records = []
    env = {**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
    with tempfile.TemporaryDirectory() as directory:
        tmp = Path(directory)
        data = np.load(MODEL / "calibration.npy", allow_pickle=False)
        for repeat in (1, 4):
            sample_path = tmp / "cal.npy"
            np.save(sample_path, np.tile(data, (repeat, 1)))
            for mode in ("full", "stream"):
                model = tmp / f"{repeat}-{mode}.rmx"
                command = [sys.executable, "-m", "benchmarks.calibration_memory",
                           "--worker", mode, str(sample_path), str(model)]
                result = json.loads(subprocess.check_output(command, cwd=ROOT, env=env))
                records.append(dict(repeat=repeat, calibration_rows=len(data)*repeat,
                                    mode=mode, **result))
        # Load the test split in the driver only after all isolated RSS measurements.
        from sklearn.datasets import load_digits
        from .train import split
        (inputs, labels, _) = split(*load_digits(return_X_y=True))
        x, y = inputs[2], labels[2]
        float_top1 = Graph.load(MODEL / "float.npz").trace(x)[-1].reshape(len(y), -1).argmax(1)
        x.astype("<f4").tofile(tmp / "test.f32")
        for record in records:
            model = tmp / f"{record['repeat']}-{record['mode']}.rmx"
            subprocess.run([ROOT / "build/rowmix-run", "run", model, tmp / "test.f32", tmp / "out"],
                           check=True, env=env)
            top1 = np.fromfile(tmp / "out", dtype="<f4").reshape(len(y), -1).argmax(1)
            record.update(test=len(y), correct=int((top1 == y).sum()),
                          matches_float=int((top1 == float_top1).sum()))
    return dict(platform=sys.platform, numpy=np.__version__, calibration_batch=8, activation_samples=65536,
                max_patches_per_group=8192, budget_bytes=8726, results=records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", nargs=3, metavar=("MODE", "SAMPLES", "MODEL"))
    parser.add_argument("--output", type=Path, default=ROOT / "benchmarks/calibration-memory.json")
    args = parser.parse_args()
    if args.worker:
        mode, path, target = args.worker
        print(json.dumps(worker(mode, path, Path(target))))
    else:
        report = experiment()
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report))
