"""Train or load float models, compile under budgets, replay native traces, measure time."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import tempfile

import numpy as np
import sklearn
from rowmix.compiler import Calibration, diagnose
from rowmix.graph import Graph
from .train import dataset, split

ROOT = Path(__file__).resolve().parents[1]


def invoke(*args, kernel="auto"):
    env = dict(os.environ, ROWMIX_KERNEL=kernel)
    return subprocess.run([str(ROOT / "build/rowmix-run"), *map(str,args)], capture_output=True, check=True, env=env).stdout


def matched_precision_comparison(calibration, mixed, x, truth):
    """Hold each layer's upgrade count fixed; change only which channels receive eight bits."""
    counts = [sum(r.bits==8 for r in rows) for rows in mixed.rows]
    results = []
    for seed in range(-1, 8):
        rng = np.random.default_rng(max(0,seed))
        other = calibration.fixed(4)
        for i,n in enumerate(calibration.graph.nodes):
            if n.weights is None:
                continue
            if seed==-1:
                score = [np.mean((w-lo.q*lo.scale)**2)-np.mean((w-hi.q*hi.scale)**2)
                         for w,(lo,hi) in zip(n.weights,calibration.options[i])]
                order = np.argsort(-np.asarray(score),kind="stable")
            else:
                order = rng.permutation(len(n.bias))
            for j in order[:counts[i]]:
                other.rows[i][j] = calibration.options[i][j][1]
        assert len(other.to_bytes())==len(mixed.to_bytes())
        pred = other.trace(x)[-1].reshape(len(x),-1).argmax(axis=1)
        results.append(int(np.sum(pred==truth)))
    return dict(weight_mse_correct=results[0], random_rows_correct=results[1:])


def digest(a):
    return hashlib.sha256(np.asarray(a, dtype="<i8").tobytes()).hexdigest()


def evaluate(name, reuse, passes):
    target = ROOT / "examples" / name
    target.mkdir(parents=True, exist_ok=True)
    path = target / "float.npz"
    if reuse:
        from sklearn.datasets import load_digits, load_wine
        data = load_digits if name == "digits" else load_wine
        sets, labels, ids = split(*data(return_X_y=True))
        graph = Graph.load(path)
    else:
        graph, sets, labels, ids = dataset(name)
        graph.save(path)
    x, truth = sets[2], labels[2]
    gold = graph.trace(x)[-1].reshape(len(x), -1)
    calibration = Calibration(graph, sets[1])
    report = dict(name=name, train=len(sets[0]), calibration=len(sets[1]), test=len(x),
                  split_sha256=[digest(i) for i in ids], float_correct=int(np.sum(gold.argmax(1)==truth)),
                  packed_fp32_parameter_bytes=sum((n.weights.size+n.bias.size)*4 for n in graph.nodes if n.weights is not None),
                  minimum_bytes=calibration.minimum, maximum_bytes=calibration.maximum, profiles=[])
    # Input and expected labels are from the held-out split. The compiler only sees sets[1].
    np.save(target / "calibration.npy", sets[1])
    x[:8].astype("<f4").tofile(target / "sample.f32")
    (target / "sample-labels.json").write_text(json.dumps(truth[:8].tolist())+"\n")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        test_in = tmp / "test.f32"; x.astype("<f4").tofile(test_in)
        for pct in (0,25,50,75,100):
            cap = calibration.minimum + (calibration.maximum-calibration.minimum)*pct//100
            model = calibration.fixed(4) if pct==0 else calibration.fixed(8) if pct==100 else calibration.fit(cap)
            blob = model.to_bytes()
            mpath = (target / "mixed.rmx") if pct==50 else (tmp / f"p{pct}.rmx")
            mpath.write_bytes(blob)
            invoke("run", mpath, test_in, tmp/"logits")
            scores = np.fromfile(tmp/"logits", dtype="<f4").reshape(gold.shape)
            refs = model.trace(x)
            np.testing.assert_array_equal(scores, (refs[-1].reshape(scores.shape)*model.scales[-1]).astype(np.float32))
            invoke("trace", mpath, test_in, tmp/"trace")
            native = np.fromfile(tmp/"trace",dtype=np.int8)
            expected = np.concatenate([a.reshape(len(x),-1) for a in refs],axis=1).ravel()
            np.testing.assert_array_equal(native, expected)
            invoke("trace", mpath, test_in, tmp/"scalar-trace",kernel="scalar")
            np.testing.assert_array_equal(np.fromfile(tmp/"scalar-trace",dtype=np.int8),expected)
            bench = json.loads(invoke("bench", mpath, test_in, passes))
            scalar = json.loads(invoke("bench", mpath, test_in, passes,kernel="scalar"))
            report["profiles"].append(dict(percent=pct, budget_bytes=cap, file_bytes=len(blob),
                arena_bytes=model.arena, arena_lower_bound=model.lower,
                rows4=sum(r.bits==4 for layer in model.rows for r in layer),
                rows8=sum(r.bits==8 for layer in model.rows for r in layer),
                correct=int(np.sum(scores.argmax(1)==truth)),
                agree_float=int(np.sum(scores.argmax(1)==gold.argmax(1))),
                native_trace_values=int(expected.size), native_logit_values=int(scores.size),
                scalar_median_us=scalar["median_us_per_sample"], **bench))
            if pct==50:
                report["mixed_layer_errors"]=diagnose(model,x)
                report["mixed_sha256"]=hashlib.sha256(blob).hexdigest()
                report["matched_layer_counts_ablation"]=matched_precision_comparison(calibration,model,x,truth)
                scores[:8].astype("<f4").tofile(target/"expected.f32")
    print(f"{name}: {report['float_correct']}/{len(x)} float | bytes, correct, median us: " +
          str([(v['file_bytes'],v['correct'],v['median_us_per_sample']) for v in report['profiles']]), flush=True)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reuse", action="store_true", help="use supplied trained graphs rather than retraining")
    p.add_argument("--passes",type=int,default=9)
    p.add_argument("--output",type=Path,default=ROOT/"benchmarks/results.json")
    args=p.parse_args()
    if not 1<=args.passes<=10000:
        p.error("passes must be in 1..10000")
    cpu=platform.processor() or "unknown"
    if Path("/proc/cpuinfo").exists():
        cpu=next((line.partition(":")[2].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                  if line.startswith("model name")),cpu)
    info=dict(numpy=np.__version__,sklearn=sklearn.__version__,python=platform.python_version(),cpu=cpu,
              machine=platform.machine(),system=platform.system(),
              compiler=subprocess.check_output(["c++","--version"],text=True).splitlines()[0],
              models=[evaluate(name,args.reuse,args.passes) for name in ("digits","wine")])
    args.output.write_text(json.dumps(info,indent=2,allow_nan=False)+"\n")


if __name__=="__main__":
    main()
