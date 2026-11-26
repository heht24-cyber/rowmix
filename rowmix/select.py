"""Select a native classifier by file size, measured speed, and held-out agreement."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np
from scipy.special import betaincinv
from .compiler import Calibration
from .cli import atomic_write
from .graph import Graph


def lower_bound(matches, n, confidence, candidates):
    if not 0 <= matches <= n or n < 1 or not 0 < confidence < 1 or candidates < 1:
        raise ValueError("invalid confidence calculation")
    if not matches:
        return 0.
    return float(betaincinv(matches,n-matches+1,(1-confidence)/candidates))


def select(graph, calibration, validation, runner, agreement=.95, confidence=.95,
           max_bytes=None, max_arena=None, max_us=None, profiles=5, passes=5, calibration_batch=None):
    if not 0<=agreement<=1 or not 0<confidence<1 or not 2<=profiles<=32 or not 1<=passes<=10000:
        raise ValueError("invalid selection parameters")
    if any(v is not None and (not np.isfinite(v) or v<0) for v in (max_bytes,max_arena,max_us)):
        raise ValueError("resource limits must be finite and nonnegative")
    h,w,classes=graph.nodes[-1].shape
    if (h,w)!=(1,1) or classes<2:
        raise ValueError("selection expects a classifier with one logit vector per sample")
    cal=Calibration(graph,calibration,batch_size=calibration_batch)
    # Generate and deduplicate the whole candidate family before evaluating validation samples.
    family={}
    for i in range(profiles):
        cap=cal.minimum+(cal.maximum-cal.minimum)*i//(profiles-1)
        model=cal.fixed(4) if i==0 else cal.fixed(8) if i==profiles-1 else cal.fit(cap)
        blob=model.to_bytes(); sha=hashlib.sha256(blob).hexdigest()
        family.setdefault(sha, (blob,cap,model.arena))
    x=np.ascontiguousarray(validation,dtype="<f4")
    refs=graph.trace(x)[-1].reshape(len(x),classes).argmax(1)
    rows=[]
    with tempfile.TemporaryDirectory() as tmp:
        tmp=Path(tmp); x.tofile(tmp/"validation.f32")
        for sha,(blob,cap,arena) in family.items():
            (tmp/"model").write_bytes(blob)
            def call(*args):
                p=subprocess.run([os.fspath(runner),*map(str,args)],capture_output=True)
                if p.returncode: raise ValueError(p.stderr.decode(errors="replace").strip())
                return p.stdout
            call("run",tmp/"model",tmp/"validation.f32",tmp/"out")
            y=np.fromfile(tmp/"out",dtype="<f4").reshape(len(x),classes)
            matches=int(np.sum(y.argmax(1)==refs))
            lower=lower_bound(matches,len(x),confidence,len(family))
            timing=json.loads(call("bench",tmp/"model",tmp/"validation.f32",passes))
            failures=[]
            if max_bytes is not None and len(blob)>max_bytes: failures.append("model_bytes")
            if max_arena is not None and arena>max_arena: failures.append("arena_bytes")
            if max_us is not None and timing["p95_pass_mean_us"]>max_us: failures.append("latency")
            if lower<agreement: failures.append("agreement_evidence")
            rows.append(dict(sha256=sha,budget_bytes=cap,model_bytes=len(blob),arena_bytes=arena,
                             matches=matches,observations=len(x),lower_agreement=lower,
                             failed=failures,**timing))
    passing=sorted((r for r in rows if not r["failed"]),key=lambda r:(r["model_bytes"],-r["lower_agreement"],r["sha256"]))
    chosen=passing[0]["sha256"] if passing else None
    report=dict(confidence=confidence,minimum_agreement=agreement,unique_candidates=len(family),
                per_candidate_alpha=(1-confidence)/len(family),selected_sha256=chosen,candidates=rows)
    return ((family[chosen][0] if chosen else None),report)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("graph");p.add_argument("calibration");p.add_argument("validation")
    p.add_argument("--runner",default="rowmix-run")
    p.add_argument("--output",required=True);p.add_argument("--report",required=True)
    p.add_argument("--min-agreement",type=float,default=.95);p.add_argument("--confidence",type=float,default=.95)
    p.add_argument("--max-bytes",type=int);p.add_argument("--max-arena",type=int);p.add_argument("--max-pass-mean-us",type=float)
    p.add_argument("--profiles",type=int,default=5);p.add_argument("--passes",type=int,default=5)
    p.add_argument("--calibration-batch", type=int, help="maximum samples per calibration forward pass")
    args=p.parse_args()
    try:
        paths=[Path(v).resolve() for v in (args.graph,args.calibration,args.validation,args.output,args.report)]
        if paths[3]==paths[4] or any(v in paths[:3] for v in paths[3:]): raise ValueError("outputs must be distinct from each other and inputs")
        blob,report=select(Graph.load(args.graph),np.load(args.calibration,mmap_mode="r",allow_pickle=False),
            np.load(args.validation,allow_pickle=False),args.runner,args.min_agreement,args.confidence,
            args.max_bytes,args.max_arena,args.max_pass_mean_us,args.profiles,args.passes,args.calibration_batch)
        atomic_write(args.report,(json.dumps(report,indent=2,allow_nan=False)+"\n").encode())
        if blob is None: p.exit(3,"no candidate met the specified limits; see report\n")
        atomic_write(args.output,blob)
        print(f"selected {report['selected_sha256']} ({len(blob)} bytes)")
    except (ValueError,OSError,KeyError) as e: p.error(str(e))


if __name__=="__main__": main()
