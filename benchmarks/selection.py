"""Demonstrate the native release selector under the same gate on two datasets."""
import json
from sklearn.datasets import load_digits,load_wine
from rowmix.graph import Graph
from rowmix.select import select
from .run import ROOT
from .train import split


def main():
    results={}
    for name,data in (("digits",load_digits),("wine",load_wine)):
        x,_,_=split(*data(return_X_y=True))
        _,report=select(Graph.load(ROOT/"examples"/name/"float.npz"),x[1],x[2],ROOT/"build/rowmix-run",
                        confidence=.95,agreement=.95,profiles=5,passes=5)
        results[name]=report
        print(name,report["selected_sha256"],[(r["matches"],round(r["lower_agreement"],4)) for r in report["candidates"]])
    (ROOT/"benchmarks/selection.json").write_text(json.dumps(results,indent=2)+"\n")


if __name__=="__main__": main()
