import itertools
from pathlib import Path
import tempfile
import unittest

import numpy as np
from rowmix.arena import layout
from rowmix.compiler import Calibration, choose_counts, pack4, quantize
from rowmix.graph import Graph, from_sklearn


def toy(seed=3, channels=7):
    rng=np.random.default_rng(seed)
    g=Graph((5,7,channels))
    def affine(op, parent, no, width, **kw):
        return g.add(op,[parent],weights=rng.normal(0,.22,(no,width)),bias=rng.normal(0,.1,no),**kw)
    a=affine("conv",0,9,channels*9,kernel=3,pad=1,relu=True)
    b=affine("conv",a,9,81,kernel=3,pad=1)
    c=g.add("add",[a,b],relu=True)
    c=g.add("maxpool",[c],kernel=2,stride=2)
    c=affine("conv",c,5,9,kernel=1,stride=2)
    affine("dense",c,3,g.nodes[c].count)
    return g,rng.normal(size=(12,5,7,channels)).astype(np.float32)


class CompilerTests(unittest.TestCase):
    def test_integer_rounding_and_nibbles(self):
        self.assertEqual(quantize(np.array([-300.,-1.5,-.5,0,.5,1.5,300.]),1).tolist(),[-127,-2,-1,0,1,2,127])
        self.assertEqual(pack4(np.array([-7,7,-1],dtype=np.int8)),bytes([0x79,0x0f]))

    def test_knapsack_against_exhaustion(self):
        rng=np.random.default_rng(2)
        for _ in range(35):
            groups=[(int(rng.integers(0,9)),sorted(rng.random(int(rng.integers(1,7))),reverse=True)) for _ in range(4)]
            for budget in (0,11,29,47):
                counts=choose_counts(groups,budget)
                score=lambda ns: sum(sum(g[:n]) for n,(_,g) in zip(ns,groups))
                cost=lambda ns: sum(n*p for n,(p,_) in zip(ns,groups))
                valid=(v for v in itertools.product(*(range(len(g)+1) for _,g in groups)) if cost(v)<=budget)
                self.assertLessEqual(cost(counts),budget)
                self.assertAlmostEqual(score(counts),max(map(score,valid)),places=10)

    def test_round_trip_and_budget(self):
        g,x=toy()
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/"g.npz"; g.save(path); copy=Graph.load(path)
            for a,b in zip(g.trace(x),copy.trace(x)):
                np.testing.assert_array_equal(a,b)
        cal=Calibration(g,x)
        for cap in np.linspace(cal.minimum,cal.maximum,9).astype(int):
            self.assertLessEqual(len(cal.fit(int(cap)).to_bytes()),cap)
        with self.assertRaisesRegex(ValueError,"at least"):
            cal.fit(cal.minimum-1)
        with self.assertRaises(ValueError):
            g.trace(np.array([[np.nan]]))

    def test_arena_does_not_overlap_live_parents(self):
        g,_=toy(); offsets,peak,lower=layout(g.nodes)
        last=list(range(len(g.nodes)))
        for i,n in enumerate(g.nodes):
            for p in n.parents: last[p]=i
        for i,n in enumerate(g.nodes):
            spans=sorted((offsets[j],offsets[j]+g.nodes[j].count) for j in range(i+1) if last[j]>=i)
            self.assertTrue(all(a[1]<=b[0] for a,b in zip(spans,spans[1:])))
        self.assertGreaterEqual(peak,lower)
        self.assertLess(peak,sum(n.count for n in g.nodes))

    def test_binary_import_and_preprocessing(self):
        from sklearn.neural_network import MLPClassifier
        rng=np.random.default_rng(9)
        x=rng.normal(size=(100,4)).astype(np.float32)
        y=np.where(x[:,0]+x[:,1]>.2,"yes","no")
        z=(x-x.mean(0))/x.std(0)
        net=MLPClassifier(hidden_layer_sizes=(7,),solver="lbfgs",random_state=7,max_iter=500).fit(z,y)
        g=from_sklearn(net); g.set_preprocess(x.mean(0),1/x.std(0))
        result=g.trace(x)[-1].reshape(len(x),2)
        np.testing.assert_array_equal(net.classes_[result.argmax(1)],net.predict(z))
        self.assertTrue(np.all(result[:,0]==0))

    def test_zero_weights_and_free_upgrades(self):
        g=Graph((1,1,1))
        g.add("dense",[0],weights=np.array([[0.],[.8]]),bias=[0.,1.])
        cal=Calibration(g,np.zeros((3,1)))
        self.assertEqual(cal.minimum,cal.maximum)
        m=cal.fit(cal.minimum)
        q=m.trace(np.zeros((1,1)))[-1]
        self.assertTrue(np.isfinite(m.scales).all())
        self.assertEqual(q.item(0),0)

    def test_dead_graph_rejected(self):
        g=Graph((1,1,3))
        for i in range(2): g.add("dense",[0],weights=np.ones((2,3)),bias=np.zeros(2))
        with self.assertRaisesRegex(ValueError,"contribute"):
            g.validate()
