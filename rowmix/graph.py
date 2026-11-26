"""Small, explicit NHWC graph shared by the importer, compiler and diagnostics."""
from dataclasses import dataclass
import json
from math import prod
from pathlib import Path

import numpy as np


def patches(x, kernel, stride, pad, constant=0, kernel_w=None):
    if pad:
        x = np.pad(x, ((0, 0), (pad, pad), (pad, pad), (0, 0)), constant_values=constant)
    view = np.lib.stride_tricks.sliding_window_view(x, (kernel, kernel_w or kernel), axis=(1, 2))
    return view[:, ::stride, ::stride].transpose(0, 1, 2, 4, 5, 3)


@dataclass
class Node:
    op: str
    parents: tuple
    shape: tuple
    relu: bool = False
    kernel: int = 1
    stride: int = 1
    pad: int = 0
    weights: object = None
    bias: object = None
    groups: int = 1
    kernel_w: int = 1

    @property
    def count(self):
        return prod(self.shape)


class Graph:
    def __init__(self, shape):
        if len(shape) != 3 or any(int(v) != v or v <= 0 for v in shape):
            raise ValueError("input shape must be three positive NHWC dimensions")
        self.nodes = [Node("input", (), tuple(map(int, shape)))]
        self.preprocess = None

    def set_preprocess(self, mean, inverse_std):
        pair = np.column_stack((mean, inverse_std)).astype(np.float64)
        if pair.shape != (self.nodes[0].shape[2], 2) or not np.isfinite(pair).all() or np.any(pair[:, 1] <= 0):
            raise ValueError("need finite channel means and positive inverse standard deviations")
        self.preprocess = pair

    def add(self, op, parents, *, weights=None, bias=None, relu=False, kernel=1, stride=1, pad=0, groups=1, kernel_w=None):
        parents = tuple(parents)
        arity = 2 if op == "add" else 1
        if len(parents) != arity or any(not isinstance(i, int) or not 0 <= i < len(self.nodes) for i in parents):
            raise ValueError("invalid parent indices")
        h, w, c = self.nodes[parents[0]].shape
        kernel_w=kernel if kernel_w is None else kernel_w
        if any(type(v) is not int for v in (kernel,kernel_w,stride,pad)) or min(kernel,kernel_w,stride)<1 or pad<0:
            raise ValueError("invalid spatial parameters")
        if op not in ("conv","maxpool") and (kernel, kernel_w, stride, pad) != (1, 1, 1, 0):
            raise ValueError("unexpected spatial options")
        if type(groups) is not int or groups < 1 or c % groups or (op != "conv" and groups != 1):
            raise ValueError("invalid group count")
        if op in ("dense", "conv"):
            weights = np.ascontiguousarray(weights, dtype=np.float64)
            bias = np.ascontiguousarray(bias, dtype=np.float64)
            width = h*w*c if op == "dense" else kernel*kernel_w*(c//groups)
            if weights.ndim != 2 or weights.shape[1] != width or weights.shape[0] < 1 or bias.shape != (weights.shape[0],):
                raise ValueError("weights must be [out, features] and bias [out]")
            if not np.isfinite(weights).all() or not np.isfinite(bias).all():
                raise ValueError("nonfinite parameters")
            c = len(bias)
            if c % groups:
                raise ValueError("group count must divide output channels")
        elif weights is not None or bias is not None:
            raise ValueError("unexpected weights")
        if op in ("conv", "maxpool"):
            if op == "maxpool" and (pad >= min(kernel,kernel_w) or relu):
                raise ValueError("invalid maxpool attributes")
            h, w = (h + 2*pad-kernel)//stride+1, (w + 2*pad-kernel_w)//stride+1
        elif op in ("dense","globalavg"):
            h, w = 1, 1
        elif op == "add":
            if self.nodes[parents[1]].shape != (h, w, c):
                raise ValueError("residual shapes must agree")
        elif op != "relu":
            raise ValueError(f"unsupported operator {op!r}")
        if min(h, w) < 1:
            raise ValueError("empty spatial output")
        self.nodes.append(Node(op, parents, (h, w, c), bool(relu), kernel, stride, pad, weights, bias, groups,kernel_w))
        return len(self.nodes)-1

    def validate(self):
        reached = {len(self.nodes)-1}
        for i in reversed(range(len(self.nodes))):
            if i in reached:
                reached.update(self.nodes[i].parents)
        if len(reached) != len(self.nodes) or len(reached) < 2:
            raise ValueError("every node must contribute to one non-input output")

    def inputs(self, x):
        a = np.asarray(x, dtype=np.float32).astype(np.float64)
        if a.ndim == 2 and a.shape[1] == self.nodes[0].count:
            a = a.reshape(-1, *self.nodes[0].shape)
        if a.ndim != 4 or a.shape[1:] != self.nodes[0].shape or not len(a) or not np.isfinite(a).all():
            raise ValueError("input must be a finite nonempty batch with the model's shape")
        if self.preprocess is not None:
            a = (a - self.preprocess[:, 0]) * self.preprocess[:, 1]
        return a

    def trace(self, x):
        values = [self.inputs(x)]
        for n in self.nodes[1:]:
            x = values[n.parents[0]]
            if n.op in ("dense", "conv"):
                out = []
                for group, weights in enumerate(np.split(n.weights,n.groups)):
                    out.append(design(n,x,group) @ weights.T)
                y = (np.concatenate(out,axis=-1) + n.bias).reshape(len(x), *n.shape)
            elif n.op == "add":
                y = x + values[n.parents[1]]
            elif n.op == "maxpool":
                y = patches(x,n.kernel,n.stride,n.pad,-np.inf,n.kernel_w).max(axis=(3,4))
            elif n.op == "globalavg":
                y = x.mean(axis=(1,2),keepdims=True)
            else:
                y = np.maximum(x,0.)
            values.append(np.maximum(y, 0) if n.relu else y)
        return values

    def save(self, path):
        self.validate()
        arrays = {}
        layers = []
        for i, n in enumerate(self.nodes[1:], 1):
            layers.append(dict(op=n.op, parents=n.parents, relu=n.relu, kernel=n.kernel,kernel_w=n.kernel_w, stride=n.stride, pad=n.pad, groups=n.groups))
            if n.weights is not None:
                arrays[f"w{i}"], arrays[f"b{i}"] = n.weights, n.bias
        spec = dict(version=1, input=self.nodes[0].shape, layers=layers)
        if self.preprocess is not None:
            arrays["preprocess"] = self.preprocess
        with Path(path).open("wb") as handle:
            np.savez_compressed(handle, spec=np.array(json.dumps(spec)), **arrays)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            spec = json.loads(str(data["spec"].item()))
            if spec["version"] != 1:
                raise ValueError("unsupported graph format")
            graph = cls(spec["input"])
            if "preprocess" in data:
                graph.set_preprocess(*data["preprocess"].T)
            for i, layer in enumerate(spec["layers"], 1):
                if layer["op"] in ("dense", "conv"):
                    layer.update(weights=data[f"w{i}"], bias=data[f"b{i}"])
                graph.add(**layer)
        graph.validate()
        return graph


def design(node, x, group=0):
    if node.op == "dense":
        return x.reshape(len(x), -1)
    channels = x.shape[-1]//node.groups
    part = x[...,group*channels:(group+1)*channels]
    return patches(part,node.kernel,node.stride,node.pad,kernel_w=node.kernel_w).reshape(-1,node.weights.shape[1])


def from_sklearn(model):
    """Import pre-softmax logits from a fitted ReLU or identity MLPClassifier.

    Binary classifiers export two logits [0, z], preserving sklearn class order.
    """
    if model.activation not in ("identity", "relu") or model.out_activation_ not in ("softmax", "logistic"):
        raise ValueError("supported classifier uses ReLU/identity and softmax/logistic output")
    if getattr(model, "_label_binarizer", None) is None or model._label_binarizer.y_type_ not in ("multiclass", "binary"):
        raise ValueError("expected fitted single-label classifier")
    g = Graph((1, 1, model.coefs_[0].shape[0]))
    for i, (w, b) in enumerate(zip(model.coefs_, model.intercepts_)):
        final = i == len(model.coefs_)-1
        if final and w.shape[1] == 1:
            w = np.column_stack((np.zeros(len(w)), w[:, 0]))
            b = np.array([0., b[0]])
        g.add("dense", [i], weights=w.T, bias=b, relu=not final and model.activation == "relu")
    return g
