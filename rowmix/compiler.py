"""Post-training calibration, packing and allocation under a whole-file byte budget."""
from dataclasses import dataclass
import struct
import zlib

import numpy as np
from .arena import layout
from .graph import design, patches

MAGIC = b"ROWMIX1\0"
PREFIX = struct.Struct("<8sII")
NODE = struct.Struct("<13Id")
ROW = struct.Struct("<Bdd")
OPS = ("input", "dense", "conv", "add", "maxpool", "relu", "globalavg")
NONE = 0xffffffff
MAX_FILE = 256 << 20


def rnd(x):
    return np.copysign(np.floor(np.abs(x) + 0.5), x)


def quantize(x, scale):
    # Clipping before rounding avoids overflow for finite, out-of-range user inputs.
    return rnd(np.clip(x / scale, -127., 127.)).astype(np.int8)


def activation_scale(x):
    v = np.abs(x.ravel())
    if not np.isfinite(v).all():
        raise ValueError("nonfinite calibration activations")
    if not np.any(v):
        return 1.
    candidates = np.unique(np.maximum(np.quantile(v, [.99, .999, 1]), np.finfo(np.float32).tiny) / 127.)
    return float(min(candidates, key=lambda s: np.mean((x - quantize(x, s)*s)**2)))


def pack4(q):
    flat = np.zeros((len(q)+1)//2*2, dtype=np.uint8)
    flat[:len(q)] = q.astype(np.uint8) & 15
    return (flat[::2] | (flat[1::2] << 4)).tobytes()


@dataclass
class Row:
    bits: int
    scale: float
    bias: float
    q: np.ndarray

    def data(self):
        payload = pack4(self.q) if self.bits == 4 else self.q.tobytes()
        return ROW.pack(self.bits, self.scale, self.bias) + payload


@dataclass
class Compiled:
    graph: object
    scales: list
    rows: list
    offsets: list
    arena: int
    lower: int

    def to_bytes(self):
        nodes = self.graph.nodes
        pre = self.graph.preprocess
        sections = [struct.pack("<III", self.arena, len(nodes), len(pre) if pre is not None else 0)]
        if pre is not None:
            sections.append(pre.astype("<f8").tobytes())
        if not 2 <= len(nodes) <= 1024 or self.arena > MAX_FILE:
            raise ValueError("runtime graph or arena limit exceeded")
        for i, n in enumerate(nodes):
            if not 0 < self.scales[i] <= float(np.finfo(np.float32).max)/127:
                raise ValueError(f"node {i} activation scale is outside the native float32 output range")
            if n.kernel>4096 or n.kernel_w>4096 or n.stride>4096 or n.pad>4096:
                raise ValueError(f"node {i} exceeds native spatial limits")
            if n.weights is not None:
                gains=[r.scale*self.scales[n.parents[0]] for r in self.rows[i]]
                if not np.isfinite(gains).all() or min(gains)<=0:
                    raise ValueError(f"node {i} has unrepresentable requantization multipliers")
            a, b = (list(n.parents) + [NONE, NONE])[:2]
            sections.append(NODE.pack(OPS.index(n.op), int(n.relu), a, b, *n.shape, self.offsets[i],
                                      n.kernel, n.stride, n.pad, n.groups, n.kernel_w, self.scales[i]))
            sections.extend(r.data() for r in self.rows[i])
        body = b"".join(sections)
        if len(body)+PREFIX.size > MAX_FILE:
            raise ValueError("model too large")
        return PREFIX.pack(MAGIC, len(body), zlib.crc32(body)) + body

    def trace(self, x):
        nodes = self.graph.nodes
        values = [quantize(self.graph.inputs(x), self.scales[0])]
        for i, n in enumerate(nodes[1:], 1):
            a = n.parents[0]
            xq = values[a]
            if n.weights is not None:
                rows = self.rows[i]
                W = np.stack([r.q for r in rows]).astype(np.int32)
                accum = np.concatenate([design(n,xq,j).astype(np.int32) @ w.T
                                        for j,w in enumerate(np.split(W,n.groups))],axis=-1)
                gain = self.scales[a] * np.array([r.scale for r in rows])
                out = (accum*gain + n.bias).reshape(len(xq), *n.shape)
            elif n.op == "add":
                b = n.parents[1]
                out = xq*self.scales[a] + values[b]*self.scales[b]
            elif n.op == "maxpool":
                out = patches(xq,n.kernel,n.stride,n.pad,-127,n.kernel_w).max(axis=(3,4))*self.scales[a]
            elif n.op == "globalavg":
                out = xq.sum(axis=(1,2),keepdims=True)*self.scales[a]/(xq.shape[1]*xq.shape[2])
            else:
                out = np.maximum(xq,0)*self.scales[a]
            values.append(quantize(np.maximum(out, 0.) if n.relu else out, self.scales[i]))
        return values


def choose_counts(groups, capacity):
    """Each layer has equal-price upgrades; combine sorted prefixes across layers."""
    states = {0: (0., ())}
    for price, gains in groups:
        prefixes = [0., *np.cumsum(gains)]
        nxt = {}
        for used, (score, counts) in states.items():
            for k, extra in enumerate(prefixes):
                size = used+k*price
                if size > capacity:
                    break
                value = score + extra
                if size not in nxt or value > nxt[size][0]:
                    nxt[size] = (value, counts+(k,))
        states = {}
        best = -float("inf")
        for size, pair in sorted(nxt.items()):
            if pair[0] > best:
                states[size] = pair
                best = pair[0]
    return max(states.values(), key=lambda pair: pair[0])[1]


class Calibration:
    def __init__(self, graph, samples, max_patches=8192, *, batch_size=None, activation_samples=65536):
        graph.validate()
        if not isinstance(max_patches, (int, np.integer)) or max_patches < 1:
            raise ValueError("calibration needs at least one patch")
        self.graph = graph
        if batch_size is None:
            traces = graph.trace(samples)
            def sample_features(i, group):
                node = graph.nodes[i]
                features = design(node, traces[node.parents[0]], group)
                if len(features) > max_patches:
                    indices = np.linspace(0, len(features) - 1, max_patches, dtype=int)
                    return features[indices]
                return features
        else:
            from .streaming import collect
            traces, features_by_group = collect(graph, np.asanyarray(samples), batch_size,
                                                activation_samples, max_patches)
            def sample_features(i, group):
                return features_by_group[i, group]
        self.scales = [activation_scale(t) for t in traces]
        self.options = [[] for n in graph.nodes]
        self.groups = []
        self.offsets, self.arena, self.lower = layout(graph.nodes)
        for i, n in enumerate(graph.nodes):
            if n.weights is None:
                continue
            width = n.weights.shape[1]
            if width > 100000:
                raise ValueError("dot product exceeds native accumulator limit")
            gains = []
            for j, (w, bias) in enumerate(zip(n.weights, n.bias)):
                per_group=len(n.bias)//n.groups
                if j % per_group == 0:
                    features = sample_features(i, j//per_group)
                    gram=features.T @ features / len(features) if width<=min(1024,len(features)) else None
                choices = []
                errors = []
                for bits, bound in ((4, 7), (8, 127)):
                    peak = np.max(np.abs(w))
                    s = max(float(peak / bound), float(np.finfo(np.float64).tiny)) if peak else 1.
                    q = rnd(np.clip(w/s, -bound, bound)).astype(np.int8)
                    choices.append(Row(bits, s, float(bias), q))
                    err = w - q*s
                    mse = err @ gram @ err if gram is not None else np.mean((features @ err)**2)
                    errors.append(float(mse / self.scales[i]**2))
                self.options[i].append(choices)
                gains.append((errors[0]-errors[1], j))
            self.groups.append((i, width - (width+1)//2, sorted(gains, key=lambda v: (-v[0], v[1]))))
        self.minimum = len(self.fixed(4).to_bytes())
        self.maximum = len(self.fixed(8).to_bytes())

    def _model(self, rows):
        return Compiled(self.graph, self.scales, rows, self.offsets, self.arena, self.lower)

    def fixed(self, bits):
        if bits not in (4, 8):
            raise ValueError("fixed precision must be 4 or 8")
        return self._model([[pair[bits == 8] for pair in layer] for layer in self.options])

    def fit(self, byte_budget):
        if not isinstance(byte_budget, (int, np.integer)) or byte_budget < self.minimum:
            raise ValueError(f"model requires at least {self.minimum} bytes")
        useful = [(i, price, [(gain, j) for gain, j in ranking if gain > 0])
                  for i, price, ranking in self.groups]
        counts = choose_counts([(price, [g for g, _ in ranking]) for _, price, ranking in useful],
                               byte_budget - self.minimum)
        model = self.fixed(4)
        for (i, _, ranking), k in zip(useful, counts):
            for _, j in ranking[:k]:
                model.rows[i][j] = self.options[i][j][1]
        assert len(model.to_bytes()) <= byte_budget
        return model


def diagnose(model, samples):
    golden = model.graph.trace(samples)
    deployed = model.trace(samples)
    report = []
    for i, (ref, q, node) in enumerate(zip(golden, deployed, model.graph.nodes)):
        decoded = q*model.scales[i]
        err = ref-decoded
        mse = float(np.mean(err**2))
        power = float(np.mean(ref**2))
        report.append(dict(node=i, op=node.op, scale=model.scales[i], rmse=float(np.sqrt(mse)),
                           max_abs=float(np.max(np.abs(err))),
                           sqnr_db=float(10*np.log10(power/mse)) if power > 0 and mse > 0 else None,
                           boundary_fraction=float(np.mean(np.abs(q.astype(np.int16)) == 127)),
                           int8_rows=sum(r.bits == 8 for r in model.rows[i]), offset=model.offsets[i]))
    return report
