"""Deterministic residual-CNN training in NumPy and a sklearn MLP import."""
import numpy as np
from sklearn.datasets import load_digits, load_wine
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from rowmix.graph import Graph, from_sklearn, patches


def split(x, y):
    ids = np.arange(len(y))
    rest, test = train_test_split(ids, test_size=.20, stratify=y, random_state=57)
    train, calibration = train_test_split(rest, test_size=.18, stratify=y[rest], random_state=58)
    return [x[i].astype(np.float32) for i in (train, calibration, test)], [y[i] for i in (train, calibration, test)], [train, calibration, test]


def conv(x, weight, bias):
    k = 3
    cols = np.ascontiguousarray(patches(x, k, 1, 1)).reshape(-1, weight.shape[1])
    result = (cols @ weight.T + bias).reshape(*x.shape[:3], len(bias))
    return result, (x.shape, cols)


def conv_grad(dy, cache, weight):
    shape, cols = cache
    b, h, w, c = shape
    d = dy.reshape(-1, len(weight))
    dw, db = d.T @ cols, d.sum(axis=0)
    expanded = (d @ weight).reshape(b, h, w, 3, 3, c)
    dx = np.zeros((b, h+2, w+2, c), dtype=np.float32)
    for y in range(3):
        for x in range(3):
            dx[:, y:y+h, x:x+w, :] += expanded[:, :, :, y, x, :]
    return dx[:, 1:-1, 1:-1], dw, db


def train_cnn(images, labels, epochs=20, seed=4):
    rng = np.random.default_rng(seed)
    p = []
    for fanin, fanout in [(9,12),(9*12,12),(9*12,12),(4*4*12,32),(32,10)]:
        p.extend([(rng.normal(size=(fanout,fanin))*np.sqrt(2/fanin)).astype(np.float32),
                  np.zeros(fanout, dtype=np.float32)])
    # Start the residual branch small, keeping a usable identity path from the first step.
    p[4] *= .15
    m, v = [np.zeros_like(a) for a in p], [np.zeros_like(a) for a in p]
    step = 0
    x = images.reshape(-1, 8, 8, 1) / np.float32(16.)
    for epoch in range(epochs):
        indices = rng.permutation(len(x))
        for start in range(0, len(x), 48):
            ids = indices[start:start+48]
            h1, c1 = conv(x[ids], p[0], p[1]); h1 = np.maximum(h1, 0.)
            h2, c2 = conv(h1, p[2], p[3]); h2 = np.maximum(h2, 0.)
            delta, c3 = conv(h2, p[4], p[5])
            h3 = np.maximum(h1 + delta, 0.)
            b = len(ids)
            windows = h3.reshape(b,4,2,4,2,12).transpose(0,1,3,5,2,4).reshape(b,4,4,12,4)
            positions = np.argmax(windows, axis=-1, keepdims=True)
            pooled = np.take_along_axis(windows, positions, axis=-1).reshape(b,-1)
            hidden = np.maximum(pooled @ p[6].T + p[7], 0.)
            logits = hidden @ p[8].T + p[9]
            exp = np.exp(logits - logits.max(axis=1,keepdims=True))
            dy = exp / exp.sum(axis=1,keepdims=True)
            dy[np.arange(b), labels[ids]] -= 1
            dy /= b
            g = [None]*len(p)
            g[8], g[9] = dy.T @ hidden, dy.sum(axis=0)
            dh = (dy @ p[8]) * (hidden > 0)
            g[6], g[7] = dh.T @ pooled, dh.sum(axis=0)
            dwindows = np.zeros_like(windows)
            np.put_along_axis(dwindows, positions, (dh @ p[6]).reshape(b,4,4,12,1), axis=-1)
            skip = dwindows.reshape(b,4,4,12,2,2).transpose(0,1,4,2,5,3).reshape(b,8,8,12) * (h3 > 0)
            dh2, g[4], g[5] = conv_grad(skip, c3, p[4])
            dh1, g[2], g[3] = conv_grad(dh2*(h2>0), c2, p[2])
            _, g[0], g[1] = conv_grad((dh1+skip)*(h1>0), c1, p[0])
            step += 1
            lr = .002 * (.35 if epoch >= epochs*3//4 else 1.)
            for i in range(len(p)):
                m[i] = .9*m[i] + .1*g[i]
                v[i] = .999*v[i] + .001*g[i]**2
                p[i] -= lr * (m[i]/(1-.9**step)) / (np.sqrt(v[i]/(1-.999**step)) + 1e-8)
    graph = Graph((8,8,1))
    graph.set_preprocess([0], [1/16])
    s = graph.add("conv", [0], weights=p[0], bias=p[1], kernel=3, pad=1, relu=True)
    r = graph.add("conv", [s], weights=p[2], bias=p[3], kernel=3, pad=1, relu=True)
    r = graph.add("conv", [r], weights=p[4], bias=p[5], kernel=3, pad=1)
    r = graph.add("add", [s,r], relu=True)
    r = graph.add("maxpool", [r], kernel=2, stride=2)
    r = graph.add("dense", [r], weights=p[6], bias=p[7], relu=True)
    graph.add("dense", [r], weights=p[8], bias=p[9])
    return graph


def dataset(name):
    if name == "digits":
        X, y = load_digits(return_X_y=True)
    else:
        X, y = load_wine(return_X_y=True)
    sets, ys, ids = split(X, y)
    if name == "digits":
        graph = train_cnn(sets[0], ys[0])
    else:
        scale = StandardScaler().fit(sets[0])
        model = MLPClassifier(hidden_layer_sizes=(32,16), solver="lbfgs", alpha=.08,
                              max_iter=600, random_state=5).fit(scale.transform(sets[0]), ys[0])
        graph = from_sklearn(model)
        graph.set_preprocess(scale.mean_, 1/scale.scale_)
        baseline = graph.trace(sets[2])[-1].reshape(len(ys[2]), -1).argmax(axis=1)
        assert np.array_equal(baseline, model.predict(scale.transform(sets[2])))
    return graph, sets, ys, ids
