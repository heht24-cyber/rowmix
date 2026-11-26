"""Gather deterministic calibration samples from a memory-mapped batch in one pass."""
import operator

import numpy as np

from .graph import design


class SamplePlan:
    """Indices in a logical dataset and their destinations in a fixed-size buffer."""

    def __init__(self, total, limit, rng=None):
        count = min(total, limit)
        if count == total:
            self.indices = np.arange(count, dtype=np.int64)
        elif rng is None:
            # Use the same patch selection as the full calibration path.
            self.indices = np.linspace(0, total - 1, count, dtype=np.int64)
        else:
            # One observation per stratum avoids always hitting the same channel
            # or spatial position in a periodic feature map.
            stride, remainder = divmod(total, count)
            boundaries = np.arange(count + 1, dtype=np.int64)
            edges = boundaries * stride + boundaries * remainder // count
            self.indices = rng.integers(edges[:-1], edges[1:])

    def window(self, offset, length):
        first, stop = self.indices.searchsorted([offset, offset + length])
        return slice(first, stop), self.indices[first:stop] - offset


def collect(graph, samples, batch_size, activation_samples, max_patches):
    try:
        batch_size = operator.index(batch_size)
        activation_samples = operator.index(activation_samples)
    except TypeError as exc:
        raise ValueError("calibration limits must be integers") from exc
    if batch_size < 1 or activation_samples < 1 or samples.ndim not in (2, 4) or not len(samples):
        raise ValueError("streaming calibration requires positive limits and a nonempty batch")

    size = len(samples)
    rng = np.random.default_rng(1701)
    activations = []
    patch_plans = {}
    features = {}
    for i, node in enumerate(graph.nodes):
        plan = SamplePlan(size * node.count, activation_samples, rng)
        activations.append((plan, np.empty(len(plan.indices), dtype=np.float64)))
        if node.weights is not None:
            positions = node.shape[0] * node.shape[1]
            plan = SamplePlan(size * positions, max_patches)
            patch_plans[i] = plan
            for group in range(node.groups):
                features[i, group] = np.empty((len(plan.indices), node.weights.shape[1]), dtype=np.float64)

    for start in range(0, size, batch_size):
        trace = graph.trace(samples[start:start + batch_size])
        for i, (node, values) in enumerate(zip(graph.nodes, trace)):
            if not np.isfinite(values).all():
                raise ValueError(f"nonfinite calibration activations at node {i}")
            plan, buffer = activations[i]
            target, local = plan.window(start * node.count, values.size)
            buffer[target] = values.ravel()[local]
            if i in patch_plans:
                positions = node.shape[0] * node.shape[1]
                target, local = patch_plans[i].window(start * positions, len(values) * positions)
                if len(local):
                    parent = trace[node.parents[0]]
                    for group in range(node.groups):
                        features[i, group][target] = design(node, parent, group)[local]
        del trace
    return [buffer for _, buffer in activations], features
