import unittest
from unittest.mock import patch

import numpy as np

from rowmix.compiler import Calibration
from rowmix.graph import Graph, design
from rowmix.onnx_import import native_input
from rowmix.streaming import collect


class StreamingTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        self.graph = g = Graph((4, 6, 4))
        # Dyadic parameters keep these fixtures exactly representable across BLAS kernels.
        conv = g.add("conv", [0], groups=2, kernel=3, kernel_w=1, pad=1,
                     weights=rng.integers(-7, 8, (4, 6)) / 8, bias=np.zeros(4))
        g.add("dense", [conv], weights=rng.integers(-9, 10, (3, g.nodes[conv].count)) / 16,
              bias=np.zeros(3))
        self.x = rng.integers(-16, 17, (19, 4, 6, 4)).astype(np.float32)

    def test_group_features_and_all_activations_match_full_trace(self):
        full = self.graph.trace(self.x)
        acts, feats = collect(self.graph, self.x, 5, 100000, 31)
        for a, values in zip(acts, full):
            np.testing.assert_array_equal(a, values.ravel())
        for group in range(2):
            full_features = design(self.graph.nodes[1], full[0], group)
            ids = np.linspace(0, len(full_features) - 1, 31, dtype=int)
            np.testing.assert_array_equal(feats[1, group], full_features[ids])
        a = Calibration(self.graph, self.x, max_patches=31)
        b = Calibration(self.graph, self.x, batch_size=5, max_patches=31)
        cap = (a.minimum + a.maximum) // 2
        self.assertEqual(a.fit(cap).to_bytes(), b.fit(cap).to_bytes())

    def test_sampling_independent_of_chunk_boundaries(self):
        a, fa = collect(self.graph, self.x, 1, 23, 29)
        with patch.object(self.graph, "trace", wraps=self.graph.trace) as trace:
            b, fb = collect(self.graph, self.x, 6, 23, 29)
        self.assertEqual([len(call.args[0]) for call in trace.call_args_list], [6, 6, 6, 1])
        for x, y in zip(a, b):
            self.assertEqual(len(x), 23)
            np.testing.assert_array_equal(x, y)
        for k in fa:
            np.testing.assert_array_equal(fa[k], fb[k])

    def test_invalid_limits_rank_and_nonfinite_unsampled_value(self):
        for batch_size, values in [(0, 1), (1, 0), (0.25, 8)]:
            with self.assertRaises(ValueError):
                collect(self.graph, self.x, batch_size, values, 2)
        self.x[-1, 0, 0, 0] = np.inf
        with self.assertRaises(ValueError):
            collect(self.graph, self.x, 3, 1, 1)
        with self.assertRaisesRegex(ValueError, "rank-4"):
            native_input(np.zeros((2, 64)), "image")
