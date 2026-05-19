import unittest

import numpy as np

from ontogenia.topology import classify_series


class TestTopology(unittest.TestCase):
    def test_u_shape_detected(self) -> None:
        steps = np.array([0, 1, 2, 3, 4], dtype=float)
        values = np.array([0.9, 0.7, 0.5, 0.7, 0.9], dtype=float)
        topo = classify_series(steps, values)
        self.assertIn(topo.shape, {"u_shape", "single_turn"})

    def test_monotonic_detected(self) -> None:
        steps = np.array([0, 1, 2, 3, 4], dtype=float)
        values = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=float)
        topo = classify_series(steps, values)
        self.assertEqual(topo.shape, "monotonic")


if __name__ == "__main__":
    unittest.main()

