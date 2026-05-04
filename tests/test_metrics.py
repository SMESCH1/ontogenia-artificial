import unittest

from ontogenia.metrics import pairwise_slln_lp_margin, slln_lp


class TestSllnLp(unittest.TestCase):
    def test_slln_alpha_half(self) -> None:
        # log p = 0, 4 tokens -> 0 / 4^0.5 = 0
        self.assertAlmostEqual(slln_lp(0.0, 4, alpha=0.5), 0.0)
        # log p = -4, 4 tokens -> -4/2 = -2
        self.assertAlmostEqual(slln_lp(-4.0, 4, alpha=0.5), -2.0)

    def test_pairwise_margin(self) -> None:
        m = pairwise_slln_lp_margin(
            -1.0, 1, -4.0, 4, alpha=0.5
        )  # -1/1 - (-4/2) = -1 - (-2) = 1
        self.assertAlmostEqual(m, 1.0)


if __name__ == "__main__":
    unittest.main()
