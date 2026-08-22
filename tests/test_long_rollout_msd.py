from pathlib import Path
import sys
import unittest

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import long_rollout_msd_utils as MODULE


class LongRolloutMSDTest(unittest.TestCase):
    def test_log_uniform_boundaries_are_strict_and_valid(self) -> None:
        self.assertEqual(MODULE.segment_boundaries(128), (0, 3, 11, 38, 128))
        boundaries = MODULE.segment_boundaries(4096)
        self.assertEqual(boundaries, (0, 8, 64, 512, 4096))
        self.assertTrue(all(left < right for left, right in zip(boundaries, boundaries[1:])))

    def test_full_state_concat_identity(self) -> None:
        torch.manual_seed(3)
        h_a, h_b = torch.randn(2, 4, 8), torch.randn(2, 4, 8)
        l_a, l_b = torch.randn(2, 4, 8), torch.randn(2, 4, 8)
        msd = MODULE.state_msd(h_a, l_a, h_b, l_b)
        self.assertAlmostEqual(msd["hl_concat"], (msd["h"] + msd["l"]) / 2, places=6)

    def test_lags_are_positive_and_fit_their_segments(self) -> None:
        boundaries = MODULE.segment_boundaries(128)
        lags = MODULE.segment_lags(boundaries, points=16)
        self.assertTrue(lags)
        self.assertTrue(all(lag > 0 for lag in lags))
        self.assertLessEqual(max(lags), max(end - start - 1 for start, end in zip(boundaries, boundaries[1:])))


if __name__ == "__main__":
    unittest.main()
