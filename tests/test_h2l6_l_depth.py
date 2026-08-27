import sys
import types
import unittest

# train.py imports coolname, which the lightweight local test environment omits.
coolname = types.ModuleType("coolname")
coolname.generate_slug = lambda _count: "stub"
sys.modules.setdefault("coolname", coolname)

from scripts.evaluate_h2l6_l_depth import DEFAULT_L_VALUES, row_key, summarize


class LDepthTest(unittest.TestCase):
    def test_default_grid_reaches_l512(self) -> None:
        self.assertEqual(DEFAULT_L_VALUES[-4:], (64, 128, 256, 512))

    def test_summary_groups_training_seeds_by_inference_depth(self) -> None:
        rows = [
            {"condition": "H2L6_h", "readout": "h", "seed": "1", "eval_l": "1", "test_exact_match": ".7", "cell_accuracy": ".9"},
            {"condition": "H2L6_h", "readout": "h", "seed": "2", "eval_l": "1", "test_exact_match": ".8", "cell_accuracy": ".95"},
        ]
        summary = summarize(rows)
        self.assertEqual(len(summary), 1)
        self.assertEqual(summary[0]["seeds"], 2)
        self.assertAlmostEqual(float(summary[0]["exact_mean"]), .75)
        self.assertEqual(row_key(rows[0]), ("H2L6_h", 1, 1))


if __name__ == "__main__":
    unittest.main()
