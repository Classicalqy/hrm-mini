import tempfile
import unittest
from pathlib import Path

from test_time_scaling import (
    CONDITION_NAMES,
    architecture_at_l,
    best_epoch_from_history,
    evaluation_spec,
    format_markdown_table,
    parse_l_values,
    steps_per_epoch_from_metadata,
    summarize_rows,
)


class TestTimeScalingTests(unittest.TestCase):
    def test_condition_names_cover_the_six_k55_training_runs(self):
        self.assertEqual(set(CONDITION_NAMES), {
            "easy_k55_hrm", "hard_k55_hrm", "easy_k55_trm", "hard_k55_trm", "easy_k55_rt", "hard_k55_rt",
        })

    def test_extracts_step_count_without_parsing_cyclic_checkpoint_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            metadata_path = Path(temp_dir) / "model_config.json"
            metadata_path.write_text(
                "data: !!python/object:omegaconf.dictconfig.DictConfig\n"
                "  _parent: &parent !!python/object:omegaconf.dictconfig.DictConfig\n"
                "    _parent: *parent\n"
                "run_metadata:\n"
                "  steps_per_epoch_per_rank: 4160\n"
            )
            steps = steps_per_epoch_from_metadata(Path(temp_dir))
        self.assertEqual(steps, 4160)

    def test_hrm_and_trm_change_only_l_cycles(self):
        config = {"arch": {"name": "hrm@HRM", "H_cycles": 2, "L_cycles": 6, "hidden_size": 512}}
        scaled = architecture_at_l(config, "hrm", 32)
        self.assertEqual(scaled["H_cycles"], 2)
        self.assertEqual(scaled["L_cycles"], 32)
        self.assertEqual(config["arch"]["L_cycles"], 6)

    def test_rt_matches_h2l_transformer_layer_budget(self):
        config = {"arch": {"name": "rt@RecurrentTransformer", "cycles": 7}}
        self.assertEqual(architecture_at_l(config, "rt", 6)["cycles"], 7)
        self.assertEqual(architecture_at_l(config, "rt", 256)["cycles"], 257)

    def test_parses_l_values(self):
        self.assertEqual(parse_l_values("6,8,16"), (6, 8, 16))
        with self.assertRaises(Exception):
            parse_l_values("6,6")

    def test_requires_the_k55_easy_and_hard_test_sets(self):
        config = {
            "data": {
                "name": "sudoku", "dataset_name": "train", "eval_dataset_name": "test",
                "eval_num_base_puzzles": 10_000, "eval_seed": 42,
                "eval_sets": {
                    "easy": {"split": "test", "eval_blank_max": 55},
                    "hard": {"split": "test", "eval_blank_min": 56},
                },
            },
        }
        name, data, eval_sets = evaluation_spec(config)
        self.assertEqual(name, "sudoku")
        self.assertNotIn("eval_sets", data)
        self.assertEqual(eval_sets["hard"]["eval_blank_min"], 56)

    def test_best_epoch_uses_highest_metric_and_breaks_ties_early(self):
        history = [
            {"_step": 160, "eval/hard_exact_match": 0.50},
            {"_step": 320, "eval/hard_exact_match": 0.70},
            {"_step": 480, "eval/hard_exact_match": 0.70},
            {"_step": 500, "eval/hard_exact_match": 0.99},
        ]
        self.assertEqual(best_epoch_from_history(history, "eval/hard_exact_match", 160), (1, 320, 0.70))

    def test_summary_contains_seed_mean_and_standard_deviation(self):
        rows = [
            {"model": "hrm", "train_band": "easy", "L_cycles": 6, "test_band": "easy", "exact_match_accuracy": 0.4},
            {"model": "hrm", "train_band": "easy", "L_cycles": 6, "test_band": "easy", "exact_match_accuracy": 0.6},
            {"model": "hrm", "train_band": "easy", "L_cycles": 6, "test_band": "hard", "exact_match_accuracy": 0.2},
            {"model": "hrm", "train_band": "easy", "L_cycles": 6, "test_band": "hard", "exact_match_accuracy": 0.4},
        ]
        summary = summarize_rows(rows)
        self.assertEqual(len(summary), 1)
        self.assertAlmostEqual(summary[0]["easy_exact_match_mean"], 0.5)
        self.assertAlmostEqual(summary[0]["hard_exact_match_mean"], 0.3)
        self.assertIn("easy exact match", format_markdown_table(summary))


if __name__ == "__main__":
    unittest.main()
