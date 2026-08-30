import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from cross_evaluate import ExperimentCheckpoint, aggregate_rows, parse_checkpoint, validate_experiment_matrix


class CrossEvaluationTests(unittest.TestCase):
    def test_parser_requires_declared_final_checkpoint_and_matching_architecture(self):
        with TemporaryDirectory() as temp_dir:
            checkpoint_dir = Path(temp_dir)
            (checkpoint_dir / "model_config.json").write_text(
                "arch:\n  name: trm@TRM\ndata:\n  name: sudoku\nepochs: 20\ncycles_per_data: 16\n"
            )
            final_checkpoint = checkpoint_dir / "epoch_19.pt"
            final_checkpoint.touch()
            parsed = parse_checkpoint(f"trm:easy:1={final_checkpoint}")
            self.assertEqual(parsed, ExperimentCheckpoint("trm", "easy", 1, final_checkpoint))

            nonfinal_checkpoint = checkpoint_dir / "epoch_18.pt"
            nonfinal_checkpoint.touch()
            with self.assertRaisesRegex(Exception, "final checkpoint"):
                parse_checkpoint(f"trm:easy:1={nonfinal_checkpoint}")

    def test_matrix_requires_each_model_band_with_same_seeds(self):
        checkpoints = [
            ExperimentCheckpoint(model, band, seed, Path(f"/{model}-{band}-{seed}.pt"))
            for model in ("hrm", "rt", "trm")
            for band in ("easy", "medium", "hard")
            for seed in (1, 2)
        ]
        self.assertEqual(validate_experiment_matrix(checkpoints), [1, 2])

        incomplete = [item for item in checkpoints if not (item.model == "trm" and item.seed == 2)]
        with self.assertRaisesRegex(ValueError, "identical"):
            validate_experiment_matrix(incomplete)

    def test_aggregate_reports_mean_and_sample_standard_deviation(self):
        rows = [
            {
                "model": "trm", "train_band": "easy", "test_band": "hard", "seed": 1,
                "exact_match_accuracy": 0.4, "total_samples": 10_000,
                "parameter_count": 100, "core_calls_per_prediction": 128,
                "total_training_steps_per_rank": 20, "world_size": 8,
            },
            {
                "model": "trm", "train_band": "easy", "test_band": "hard", "seed": 2,
                "exact_match_accuracy": 0.6, "total_samples": 10_000,
                "parameter_count": 100, "core_calls_per_prediction": 128,
                "total_training_steps_per_rank": 20, "world_size": 8,
            },
        ]
        aggregate = aggregate_rows(rows)
        self.assertEqual(len(aggregate), 1)
        self.assertAlmostEqual(aggregate[0]["exact_match_accuracy_mean"], 0.5)
        self.assertAlmostEqual(aggregate[0]["exact_match_accuracy_std"], 2 ** 0.5 / 10)
        self.assertEqual(aggregate[0]["num_seeds"], 2)


if __name__ == "__main__":
    unittest.main()
