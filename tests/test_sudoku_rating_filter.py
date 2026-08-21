import unittest
from unittest.mock import patch

try:
    from datasets import Dataset
    from dataset.sudoku import create_dataloader
except ModuleNotFoundError:
    Dataset = None
    create_dataloader = None


def _example(rating: int) -> dict[str, object]:
    return {
        "source": "synthetic",
        "question": "." * 81,
        "answer": "123456789" * 9,
        "rating": rating,
    }


@unittest.skipUnless(Dataset is not None, "requires the project's datasets dependency")
class SudokuRatingFilterTests(unittest.TestCase):
    def setUp(self):
        examples = [_example(rating) for rating in (1, 10, 15, 20, 30, 40)]
        self.dataset = Dataset.from_list(examples)

    @patch("dataset.sudoku.load_dataset")
    def test_medium_filter_happens_before_selection_and_repeat(self, load_dataset):
        load_dataset.return_value = self.dataset
        loader, metadata = create_dataloader(
            "train", batch_size=2, rank=0, world_size=1, dataset_name="unused",
            rating_min=10, rating_max=30, num_base_puzzles=3, repeat=2,
            augment=False, num_workers=0, seed=1,
        )
        self.assertEqual(len(loader.dataset), 6)
        self.assertEqual(len(loader), 3)
        self.assertTrue(set(loader.dataset["rating"]).issubset({10, 15, 20, 30}))
        x, y = next(iter(loader))
        self.assertEqual(tuple(x.shape), (2, 82))
        self.assertEqual(tuple(y.shape), (2, 82))
        self.assertEqual(metadata, {"vocab_size": 10, "seq_len": 82, "is_causal": False})

    @patch("dataset.sudoku.load_dataset")
    def test_evaluation_uses_the_fixed_hard_dataset_without_rating_filter(self, load_dataset):
        load_dataset.return_value = self.dataset
        loader, _ = create_dataloader(
            "test_hard", batch_size=2, rank=0, world_size=1,
            dataset_name="medium-train", eval_dataset_name="fixed-hard-test",
            rating_min=10, rating_max=30, num_base_puzzles=3, num_workers=0,
        )
        self.assertEqual(load_dataset.call_args.args[0], "fixed-hard-test")
        self.assertEqual(len(loader.dataset), len(self.dataset))

    @patch("dataset.sudoku.load_dataset")
    def test_evaluation_can_use_its_own_rating_bounds(self, load_dataset):
        load_dataset.return_value = self.dataset
        loader, _ = create_dataloader(
            "test_hard", batch_size=2, rank=0, world_size=1,
            dataset_name="easy-train", eval_dataset_name="fixed-test",
            rating_min=0, rating_max=15,
            eval_rating_min=16, eval_rating_max=30, num_workers=0,
        )
        self.assertEqual(load_dataset.call_args.args[0], "fixed-test")
        self.assertEqual(loader.dataset["rating"], [20, 30])


if __name__ == "__main__":
    unittest.main()
