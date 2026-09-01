import unittest
from unittest.mock import patch

try:
    from datasets import Dataset
    from dataset.sudoku import create_dataloader
except ModuleNotFoundError:
    Dataset = None
    create_dataloader = None


def _example(rating: int, blank_count: int = 81) -> dict[str, object]:
    return {
        "source": "synthetic",
        "question": "." * blank_count + "1" * (81 - blank_count),
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

    @patch("dataset.sudoku.load_dataset")
    def test_evaluation_uses_a_fixed_size_deterministic_sample(self, load_dataset):
        load_dataset.return_value = self.dataset
        first, _ = create_dataloader(
            "test", batch_size=2, rank=0, world_size=1, dataset_name="unused",
            eval_rating_min=16, eval_rating_max=40,
            eval_num_base_puzzles=2, eval_seed=7, num_workers=0,
        )
        second, _ = create_dataloader(
            "test", batch_size=2, rank=0, world_size=1, dataset_name="unused",
            eval_rating_min=16, eval_rating_max=40,
            eval_num_base_puzzles=2, eval_seed=7, num_workers=0,
        )
        self.assertEqual(len(first.dataset), 2)
        self.assertEqual(first.dataset["rating"], second.dataset["rating"])
        self.assertTrue(set(first.dataset["rating"]).issubset({20, 30, 40}))

    @patch("dataset.sudoku.load_dataset")
    def test_blank_count_filters_apply_to_training_before_selection(self, load_dataset):
        load_dataset.return_value = Dataset.from_list([
            _example(rating=rating, blank_count=blanks)
            for rating, blanks in ((1, 52), (2, 53), (3, 54), (4, 57), (5, 58))
        ])
        loader, _ = create_dataloader(
            "train", batch_size=2, rank=0, world_size=1, dataset_name="unused",
            blank_min=54, blank_max=57, num_base_puzzles=2, augment=False,
            num_workers=0, seed=1,
        )
        self.assertEqual(len(loader.dataset), 2)
        self.assertTrue(all(54 <= question.count(".") <= 57 for question in loader.dataset["question"]))

    @patch("dataset.sudoku.load_dataset")
    def test_blank_count_filters_apply_only_to_evaluation(self, load_dataset):
        load_dataset.return_value = Dataset.from_list([
            _example(rating=rating, blank_count=blanks)
            for rating, blanks in ((1, 52), (2, 53), (3, 54), (4, 57), (5, 58))
        ])
        loader, _ = create_dataloader(
            "test", batch_size=2, rank=0, world_size=1, dataset_name="train",
            eval_dataset_name="held-out-test", blank_min=54, blank_max=57,
            eval_blank_max=53, num_workers=0,
        )
        self.assertEqual(load_dataset.call_args.args[0], "held-out-test")
        self.assertTrue(all(question.count(".") <= 53 for question in loader.dataset["question"]))


if __name__ == "__main__":
    unittest.main()
