import tempfile
import unittest
from pathlib import Path

import numpy as np

from dataset.sudoku_easy import (
    EasySudokuDataset,
    _count_solutions,
    _load_base_bank,
    create_dataloader,
    generate_puzzle,
    prepare_base_bank,
)


class EasySudokuTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary_directory = tempfile.TemporaryDirectory()
        cls.cache_path = str(Path(cls.temporary_directory.name) / "base_bank_seed{seed}.npz")
        cls.path = prepare_base_bank(
            seed=1,
            rank=0,
            base_bank_size=16,
            clues_min=32,
            clues_max=40,
            base_bank_workers=1,
            cache_path=cls.cache_path,
        )
        cls.puzzles, cls.solutions = _load_base_bank(cls.path, 16)

    @classmethod
    def tearDownClass(cls):
        cls.temporary_directory.cleanup()

    def test_base_bank_contains_unique_valid_puzzles(self):
        for puzzle, solution in zip(self.puzzles, self.solutions):
            self.assertTrue(np.all(puzzle[puzzle != 0] == solution[puzzle != 0]))
            self.assertGreaterEqual(np.count_nonzero(puzzle), 32)
            self.assertLessEqual(np.count_nonzero(puzzle), 40)
            self.assertEqual(_count_solutions(puzzle), 1)
            for index in range(9):
                self.assertEqual(set(solution[index]), set(range(1, 10)))
                self.assertEqual(set(solution[:, index]), set(range(1, 10)))

    def test_generated_puzzle_is_unique_and_matches_solution(self):
        puzzle, solution = generate_puzzle(self.puzzles, self.solutions, seed=1, epoch=0, index=0)
        self.assertEqual(puzzle.shape, (9, 9))
        self.assertTrue(np.all((puzzle >= 0) & (puzzle <= 9)))
        self.assertTrue(np.all(puzzle[puzzle != 0] == solution[puzzle != 0]))
        self.assertEqual(_count_solutions(puzzle), 1)

    def test_sample_identity_is_reproducible_and_epoch_sensitive(self):
        first = generate_puzzle(self.puzzles, self.solutions, seed=7, epoch=2, index=11)
        second = generate_puzzle(self.puzzles, self.solutions, seed=7, epoch=2, index=11)
        later = generate_puzzle(self.puzzles, self.solutions, seed=7, epoch=3, index=11)
        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])
        self.assertFalse(np.array_equal(first[0], later[0]))

    def test_base_bank_is_cached_and_virtual_dataset_has_correct_tokens(self):
        self.assertEqual(
            prepare_base_bank(
                seed=1, rank=0, base_bank_size=16, clues_min=32, clues_max=40,
                base_bank_workers=1, cache_path=self.cache_path,
            ),
            self.path,
        )
        dataset = EasySudokuDataset(200_000, seed=3, base_puzzles=self.puzzles, base_solutions=self.solutions)
        self.assertEqual(len(dataset), 200_000)
        x, y = dataset[0]
        self.assertEqual(x.shape, (82,))
        self.assertEqual(y.shape, (82,))
        self.assertEqual(x[0], 0)
        self.assertEqual(y[0], 0)
        dataset.set_epoch(1)
        next_x, _ = dataset[0]
        self.assertFalse(np.array_equal(x, next_x))

    def test_default_ddp_budget_has_260_local_batches(self):
        loader, metadata = create_dataloader(
            "train", batch_size=96, rank=0, world_size=8, dataset_size=200_000,
            base_bank_size=16, clues_min=32, clues_max=40, base_bank_workers=1,
            cache_path=self.cache_path, num_workers=0, seed=1,
        )
        self.assertEqual(len(loader), 260)
        x, y = next(iter(loader))
        self.assertEqual(x.shape, (96, 82))
        self.assertEqual(y.shape, (96, 82))
        self.assertEqual(metadata, {"vocab_size": 10, "seq_len": 82, "is_causal": False})


if __name__ == "__main__":
    unittest.main()
