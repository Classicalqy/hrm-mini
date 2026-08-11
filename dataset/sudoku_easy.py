"""Easy Sudoku data built from a cached bank of independent base puzzles."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Value
from pathlib import Path
import os

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, DistributedSampler


_FULL_MASK = (1 << 9) - 1


def _count_solutions(board: np.ndarray, limit: int = 2) -> int:
    """Count solutions using bitmasks and MRV, stopping once ``limit`` is hit."""
    cells = board.reshape(-1).astype(np.int8, copy=True)
    row_masks = np.zeros(9, dtype=np.int16)
    col_masks = np.zeros(9, dtype=np.int16)
    box_masks = np.zeros(9, dtype=np.int16)

    for pos, value in enumerate(cells):
        if value == 0:
            continue
        bit = 1 << (int(value) - 1)
        row, col = divmod(pos, 9)
        box = (row // 3) * 3 + col // 3
        if row_masks[row] & bit or col_masks[col] & bit or box_masks[box] & bit:
            return 0
        row_masks[row] |= bit
        col_masks[col] |= bit
        box_masks[box] |= bit

    def search(found: int) -> int:
        if found >= limit:
            return found

        best_pos = -1
        best_candidates = 0
        best_count = 10
        for pos, value in enumerate(cells):
            if value != 0:
                continue
            row, col = divmod(pos, 9)
            box = (row // 3) * 3 + col // 3
            candidates = _FULL_MASK & ~(int(row_masks[row]) | int(col_masks[col]) | int(box_masks[box]))
            count = candidates.bit_count()
            if count == 0:
                return found
            if count < best_count:
                best_pos, best_candidates, best_count = pos, candidates, count
                if count == 1:
                    break

        if best_pos == -1:
            return found + 1

        row, col = divmod(best_pos, 9)
        box = (row // 3) * 3 + col // 3
        while best_candidates and found < limit:
            bit = best_candidates & -best_candidates
            best_candidates ^= bit
            cells[best_pos] = bit.bit_length()
            row_masks[row] |= bit
            col_masks[col] |= bit
            box_masks[box] |= bit
            found = search(found)
            row_masks[row] ^= bit
            col_masks[col] ^= bit
            box_masks[box] ^= bit
            cells[best_pos] = 0
        return found

    return search(0)


def _random_complete_solution(rng: np.random.Generator) -> np.ndarray:
    """Generate a full Sudoku board with randomized MRV backtracking."""
    cells = np.zeros(81, dtype=np.int8)
    row_masks = np.zeros(9, dtype=np.int16)
    col_masks = np.zeros(9, dtype=np.int16)
    box_masks = np.zeros(9, dtype=np.int16)

    def solve() -> bool:
        best_pos = -1
        best_candidates = 0
        best_count = 10
        for pos, value in enumerate(cells):
            if value != 0:
                continue
            row, col = divmod(pos, 9)
            box = (row // 3) * 3 + col // 3
            candidates = _FULL_MASK & ~(int(row_masks[row]) | int(col_masks[col]) | int(box_masks[box]))
            count = candidates.bit_count()
            if count == 0:
                return False
            if count < best_count:
                best_pos, best_candidates, best_count = pos, candidates, count
                if count == 1:
                    break

        if best_pos == -1:
            return True

        row, col = divmod(best_pos, 9)
        box = (row // 3) * 3 + col // 3
        bits: list[int] = []
        while best_candidates:
            bit = best_candidates & -best_candidates
            best_candidates ^= bit
            bits.append(bit)
        rng.shuffle(bits)
        for bit in bits:
            cells[best_pos] = bit.bit_length()
            row_masks[row] |= bit
            col_masks[col] |= bit
            box_masks[box] |= bit
            if solve():
                return True
            row_masks[row] ^= bit
            col_masks[col] ^= bit
            box_masks[box] ^= bit
            cells[best_pos] = 0
        return False

    if not solve():  # Defensive: an empty Sudoku always has a solution.
        raise RuntimeError("random Sudoku solver failed")
    return cells.reshape(9, 9)


def _base_rng(seed: int, index: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([seed, 0xEA57, index]))


def _generate_base_puzzle(task: tuple[int, int, int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Generate one independently solved, uniquely-solvable base puzzle."""
    seed, index, clues_min, clues_max, max_attempts = task
    rng = _base_rng(seed, index)
    for _ in range(max_attempts):
        solution = _random_complete_solution(rng)
        puzzle = solution.copy()
        clue_count = int(rng.integers(clues_min, clues_max + 1))
        puzzle.reshape(-1)[rng.choice(81, size=81 - clue_count, replace=False)] = 0
        if _count_solutions(puzzle) == 1:
            return puzzle, solution
    raise RuntimeError(
        f"base puzzle {index} did not become unique after {max_attempts} attempts; "
        "increase clues_min or max_attempts"
    )


def _bank_path(cache_path: str, seed: int, base_bank_size: int, clues_min: int, clues_max: int) -> Path:
    return Path(cache_path.format(
        seed=seed,
        base_bank_size=base_bank_size,
        clues_min=clues_min,
        clues_max=clues_max,
    ))


def _load_base_bank(path: Path, base_bank_size: int) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as bank:
        puzzles = bank["puzzles"].astype(np.int8, copy=False)
        solutions = bank["solutions"].astype(np.int8, copy=False)
    expected_shape = (base_bank_size, 9, 9)
    if puzzles.shape != expected_shape or solutions.shape != expected_shape:
        raise ValueError(f"invalid Easy Sudoku base bank {path}: expected {expected_shape}")
    return puzzles, solutions


def prepare_base_bank(
    *,
    seed: int,
    rank: int,
    base_bank_size: int = 10_000,
    clues_min: int = 32,
    clues_max: int = 40,
    max_attempts: int = 1_024,
    base_bank_workers: int = 8,
    cache_path: str = "./downloaded-datasets/sudoku_easy_base_bank_seed{seed}_n{base_bank_size}_c{clues_min}-{clues_max}.npz",
    **_: object,
) -> Path:
    """Build a deterministic base bank on rank 0 before optimizer step 0."""
    path = _bank_path(cache_path, seed, base_bank_size, clues_min, clues_max)
    if path.exists():
        _load_base_bank(path, base_bank_size)
        return path
    if rank != 0:
        return path
    if not 0 < clues_min <= clues_max <= 81:
        raise ValueError("clue bounds must satisfy 0 < clues_min <= clues_max <= 81")

    path.parent.mkdir(parents=True, exist_ok=True)
    workers = max(1, min(base_bank_workers, base_bank_size, os.cpu_count() or 1))
    print(f"Building Easy Sudoku base bank ({base_bank_size} puzzles, {workers} CPU workers): {path}", flush=True)
    tasks = ((seed, index, clues_min, clues_max, max_attempts) for index in range(base_bank_size))
    if workers == 1:
        generated = map(_generate_base_puzzle, tasks)
        pairs = list(generated)
    else:
        try:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                pairs = list(executor.map(_generate_base_puzzle, tasks, chunksize=16))
        except OSError as error:
            # Some restricted environments disallow process semaphores. Falling
            # back keeps data generation correct, albeit slower.
            print(f"Parallel base-bank generation unavailable ({error}); using one CPU worker.", flush=True)
            tasks = ((seed, index, clues_min, clues_max, max_attempts) for index in range(base_bank_size))
            pairs = list(map(_generate_base_puzzle, tasks))
    puzzles = np.stack([puzzle for puzzle, _ in pairs])
    solutions = np.stack([solution for _, solution in pairs])

    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("wb") as output:
        np.savez_compressed(output, puzzles=puzzles, solutions=solutions)
    os.replace(temporary_path, path)
    return path


def _rng_for(seed: int, epoch: int, index: int) -> np.random.Generator:
    # Independent of DataLoader worker order and persistent-worker scheduling.
    return np.random.default_rng(np.random.SeedSequence([seed, epoch, index]))


def _apply_random_symmetry(
    puzzle: np.ndarray, solution: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the same Sudoku-preserving transformation to puzzle and solution."""
    bands = rng.permutation(3)
    rows = np.concatenate([band * 3 + rng.permutation(3) for band in bands])
    stacks = rng.permutation(3)
    cols = np.concatenate([stack * 3 + rng.permutation(3) for stack in stacks])
    puzzle, solution = puzzle[rows][:, cols], solution[rows][:, cols]
    if rng.integers(2):
        puzzle, solution = puzzle.T, solution.T

    digit_map = np.empty(10, dtype=np.int8)
    digit_map[0] = 0
    digit_map[1:] = rng.permutation(np.arange(1, 10, dtype=np.int8))
    return digit_map[puzzle], digit_map[solution]


def generate_puzzle(
    base_puzzles: np.ndarray,
    base_solutions: np.ndarray,
    seed: int,
    epoch: int,
    index: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample a base puzzle then create a deterministic, symmetry-augmented variant."""
    rng = _rng_for(seed, epoch, index)
    base_index = int(rng.integers(len(base_puzzles)))
    return _apply_random_symmetry(base_puzzles[base_index], base_solutions[base_index], rng)


class EasySudokuDataset(Dataset[tuple[np.ndarray, np.ndarray]]):
    """A virtual dataset that draws new deterministic variants from a base bank."""

    def __init__(self, size: int, seed: int, base_puzzles: np.ndarray, base_solutions: np.ndarray) -> None:
        self.size = size
        self.seed = seed
        self.base_puzzles = base_puzzles
        self.base_solutions = base_solutions
        # Shared memory lets persistent DataLoader workers observe epoch changes.
        self._epoch = Value("q", 0)

    def __len__(self) -> int:
        return self.size

    def set_epoch(self, epoch: int) -> None:
        with self._epoch.get_lock():
            self._epoch.value = epoch

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        with self._epoch.get_lock():
            epoch = self._epoch.value
        puzzle, solution = generate_puzzle(self.base_puzzles, self.base_solutions, self.seed, epoch, index)
        return np.pad(puzzle.reshape(-1), (1, 0)), np.pad(solution.reshape(-1), (1, 0))


def _collate(batch: list[tuple[np.ndarray, np.ndarray]]) -> tuple[Tensor, Tensor]:
    xs, ys = zip(*batch)
    return torch.from_numpy(np.stack(xs).astype(np.int32, copy=False)), torch.from_numpy(np.stack(ys).astype(np.int32, copy=False))


def create_dataloader(
    split: str,
    batch_size: int,
    rank: int,
    world_size: int,
    dataset_size: int = 200_000,
    base_bank_size: int = 10_000,
    clues_min: int = 32,
    clues_max: int = 40,
    max_attempts: int = 1_024,
    base_bank_workers: int = 8,
    cache_path: str = "./downloaded-datasets/sudoku_easy_base_bank_seed{seed}_n{base_bank_size}_c{clues_min}-{clues_max}.npz",
    num_workers: int = 1,
    seed: int = 42,
):
    if split != "train":
        raise ValueError("sudoku_easy only provides the training split; configure a separate hard evaluation dataset")

    path = _bank_path(cache_path, seed, base_bank_size, clues_min, clues_max)
    if not path.exists():
        prepare_base_bank(
            seed=seed, rank=rank, base_bank_size=base_bank_size, clues_min=clues_min, clues_max=clues_max,
            max_attempts=max_attempts, base_bank_workers=base_bank_workers, cache_path=cache_path,
        )
    base_puzzles, base_solutions = _load_base_bank(path, base_bank_size)
    dataset = EasySudokuDataset(dataset_size, seed, base_puzzles, base_solutions)
    loader_kwargs: dict[str, object] = {
        "batch_size": batch_size,
        "collate_fn": _collate,
        "sampler": DistributedSampler(
            dataset, rank=rank, num_replicas=world_size, shuffle=True, drop_last=True, seed=seed
        ),
        "drop_last": True,
        "pin_memory": True,
        "num_workers": num_workers,
    }
    if num_workers > 0:
        loader_kwargs.update(persistent_workers=True, prefetch_factor=2)

    return DataLoader(dataset, **loader_kwargs), {
        "vocab_size": 10,
        "seq_len": 82,
        "is_causal": False,
    }
