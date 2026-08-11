from functools import partial

import numpy as np
import torch
from torch import Tensor
from datasets import load_dataset, Features, Value
from torch.utils.data import Dataset, DataLoader, DistributedSampler

def shuffle_sudoku(board: np.ndarray, solution: np.ndarray):
    # Create a random digit mapping: a permutation of 1..9, with zero (blank) unchanged
    digit_map = np.pad(np.random.permutation(np.arange(1, 10)), (1, 0))
    
    # Randomly decide whether to transpose.
    transpose_flag = np.random.rand() < 0.5

    # Generate a valid row permutation:
    # - Shuffle the 3 bands (each band = 3 rows) and for each band, shuffle its 3 rows.
    bands = np.random.permutation(3)
    row_perm = np.concatenate([b * 3 + np.random.permutation(3) for b in bands])

    # Similarly for columns (stacks).
    stacks = np.random.permutation(3)
    col_perm = np.concatenate([s * 3 + np.random.permutation(3) for s in stacks])

    # Build an 81->81 mapping. For each new cell at (i, j)
    # (row index = i // 9, col index = i % 9),
    # its value comes from old row = row_perm[i//9] and old col = col_perm[i%9].
    mapping = np.array([row_perm[i // 9] * 9 + col_perm[i % 9] for i in range(81)])

    def apply_transformation(x: np.ndarray) -> np.ndarray:
        # Apply transpose flag
        if transpose_flag:
            x = x.T
        # Apply the position mapping.
        new_board = x.flatten()[mapping].reshape(9, 9).copy()
        # Apply digit mapping
        return digit_map[new_board]

    return apply_transformation(board), apply_transformation(solution)

def collate_fn(batch: list[dict[str, str]], augment: bool) -> tuple[Tensor, Tensor]:
    xs, ys = [], []
    for item in batch:
        board = np.frombuffer(item["question"].replace('.', '0').encode(), dtype=np.uint8).reshape(9, 9) - ord('0')
        solution = np.frombuffer(item["answer"].encode(), dtype=np.uint8).reshape(9, 9) - ord('0')
        if augment:
            board, solution = shuffle_sudoku(board, solution)

        # Convert and flatten
        board = board.flatten().astype(np.int32)
        solution = solution.flatten().astype(np.int32)
        # Pad a BOS token
        xs.append(np.pad(board, (1, 0)))
        ys.append(np.pad(solution, (1, 0)))

    return torch.from_numpy(np.stack(xs, axis=0)), torch.from_numpy(np.stack(ys, axis=0))

def _worker_init_fn(worker_id: int, base_seed: int):
    np.random.seed(base_seed + worker_id)


def _select_training_puzzles(
    dataset: Dataset,
    seed: int,
    rating_min: int | None,
    rating_max: int | None,
    num_base_puzzles: int | None,
) -> Dataset:
    """Filter official Sudoku-Extreme train data before repeat/augmentation."""
    if rating_min is not None or rating_max is not None:
        if "rating" not in dataset.column_names:
            raise ValueError("rating filtering requires a dataset with a 'rating' column")
        lower = rating_min if rating_min is not None else -np.inf
        upper = rating_max if rating_max is not None else np.inf
        dataset = dataset.filter(lambda rating: lower <= rating <= upper, input_columns="rating")  # pyright: ignore[reportAttributeAccessIssue]
        if len(dataset) == 0:
            raise ValueError(f"no training puzzles have ratings in [{rating_min}, {rating_max}]")

    if num_base_puzzles is not None:
        if num_base_puzzles <= 0:
            raise ValueError("num_base_puzzles must be positive")
        if len(dataset) < num_base_puzzles:
            raise ValueError(f"requested {num_base_puzzles} base puzzles but only {len(dataset)} are available")
        dataset = dataset.shuffle(seed=seed).select(range(num_base_puzzles))  # pyright: ignore[reportAttributeAccessIssue]
    return dataset


def create_dataloader(
    split: str,
    batch_size: int,
    rank: int,
    world_size: int,
    dataset_name: str,
    eval_dataset_name: str | None = None,
    augment: bool = False,
    repeat: int = 1,
    rating_min: int | None = None,
    rating_max: int | None = None,
    num_base_puzzles: int | None = None,
    num_workers: int = 1,
    seed: int = 42,
):
    is_train = split == "train"
    source_dataset_name = dataset_name if is_train or eval_dataset_name is None else eval_dataset_name
    dataset: Dataset = load_dataset(source_dataset_name, split=split, features = Features({
        "source": Value("string"),
        "question": Value("string"),
        "answer": Value("string"),
        "rating": Value("int64"),
    }))  # pyright: ignore[reportAssignmentType]
    if is_train:
        dataset = _select_training_puzzles(dataset, seed, rating_min, rating_max, num_base_puzzles)
        dataset = dataset.repeat(repeat)  # pyright: ignore[reportAssignmentType]

    loader_kwargs: dict[str, object] = {
        "batch_size": batch_size,
        "collate_fn": partial(collate_fn, augment=augment and is_train),
        "sampler": DistributedSampler(
            dataset, rank=rank, num_replicas=world_size,
            shuffle=is_train, drop_last=True, seed=seed,
        ),
        "drop_last": True,
        "pin_memory": True,
        "worker_init_fn": partial(_worker_init_fn, base_seed=seed),
        "num_workers": num_workers,
    }
    if num_workers > 0:
        loader_kwargs.update(persistent_workers=True, prefetch_factor=2)

    return DataLoader(dataset, **loader_kwargs), {
        # Dataset metadata
        "vocab_size": 10,
        "seq_len": 82,
        "is_causal": False
    }
