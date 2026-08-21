import numpy as np
import torch
from torch import Tensor
from datasets import load_dataset, Features, Value
from torch.utils.data import DataLoader, DistributedSampler

VOCAB = "~# SGo"
CHAR_TO_ID = {char: idx for idx, char in enumerate(VOCAB)}

def collate_fn(batch: list[dict[str, str]]) -> tuple[Tensor, Tensor]:
    xs, ys = [], []
    for item in batch:
        # Pad BOS, then tokenize
        board_ids = [CHAR_TO_ID[char] for char in "~" + item["question"]]
        solution_ids = [CHAR_TO_ID[char] for char in "~" + item["answer"]]

        board = np.array(board_ids, dtype=np.int32)
        solution = np.array(solution_ids, dtype=np.int32)

        xs.append(board)
        ys.append(solution)

    return torch.from_numpy(np.stack(xs, axis=0)), torch.from_numpy(np.stack(ys, axis=0))

def create_dataloader(
    split: str,
    batch_size: int,
    rank: int,
    world_size: int,
    dataset_name: str,
    repeat: int = 1,
    seed: int = 42,
    drop_last: bool = True,
):
    is_train = split == "train"
    dataset = load_dataset(dataset_name, split=split, features = Features({
        "question": Value("string"),
        "answer": Value("string"),
    }))
    dataset = dataset.repeat(repeat if is_train else 1)  # pyright: ignore[reportAssignmentType]

    return DataLoader(
        dataset,  # pyright: ignore[reportArgumentType]
        batch_size=batch_size,
        collate_fn=collate_fn,

        sampler=DistributedSampler(dataset,  # pyright: ignore[reportArgumentType]
                                   rank=rank, num_replicas=world_size,
                                   shuffle=is_train, drop_last=drop_last,
                                   seed=seed),
        drop_last=drop_last,

        pin_memory=True,
        persistent_workers=True,

        num_workers=1,
        prefetch_factor=2
    ), {
        # Dataset metadata
        "vocab_size": len(VOCAB),
        "seq_len": 901,
        "is_causal": False
    }
