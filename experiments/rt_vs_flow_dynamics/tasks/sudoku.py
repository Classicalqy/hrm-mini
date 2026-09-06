from __future__ import annotations

from hashlib import sha256
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


class SyntheticSudokuDataset(Dataset[tuple[Tensor, Tensor]]):
    """Small deterministic tensor-only dataset used by smoke tests."""

    def __init__(self, count: int, seed: int) -> None:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        targets = torch.randint(1, 10, (count, 82), generator=generator, dtype=torch.long)
        targets[:, 0] = 0
        keep = torch.rand(count, 82, generator=generator) < 0.32
        keep[:, 0] = True
        inputs = torch.where(keep, targets, torch.zeros_like(targets))
        self.inputs, self.targets = inputs, targets

    def __len__(self) -> int:
        return self.inputs.shape[0]

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        return self.inputs[index], self.targets[index]


def create_sudoku_loaders(
    config: dict[str, Any], rank: int = 0, world_size: int = 1, smoke: bool = False
) -> tuple[DataLoader, DataLoader, dict[str, int | bool]]:
    if smoke:
        batch_size = int(config.get("batch_size", 4))
        train = SyntheticSudokuDataset(int(config.get("train_count", 16)), int(config.get("seed", 1)))
        test = SyntheticSudokuDataset(int(config.get("test_count", 8)), int(config.get("seed", 1)) + 1)
        return (
            DataLoader(train, batch_size=batch_size, shuffle=False),
            DataLoader(test, batch_size=batch_size, shuffle=False),
            {"vocab_size": 10, "seq_len": 82, "is_causal": False},
        )

    from dataset.sudoku import create_dataloader

    common = dict(config)
    batch_size = int(common.pop("batch_size"))
    seed = int(common.pop("seed", 42))
    eval_split = str(common.pop("eval_split", "test_hard"))
    train_loader, metadata = create_dataloader(
        "train", batch_size, rank=rank, world_size=world_size, seed=seed, **common
    )
    eval_config = dict(common)
    eval_config["augment"] = False
    eval_config["repeat"] = 1
    eval_loader, _ = create_dataloader(
        eval_split, batch_size, rank=rank, world_size=world_size, seed=seed, **eval_config
    )
    return train_loader, eval_loader, metadata


def sudoku_violations(predictions: Tensor) -> Tensor:
    grid = predictions[:, 1:].reshape(-1, 9, 9)
    digits = torch.arange(1, 10, device=grid.device)
    one_hot = grid.unsqueeze(-1) == digits

    def duplicate_pairs(counts: Tensor) -> Tensor:
        return (counts * (counts - 1) // 2).sum(dim=tuple(range(1, counts.ndim)))

    rows = one_hot.sum(dim=2)
    columns = one_hot.sum(dim=1)
    boxes = (
        one_hot.reshape(grid.shape[0], 3, 3, 3, 3, 9)
        .permute(0, 1, 3, 2, 4, 5)
        .reshape(grid.shape[0], 3, 3, 9, 9)
        .sum(dim=3)
    )
    return (grid == 0).sum(dim=(1, 2)) + duplicate_pairs(rows) + duplicate_pairs(columns) + duplicate_pairs(boxes)


def correct_margin(logits: Tensor, targets: Tensor) -> Tensor:
    logits = logits[:, 1:].to(torch.float32)
    targets = targets[:, 1:].long()
    correct = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    alternatives = logits.scatter(-1, targets.unsqueeze(-1), float("-inf")).amax(dim=-1)
    return (correct - alternatives).mean(dim=-1)


def tensor_dataset_fingerprint(inputs: Tensor, targets: Tensor) -> str:
    digest = sha256()
    digest.update(inputs.detach().cpu().numpy().tobytes())
    digest.update(targets.detach().cpu().numpy().tobytes())
    return digest.hexdigest()
