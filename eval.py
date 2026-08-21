import argparse
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn
import tqdm
import yaml

from arch.layers import Carry
from train import TrainConfig, load_module, run_inference


def evaluate_checkpoint(
    ckpt: str,
    split: str = "test",
    eval_rating_min: int | None = None,
    eval_rating_max: int | None = None,
    eval_dataset_name: str | None = None,
    output: str | None = None,
    save_samples: bool = False,
) -> dict[str, float | int]:
    """Evaluate one checkpoint and return exact-match metrics.

    The ``eval_rating_*`` arguments filter only the evaluation split. They do
    not alter the training rating bounds stored in ``model_config.json``.
    """
    with open(os.path.join(os.path.dirname(ckpt), "model_config.json"), "r") as f:
        config = TrainConfig(**yaml.safe_load(f))

    create_dataloader = load_module(f"dataset.{config.data.name}@create_dataloader")
    data_kwargs = dict(config.data.__pydantic_extra__ or {})
    if eval_rating_min is not None:
        data_kwargs["eval_rating_min"] = eval_rating_min
    if eval_rating_max is not None:
        data_kwargs["eval_rating_max"] = eval_rating_max
    if eval_dataset_name is not None:
        data_kwargs["eval_dataset_name"] = eval_dataset_name

    # Do not silently discard a final partial batch during standalone eval.
    eval_loader, metadata = create_dataloader(
        split,
        config.local_batch_size,
        rank=0,
        world_size=1,
        drop_last=False,
        **data_kwargs,  # pyright: ignore[reportCallIssue]
    )

    model_cls = load_module(f"arch.{config.arch.name}")
    with torch.device("cuda"):
        model = model_cls(config.arch.__pydantic_extra__ | metadata)
        state_dict = torch.load(ckpt, map_location="cuda", weights_only=True)
        model.load_state_dict(state_dict, assign=True)
        model: nn.Module = torch.compile(model, dynamic=False, fullgraph=True)  # pyright: ignore[reportAssignmentType]
        model.eval()

    total_correct = 0
    total_samples = 0
    correctness = []
    samples = [] if save_samples else None

    print(f"Starting evaluation on '{split}' split...")
    for x, y in tqdm.tqdm(eval_loader):
        if samples is not None:
            samples.append(x.numpy())

        x, y = x.cuda(), y.cuda()
        carry: Carry = model.initial_carry  # pyright: ignore[reportAssignmentType]
        y_hat = None
        for _ in range(config.cycles_per_data):
            carry, y_hat = run_inference(model, carry, x)
        batch_correctness = torch.all(y_hat == y, dim=-1)
        correctness.append(batch_correctness.cpu().numpy())
        total_correct += batch_correctness.sum().item()
        total_samples += y.shape[0]

    if total_samples == 0:
        raise RuntimeError(f"evaluation split '{split}' is empty")

    accuracy = total_correct / total_samples
    if output is not None:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result = {
            "correctness": np.concat(correctness, axis=0),
            "total_samples": np.array(total_samples),
            "exact_match_accuracy": np.array(accuracy),
        }
        if samples is not None:
            result["samples"] = np.concat(samples, axis=0)
        np.savez(output_path, **result)

    print("\n--- Results ---")
    print(f"Total Samples: {total_samples}")
    print(f"Exact Match Accuracy: {accuracy:.4f}")
    return {"total_samples": total_samples, "exact_match_accuracy": accuracy}


def evaluate():
    parser = argparse.ArgumentParser(description="Evaluate a trained model.")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to the saved checkpoint (.pt file)")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to evaluate on")
    parser.add_argument("--eval-rating-min", type=int, help="Optional inclusive lower Sudoku rating bound for evaluation")
    parser.add_argument("--eval-rating-max", type=int, help="Optional inclusive upper Sudoku rating bound for evaluation")
    parser.add_argument("--eval-dataset-name", help="Optional dataset path/name to use instead of the checkpoint's eval_dataset_name")
    parser.add_argument("--output", type=str, help="Optional .npz output path (default: checkpoint-dir/eval_result.npz)")
    parser.add_argument("--save-samples", action="store_true", help="Include input samples in the optional .npz output")
    args = parser.parse_args()
    evaluate_checkpoint(
        args.ckpt,
        split=args.split,
        eval_rating_min=args.eval_rating_min,
        eval_rating_max=args.eval_rating_max,
        eval_dataset_name=args.eval_dataset_name,
        # Preserve the original CLI behaviour: without extra flags it writes
        # eval_result.npz, including the input samples.
        output=args.output or os.path.join(os.path.dirname(args.ckpt), "eval_result.npz"),
        save_samples=args.save_samples or args.output is None,
    )


if __name__ == "__main__":
    evaluate()
