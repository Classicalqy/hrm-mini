import argparse
import yaml
import os
from pathlib import Path

import torch
from torch import nn
import tqdm
import numpy as np

from arch.layers import Carry
from train import TrainConfig, load_module, run_inference

# --- Main Eval Logic ---
def evaluate():
    parser = argparse.ArgumentParser(description="Evaluate a trained model.")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to the saved checkpoint (.pt file)")
    parser.add_argument("--split", type=str, default=None, help="Dataset split to evaluate using the training dataset")
    parser.add_argument("--eval-name", type=str, default=None, help="Named evaluation from the checkpoint config")
    parser.add_argument("--rollout-cycles", type=int, nargs="+", default=None, help="External rollout counts to evaluate")
    args = parser.parse_args()

    # Load config
    with open(os.path.join(os.path.dirname(args.ckpt), "model_config.json"), "r") as f:
        config_dict = yaml.safe_load(f)
    config = TrainConfig(**config_dict)

    if args.eval_name is not None and args.split is not None:
        parser.error("--eval-name and --split cannot be used together")
    if args.eval_name is not None:
        evaluations = [evaluation for evaluation in config.evals if evaluation.name == args.eval_name]
        if not evaluations:
            parser.error(f"unknown evaluation '{args.eval_name}'; available: {[evaluation.name for evaluation in config.evals]}")
        evaluation = evaluations[0]
        data, split, eval_label = evaluation.data, evaluation.split, evaluation.name
    else:
        data, split, eval_label = config.data, args.split or "test", args.split or "test"

    create_dataloader = load_module(f"dataset.{data.name}@create_dataloader")
    eval_loader, metadata = create_dataloader(
        split, config.local_batch_size, rank=0, world_size=1, seed=0, **(data.__pydantic_extra__ or {})
    )
    rollout_cycles = sorted(set(args.rollout_cycles or [config.cycles_per_data]))
    if not rollout_cycles or rollout_cycles[0] < 1:
        parser.error("--rollout-cycles must contain positive integers")

    # Initialize Model
    model_cls = load_module(f"arch.{config.arch.name}")
    with torch.device("cuda"):
        model = model_cls(config.arch.__pydantic_extra__ | metadata)
        # Load Checkpoint
        state_dict = torch.load(args.ckpt, map_location="cuda", weights_only=True)
        model.load_state_dict(state_dict, assign=True)
        model: nn.Module = torch.compile(model, dynamic=False, fullgraph=True)  # pyright: ignore[reportAssignmentType]

        model.eval()

    # Evaluation Loop
    total_samples = 0
    exact_correct = {cycles: 0 for cycles in rollout_cycles}
    position_correct = {cycles: 0 for cycles in rollout_cycles}
    correctness = []
    samples = []

    print(f"Starting evaluation '{eval_label}' on '{split}' split for rollouts {rollout_cycles}...")
    for x, y in tqdm.tqdm(eval_loader):
        samples.append(x.numpy())

        x, y = x.cuda(), y.cuda()
        carry: Carry = model.initial_carry  # pyright: ignore[reportAssignmentType]
        for cycles in range(1, rollout_cycles[-1] + 1):
            carry, y_hat = run_inference(model, carry, x)
            if cycles in exact_correct:
                is_correct = torch.all(y_hat == y, dim=-1)
                exact_correct[cycles] += is_correct.sum().item()
                position_correct[cycles] += (y_hat == y).sum().item()
                if cycles == rollout_cycles[-1]:
                    correctness.append(is_correct.cpu().numpy())
        total_samples += y.shape[0]

    result_path = Path(args.ckpt).with_name(f"eval_result_{Path(args.ckpt).stem}_{eval_label}.npz")
    np.savez(result_path,
             correctness=np.concat(correctness, axis=0),
             samples=np.concat(samples, axis=0),
             rollout_cycles=np.asarray(rollout_cycles),
             exact_match=np.asarray([exact_correct[cycles] / total_samples for cycles in rollout_cycles]),
             per_position_accuracy=np.asarray([position_correct[cycles] / (total_samples * 82) for cycles in rollout_cycles]))

    print(f"\n--- Results ---")
    print(f"Total Samples: {total_samples}")
    for cycles in rollout_cycles:
        print(
            f"Rollout {cycles}: exact match={exact_correct[cycles] / total_samples:.4f}, "
            f"per-position accuracy={position_correct[cycles] / (total_samples * 82):.4f}"
        )
    print(f"Saved results to {result_path}")

if __name__ == "__main__":
    evaluate()
