import argparse
import yaml
import os

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
    parser.add_argument("--split", type=str, default="test", help="Dataset split to evaluate on")
    args = parser.parse_args()

    # Load config
    with open(os.path.join(os.path.dirname(args.ckpt), "model_config.json"), "r") as f:
        config_dict = yaml.safe_load(f)
    config = TrainConfig(**config_dict)

    create_dataloader = load_module(f"dataset.{config.data.name}@create_dataloader")
    eval_loader, metadata = create_dataloader(
        args.split, config.local_batch_size, rank=0, world_size=1, **config.data.__pydantic_extra__  # pyright: ignore[reportCallIssue]
    )

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
    total_correct = 0
    total_samples = 0
    correctness = []
    samples = []

    print(f"Starting evaluation on '{args.split}' split...")
    for x, y in tqdm.tqdm(eval_loader):
        samples.append(x.numpy())

        x, y = x.cuda(), y.cuda()
        carry: Carry = model.initial_carry  # pyright: ignore[reportAssignmentType]
        y_hat = None
        for _ in range(config.cycles_per_data):
            carry, y_hat = run_inference(model, carry, x)
        correctness.append(torch.all(y_hat == y, dim=-1).cpu().numpy())
        total_correct += torch.all(y_hat == y, dim=-1).sum().item()
        total_samples += y.shape[0]

    np.savez(os.path.join(os.path.dirname(args.ckpt), "eval_result.npz"),
             correctness=np.concat(correctness, axis=0),
             samples=np.concat(samples, axis=0))

    print(f"\n--- Results ---")
    print(f"Total Samples: {total_samples}")
    print(f"Exact Match Accuracy: {total_correct / total_samples:.4f}")

if __name__ == "__main__":
    evaluate()
