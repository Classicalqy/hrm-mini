from typing import Any, Optional
import os
import importlib
import math
import yaml

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import nn
from torch import Tensor
import torch.nn.functional as F

import pydantic
import hydra
import tqdm
import wandb
import coolname
from hydra.core.hydra_config import HydraConfig

from adam_atan2 import AdamATan2
from arch.layers import Carry

class ArchConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='allow')
    name: str

class DataConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='allow')
    name: str

class TrainConfig(pydantic.BaseModel):
    arch: ArchConfig
    data: DataConfig

    seeds: list[int] = [42]

    cycles_per_data: int
    epochs: int

    local_batch_size: int

    lr: float
    lr_warmup_steps: int
    lr_min_ratio: float

    weight_decay: float
    beta1: float
    beta2: float
    ema: Optional[float] = None

    log_interval: int = 5

# [Utils]
def load_module(identifier: str):
    module_path, class_name = identifier.split('@')
    # Import the module
    module = importlib.import_module(module_path)
    return getattr(module, class_name)

# [Training and Inference Step]
def train_step(model: nn.Module, carry: Carry, opt: torch.optim.Optimizer, x: Tensor, y: Tensor):
    carry, y_hat = model(carry, x)
    # loss (f32 for CrossEntropy)
    loss = F.cross_entropy(y_hat.view(-1, y_hat.shape[-1]).to(torch.float32), y.view(-1).long(), reduction="mean")
    loss.backward()
    opt.step()
    opt.zero_grad()

    # metrics
    with torch.no_grad():
        preds = torch.argmax(y_hat, dim=-1)
        metrics = {
            "loss": loss.detach(),
            "per_position_accuracy": torch.mean(preds == y, dtype=torch.float32),
            "exact_match": torch.mean(torch.all(preds == y, dim=-1), dtype=torch.float32)
        }

    return carry, metrics

@torch.inference_mode()
def run_inference(model: nn.Module, carry: Carry, x: Tensor):
    carry, y_hat = model(carry, x)
    return carry, torch.argmax(y_hat, dim=-1)

def update_lr(config: TrainConfig, optim: torch.optim.Optimizer, step: int, total_steps: int) -> float:
    # Linear warmup cosine schedule
    if step < config.lr_warmup_steps:
        lr = config.lr * min(1.0, step / config.lr_warmup_steps)
    else:
        progress = (step - config.lr_warmup_steps) / (total_steps - config.lr_warmup_steps)
        lr = config.lr * (config.lr_min_ratio + max(0.0, (1 - config.lr_min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))))

    tensor_lr = torch.tensor(lr, dtype=torch.get_default_dtype(), device="cpu")
    for param_group in optim.param_groups:
        param_group["lr"] = tensor_lr

    return lr


def train_single_seed(config: TrainConfig, seed: int, group_name: str, WORLD_SIZE: int, RANK: int):
    """Run a full training run for a single seed."""
    # Set random seeds
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Initialize Dataloader
    create_dataloader = load_module(f"dataset.{config.data.name}@create_dataloader")
    train_loader, train_metadata = create_dataloader("train", config.local_batch_size, rank=RANK, world_size=WORLD_SIZE, seed=seed, **config.data.__pydantic_extra__)  # pyright: ignore[reportCallIssue]
    eval_loaders = {split_name: create_dataloader(split_name, config.local_batch_size, rank=RANK, world_size=WORLD_SIZE, seed=seed, **config.data.__pydantic_extra__)[0] for split_name in ["test_hard"]}  # pyright: ignore[reportCallIssue]

    total_steps = int(config.cycles_per_data * len(train_loader) * config.epochs)

    # Initialize Model and Optimizer
    model_cls = load_module(f"arch.{config.arch.name}")
    with torch.device("cuda"):
        model: nn.Module = model_cls(config.arch.__pydantic_extra__ | train_metadata)
        model = torch.compile(model, dynamic=False, fullgraph=True)  # pyright: ignore[reportAssignmentType]

        # DDP Wrap
        model = DDP(model, static_graph=True)

    optim = AdamATan2(
        model.parameters(),
        lr=torch.tensor(0.0, dtype=torch.get_default_dtype(), device="cpu"),
        betas=(config.beta1, config.beta2),
        weight_decay=config.weight_decay,
        ema=config.ema
    )

    # Initialize checkpointing
    run_name = f"{group_name}/seed_{seed}"
    checkpoint_dir = os.path.join("checkpoints", run_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    with open(os.path.join(checkpoint_dir, "model_config.json"), "w") as f:
        yaml.dump(config.model_dump(), f)

    # -----Train & Eval loop
    progress_bar = None
    if RANK == 0:
        progress_bar = tqdm.tqdm(total=total_steps, desc=f"seed={seed}")

        wandb.init(project=config.data.name,
                   name=run_name,
                   group=group_name,
                   config=config.model_dump() | {"seed": seed},
                   settings=wandb.Settings(x_disable_stats=True))
        if wandb.run is not None:
            wandb.run.log_code()

    step = 0
    for epoch in range(config.epochs):
        model.train()
        for x, y in train_loader:
            x = x.cuda()
            y = y.cuda()

            metrics = {}
            lr = None
            carry: Carry = model.module.initial_carry
            for _ in range(config.cycles_per_data):
                step += 1
                lr = update_lr(config, optim, step, total_steps)

                carry, metrics = train_step(model, carry, optim, x, y)

            if RANK == 0 and progress_bar is not None and step - progress_bar.n >= config.log_interval:
                progress_bar.update(step - progress_bar.n)
                wandb.log({f"train/{k}": v.item() for k, v in metrics.items()} | {"train/lr": lr}, step=step)

            del x, y, carry, metrics

        # Eval
        model.eval()
        optim.swap_ema()

        # Save model
        # Clean '_orig_mod.' prefix added by torch.compile for easier downstream loading
        torch.save({k.replace("_orig_mod.", ""): v for k, v in model.module.state_dict().items()},
                   os.path.join(checkpoint_dir, f"epoch_{epoch}.pt"))

        for eval_name, eval_loader in eval_loaders.items():
            num_total_correct = torch.zeros(2, dtype=torch.long, device="cuda")
            for x, y in eval_loader:
                # Run inference
                carry: Carry = model.module.initial_carry
                y_hat = None
                for _ in range(config.cycles_per_data):
                    carry, y_hat = run_inference(model, carry, x.cuda())

                num_total_correct[0] += torch.all(y_hat == y.cuda(), dim=-1).sum()
                num_total_correct[1] += y.shape[0]

                del carry, y_hat

            # Reduce and log
            dist.reduce(num_total_correct, dst=0)
            num_total_correct = num_total_correct.cpu().tolist()
            if RANK == 0:
                wandb.log({f"eval/{eval_name}_exact_match": num_total_correct[0] / num_total_correct[1]}, step=step)

        optim.swap_ema()  # Swap EMA back

    # Close progress bar and wandb run for this seed
    if progress_bar is not None:
        progress_bar.close()
    if RANK == 0:
        wandb.finish()


# [Training Loop]
@hydra.main(config_path="config", version_base=None)
def train(config_dict: dict[str, Any]):
    WORLD_SIZE = 1
    RANK = 0
    DEVICE_ID = 0

    # Initialize distributed training if in distributed environment (e.g. torchrun)
    if "LOCAL_RANK" in os.environ:
        # Initialize distributed, default device and dtype
        dist.init_process_group(backend="nccl")

        WORLD_SIZE = dist.get_world_size()
        RANK = dist.get_rank()
        DEVICE_ID = int(os.environ["LOCAL_RANK"])

        torch.cuda.set_device(DEVICE_ID)

    # Load config
    config = TrainConfig(**config_dict)

    # Generate a shared group name for all seeds in this run
    group_name = os.environ.get("MLP_TASK_NAME", f"{HydraConfig.get().job.config_name} {coolname.generate_slug(2)}")

    for seed in config.seeds:
        train_single_seed(config, seed, group_name, WORLD_SIZE, RANK)

if __name__ == "__main__":
    train()
