from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from .models.flow import FlowMatchingTransformer
from .models.native_rt import NativeRTAdapter
from .trace import (
    collect_sudoku_trace,
    detect_fixed_points,
    file_sha256,
    perturb_wrong_fixed_points,
    save_trace,
    write_rows,
)
from .train import (
    build_matched_model,
    load_config,
    resolve_device,
    train_sudoku,
    train_toy,
)
from .tasks.sudoku import create_sudoku_loaders


PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_SMOKE_CONFIG = PACKAGE_ROOT / "configs" / "smoke.yaml"
DEFAULT_FULL_CONFIG = PACKAGE_ROOT / "configs" / "full.yaml"


def _checkpoint_metadata(checkpoint: Path) -> dict[str, Any]:
    metadata_path = checkpoint.parent / "model_config.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"missing checkpoint metadata: {metadata_path}")
    return json.loads(metadata_path.read_text())


def _load_matched(checkpoint: Path, config: dict[str, Any], device: torch.device) -> tuple[torch.nn.Module, str, int]:
    metadata = _checkpoint_metadata(checkpoint)
    condition = str(metadata["condition"])
    seed = int(metadata["seed"])
    stored_model_config = metadata.get("config", {}).get("model")
    if stored_model_config is not None and stored_model_config != config["model"]:
        raise ValueError("checkpoint model config does not match the requested experiment config")
    model = build_matched_model(config, metadata["model_metadata"], condition).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.eval()
    return model, condition, seed


def _eval_loader(config: dict[str, Any]):
    sudoku_config = dict(config["sudoku"])
    smoke = bool(sudoku_config.pop("smoke", False))
    return create_sudoku_loaders(sudoku_config, smoke=smoke)[1]


def trace_checkpoint(
    config: dict[str, Any], checkpoint: Path, native: bool = False, perturb: bool = False
) -> list[Path]:
    device = resolve_device(config)
    if native:
        model = NativeRTAdapter.from_checkpoint(checkpoint, device)
        condition, seed = "native_rt", int(checkpoint.parent.name.removeprefix("seed_"))
        loader = _eval_loader(config)
        first_inputs = next(iter(loader))[0].to(device)
        model.validate_group_parity(first_inputs)
        step_counts = [int(config["trace"]["native_steps"])]
    else:
        model, condition, seed = _load_matched(checkpoint, config, device)
        loader = _eval_loader(config)
        if isinstance(model, FlowMatchingTransformer):
            step_counts = [int(value) for value in config["trace"]["flow_nfe"]]
        else:
            step_counts = [int(config["trace"]["matched_steps"])]
    output_paths = []
    for steps in step_counts:
        arrays = collect_sudoku_trace(
            model, loader, steps, int(config["trace"]["sample_count"]),
            int(config["trace"]["projection_dim"]), int(seed) + 20260906, device,
        )
        output_path = Path(config["experiment"]["output_root"]) / "traces" / f"{condition}_seed_{seed}_steps_{steps}.npz"
        save_trace(output_path, arrays, {
            "condition": condition, "seed": seed, "steps": steps,
            "checkpoint": str(checkpoint), "checkpoint_sha256": file_sha256(checkpoint),
            "model_kind": "native_rt" if native else config["conditions"][condition]["kind"],
            "config": config,
        })
        output_paths.append(output_path)
        if perturb and not isinstance(model, FlowMatchingTransformer):
            trace_config = config["trace"]
            onset = detect_fixed_points(
                arrays, int(trace_config["fixed_window"]), float(trace_config["state_eps"]),
                float(trace_config["logit_eps"]),
            )
            perturbation, jacobian = perturb_wrong_fixed_points(
                model, arrays, onset, steps, int(trace_config["perturb_continue_steps"]),
                [float(value) for value in trace_config["perturb_sigmas"]],
                int(trace_config["perturb_directions"]), int(trace_config["perturb_max_puzzles"]),
                int(trace_config["jacobian_max_puzzles"]), int(trace_config["jacobian_iterations"]),
                int(seed) + 991, device,
            )
            write_rows(output_path.with_name(output_path.stem + "_perturbation.csv"), perturbation)
            write_rows(output_path.with_name(output_path.stem + "_jacobian.csv"), jacobian)
    return output_paths


def run_smoke(config_path: Path) -> None:
    from .analyze import analyze_traces, plot_toy_results

    config = load_config(config_path)
    train_toy(config); plot_toy_results(config)
    train_sudoku(config, ["matched_rt", "rt_smooth", "flow_linear", "flow_bent_05"])
    checkpoint_root = Path(config["experiment"]["checkpoint_root"])
    epoch = int(config["schedule"]["epochs"]) - 1
    for condition in ("matched_rt", "flow_linear"):
        checkpoint = checkpoint_root / condition / "seed_1" / f"epoch_{epoch}.pt"
        trace_checkpoint(config, checkpoint, native=False, perturb=condition == "matched_rt")
    analyze_traces(config)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RT versus Flow dynamics experiment")
    subparsers = parser.add_subparsers(dest="command", required=True)
    smoke = subparsers.add_parser("smoke", help="Run the complete CPU smoke pipeline")
    smoke.add_argument("--config", type=Path, default=DEFAULT_SMOKE_CONFIG)
    toy = subparsers.add_parser("toy", help="Train and evaluate analytic 2-D trajectories")
    toy.add_argument("--config", type=Path, default=DEFAULT_FULL_CONFIG)
    train = subparsers.add_parser("train-sudoku", help="Train controlled Sudoku conditions")
    train.add_argument("--config", type=Path, default=DEFAULT_FULL_CONFIG)
    train.add_argument("--condition", nargs="+", default=["matched_rt", "flow_linear", "rt_smooth"])
    trace = subparsers.add_parser("trace-sudoku", help="Trace a matched or native RT checkpoint")
    trace.add_argument("--config", type=Path, default=DEFAULT_FULL_CONFIG)
    trace.add_argument("--checkpoint", type=Path, required=True)
    trace.add_argument("--native", action="store_true")
    trace.add_argument("--perturb", action="store_true")
    analyze = subparsers.add_parser("analyze", help="Aggregate saved traces")
    analyze.add_argument("--config", type=Path, default=DEFAULT_FULL_CONFIG)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "smoke":
        run_smoke(args.config)
    elif args.command == "toy":
        from .analyze import plot_toy_results
        config = load_config(args.config); train_toy(config); plot_toy_results(config)
    elif args.command == "train-sudoku":
        train_sudoku(load_config(args.config), list(args.condition))
    elif args.command == "trace-sudoku":
        trace_checkpoint(load_config(args.config), args.checkpoint, native=args.native, perturb=args.perturb)
    elif args.command == "analyze":
        from .analyze import analyze_traces
        analyze_traces(load_config(args.config))


if __name__ == "__main__":
    main()
