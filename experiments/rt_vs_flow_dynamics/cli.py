from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics
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
    evaluate_sudoku,
    train_sudoku,
    train_toy,
)
from .tasks.sudoku import create_sudoku_loaders


PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_SMOKE_CONFIG = PACKAGE_ROOT / "configs" / "smoke.yaml"
DEFAULT_FULL_CONFIG = PACKAGE_ROOT / "configs" / "full.yaml"
DEFAULT_FLOW_V2_CONFIG = PACKAGE_ROOT / "configs" / "flow_v2.yaml"
DEFAULT_FLOW_V2_OVERFIT_CONFIG = PACKAGE_ROOT / "configs" / "flow_v2_overfit.yaml"


def _override_seeds(config: dict[str, Any], seeds: list[int] | None) -> dict[str, Any]:
    if seeds is None:
        return config
    updated = dict(config)
    updated["experiment"] = dict(config["experiment"]) | {"seeds": list(seeds)}
    return updated


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
    return create_sudoku_loaders(
        sudoku_config, smoke=smoke, eval_partition=None if smoke else "test"
    )[1]


def trace_checkpoint(
    config: dict[str, Any], checkpoint: Path, native: bool = False, perturb: bool = False,
    label: str | None = None,
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
    source_condition = condition
    output_condition = label or condition
    output_paths = []
    for steps in step_counts:
        arrays = collect_sudoku_trace(
            model, loader, steps, int(config["trace"]["sample_count"]),
            int(config["trace"]["projection_dim"]), int(seed) + 20260906, device,
        )
        output_path = Path(config["experiment"]["output_root"]) / "traces" / f"{output_condition}_seed_{seed}_steps_{steps}.npz"
        save_trace(output_path, arrays, {
            "condition": output_condition, "source_condition": source_condition,
            "seed": seed, "steps": steps,
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


def trace_fair_comparison(
    config: dict[str, Any], flow_condition: str, rt_condition: str = "matched_rt"
) -> list[Path]:
    root = Path(config["experiment"]["output_root"])
    gate_path = root / "evaluation" / "flow_gate.json"
    if not gate_path.is_file() or not bool(json.loads(gate_path.read_text()).get("passed")):
        raise RuntimeError("Flow performance gate must pass before comparison traces are generated")
    table = root / "evaluation" / f"fair_comparison_{flow_condition}_vs_{rt_condition}.csv"
    with table.open() as handle:
        rows = list(csv.DictReader(handle))
    output: list[Path] = []
    for row in rows:
        output += trace_checkpoint(
            config, Path(row["flow_budget_checkpoint"]), label=f"budget_{flow_condition}"
        )
        output += trace_checkpoint(
            config, Path(row["rt_budget_checkpoint"]), label=f"budget_{rt_condition}"
        )
        if row["accuracy_match_valid"].lower() == "true":
            output += trace_checkpoint(
                config, Path(row["flow_budget_checkpoint"]), label=f"accuracy_{flow_condition}"
            )
            output += trace_checkpoint(
                config, Path(row["rt_accuracy_matched_checkpoint"]), label=f"accuracy_{rt_condition}"
            )
    return output


def diagnose_flow(config: dict[str, Any], conditions: list[str]) -> Path:
    output = train_sudoku(config, conditions)
    required = float(config["analysis"].get("overfit_exact_match", 0.95))
    failures = []
    for condition in conditions:
        for seed in config["experiment"]["seeds"]:
            path = output / f"training_{condition}_seed_{seed}.csv"
            with path.open() as handle:
                best = max(float(row["eval_exact_match"]) for row in csv.DictReader(handle))
            if best < required:
                failures.append(f"{condition}/seed_{seed}={best:.3f}")
    if failures:
        raise RuntimeError(
            f"Flow overfit gate requires exact match >= {required:.3f}; failed: {', '.join(failures)}"
        )
    return output


def evaluate_checkpoints(
    config: dict[str, Any], condition: str, checkpoint_root_override: Path | None = None
) -> Path:
    device = resolve_device(config)
    checkpoint_root = checkpoint_root_override or Path(config["experiment"]["checkpoint_root"])
    output_root = Path(config["experiment"]["output_root"]) / "evaluation"
    output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for seed in config["experiment"]["seeds"]:
        sudoku_config = dict(config["sudoku"]); sudoku_config["seed"] = int(seed)
        smoke = bool(sudoku_config.pop("smoke", False))
        _, dev_loader, dev_metadata = create_sudoku_loaders(
            sudoku_config, smoke=smoke, eval_partition=None if smoke else "dev"
        )
        _, test_loader, test_metadata = create_sudoku_loaders(
            sudoku_config, smoke=smoke, eval_partition=None if smoke else "test"
        )
        for checkpoint in sorted(
            (checkpoint_root / condition / f"seed_{seed}").glob("epoch_*.pt"),
            key=lambda path: int(path.stem.removeprefix("epoch_")),
        ):
            model, loaded_condition, loaded_seed = _load_matched(checkpoint, config, device)
            if loaded_condition != condition or loaded_seed != int(seed):
                raise ValueError(f"checkpoint metadata mismatch: {checkpoint}")
            steps = int(config["trace"]["matched_steps"])
            dev = evaluate_sudoku(model, dev_loader, steps, device)
            test = evaluate_sudoku(model, test_loader, steps, device)
            rows.append({
                "condition": condition, "seed": int(seed),
                "epoch": int(checkpoint.stem.removeprefix("epoch_")), "checkpoint": str(checkpoint),
                **{f"dev_{key}": value for key, value in dev.items() if key not in {"correct", "total"}},
                **{f"test_{key}": value for key, value in test.items() if key not in {"correct", "total"}},
                "dev_samples": dev["total"], "test_samples": test["total"],
                "dev_fingerprint": dev_metadata.get("evaluation_dataset_fingerprint"),
                "test_fingerprint": test_metadata.get("evaluation_dataset_fingerprint"),
            })
    if not rows:
        raise FileNotFoundError(f"no checkpoints found for {condition!r} under {checkpoint_root}")
    path = output_root / f"checkpoint_evaluation_{condition}.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    return path


def summarize_fair_comparison(
    config: dict[str, Any], flow_condition: str, rt_condition: str = "matched_rt"
) -> tuple[Path, bool]:
    evaluation_root = Path(config["experiment"]["output_root"]) / "evaluation"

    def read(condition: str) -> list[dict[str, str]]:
        path = evaluation_root / f"checkpoint_evaluation_{condition}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"evaluate checkpoints first: {path}")
        with path.open() as handle:
            return list(csv.DictReader(handle))

    flow_rows, rt_rows = read(flow_condition), read(rt_condition)
    seeds = sorted({int(row["seed"]) for row in flow_rows} & {int(row["seed"]) for row in rt_rows})
    if not seeds:
        raise ValueError("Flow and RT evaluation tables have no common seeds")

    def best(rows: list[dict[str, str]], seed: int) -> dict[str, str]:
        candidates = [row for row in rows if int(row["seed"]) == seed]
        return max(candidates, key=lambda row: (
            float(row["dev_exact_match"]), float(row["dev_cell_accuracy"]),
            -float(row["dev_constraint_violations"]), -int(row["epoch"]),
        ))

    output_rows: list[dict[str, object]] = []
    flow_test = []
    rt_test = []
    tolerance = float(config["analysis"]["flow_gate"]["accuracy_match_tolerance"])
    for seed in seeds:
        selected_flow, selected_rt = best(flow_rows, seed), best(rt_rows, seed)
        flow_test.append(float(selected_flow["test_exact_match"]))
        rt_test.append(float(selected_rt["test_exact_match"]))
        candidates = [row for row in rt_rows if int(row["seed"]) == seed]
        accuracy_rt = min(
            candidates,
            key=lambda row: (abs(float(row["dev_exact_match"]) - float(selected_flow["dev_exact_match"])), int(row["epoch"])),
        )
        accuracy_gap = abs(float(accuracy_rt["dev_exact_match"]) - float(selected_flow["dev_exact_match"]))
        output_rows.append({
            "seed": seed, "flow_condition": flow_condition,
            "flow_budget_epoch": int(selected_flow["epoch"]),
            "flow_budget_checkpoint": selected_flow["checkpoint"],
            "flow_budget_dev_exact_match": float(selected_flow["dev_exact_match"]),
            "flow_budget_test_exact_match": float(selected_flow["test_exact_match"]),
            "rt_budget_epoch": int(selected_rt["epoch"]),
            "rt_budget_checkpoint": selected_rt["checkpoint"],
            "rt_budget_dev_exact_match": float(selected_rt["dev_exact_match"]),
            "rt_budget_test_exact_match": float(selected_rt["test_exact_match"]),
            "rt_accuracy_matched_epoch": int(accuracy_rt["epoch"]),
            "rt_accuracy_matched_checkpoint": accuracy_rt["checkpoint"],
            "rt_accuracy_matched_dev_exact_match": float(accuracy_rt["dev_exact_match"]),
            "accuracy_match_gap": accuracy_gap,
            "accuracy_match_valid": accuracy_gap <= tolerance,
        })
    path = evaluation_root / f"fair_comparison_{flow_condition}_vs_{rt_condition}.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0])); writer.writeheader(); writer.writerows(output_rows)

    rules = config["analysis"]["flow_gate"]
    gap = statistics.mean(rt_test) - statistics.mean(flow_test)
    passed = (
        gap <= float(rules["max_mean_gap"])
        and min(flow_test) >= float(rules["minimum_seed_exact_match"])
        and max(flow_test) - min(flow_test) <= float(rules["max_seed_range"])
    )
    gate = {
        "passed": passed, "flow_condition": flow_condition, "rt_condition": rt_condition,
        "seeds": seeds, "flow_mean_test_exact_match": statistics.mean(flow_test),
        "rt_mean_test_exact_match": statistics.mean(rt_test), "rt_minus_flow_gap": gap,
        "flow_min_test_exact_match": min(flow_test), "flow_seed_range": max(flow_test) - min(flow_test),
        "rules": rules,
    }
    gate_path = evaluation_root / "flow_gate.json"
    gate_path.write_text(json.dumps(gate, indent=2) + "\n")
    return path, passed


def select_flow_condition(config: dict[str, Any], conditions: list[str], seed: int = 0) -> Path:
    evaluation_root = Path(config["experiment"]["output_root"]) / "evaluation"
    candidates: list[dict[str, object]] = []
    for condition in conditions:
        path = evaluation_root / f"checkpoint_evaluation_{condition}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"evaluate checkpoints first: {path}")
        with path.open() as handle:
            rows = [row for row in csv.DictReader(handle) if int(row["seed"]) == seed]
        if not rows:
            raise ValueError(f"no seed {seed} rows for {condition}")
        best = max(rows, key=lambda row: (
            float(row["dev_exact_match"]), float(row["dev_cell_accuracy"]),
            -float(row["dev_constraint_violations"]), -int(row["epoch"]),
        ))
        candidates.append({
            "condition": condition, "seed": seed, "epoch": int(best["epoch"]),
            "checkpoint": best["checkpoint"], "dev_exact_match": float(best["dev_exact_match"]),
            "dev_cell_accuracy": float(best["dev_cell_accuracy"]),
            "dev_constraint_violations": float(best["dev_constraint_violations"]),
        })
    selected = max(candidates, key=lambda row: (
        float(row["dev_exact_match"]), float(row["dev_cell_accuracy"]),
        -float(row["dev_constraint_violations"]), -int(row["epoch"]),
    ))
    output = evaluation_root / "selected_flow_condition.json"
    output.write_text(json.dumps({"selected": selected, "candidates": candidates}, indent=2) + "\n")
    return output


def run_smoke(config_path: Path) -> None:
    from .analyze import analyze_traces, plot_toy_results

    config = load_config(config_path)
    train_toy(config); plot_toy_results(config)
    train_sudoku(config, [
        "matched_rt", "rt_smooth", "flow_linear", "flow_bent_05",
        "flow_v2_teacher", "flow_v2_onpolicy",
    ])
    checkpoint_root = Path(config["experiment"]["checkpoint_root"])
    epoch = int(config["schedule"]["epochs"]) - 1
    for condition in ("matched_rt", "flow_linear", "flow_v2_onpolicy"):
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
    train.add_argument("--seed", nargs="+", type=int, help="Override experiment.seeds without editing YAML")
    diagnose = subparsers.add_parser("diagnose-flow", help="Run the real-Sudoku Flow overfit gate")
    diagnose.add_argument("--config", type=Path, default=DEFAULT_FLOW_V2_OVERFIT_CONFIG)
    diagnose.add_argument("--condition", nargs="+", default=["flow_v2_teacher", "flow_v2_onpolicy"])
    diagnose.add_argument("--seed", nargs="+", type=int, help="Override experiment.seeds without editing YAML")
    evaluate = subparsers.add_parser("evaluate-checkpoints", help="Evaluate every epoch on fixed dev/test partitions")
    evaluate.add_argument("--config", type=Path, default=DEFAULT_FLOW_V2_CONFIG)
    evaluate.add_argument("--condition", required=True)
    evaluate.add_argument("--checkpoint-root", type=Path, help="Override the checkpoint root (for V1 RT checkpoints)")
    evaluate.add_argument("--seed", nargs="+", type=int, help="Override experiment.seeds without editing YAML")
    compare = subparsers.add_parser("compare-flow", help="Apply the preregistered Flow gate and match RT checkpoints")
    compare.add_argument("--config", type=Path, default=DEFAULT_FLOW_V2_CONFIG)
    compare.add_argument("--flow-condition", required=True)
    compare.add_argument("--rt-condition", default="matched_rt")
    select = subparsers.add_parser("select-flow", help="Select teacher or on-policy Flow using development seed 0")
    select.add_argument("--config", type=Path, default=DEFAULT_FLOW_V2_CONFIG)
    select.add_argument("--condition", nargs="+", default=["flow_v2_teacher", "flow_v2_onpolicy"])
    select.add_argument("--seed", type=int, default=0)
    trace = subparsers.add_parser("trace-sudoku", help="Trace a matched or native RT checkpoint")
    trace.add_argument("--config", type=Path, default=DEFAULT_FULL_CONFIG)
    trace.add_argument("--checkpoint", type=Path, required=True)
    trace.add_argument("--native", action="store_true")
    trace.add_argument("--perturb", action="store_true")
    trace.add_argument("--label", help="Analysis label used to keep multiple checkpoint selections separate")
    trace_comparison = subparsers.add_parser("trace-comparison", help="Trace budget- and accuracy-matched checkpoint pairs")
    trace_comparison.add_argument("--config", type=Path, default=DEFAULT_FLOW_V2_CONFIG)
    trace_comparison.add_argument("--flow-condition", required=True)
    trace_comparison.add_argument("--rt-condition", default="matched_rt")
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
        train_sudoku(_override_seeds(load_config(args.config), args.seed), list(args.condition))
    elif args.command == "diagnose-flow":
        diagnose_flow(_override_seeds(load_config(args.config), args.seed), list(args.condition))
    elif args.command == "evaluate-checkpoints":
        evaluate_checkpoints(
            _override_seeds(load_config(args.config), args.seed), args.condition, args.checkpoint_root
        )
    elif args.command == "compare-flow":
        _, passed = summarize_fair_comparison(load_config(args.config), args.flow_condition, args.rt_condition)
        if not passed:
            raise RuntimeError("Flow did not pass the preregistered performance gate; do not interpret trajectory comparisons")
    elif args.command == "select-flow":
        select_flow_condition(load_config(args.config), list(args.condition), args.seed)
    elif args.command == "trace-sudoku":
        trace_checkpoint(
            load_config(args.config), args.checkpoint, native=args.native,
            perturb=args.perturb, label=args.label,
        )
    elif args.command == "trace-comparison":
        trace_fair_comparison(load_config(args.config), args.flow_condition, args.rt_condition)
    elif args.command == "analyze":
        from .analyze import analyze_traces
        config = load_config(args.config)
        if "flow_gate" in config["analysis"]:
            gate_path = Path(config["experiment"]["output_root"]) / "evaluation" / "flow_gate.json"
            if not gate_path.is_file() or not bool(json.loads(gate_path.read_text()).get("passed")):
                raise RuntimeError("Flow performance gate has not passed; trajectory analysis is intentionally disabled")
        analyze_traces(config)


if __name__ == "__main__":
    main()
