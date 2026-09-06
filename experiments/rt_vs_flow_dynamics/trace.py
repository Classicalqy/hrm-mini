from __future__ import annotations

import csv
from hashlib import sha256
import json
from pathlib import Path
import subprocess
from typing import Any, Iterable

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .tasks.sudoku import correct_margin, sudoku_violations, tensor_dataset_fingerprint


def file_sha256(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def make_projection(input_size: int, output_size: int, seed: int, device: torch.device) -> Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    projection = torch.randn(input_size, output_size, generator=generator) / output_size**0.5
    return projection.to(device)


def project_hidden(state: Tensor, input_ids: Tensor, projection: Tensor) -> Tensor:
    projected = state.to(torch.float32) @ projection
    all_tokens = projected.mean(dim=1)
    blank_mask = input_ids.eq(0); blank_mask[:, 0] = False
    count = blank_mask.sum(dim=1, keepdim=True).clamp_min(1)
    blank_tokens = (projected * blank_mask.unsqueeze(-1)).sum(dim=1) / count
    return torch.cat((all_tokens, blank_tokens), dim=-1)


def project_probabilities(logits: Tensor, projection: Tensor) -> Tensor:
    probabilities = logits.to(torch.float32).softmax(dim=-1)
    return probabilities.flatten(start_dim=1) @ projection


@torch.inference_mode()
def collect_sudoku_trace(
    model: nn.Module,
    loader: Iterable[tuple[Tensor, Tensor]],
    steps: int,
    sample_count: int,
    projection_dim: int,
    seed: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    hidden_projection = make_projection(int(model.hidden_size), projection_dim, seed, device)
    probability_projection = make_projection(82 * int(model.vocab_size), projection_dim, seed + 1, device)
    collected: dict[str, list[np.ndarray]] = {}
    remaining = sample_count
    for inputs, targets in loader:
        if remaining <= 0:
            break
        inputs, targets = inputs[:remaining].to(device), targets[:remaining].to(device)
        state = model.initial_state(inputs)
        logits = model.decode(state)
        previous_state, previous_probabilities = state, logits.to(torch.float32).softmax(dim=-1)
        batch: dict[str, list[Tensor]] = {
            "projected_state": [], "projected_probability": [], "prediction": [],
            "exact_match": [], "cell_accuracy": [], "constraint_violations": [],
            "correct_margin": [], "state_residual": [], "logit_change": [],
        }

        def record(current_state: Tensor, current_logits: Tensor, residual: Tensor, probability_change: Tensor) -> None:
            predictions = current_logits.argmax(dim=-1)
            batch["projected_state"].append(project_hidden(current_state, inputs, hidden_projection))
            batch["projected_probability"].append(project_probabilities(current_logits, probability_projection))
            batch["prediction"].append(predictions.to(torch.uint8))
            batch["exact_match"].append((predictions == targets).all(dim=-1))
            batch["cell_accuracy"].append((predictions == targets).float().mean(dim=-1))
            batch["constraint_violations"].append(sudoku_violations(predictions).to(torch.float32))
            batch["correct_margin"].append(correct_margin(current_logits, targets))
            batch["state_residual"].append(residual)
            batch["logit_change"].append(probability_change)

        zeros = torch.zeros(inputs.shape[0], device=device)
        record(state, logits, zeros, zeros)
        for index in range(steps):
            output = model.step(state, inputs, index, steps)
            state, logits = output.state, output.logits
            residual = (state.to(torch.float32) - previous_state.to(torch.float32)).flatten(1).norm(dim=1)
            residual /= previous_state.to(torch.float32).flatten(1).norm(dim=1).clamp_min(1e-8)
            probabilities = logits.to(torch.float32).softmax(dim=-1)
            probability_change = (probabilities - previous_probabilities).abs().mean(dim=(1, 2))
            record(state, logits, residual, probability_change)
            previous_state, previous_probabilities = state, probabilities
        prediction_tensor = torch.stack(batch["prediction"], dim=1)
        flip_count = torch.zeros(prediction_tensor.shape[:2], device=device, dtype=torch.int16)
        flip_count[:, 1:] = (prediction_tensor[:, 1:] != prediction_tensor[:, :-1]).sum(dim=-1).to(torch.int16)
        collected.setdefault("prediction_flip_count", []).append(flip_count.cpu().numpy())
        for key, values in batch.items():
            array = torch.stack(values, dim=1).cpu().numpy()
            if key in {"projected_state", "projected_probability"}:
                array = array.astype(np.float16)
            collected.setdefault(key, []).append(array)
        collected.setdefault("input_ids", []).append(inputs.cpu().numpy().astype(np.uint8))
        collected.setdefault("targets", []).append(targets.cpu().numpy().astype(np.uint8))
        remaining -= inputs.shape[0]
    if remaining > 0:
        raise RuntimeError(f"requested {sample_count} trace samples but loader supplied only {sample_count - remaining}")
    result = {key: np.concatenate(values, axis=0) for key, values in collected.items()}
    result["sample_id"] = np.arange(sample_count, dtype=np.int32)
    return result


def detect_fixed_points(
    arrays: dict[str, np.ndarray], window: int = 8, state_eps: float = 1e-3, logit_eps: float = 1e-4
) -> np.ndarray:
    residual = arrays["state_residual"]
    logit_change = arrays["logit_change"]
    predictions = arrays["prediction"]
    exact = arrays["exact_match"]
    onset = np.full(residual.shape[0], -1, dtype=np.int32)
    for puzzle in range(residual.shape[0]):
        for end in range(window, residual.shape[1]):
            start = end - window + 1
            stable_prediction = np.all(predictions[puzzle, start:end + 1] == predictions[puzzle, end])
            if (
                stable_prediction
                and np.all(residual[puzzle, start:end + 1] < state_eps)
                and np.all(logit_change[puzzle, start:end + 1] < logit_eps)
                and not bool(exact[puzzle, end])
            ):
                onset[puzzle] = end
                break
    return onset


def trajectory_metrics(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    path = arrays["projected_probability"].astype(np.float32)
    delta = np.diff(path, axis=1)
    step_norm = np.linalg.norm(delta, axis=-1)
    displacement = np.linalg.norm(path[:, -1] - path[:, 0], axis=-1)
    tortuosity = step_norm.sum(axis=1) / np.clip(displacement, 1e-8, None)
    directions = delta / np.clip(step_norm[..., None], 1e-8, None)
    direction_autocorrelation = (directions[:, 1:] * directions[:, :-1]).sum(axis=-1).mean(axis=1)
    margin_delta = np.diff(arrays["correct_margin"].astype(np.float32), axis=1)
    backtrack_rate = (margin_delta < 0).mean(axis=1)
    prediction = arrays["prediction"]
    reversals = (
        (prediction[:, 2:] == prediction[:, :-2]) & (prediction[:, 1:-1] != prediction[:, :-2])
    ).sum(axis=(1, 2))
    ranks = np.empty(path.shape[0], dtype=np.float32)
    for index, values in enumerate(delta):
        singular = np.linalg.svd(values, compute_uv=False)
        eigenvalues = singular.astype(np.float64) ** 2
        ranks[index] = eigenvalues.sum() ** 2 / max(np.square(eigenvalues).sum(), 1e-12)
    return {
        "tortuosity": tortuosity,
        "direction_autocorrelation": direction_autocorrelation,
        "backtrack_rate": backtrack_rate,
        "prediction_reversals": reversals.astype(np.float32),
        "effective_rank": ranks,
    }


def save_trace(
    path: str | Path, arrays: dict[str, np.ndarray], metadata: dict[str, Any]
) -> tuple[Path, Path]:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    metadata = dict(metadata)
    metadata["dataset_fingerprint"] = tensor_dataset_fingerprint(
        torch.from_numpy(arrays["input_ids"]), torch.from_numpy(arrays["targets"])
    )
    metadata["git_revision"] = git_revision()
    json_path = path.with_suffix(".json")
    json_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return path, json_path


def _state_at(model: nn.Module, inputs: Tensor, onset: int, total_steps: int) -> Tensor:
    state = model.initial_state(inputs)
    for index in range(onset):
        state = model.step(state, inputs, index, total_steps).state
    return state


def top_jacobian_singular_value(
    model: nn.Module, state: Tensor, inputs: Tensor, step_index: int, total_steps: int,
    iterations: int, generator: torch.Generator,
) -> float:
    state = state.detach().to(torch.float32)

    def transition(value: Tensor) -> Tensor:
        return model.step(value, inputs, step_index, total_steps).state.to(torch.float32)

    vector = torch.randn(state.shape, generator=generator, device=state.device, dtype=state.dtype)
    vector = vector / vector.norm().clamp_min(1e-12)
    sigma = torch.zeros((), device=state.device)
    for _ in range(iterations):
        _, vjp = torch.func.vjp(transition, state)
        jt_vector = vjp(vector)[0]
        right = jt_vector / jt_vector.norm().clamp_min(1e-12)
        _, j_right = torch.func.jvp(transition, (state,), (right,))
        sigma = j_right.norm()
        vector = j_right / sigma.clamp_min(1e-12)
    return float(sigma)


def perturb_wrong_fixed_points(
    model: nn.Module,
    arrays: dict[str, np.ndarray],
    onset: np.ndarray,
    total_steps: int,
    continue_steps: int,
    sigmas: list[float],
    directions: int,
    max_puzzles: int,
    jacobian_max_puzzles: int,
    jacobian_iterations: int,
    seed: int,
    device: torch.device,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    candidates = np.flatnonzero(onset >= 0)[:max_puzzles]
    generator = torch.Generator(device=device.type).manual_seed(seed)
    perturbation_rows: list[dict[str, object]] = []
    jacobian_rows: list[dict[str, object]] = []
    for order, puzzle in enumerate(candidates):
        inputs = torch.from_numpy(arrays["input_ids"][puzzle:puzzle + 1].astype(np.int64)).to(device)
        targets = torch.from_numpy(arrays["targets"][puzzle:puzzle + 1].astype(np.int64)).to(device)
        fixed_state = _state_at(model, inputs, int(onset[puzzle]), total_steps)
        reference = torch.from_numpy(arrays["prediction"][puzzle, onset[puzzle]].astype(np.int64)).to(device)
        if order < jacobian_max_puzzles:
            value = top_jacobian_singular_value(
                model, fixed_state, inputs, int(onset[puzzle]), total_steps, jacobian_iterations, generator
            )
            jacobian_rows.append({"sample_id": int(puzzle), "onset": int(onset[puzzle]), "top_singular_value": value})
        state_rms = fixed_state.to(torch.float32).square().mean().sqrt()
        for sigma in sigmas:
            expanded_state = fixed_state.expand(directions, -1, -1).clone()
            noise = torch.randn(expanded_state.shape, generator=generator, device=device, dtype=torch.float32)
            noise = noise / noise.flatten(1).norm(dim=1).view(-1, 1, 1).clamp_min(1e-12)
            noise *= float(sigma) * state_rms * expanded_state[0].numel() ** 0.5
            state = expanded_state + noise.to(expanded_state.dtype)
            repeated_inputs = inputs.expand(directions, -1)
            for offset in range(continue_steps):
                state = model.step(
                    state, repeated_inputs, int(onset[puzzle]) + offset, total_steps
                ).state
            predictions = model.decode(state).argmax(dim=-1)
            same = (predictions == reference).all(dim=-1)
            correct = (predictions == targets.expand_as(predictions)).all(dim=-1)
            perturbation_rows.append({
                "sample_id": int(puzzle), "onset": int(onset[puzzle]), "sigma": float(sigma),
                "directions": directions, "return_rate": float(same.float().mean()),
                "correct_rate": float(correct.float().mean()),
                "escape_rate": float((~same).float().mean()),
            })
    return perturbation_rows, jacobian_rows


def write_rows(path: str | Path, rows: list[dict[str, object]]) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
