from __future__ import annotations

import json
import csv
import types

import numpy as np
import torch

from arch.rt import RecurrentTransformer
from experiments.rt_vs_flow_dynamics.models.common import (
    StepOutput,
    count_trainable_parameters,
    make_orthogonal_codebook,
    sinusoidal_time_embedding,
)
from experiments.rt_vs_flow_dynamics.models.flow import (
    FlowMatchingTransformer,
    interpolant,
    reconstruct_endpoint,
    relative_velocity_loss,
)
from dataset.sudoku import evaluation_partition_for_question
from experiments.rt_vs_flow_dynamics.models.matched_rt import MatchedRecurrentTransformer
from experiments.rt_vs_flow_dynamics.models.native_rt import NativeRTAdapter
from experiments.rt_vs_flow_dynamics.tasks.toy import generate_toy_trajectories
from experiments.rt_vs_flow_dynamics.trace import (
    detect_fixed_points,
    perturb_wrong_fixed_points,
    top_jacobian_singular_value,
    trajectory_metrics,
)
from experiments.rt_vs_flow_dynamics.cli import summarize_fair_comparison
from experiments.rt_vs_flow_dynamics.train import flow_on_policy_ratio


def tiny_config() -> dict[str, object]:
    return {
        "vocab_size": 10, "seq_len": 82, "is_causal": False,
        "num_layers": 1, "hidden_size": 32, "intermediate_size": 64, "head_dim": 16,
        "norm_eps": 1e-6, "rope_theta": 10000.0, "forward_dtype": "float32",
        "codebook_seed": 7, "decoder_temperature": 10.0,
    }


def test_fixed_codebook_and_time_embedding_are_deterministic() -> None:
    codebook = make_orthogonal_codebook(10, 32, 7)
    torch.testing.assert_close(codebook @ codebook.T, torch.eye(10), atol=1e-5, rtol=1e-5)
    assert torch.equal(codebook, make_orthogonal_codebook(10, 32, 7))
    embedding = sinusoidal_time_embedding(torch.tensor([0.0, 0.5]), 33)
    assert embedding.shape == (2, 33)
    assert torch.isfinite(embedding).all()


def test_interpolant_has_exact_endpoints_and_derivative() -> None:
    z0 = torch.randn(2, 3, 32)
    z1 = torch.randn(2, 3, 32)
    start, _ = interpolant(z0, z1, torch.zeros(2), beta=.5)
    end, _ = interpolant(z0, z1, torch.ones(2), beta=.5)
    torch.testing.assert_close(start, z0); torch.testing.assert_close(end, z1)
    t = torch.full((2,), .37); epsilon = 1e-3
    before, _ = interpolant(z0, z1, t - epsilon, beta=.5)
    after, _ = interpolant(z0, z1, t + epsilon, beta=.5)
    _, velocity = interpolant(z0, z1, t, beta=.5)
    torch.testing.assert_close((after - before) / (2 * epsilon), velocity, atol=2e-3, rtol=2e-3)


def test_flow_endpoint_reconstruction_is_exact_for_straight_and_bent_paths() -> None:
    z0 = torch.randn(3, 5, 32)
    z1 = torch.randn(3, 5, 32)
    time = torch.tensor([0.0, 0.37, 0.91])
    for beta in (0.0, 0.5, 1.0):
        path, velocity = interpolant(z0, z1, time, beta)
        endpoint = reconstruct_endpoint(path, velocity, z0, z1, time, beta)
        torch.testing.assert_close(endpoint, z1, atol=2e-5, rtol=2e-5)


def test_relative_flow_loss_does_not_shrink_with_hidden_size() -> None:
    for hidden_size in (32, 512):
        target = torch.randn(4, 7, hidden_size)
        loss = relative_velocity_loss(torch.zeros_like(target), target)
        torch.testing.assert_close(loss, torch.ones_like(loss), atol=1e-6, rtol=1e-6)


def test_controlled_models_have_parameter_parity() -> None:
    rt = MatchedRecurrentTransformer(tiny_config())
    flow = FlowMatchingTransformer(tiny_config())
    assert count_trainable_parameters(rt) == count_trainable_parameters(flow)


def test_flow_euler_solver_integrates_constant_velocity() -> None:
    model = FlowMatchingTransformer(tiny_config())

    def constant_velocity(self, state, input_ids, t):
        del input_ids, t
        return torch.ones_like(state) * .25

    model.velocity = types.MethodType(constant_velocity, model)
    inputs = torch.zeros(2, 82, dtype=torch.long)
    initial = model.initial_state(inputs).clone()
    result_8 = model.rollout(inputs, 8).state
    result_32 = model.rollout(inputs, 32).state
    expected = initial + .25
    torch.testing.assert_close(result_8, expected)
    torch.testing.assert_close(result_32, expected)


def test_flow_v2_chunks_use_global_times_and_match_rollout_carry() -> None:
    model = FlowMatchingTransformer(tiny_config())
    observed_times: list[float] = []

    def constant_velocity(self, state, input_ids, t):
        del input_ids
        observed_times.extend(float(value) for value in t)
        return torch.ones_like(state) * .25

    model.velocity = types.MethodType(constant_velocity, model)
    inputs = torch.zeros(1, 82, dtype=torch.long)
    targets = torch.ones(1, 82, dtype=torch.long)
    state = model.initial_state(inputs).clone()
    for start in (0, 2):
        _, _, state, _ = model(
            inputs, targets, state, start_step=start, total_steps=4, micro_steps=2,
            on_policy_ratio=1.0,
        )
    expected = model.initial_state(inputs) + .25
    torch.testing.assert_close(state, expected)
    assert observed_times == [0.0, 0.25, 0.5, 0.75]


def test_flow_v2_rejects_invalid_on_policy_ratio() -> None:
    model = FlowMatchingTransformer(tiny_config())
    inputs = torch.zeros(1, 82, dtype=torch.long)
    with np.testing.assert_raises(ValueError):
        model(inputs, inputs, model.initial_state(inputs), 0, 4, 2, on_policy_ratio=1.1)


def test_flow_on_policy_schedule_boundaries() -> None:
    assert flow_on_policy_ratio("teacher", 1.0) == 0.0
    assert flow_on_policy_ratio("onpolicy", 0.2) == 0.0
    assert flow_on_policy_ratio("onpolicy", 0.4) == .5
    assert flow_on_policy_ratio("onpolicy", 0.6) == 1.0


def test_evaluation_partition_is_stable_and_disjoint() -> None:
    questions = [f"puzzle-{index}" for index in range(1000)]
    first = [evaluation_partition_for_question(value, .2, 20260908) for value in questions]
    second = [evaluation_partition_for_question(value, .2, 20260908) for value in questions]
    assert first == second
    assert set(first) == {"dev", "test"}
    assert 150 <= first.count("dev") <= 250


def test_fair_comparison_gate_and_accuracy_matching(tmp_path) -> None:
    evaluation = tmp_path / "evaluation"
    evaluation.mkdir()
    fields = [
        "condition", "seed", "epoch", "checkpoint", "dev_exact_match",
        "dev_cell_accuracy", "dev_constraint_violations", "test_exact_match",
    ]
    for condition, offset in (("flow", 0.0), ("matched_rt", 0.02)):
        rows = []
        for seed in (1, 2, 3):
            for epoch, dev in ((0, .58), (1, .65 + offset)):
                rows.append({
                    "condition": condition, "seed": seed, "epoch": epoch,
                    "checkpoint": f"{condition}-{seed}-{epoch}.pt", "dev_exact_match": dev,
                    "dev_cell_accuracy": .8 + dev / 10, "dev_constraint_violations": 2.0,
                    "test_exact_match": dev - .01,
                })
        with (evaluation / f"checkpoint_evaluation_{condition}.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    config = {
        "experiment": {"output_root": str(tmp_path)},
        "analysis": {"flow_gate": {
            "max_mean_gap": .05, "minimum_seed_exact_match": .5,
            "max_seed_range": .1, "accuracy_match_tolerance": .03,
        }},
    }
    path, passed = summarize_fair_comparison(config, "flow")
    assert passed and path.is_file()
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    assert all(row["accuracy_match_valid"] == "True" for row in rows)
    assert json.loads((evaluation / "flow_gate.json").read_text())["passed"] is True


def test_native_adapter_matches_group_forward() -> None:
    config = tiny_config() | {"cycles": 3, "bptt": True}
    model = RecurrentTransformer(config)
    adapter = NativeRTAdapter(model, cycles=3)
    inputs = torch.randint(0, 10, (2, 82))
    adapter.validate_group_parity(inputs, atol=1e-5, rtol=1e-5)


def test_native_adapter_loads_repository_checkpoint_layout(tmp_path) -> None:
    config = tiny_config() | {"cycles": 3, "bptt": True}
    model = RecurrentTransformer(config)
    checkpoint_dir = tmp_path / "seed_3"
    checkpoint_dir.mkdir()
    checkpoint = checkpoint_dir / "epoch_0.pt"
    torch.save(model.state_dict(), checkpoint)
    (checkpoint_dir / "model_config.json").write_text(json.dumps({
        "arch": {"name": "rt@RecurrentTransformer"} | config
    }))
    loaded = NativeRTAdapter.from_checkpoint(checkpoint)
    inputs = torch.randint(0, 10, (2, 82))
    loaded.validate_group_parity(inputs, atol=1e-5, rtol=1e-5)


def _fixed_point_arrays(kind: str) -> dict[str, np.ndarray]:
    steps = 12
    prediction = np.zeros((1, steps, 3), dtype=np.uint8)
    residual = np.full((1, steps), 1e-5, dtype=np.float32)
    change = np.full((1, steps), 1e-6, dtype=np.float32)
    exact = np.zeros((1, steps), dtype=bool)
    if kind == "correct":
        exact[:] = True
    elif kind == "oscillating":
        prediction[:, 1::2] = 1
    elif kind == "drifting":
        residual[:] = 1e-2
    return {"prediction": prediction, "state_residual": residual, "logit_change": change, "exact_match": exact}


def test_fixed_point_detector_rejects_correct_oscillating_and_drifting() -> None:
    assert detect_fixed_points(_fixed_point_arrays("wrong"), window=4)[0] >= 0
    assert detect_fixed_points(_fixed_point_arrays("correct"), window=4)[0] == -1
    assert detect_fixed_points(_fixed_point_arrays("oscillating"), window=4)[0] == -1
    assert detect_fixed_points(_fixed_point_arrays("drifting"), window=4)[0] == -1


def test_trajectory_metrics_are_rotation_invariant_and_finite() -> None:
    generator = np.random.default_rng(4)
    path = generator.normal(size=(3, 12, 6)).astype(np.float32)
    q, _ = np.linalg.qr(generator.normal(size=(6, 6)))
    base = {
        "projected_probability": path,
        "correct_margin": generator.normal(size=(3, 12)).astype(np.float32),
        "prediction": generator.integers(0, 10, size=(3, 12, 5), dtype=np.uint8),
    }
    rotated = dict(base); rotated["projected_probability"] = path @ q
    first, second = trajectory_metrics(base), trajectory_metrics(rotated)
    for key in ("tortuosity", "direction_autocorrelation", "effective_rank"):
        np.testing.assert_allclose(first[key], second[key], rtol=2e-5, atol=2e-5)
        assert np.isfinite(first[key]).all()


def test_toy_paths_have_exact_endpoints() -> None:
    data = generate_toy_trajectories(50, 32, 9)
    np.testing.assert_allclose(data.path[:, 0], data.context[:, :2], atol=1e-6)
    np.testing.assert_allclose(data.path[:, -1], data.context[:, 2:4], atol=1e-5)


def test_jacobian_power_iteration_on_identity_transition() -> None:
    class IdentityTransition(torch.nn.Module):
        def step(self, state, inputs, step_index, total_steps):
            del inputs, step_index, total_steps
            return StepOutput(state=state, logits=torch.empty(0), update=torch.zeros_like(state))

    state = torch.randn(1, 3, 4)
    value = top_jacobian_singular_value(
        IdentityTransition(), state, torch.zeros(1, 3, dtype=torch.long), 0, 1, 4,
        torch.Generator().manual_seed(3),
    )
    assert abs(value - 1.0) < 1e-5


def test_perturbation_analysis_runs_on_contracting_wrong_state() -> None:
    class ContractingTransition(torch.nn.Module):
        def initial_state(self, inputs):
            return torch.zeros(inputs.shape[0], inputs.shape[1], 4)

        def decode(self, state):
            logits = torch.zeros(*state.shape[:-1], 2)
            logits[..., 0] = 1.0
            return logits

        def step(self, state, inputs, step_index, total_steps):
            del inputs, step_index, total_steps
            next_state = .5 * state
            return StepOutput(next_state, self.decode(next_state), next_state - state)

    arrays = {
        "input_ids": np.zeros((1, 3), dtype=np.uint8),
        "targets": np.ones((1, 3), dtype=np.uint8),
        "prediction": np.zeros((1, 8, 3), dtype=np.uint8),
    }
    perturbation, jacobian = perturb_wrong_fixed_points(
        ContractingTransition(), arrays, np.asarray([4]), total_steps=8, continue_steps=2,
        sigmas=[.01], directions=2, max_puzzles=1, jacobian_max_puzzles=1,
        jacobian_iterations=4, seed=5, device=torch.device("cpu"),
    )
    assert len(perturbation) == len(jacobian) == 1
    assert perturbation[0]["return_rate"] == 1.0
    assert abs(jacobian[0]["top_singular_value"] - .5) < 1e-5
