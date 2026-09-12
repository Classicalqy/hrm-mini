"""Pure, dependency-light helpers for full-state long-rollout MSD analysis."""

from __future__ import annotations

import math

import numpy as np
import torch


STATE_NAMES = ("h", "l", "h_plus_l", "hl_concat")


def segment_boundaries(total_blocks: int) -> tuple[int, int, int, int, int]:
    """Four log-uniform H-boundary windows, including the initial boundary zero."""
    if total_blocks < 4:
        raise ValueError(f"Need at least four rollout blocks, got {total_blocks}.")
    boundaries = [0]
    for power in (0.25, 0.5, 0.75):
        boundary = round(total_blocks ** power)
        boundary = max(boundary, boundaries[-1] + 1)
        # Leave a strictly increasing boundary for each remaining log segment
        # plus the final endpoint N.
        boundary = min(boundary, total_blocks - (4 - len(boundaries)))
        boundaries.append(boundary)
    boundaries.append(total_blocks)
    if any(left >= right for left, right in zip(boundaries, boundaries[1:])):
        raise AssertionError(f"Invalid segment boundaries: {boundaries}")
    return tuple(boundaries)  # type: ignore[return-value]


def log_spaced_lags(max_lag: int, points: int) -> tuple[int, ...]:
    if max_lag < 1:
        return ()
    count = min(points, max_lag)
    values = np.logspace(0, math.log2(max_lag), num=count, base=2.0)
    return tuple(sorted({max(1, min(max_lag, int(round(value)))) for value in values}))


def segment_lags(boundaries: tuple[int, int, int, int, int], points: int) -> tuple[int, ...]:
    return tuple(sorted({
        lag for start, end in zip(boundaries, boundaries[1:])
        for lag in log_spaced_lags(end - start - 1, points)
    }))


def state_msd(z_h_a: torch.Tensor, z_l_a: torch.Tensor, z_h_b: torch.Tensor, z_l_b: torch.Tensor) -> dict[str, float]:
    """Full-state per-coordinate MSD, without forming a projection or reduced state."""
    delta_h = (z_h_b - z_h_a).float()
    delta_l = (z_l_b - z_l_a).float()
    h = torch.mean(torch.square(delta_h)).item()
    l = torch.mean(torch.square(delta_l)).item()
    h_plus_l = torch.mean(torch.square(delta_h + delta_l)).item()
    # This is exactly mean(square(cat(delta_h, delta_l))) but avoids a temporary
    # doubled-width tensor; it still includes every token and hidden coordinate.
    hl_concat = (torch.square(delta_h).sum() + torch.square(delta_l).sum()).div(delta_h.numel() + delta_l.numel()).item()
    return {"h": h, "l": l, "h_plus_l": h_plus_l, "hl_concat": hl_concat}


def state_msd_per_puzzle(
    z_h_a: torch.Tensor, z_l_a: torch.Tensor, z_h_b: torch.Tensor, z_l_b: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Return exact full-state per-coordinate MSD for every puzzle in a batch.

    The return tensors have shape ``[batch]``.  Token and hidden dimensions are
    averaged, but the puzzle dimension is intentionally retained for bootstrap
    inference over input instances.
    """
    delta_h = (z_h_b - z_h_a).float()
    delta_l = (z_l_b - z_l_a).float()
    h = torch.mean(torch.square(delta_h), dim=(-2, -1))
    l = torch.mean(torch.square(delta_l), dim=(-2, -1))
    h_plus_l = torch.mean(torch.square(delta_h + delta_l), dim=(-2, -1))
    hl_concat = (torch.sum(torch.square(delta_h), dim=(-2, -1)) + torch.sum(torch.square(delta_l), dim=(-2, -1)))
    hl_concat = hl_concat / (delta_h.shape[-2] * (delta_h.shape[-1] + delta_l.shape[-1]))
    return {"h": h, "l": l, "h_plus_l": h_plus_l, "hl_concat": hl_concat}


def rt_state_msd(z_a: torch.Tensor, z_b: torch.Tensor) -> float:
    return torch.mean(torch.square((z_b - z_a).float())).item()


def rt_state_msd_per_puzzle(z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
    """Per-puzzle counterpart of :func:`rt_state_msd`."""
    return torch.mean(torch.square((z_b - z_a).float()), dim=(-2, -1))


def local_log_slope(values: np.ndarray, lags: np.ndarray) -> np.ndarray:
    """Central finite-difference d(log MSD)/d(log lag) along the last axis."""
    if values.shape[-1] != len(lags):
        raise ValueError("The last values dimension must equal the number of lags.")
    if len(lags) < 2:
        return np.full_like(values, np.nan, dtype=np.float64)
    log_values = np.log(values)
    log_lags = np.log(lags.astype(np.float64))
    result = np.empty_like(log_values, dtype=np.float64)
    result[..., 0] = (log_values[..., 1] - log_values[..., 0]) / (log_lags[1] - log_lags[0])
    result[..., -1] = (log_values[..., -1] - log_values[..., -2]) / (log_lags[-1] - log_lags[-2])
    if len(lags) > 2:
        result[..., 1:-1] = (log_values[..., 2:] - log_values[..., :-2]) / (log_lags[2:] - log_lags[:-2])
    return result


def segment_for_pair(t: int, lag: int, boundaries: tuple[int, int, int, int, int]) -> int | None:
    """Return a segment only when both endpoints remain inside that segment."""
    for index, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
        if start <= t and t + lag < end:
            return index
    return None
