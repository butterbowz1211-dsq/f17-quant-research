# Research project author: Gilmour Jia
"""Bounded weight algebra for direct-weight portfolio policies.

The policy emits a vector over (stocks, cash) that must map onto the feasible
set of an actual account: weights sum to one, every stock is capped, and
positions that cannot be traded are frozen at their current weight.

Everything here is pure numpy and has no knowledge of prices or dates, so it
can be unit-tested in isolation.
"""
from __future__ import annotations

import numpy as np

__all__ = ["project_capped_simplex", "blend_targets", "no_trade_mask", "feasible_bounds"]


def project_capped_simplex(v, lower, upper, total=1.0, tol=1e-10, max_iter=200):
    """Project ``v`` onto {w : sum(w)=total, lower<=w<=upper}.

    Bisection on the dual variable (capped-simplex algorithm). Bounds must be
    able to admit ``total`` or the result is returned with ok=False.
    Returns (w, ok).
    """
    v = np.asarray(v, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    if not (v.shape == lower.shape == upper.shape):
        raise ValueError("v, lower and upper must share a shape")
    if np.any(lower > upper + 1e-12):
        raise ValueError("lower bound exceeds upper bound")
    if total < float(lower.sum()) - 1e-9 or total > float(upper.sum()) + 1e-9:
        return np.clip(v, lower, upper), False

    # w(tau) = clip(v - tau, lower, upper) has sum(w) non-increasing in tau.
    lo_tau = float(np.min(v - upper)) - 1.0
    hi_tau = float(np.max(v - lower)) + 1.0
    w = np.clip(v - hi_tau, lower, upper)
    for _ in range(max_iter):
        tau = 0.5 * (lo_tau + hi_tau)
        w = np.clip(v - tau, lower, upper)
        s = float(w.sum())
        if abs(s - total) <= tol:
            break
        if s > total:
            lo_tau = tau
        else:
            hi_tau = tau
    drift = total - float(w.sum())
    if abs(drift) > 0:
        w = _rebalance_residual(w, lower, upper, drift)
    return w, True


def _rebalance_residual(w, lower, upper, drift):
    """Push a rounding residual into available slack, greedily."""
    remaining = float(drift)
    for _ in range(50):
        if abs(remaining) <= 1e-14:
            break
        slack = (upper - w) if remaining > 0 else (w - lower)
        room = slack > 1e-15
        if not room.any():
            break
        idx = np.flatnonzero(room)
        total_slack = float(slack[idx].sum())
        move = min(abs(remaining), total_slack)
        signed = move if remaining > 0 else -move
        w[idx] += signed * (slack[idx] / total_slack)
        remaining -= signed
    return w


def feasible_bounds(tradeable, locked, cap, cash_locked=False):
    """Bounds implied by frozen positions, for a vector ending in cash.

    ``tradeable`` and ``locked`` include the cash slot as the final element.
    Frozen stock positions keep exactly their locked weight. Cash is normally
    movable, so its bounds stay [0, 1].
    """
    tradeable = np.asarray(tradeable, dtype=bool)
    locked = np.asarray(locked, dtype=float)
    lower = np.where(tradeable, 0.0, locked)
    upper = np.where(tradeable, 1.0, locked)
    upper = np.where(tradeable, np.minimum(upper, cap), upper)
    if not cash_locked:
        last = len(lower) - 1
        lower[last] = 0.0
        upper[last] = 1.0
    return lower, upper


def blend_targets(current, desired, partial):
    """Move a fraction ``partial`` of the way from current to desired."""
    if not 0.0 < partial <= 1.0:
        raise ValueError("partial must be in (0, 1]")
    current = np.asarray(current, dtype=float)
    desired = np.asarray(desired, dtype=float)
    return current + partial * (desired - current)


def no_trade_mask(current, target, nav, threshold):
    """Positions whose notional gap is too small to be worth trading."""
    current = np.asarray(current, dtype=float)
    target = np.asarray(target, dtype=float)
    if threshold <= 0 or nav <= 0:
        return np.zeros(current.shape, dtype=bool)
    return np.abs(target - current) <= threshold
