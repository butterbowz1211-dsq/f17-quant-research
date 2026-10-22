# Research project author: Gilmour Jia
import numpy as np
import pandas as pd
from spec import SPEC


def cap_allocate(preference, cap=0.03):
    p = np.asarray(preference, float)
    if p.ndim != 1 or not np.isfinite(p).all() or (p < 0).any() or not (0 < cap <= 1):
        raise ValueError("Invalid allocation")
    out = np.zeros(len(p))
    free = p > 0
    remaining = min(1.0, int(free.sum()) * cap)
    while free.any() and remaining > 1e-14:
        alloc = p[free] / p[free].sum() * remaining
        idx = np.flatnonzero(free)
        bound = alloc >= cap
        if not bound.any():
            out[idx] = alloc
            break
        out[idx[bound]] = cap
        remaining -= int(bound.sum()) * cap
        free[idx[bound]] = False
    return out


def readout(scores, previous=(), mode="softmax"):
    """Selection state and account holdings are deliberately separate."""
    s = pd.Series(scores, dtype=float)
    s = s[np.isfinite(s)]
    # Stable and explicit tie rule independent of caller row order.
    s = s.sort_index().sort_values(ascending=False, kind="stable")
    if len(s) < SPEC["portfolio"]["minimum_candidates"]:
        return None, list(previous)  # Skip decision; do not liquidate silently.
    n = min(50, len(s))
    band = set(s.iloc[:80].index)
    kept = [c for c in previous if c in band][:n]
    selected = kept + [c for c in s.index if c not in kept][:n-len(kept)]
    v = s.reindex(selected)
    if mode == "equal":
        p = np.ones(len(v))
    elif mode == "softmax":
        rank = v.rank(method="average")
        z = (rank - rank.min()) / (rank.max() - rank.min()) if rank.max() > rank.min() else rank * 0
        p = np.exp(z.to_numpy() - z.max())  # frozen T=1
    else:
        raise ValueError("Unsupported readout")
    return pd.Series(cap_allocate(p), index=v.index), selected
