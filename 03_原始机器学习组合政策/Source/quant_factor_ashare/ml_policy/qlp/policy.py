# Research project author: Gilmour Jia
"""Direct-weight policy network: features in, portfolio weights out.

The model answers one question per decision date: given each stock's own
state, what its linked names are doing, and the market/account state, what
fraction of NAV should this name carry?  There is no intermediate expected
return and no separate optimiser -- the weights are the output.

Two structural choices matter downstream:

1. the encoder is shared across stocks, so the parameter count does not grow
   with the size of the cross-section and the same map is applied on every
   date;
2. cash is a peer node in the softmax, so the model decides its own exposure
   instead of being forced fully invested.

The softmax over hundreds of names is dense.  Temperature controls the
effective breadth, and top_k keeps only the k largest weights, renormalised,
using a straight-through estimator so dropped names still receive a gradient
through the surrogate.  top_k=0 means "hold everything the softmax touches";
top_k=10 means "at most ten positions, cash included".
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn

__all__ = ["PolicyConfig", "DirectWeightPolicy", "capped_simplex_soft",
           "listnet_loss", "net_growth_surrogate", "portfolio_growth",
           "topk_weights", "TURNOVER_SCALE"]

# Weight-space cost coefficient used inside the training objective.  The
# surrogate charges costs per unit of |dw| once per decision, so the accounting
# is in weight space rather than in shares: it is a training signal, never a
# reported number.
TURNOVER_SCALE = 1.0


@dataclass(frozen=True)
class PolicyConfig:
    # Widths, not names.  Defaults mirror qlp.dataset's design matrix: the own
    # block (ranked features + their missingness flags + the peer-validity flag),
    # the whole peer block (contemporaneous and lagged copies), the market block
    # and the account-state block.
    n_own: int = 21
    n_neighbour: int = 40
    n_market: int = 6
    n_state: int = 3
    n_prev: int = 0             # per-stock previous-weight feature (EIIE-style)
    hidden: tuple = (32, 16)
    cap: float = 0.03
    cost_bp: float = 25.0
    l2: float = 1e-3
    dropout: float = 0.10
    temperature: float = 1.0    # controls breadth; annealed by the trainer
    top_k: int = 0              # 0 = dense; else keep the k largest weights
    top_g: int = 0              # 0 = dense; else softmax only over the top-G scores
    cap_soft: bool = False      # apply the differentiable cap during training

    @property
    def n_input(self) -> int:
        return (self.n_own + self.n_neighbour + self.n_market + self.n_state
                + self.n_prev)


def topk_weights(w: torch.Tensor, k: int) -> torch.Tensor:
    """Keep the k largest entries of w (cash included) and renormalise.

    Straight-through: the forward value is the sparse portfolio, the backward
    pass sees the dense softmax so dropped names are not silently frozen out.
    """
    if k <= 0 or k >= w.numel():
        return w
    vals, pos = torch.topk(w, k)
    sparse = torch.zeros_like(w).scatter(0, pos, vals)
    sparse = sparse / sparse.sum().clamp_min(1e-12)
    return w + (sparse - w).detach()


class DirectWeightPolicy(nn.Module):
    """Shared per-stock encoder plus a cash node, then a normalised allocation."""

    def __init__(self, cfg: PolicyConfig = PolicyConfig()):
        super().__init__()
        self.cfg = cfg
        layers, d = [], cfg.n_input
        for h in cfg.hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(cfg.dropout)]
            d = h
        layers += [nn.Linear(d, 1)]
        self.encoder = nn.Sequential(*layers)
        self.cash_logit = nn.Parameter(torch.zeros(1))

    def scores(self, x: torch.Tensor) -> torch.Tensor:
        """Raw per-stock scores, before temperature and before the top-G mask."""
        return self.encoder(x).squeeze(-1)

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        """x: (n_stocks, n_input) -> (n_stocks + 1,) including cash.

        top_g > 0 applies the AlphaPortfolio-style hard truncation before the
        softmax: only the G highest scores compete for weight.  This is a
        forward-only mask; the auxiliary ListNet loss is computed on the
        unmasked scores, so names outside the current top G can still receive a
        direct gradient and re-enter on a later step.
        """
        z = self.scores(x) / self.cfg.temperature
        g = int(self.cfg.top_g)
        if 0 < g < z.numel():
            vals, pos = torch.topk(z, g)
            z = torch.full_like(z, float("-inf")).scatter(0, pos, vals)
        return torch.cat([z, self.cash_logit])

    def weights(self, x: torch.Tensor) -> torch.Tensor:
        """Weights over (stocks..., cash); sums to 1."""
        w = torch.softmax(self.logits(x), dim=0)
        if self.cfg.top_k:
            w = topk_weights(w, int(self.cfg.top_k))
        if self.cfg.cap_soft:
            w = capped_simplex_soft(w, self.cfg.cap)
        return w

    @torch.no_grad()
    def top_names(self, x: torch.Tensor, k: int):
        """Exact top-k allocation: k largest stock weights, capped, rest cash.

        This is the read-out control.  Temperature cannot hold breadth fixed
        across seeds, because annealing leaves each seed's logits at a different
        scale: two models that both look reasonable can hold 27 names and 795
        names at the same nominal temperature.  Selecting the k largest weights
        makes the holding count an explicit, reproducible choice, and whatever
        the cap forces out stays in cash rather than being spread over names the
        model did not choose.
        """
        w = torch.softmax(self.logits(x), dim=0)[:-1]
        k = int(min(k, w.numel()))
        vals, pos = torch.topk(w, k)
        out = torch.zeros_like(w)
        out[pos] = vals
        # The k selected names absorb the whole book, then the caller's cap
        # decides how much of it can actually be deployed.  Renormalising here
        # rather than only when the sum exceeds one keeps a cold portfolio from
        # silently parking most of NAV in cash just because the softmax was
        # diffuse.
        total = out.sum()
        if float(total) > 0:
            out = out / total
        return out, pos.cpu().numpy()

    @torch.no_grad()
    def effective_breadth(self, x: torch.Tensor) -> dict:
        """Diagnostics: how many names the allocation actually touches."""
        w = self.weights(x)
        s = w[:-1]
        nz = s > 0
        inv = float((s[nz] ** 2).sum()) if bool(nz.any()) else 0.0
        return dict(n_nonzero=int(nz.sum()),
                    n_above_1e4=int((s > 1e-4).sum()),
                    effective_n=(1.0 / inv) if inv > 0 else 0.0,
                    max_weight=float(s.max()) if bool(nz.any()) else 0.0,
                    cash_weight=float(w[-1]))


def capped_simplex_soft(w: torch.Tensor, cap: float) -> torch.Tensor:
    """Differentiable cap: shrink the excess, redistribute into slack.

    Same feasible set as the engine's hard capped-simplex projection, but this
    version keeps a gradient path so it can be used inside training.
    """
    over = torch.clamp(w - cap, min=0.0)
    if float(over.detach().sum()) <= 0.0:
        return w
    w2 = w - over
    room = torch.clamp(cap - w2, min=0.0)
    denom = room.sum()
    if float(denom.detach()) <= 0.0:
        return w2
    return w2 + room * (over.sum() / denom)


def listnet_loss(scores: torch.Tensor, returns: torch.Tensor,
                  temperature: float = 1.0, label: str = "rank") -> torch.Tensor:
    """Top-1 ListNet cross-entropy between forward returns and scores.

    scores and returns index the same stocks on one decision date.  The target
    distribution is built from the cross-sectional rank of the realised return
    (ties broken by index), which keeps a single extreme winner from turning the
    target into a one-hot vector.  The loss is divided by log(n) so its initial
    value is O(1) and one coefficient is comparable across dates with different
    cross-section sizes.

    This is an auxiliary training signal only; it is never used to select the
    portfolio or to report performance.
    """
    if scores.shape != returns.shape:
        raise ValueError("scores and returns must describe the same universe")
    n = int(scores.numel())
    if n < 2:
        return scores.sum() * 0.0
    if temperature <= 0:
        raise ValueError("listnet temperature must be positive")
    if label == "rank":
        order = torch.argsort(torch.argsort(returns))
        target = order.to(scores.dtype) / float(n - 1)
    elif label == "return":
        target = returns.to(scores.dtype)
    else:
        raise ValueError("listnet label must be 'rank' or 'return'")
    p = torch.softmax(target / temperature, dim=0)
    log_q = torch.log_softmax(scores / temperature, dim=0)
    return -(p * log_q).sum() / math.log(float(n))


def net_growth_surrogate(weights: torch.Tensor, rets: torch.Tensor,
                         prev_w: torch.Tensor, cost_bp: float) -> torch.Tensor:
    """Differentiable stand-in for net log growth over one holding period.

    Both vectors index (stocks..., cash).  Costs are charged once per decision on
    the traded notional.  This is a training objective: every reported number is
    recomputed by the hard executor.
    """
    if weights.shape != prev_w.shape:
        raise ValueError("weights and prev_w must describe the same universe")
    gross = (weights[:-1] * rets).sum()
    turnover = (weights - prev_w).abs().sum() * TURNOVER_SCALE
    return gross - (cost_bp / 1e4) * turnover


def portfolio_growth(weights: torch.Tensor, rets: torch.Tensor, cost_bp: float,
                     active_prev: torch.Tensor | None = None) -> torch.Tensor:
    """Growth of a portfolio on *today's* cross-section, net of trades.

    The set of names eligible on one decision date is not the set eligible on the
    next, so the allocation vector changes length.  A share the model wants to
    keep costs nothing; a share it newly buys costs its weight; and a share it
    drops -- including one that leaves the cross-section entirely -- costs what
    it was carrying.  ``active_prev[j]`` is the previous weight of today's j-th
    stock (zero if it was not held), so trades are measured on today's axis and
    nothing has to be aligned across dates.
    """
    w = weights[:-1]
    gross = (w * rets).sum()
    buys = torch.clamp(w - active_prev, min=0.0).sum() if active_prev is not None \
        else w.sum()
    dropped = torch.clamp(active_prev - w, min=0.0).sum() if active_prev is not None \
        else torch.zeros((), dtype=w.dtype, device=w.device)
    return gross - (cost_bp / 1e4) * (buys + dropped) * TURNOVER_SCALE