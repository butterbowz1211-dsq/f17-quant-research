# Research project author: Gilmour Jia
"""Contract tests for the direct-weight policy network.

Two properties the research depends on, and which are cheap to break by
accident: the allocation is always a valid portfolio, and the training
objective actually reaches every parameter.  A silent zero gradient would make
an entire training run look plausible while learning nothing.

Widths here use the shipped convention: ``n_own`` is the width of the own
block (ranked features + their missing flags + the peer-validity flag), and
``n_neighbour`` is the whole peer block (contemporaneous and lagged copies).
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import torch

from qlp.policy import (DirectWeightPolicy, PolicyConfig, capped_simplex_soft,
                        listnet_loss, net_growth_surrogate, portfolio_growth,
                        topk_weights)

BASE = dict(n_own=21, n_neighbour=40, n_market=6, n_state=3)


def cfg(**over):
    return PolicyConfig(**dict(BASE, **over))


def make_inputs(n: int = 60, n_input: int = 70, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, n_input, generator=g)
    r = torch.randn(n, generator=g) * 0.05
    return x, r


# ------------------------------------------------------------ the portfolio ---

def test_weights_are_a_valid_portfolio():
    x, _ = make_inputs()
    p = DirectWeightPolicy(cfg())
    w = p.weights(x)
    assert w.shape == (x.shape[0] + 1,)
    assert float(w.sum()) == pytest.approx(1.0, abs=1e-6)
    assert float(w.min()) >= 0.0


def test_cash_is_a_peer_not_a_residual():
    """Cash must be able to take essentially the whole book.

    Note the logit has to be pushed down *relative to the stocks*: softmax is
    shift-invariant, so a lone negative cash logit with all stock logits at zero
    is not a meaningful test.
    """
    x, _ = make_inputs(n=20)
    p = DirectWeightPolicy(cfg())
    with torch.no_grad():
        p.encoder[-1].bias.fill_(-5.0)     # every stock pushed well down
        p.cash_logit.fill_(10.0)
    w = p.weights(x)
    assert float(w[-1]) > 0.999


def test_temperature_controls_breadth_monotonically():
    """Colder temperature, fewer names above a practical weight threshold.

    The encoder is perturbed first: at initialisation every logit is zero, the
    softmax is uniform, and temperature would have nothing to act on.
    """
    x, _ = make_inputs(n=400, seed=3)
    # One fixed logit vector (scaled up so it has a real spread), read out at
    # several temperatures.  Re-drawing the encoder per temperature would change
    # the logits as well and would test nothing.
    torch.manual_seed(0)
    p = DirectWeightPolicy(cfg(temperature=1.0))
    with torch.no_grad():
        for q in p.encoder:
            if hasattr(q, "weight"):
                q.weight.mul_(3.0)
        z = p.encoder(x).squeeze(-1).detach()
    breadth = []
    for temp in (4.0, 1.0, 0.25, 0.05):
        w = torch.softmax(z / temp, dim=0)
        # Effective number of positions, 1 / sum(w^2): the smooth version of
        # "how many names does this allocation really hold", and the quantity the
        # anneal is actually steering.
        breadth.append(float(1.0 / (w ** 2).sum()))
    # Colder temperature concentrates the allocation, so effective breadth falls.
    assert breadth == sorted(breadth, reverse=True), breadth
    assert breadth[-1] < breadth[0] * 0.9, breadth


def test_topk_keeps_at_most_k_positions():
    x, _ = make_inputs(n=120)
    p = DirectWeightPolicy(cfg(top_k=11))
    w = p.weights(x)
    assert int((w > 0).sum()) <= 11
    assert float(w.sum()) == pytest.approx(1.0, abs=1e-6)


def test_topk_is_straight_through():
    """Dropped names must still receive gradient, or they can never come back."""
    x, r = make_inputs(n=40)
    p = DirectWeightPolicy(cfg(top_k=5))
    w = p.weights(x)
    loss = net_growth_surrogate(w, r, torch.zeros_like(w), 25.0)
    loss.backward()
    grads = [q for q in p.parameters() if q.grad is not None]
    assert grads
    assert any(float(q.grad.abs().sum()) > 0 for q in grads)


def test_topk_helper_is_a_noop_for_large_k():
    w = torch.tensor([0.4, 0.3, 0.2, 0.1])
    assert torch.allclose(topk_weights(w, 9), w)
    assert torch.allclose(topk_weights(w, 0), w)


def test_prev_weight_input_widens_the_encoder():
    """The EIIE-style previous-weight column must be a real, separate input."""
    c = cfg(n_prev=1)
    assert c.n_input == 71
    x, _ = make_inputs(n=40, n_input=71)
    p = DirectWeightPolicy(c)
    w = p.weights(x)
    assert w.shape == (41,)
    assert float(w.sum()) == pytest.approx(1.0, abs=1e-6)


def test_parameter_count_is_independent_of_cross_section():
    p = DirectWeightPolicy(cfg())
    n = sum(q.numel() for q in p.parameters())
    small = p.weights(make_inputs(n=5)[0])
    large = p.weights(make_inputs(n=500)[0])
    assert small.shape == (6,)
    assert large.shape == (501,)
    assert n == sum(q.numel() for q in DirectWeightPolicy(cfg()).parameters())


# ---------------------------------------------------------- the objective ---

def test_surrogate_gradient_reaches_every_parameter():
    x, r = make_inputs(n=80)
    p = DirectWeightPolicy(cfg())
    prev = torch.zeros(x.shape[0] + 1)
    loss = net_growth_surrogate(p.weights(x), r, prev, 25.0)
    loss.backward()
    for name, param in p.named_parameters():
        assert param.grad is not None, name
        assert torch.isfinite(param.grad).all(), name
    touched = sum(1 for _, q in p.named_parameters() if float(q.grad.abs().sum()) > 0)
    assert touched == len(list(p.named_parameters()))


def test_surrogate_charges_turnover():
    r = torch.zeros(4)
    w = torch.tensor([0.25, 0.25, 0.25, 0.25, 0.0])
    flat = net_growth_surrogate(w, r, w.clone(), 25.0)
    churn = net_growth_surrogate(w, r, torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0]), 25.0)
    assert float(flat) == pytest.approx(0.0, abs=1e-12)
    assert float(churn) == pytest.approx(-0.0050, rel=1e-6)


def test_surrogate_rejects_mismatched_universes():
    with pytest.raises(ValueError):
        net_growth_surrogate(torch.zeros(5), torch.zeros(4), torch.zeros(6), 25.0)


def test_portfolio_growth_charges_entries_and_exits():
    r = torch.zeros(3)
    w = torch.tensor([0.3, 0.3, 0.3, 0.1])
    held = torch.tensor([0.3, 0.0, 0.0])
    # One kept name costs nothing, one new name costs 0.3, one exit costs 0.3.
    got = float(portfolio_growth(w, r, 25.0, active_prev=held))
    assert got == pytest.approx(-(25.0 / 1e4) * 0.6, rel=1e-6)


def test_portfolio_growth_without_prior_holdings_charges_the_whole_build():
    r = torch.zeros(2)
    w = torch.tensor([0.4, 0.4, 0.2])
    got = float(portfolio_growth(w, r, 25.0, active_prev=None))
    assert got == pytest.approx(-(25.0 / 1e4) * 0.8, rel=1e-6)


def test_portfolio_growth_earns_the_returns():
    w = torch.tensor([0.5, 0.5, 0.0])
    r = torch.tensor([0.10, -0.02])
    held = w[:-1].clone()
    got = float(portfolio_growth(w, r, 0.0, active_prev=held))
    assert got == pytest.approx(0.5 * 0.10 + 0.5 * -0.02, rel=1e-6)


# ------------------------------------------------------------- top-G scope ---

def test_top_g_keeps_at_most_g_stocks_plus_cash():
    x, _ = make_inputs(n=120)
    p = DirectWeightPolicy(cfg(top_g=10))
    w = p.weights(x)
    assert int((w[:-1] > 0).sum()) <= 10
    assert float(w.sum()) == pytest.approx(1.0, abs=1e-6)


def test_top_g_masks_scores_before_softmax():
    x, _ = make_inputs(n=50)
    p = DirectWeightPolicy(cfg(top_g=7))
    z = p.logits(x)[:-1]
    assert int(torch.isfinite(z).sum()) == 7
    assert int((z == float("-inf")).sum()) == 43


def test_top_g_does_not_mask_when_g_exceeds_universe():
    x, _ = make_inputs(n=20)
    p = DirectWeightPolicy(cfg(top_g=50))
    assert torch.isfinite(p.logits(x)[:-1]).all()


# --------------------------------------------------------------- listnet ---

def test_listnet_prefers_the_correct_ranking():
    r = torch.linspace(-0.10, 0.10, 60)
    good = listnet_loss(r.clone(), r, temperature=0.5)
    bad = listnet_loss(-r.clone(), r, temperature=0.5)
    assert float(good) < float(bad)


def test_listnet_reaches_every_score_even_when_top_g_is_small():
    """The portfolio loss cannot reach a dropped name; ListNet must."""
    x, r = make_inputs(n=40)
    p = DirectWeightPolicy(cfg(top_g=5))
    z = p.scores(x).detach().requires_grad_(True)
    loss = listnet_loss(z, r, temperature=0.5)
    loss.backward()
    assert z.grad is not None
    assert torch.isfinite(z.grad).all()
    assert float(z.grad.abs().min()) > 0


def test_listnet_rejects_mismatched_shapes_and_bad_temperature():
    with pytest.raises(ValueError):
        listnet_loss(torch.zeros(5), torch.zeros(4))
    with pytest.raises(ValueError):
        listnet_loss(torch.zeros(5), torch.zeros(5), temperature=0.0)


# -------------------------------------------------------------- the cap ---

def test_capped_simplex_soft_respects_the_cap():
    w = torch.tensor([0.5, 0.3, 0.1, 0.1])
    out = capped_simplex_soft(w, 0.25)
    assert float(out.max()) <= 0.25 + 1e-6
    assert float(out.sum()) == pytest.approx(1.0, abs=1e-6)


def test_capped_simplex_soft_is_a_noop_when_feasible():
    w = torch.tensor([0.2, 0.2, 0.2, 0.2, 0.2])
    assert torch.allclose(capped_simplex_soft(w, 0.25), w)


# ------------------------------------------------------------- temperatures ---

def test_temperature_is_replaceable_while_training():
    """The anneal swaps cfg mid-run; the encoder must survive that."""
    p = DirectWeightPolicy(cfg(temperature=1.0))
    p.cfg = dataclasses.replace(p.cfg, temperature=0.05)
    w = p.weights(make_inputs(n=30)[0])
    assert float(w.sum()) == pytest.approx(1.0, abs=1e-6)
