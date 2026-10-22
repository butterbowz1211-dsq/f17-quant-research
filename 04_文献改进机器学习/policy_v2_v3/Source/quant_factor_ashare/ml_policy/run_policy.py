# Research project author: Gilmour Jia
"""Walk-forward training and evaluation of the direct-weight policy.

Protocol, fixed before any result is looked at:

* expanding folds from qlp.splits: 6 years train, 2 years validation, 1 year
  test, anchored on the first year the universe is usable;
* the training objective is the differentiable surrogate in qlp.policy.
  Every performance number in the report comes from the hard executor in
  qlp.execution.  The gap between the two is reported, never hidden;
* model selection uses the validation window only.  The test window is touched
  once, at the end, for the report;
* market/state standardisation is fitted on the training slice and then frozen;
* seeds are averaged at the target-weight level and the ledger is run once, so
  transactions costs cannot be double counted;
* holding count is a reported decision, not an emergent accident: the same
  trained ensemble is read out at each requested count.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from qlp import execution as ex
from qlp import weights as wt
from qlp.data import build_prep
from qlp.dataset import DatasetConfig, assemble, make_sample, standardize_state
from qlp.policy import (DirectWeightPolicy, PolicyConfig, listnet_loss,
                        portfolio_growth)
from qlp.splits import expanding_folds, holdout_fold

HERE = Path(__file__).resolve().parent
ART = HERE / "artifacts"
ART.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------- metrics ---

def perf_metrics(net, periods_per_year: float = 252.0) -> dict:
    """Annualised performance of a daily net-return series."""
    net = np.nan_to_num(np.asarray(net, dtype=float), nan=0.0, posinf=0.0,
                        neginf=0.0)
    n = net.size
    if n == 0:
        return dict(n_days=0)
    growth = float(np.prod(1.0 + net))
    years = n / periods_per_year
    cagr = growth ** (1.0 / years) - 1.0 if growth > 0 else -1.0
    vol = float(np.std(net, ddof=1) * np.sqrt(periods_per_year)) if n > 1 else 0.0
    curve = np.cumprod(1.0 + net)
    peak = np.maximum.accumulate(curve)
    dd = float(np.min(curve / peak - 1.0))
    downside = net[net < 0]
    dvol = (float(np.std(downside, ddof=1) * np.sqrt(periods_per_year))
            if downside.size > 1 else 0.0)
    return dict(n_days=int(n), total_return=growth - 1.0, cagr=float(cagr),
                vol=vol,
                sharpe=float(cagr / vol) if vol > 1e-12 else 0.0,
                max_drawdown=dd,
                sortino=float(cagr / dvol) if dvol > 1e-12 else 0.0,
                hit_rate=float((net > 0).mean()),
                skew=float(pd.Series(net).skew()),
                worst_day=float(net.min()), best_day=float(net.max()))


def ledger_metrics(ledger: dict) -> dict:
    """Headline plus operating statistics for one hard-executor ledger."""
    daily = ledger["daily"].copy()
    if daily.empty:
        return dict(n_days=0)
    out = perf_metrics(daily["net"].to_numpy())
    n = max(1, len(daily))
    out.update(dict(
        mean_holdings=float(daily["n_holdings"].mean()),
        mean_cash_weight=float(daily["cash_weight"].mean()),
        annual_two_way_turnover=float(daily["turnover_currency"].sum()
                                      / max(1e-9, daily["nav"].mean())
                                      / (n / 252.0)),
        total_cost_bp_of_capital=float(daily["cost_currency"].sum()
                                       / max(1e-9, daily["nav"].iloc[0]) * 1e4),
        blocked_buy_days=int((daily["blocked_buy"] > 0).sum()),
        blocked_sell_days=int((daily["blocked_sell"] > 0).sum()),
        stale_weight_mean=float(daily["stale_weight"].mean()),
        stale_weight_max=float(daily["stale_weight"].max()),
        final_nav=float(daily["nav"].iloc[-1]),
    ))
    return out


def slice_ledger(ledger: dict, start, end) -> dict:
    """Restrict a ledger to a date window, keeping the full row structure."""
    daily = ledger["daily"]
    sel = (daily.index >= pd.Timestamp(start)) & (daily.index <= pd.Timestamp(end))
    out = dict(ledger)
    out["daily"] = daily.loc[sel]
    return out


def _jsonable(obj, depth: int = 0):
    """Convert a report into something json.dumps renders readably.

    ``default=str`` silently turns a numpy datetime64 into "1970-01-01" -- the
    epoch, not the date -- which is exactly the kind of wrong number that looks
    like data.  Dates are normalised here instead.
    """
    if depth > 8:
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _jsonable(v, depth + 1) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v, depth + 1) for v in obj]
    if isinstance(obj, (np.datetime64, pd.Timestamp)):
        return str(pd.Timestamp(obj).date())
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


# --------------------------------------------------------- model plumbing ---

def build_samples(a, t_list, cache: dict):
    """Assemble Sample objects lazily and memoise them, keyed by date index."""
    out = []
    for t in t_list:
        key = int(t)
        s = cache.get(key)
        if s is None:
            s = make_sample(a, int(t), slot=int(t) // max(1, int(a.rb)))
            cache[key] = s
        out.append(s)
    return out


def _penalty(policy, cfg: PolicyConfig):
    """L2 on the encoder weights.

    An L1 penalty on the allocation was tried and is deliberately absent: the
    cost of dropping one of eight hundred names is negligible while the cost of
    shrinking every weight is bounded, so L1 moves weight into cash without
    making the book sparse.  Breadth is controlled by the read-out holding
    count instead.
    """
    if not cfg.l2:
        return torch.zeros((), device=next(policy.parameters()).device)
    acc = None
    for name, p in policy.named_parameters():
        if "weight" not in name:
            continue
        term = (p ** 2).sum()
        acc = term if acc is None else acc + term
    if acc is None:
        return torch.zeros((), device=next(policy.parameters()).device)
    return 0.5 * cfg.l2 * acc


def _batch_objective(policy, samples, held, device, listnet_weight: float = 0.0,
                     listnet_temp: float = 1.0, account_state: bool = False,
                     prev_weight: bool = False,
                     prev_cash: float = 0.0, prev_hold: float = 0.0):
    """Sum of surrogate net growth over one batch, plus the carried holdings.

    ``held`` maps a security index to the weight it was left carrying at the end
    of the previous decision, so a name that leaves the eligible cross-section
    and comes back is still charged for having been sold.

    When ``account_state`` is on, the last three input columns are overwritten
    on the fly with the cash weight and holding count of the previous decision
    in this same chronological pass.  The samples are built once with zeros in
    that block, which is correct for an isolated decision but wrong for a path:
    the policy should see the book it is actually carrying.
    """
    total = None
    aux = None
    for s in samples:
        x = torch.from_numpy(s.feat).to(device)
        idx = torch.from_numpy(s.ids.astype(np.int64)).to(device)
        if account_state or prev_weight:
            x = x.clone()
        if account_state:
            n_prev = 1 if prev_weight else 0
            x[:, -3 - n_prev] = float(prev_cash)
            x[:, -1 - n_prev] = math.log1p(max(0.0, float(prev_hold)))
        if prev_weight:
            x[:, -1] = held[idx]
        r = torch.from_numpy(s.returns).to(device)
        w = policy.weights(x)
        term = portfolio_growth(w, r, policy.cfg.cost_bp, active_prev=held[idx])
        total = term if total is None else total + term
        if listnet_weight:
            z = policy.scores(x)
            a = listnet_loss(z, r, temperature=listnet_temp)
            aux = a if aux is None else aux + a
        prev_cash = float(w[-1].detach())
        prev_hold = float((w[:-1].detach() > 1e-4).sum())
        held = held.clone()
        held.zero_()
        held[idx] = w[:-1].detach()
    return total, held, aux, prev_cash, prev_hold


@torch.no_grad()
def evaluate_surrogate(policy, samples, device, rb: int, n_symbols: int) -> dict:
    """Training-view diagnostics only; never a reported performance number.

    Besides the surrogate growth path, this computes the cross-sectional rank IC
    of the model's raw scores against the realised forward return on every
    validation decision.  That IC is the alternative early-stopping signal: it
    asks whether the score ranks stocks, not whether one particular allocation
    happened to make money over two dozen overlapping periods.
    """
    policy.eval()
    if not samples:
        return dict(n_decisions=0)
    held = torch.zeros(int(n_symbols), device=device)
    growth, breadth, cash, tops, ics = [], [], [], [], []
    for s in samples:
        x = torch.from_numpy(s.feat).to(device)
        r = torch.from_numpy(s.returns).to(device)
        w = policy.weights(x)
        idx = torch.from_numpy(s.ids.astype(np.int64)).to(device)
        growth.append(float(portfolio_growth(w, r, policy.cfg.cost_bp,
                                             active_prev=held[idx]).item()))
        sw = w[:-1]
        breadth.append(float((sw > 1e-4).sum().item()))
        cash.append(float(w[-1].item()))
        tops.append(float(sw.max().item()))
        z_all = policy.scores(x)
        # Rank only the names that can actually receive weight: with top-G the
        # excluded scores receive no portfolio gradient, so including them would
        # dilute the validation IC with untrained noise.
        active = torch.isfinite(policy.logits(x)[:-1])
        z = z_all[active].detach().cpu().numpy()
        realised = np.asarray(s.returns, dtype=np.float64)[active.cpu().numpy()]
        ok = np.isfinite(z) & np.isfinite(realised)
        if int(ok.sum()) >= 10:
            zr = _rankdata(z[ok])
            rr = _rankdata(realised[ok])
            if float(zr.std()) > 1e-12 and float(rr.std()) > 1e-12:
                ic = float(np.corrcoef(zr, rr)[0, 1])
                if np.isfinite(ic):
                    ics.append(ic)
        held = held.clone()
        held.zero_()
        held[idx] = sw.detach()
    g = np.asarray(growth, dtype=float)
    per_year = 252.0 / max(1, int(rb))
    mu = float(g.mean())
    sd = float(g.std(ddof=1)) if g.size > 1 else 0.0
    out = dict(n_decisions=int(g.size), mean_log_growth=mu,
               ann_return_approx=float(np.expm1(mu * per_year)),
               ann_vol_approx=float(sd * np.sqrt(per_year)),
               info_ratio=float(mu / sd * np.sqrt(per_year)) if sd > 1e-12 else 0.0,
               mean_breadth=float(np.mean(breadth)), mean_cash=float(np.mean(cash)),
               mean_top_weight=float(np.mean(tops)))
    if ics:
        arr = np.asarray(ics, dtype=float)
        out.update(n_ic=int(arr.size), rank_ic=float(arr.mean()),
                   rank_ic_std=float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
                   rank_ic_ir=float(arr.mean() / arr.std(ddof=1))
                   if arr.size > 1 and arr.std(ddof=1) > 1e-12 else 0.0,
                   rank_ic_t=float(arr.mean() / (arr.std(ddof=1) / np.sqrt(arr.size)))
                   if arr.size > 1 and arr.std(ddof=1) > 1e-12 else 0.0,
                   rank_ic_hit=float((arr > 0).mean()))
    else:
        out.update(n_ic=0, rank_ic=None, rank_ic_std=None, rank_ic_ir=None,
                   rank_ic_t=None, rank_ic_hit=None)
    return out


def selection_score(v: dict, risk_aversion: float = 0.5,
                    metric: str = "score") -> float:
    """Score a validation result for early stopping.

    The original metric is the risk-adjusted surrogate growth.  Mean log growth
    alone will happily pick the luckiest seed: with a couple of dozen validation
    decisions the standard error is large, and two portfolios with identical
    expectation can differ by a full point of annualised return through noise.
    The variance penalty prefers the reproducible seed without turning selection
    into a second, unlogged objective.

    The rank-IC metrics use the validation cross-section instead: they ask
    whether the score orders stocks, not whether one allocation path happened to
    make money.  The portfolio path is still reported; it just stops deciding
    which epoch is kept.  Growth is the unpenalised ablation.
    """
    if not v:
        return float("-inf")
    if metric == "score":
        if not v.get("n_decisions"):
            return float("-inf")
        return float(v["ann_return_approx"]) - risk_aversion * float(v["ann_vol_approx"])
    if metric == "growth":
        if not v.get("n_decisions"):
            return float("-inf")
        return float(v["mean_log_growth"])
    if metric in ("ic", "icir"):
        if not v.get("n_ic") or v.get("rank_ic") is None:
            return float("-inf")
        if metric == "ic":
            return float(v["rank_ic"])
        return float(v["rank_ic_ir"])
    raise ValueError("selection metric must be score, growth, ic or icir; got %r"
                     % (metric,))


def train_one_seed(train_samples, valid_samples, cfg: PolicyConfig, *, seed: int,
                   epochs: int, lr: float, batch: int, patience: int,
                   temp_end: float | None, rb: int, n_symbols: int, device: str,
                   listnet_weight: float = 0.0, listnet_temp: float = 1.0,
                   chronological: bool = False, account_state: bool = False,
                   prev_weight: bool = False, selection_metric: str = "score",
                   verbose: bool = True):
    """Train one seed.  Early stopping is decided on the validation window."""
    if (account_state or prev_weight) and not chronological:
        raise ValueError("account_state/prev_weight require chronological training: "
                         "a random path has no meaningful previous book")
    if selection_metric in ("ic", "icir") and (account_state or prev_weight):
        raise ValueError("rank-IC selection is not path-aware for account_state/"
                         "prev_weight; use --selection score until the validation "
                         "walk feeds the previous book")
    torch.manual_seed(seed)
    np.random.seed(seed)
    policy = DirectWeightPolicy(cfg).to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    n_par = int(sum(p.numel() for p in policy.parameters()))
    best = dict(score=-np.inf, state=None, epoch=-1, temperature=float(cfg.temperature),
                valid=None)
    history = []
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        if temp_end is not None and epochs > 1:
            frac = (epoch - 1) / (epochs - 1)
            policy.cfg = dataclasses.replace(
                cfg, temperature=float(cfg.temperature
                                     + (temp_end - cfg.temperature) * frac))
        policy.train()
        if chronological:
            order = np.arange(len(train_samples))
        else:
            order = np.random.permutation(len(train_samples))
        opt.zero_grad(set_to_none=True)
        held = torch.zeros(int(n_symbols), device=device)
        prev_cash, prev_hold = 0.0, 0.0
        n_batch = 0
        for i in range(0, len(order), batch):
            chunk = [train_samples[j] for j in order[i:i + batch]]
            loss, held, aux, prev_cash, prev_hold = _batch_objective(
                policy, chunk, held, device, listnet_weight=listnet_weight,
                listnet_temp=listnet_temp, account_state=account_state,
                prev_weight=prev_weight, prev_cash=prev_cash, prev_hold=prev_hold)
            if loss is None:
                continue
            objective = -(loss / len(chunk)) + _penalty(policy, cfg)
            if listnet_weight and aux is not None:
                objective = objective + listnet_weight * aux / len(chunk)
            objective.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            held = held.detach()
            n_batch += 1
        if n_batch == 0:
            break
        v = (evaluate_surrogate(policy, valid_samples, device, rb, n_symbols)
             if valid_samples else None)
        # "last" is a deliberate ablation: it ignores validation and keeps the
        # final epoch.  That separates "the model at epoch 24 is bad" from
        # "the validation objective stopped us at epoch 10".
        score = (float(epoch) if selection_metric == "last"
                 else selection_score(v, metric=selection_metric))
        history.append(dict(epoch=epoch, valid_score=score,
                            valid_ir=(v or {}).get("info_ratio"),
                            valid_growth=(v or {}).get("mean_log_growth"),
                            valid_ic=(v or {}).get("rank_ic"),
                            valid_ic_ir=(v or {}).get("rank_ic_ir"),
                            valid_breadth=(v or {}).get("mean_breadth"),
                            valid_cash=(v or {}).get("mean_cash"),
                            temperature=float(policy.cfg.temperature)))
        if score > best["score"] + 1e-9:
            best = dict(score=score, epoch=epoch, valid=v,
                        temperature=float(policy.cfg.temperature),
                        state={k: t.detach().clone()
                               for k, t in policy.state_dict().items()})
        elif epoch - best["epoch"] >= patience:
            if verbose:
                print("      early stop at epoch %d (best %d)" % (epoch, best["epoch"]),
                      flush=True)
            break
        if verbose and (epoch % 10 == 0 or epoch == 1):
            vic = (v or {}).get("rank_ic")
            print("      epoch %3d  valid_IR %+.3f  IC %s  breadth %6.1f  cash %.3f  T %.3f"
                  % (epoch, (v or {}).get("info_ratio", float("nan")),
                     ("%+.4f" % vic) if vic is not None else "   n/a",
                     (v or {}).get("mean_breadth", float("nan")),
                     (v or {}).get("mean_cash", float("nan")),
                     float(policy.cfg.temperature)), flush=True)
    if best["state"] is not None:
        policy.load_state_dict(best["state"])
        policy.cfg = dataclasses.replace(cfg, temperature=float(best["temperature"]))
    policy.eval()
    bv = best.get("valid") or {}
    if verbose and bv:
        print("      selected epoch %d: growth %.5f/decision  ann_ret %.3f  ann_vol %.3f"
              "  IR %.3f  IC %s  breadth %.0f  cash %.3f"
              % (best["epoch"], bv["mean_log_growth"], bv["ann_return_approx"],
                 bv["ann_vol_approx"], bv["info_ratio"],
                 ("%+.4f" % bv["rank_ic"]) if bv.get("rank_ic") is not None else "n/a",
                 bv["mean_breadth"], bv["mean_cash"]), flush=True)
    return dict(policy=policy, n_params=n_par, best_epoch=int(best["epoch"]),
                best_valid_score=float(best["score"]),
                best_valid_ir=float(bv.get("info_ratio", 0.0)),
                best_valid_ic=(float(bv["rank_ic"])
                               if bv.get("rank_ic") is not None else None),
                best_valid=bv, history=history, seconds=float(time.time() - t0))


@torch.no_grad()
def target_weights_for_dates(policy, a, t_list, cap: float, device: str,
                             top_names: int = 0, account_state: bool = False):
    """Model weights on each decision date, hard-capped, cash left as residual.

    ``top_names`` > 0 keeps only that many names, which is the only way to make
    the holding count the same across seeds; see DirectWeightPolicy.top_names.

    When the policy was trained with account state, the read-out has to walk the
    dates in the same order and feed it the same book (cash / holding count) it
    saw in training.  Leaving those two columns at their build-time zeros makes
    the reported portfolio a different policy from the one selected.
    """
    policy.eval()
    grid = np.zeros((len(a.dates), len(a.symbols)), dtype=np.float64)
    prev_w = np.zeros(len(a.symbols), dtype=np.float64)
    prev_cash, prev_hold = 0.0, 0.0
    for t in t_list:
        t = int(t)
        s = make_sample(a, t, slot=t // max(1, int(a.rb)),
                        cash_w=prev_cash if account_state else 0.0,
                        n_hold=prev_hold if account_state else 0.0,
                        prev_w=prev_w if a.with_prev_weight else None)
        x = torch.from_numpy(s.feat).to(device)
        if top_names:
            stock, _ = policy.top_names(x, top_names)
            stock_w = stock.cpu().numpy().astype(np.float64)
        else:
            w = policy.weights(x).cpu().numpy().astype(np.float64)
            stock_w = wt.project_capped_simplex(
                w[:-1], np.zeros(w.size - 1), np.full(w.size - 1, cap),
                total=float(w.sum()))[0]
        grid[t, s.ids] = stock_w
        if a.with_prev_weight:
            prev_w[s.ids] = stock_w
        if account_state:
            prev_cash = max(0.0, 1.0 - float(stock_w.sum()))
            prev_hold = float((stock_w > 1e-4).sum())
    return grid


def hold_forward(grid: np.ndarray, t_list, rebalance: int) -> np.ndarray:
    """Carry each decision date's target forward until the next decision date."""
    ts = [int(t) for t in t_list]
    if not ts:
        return np.zeros_like(grid)
    out = np.zeros_like(grid)
    for i, t in enumerate(ts):
        stop = ts[i + 1] if i + 1 < len(ts) else min(grid.shape[0], t + int(rebalance))
        out[t:stop] = grid[t]
    return out


def _rankdata(v: np.ndarray) -> np.ndarray:
    """Average-rank transform without a scipy dependency."""
    v = np.asarray(v, dtype=np.float64)
    order = np.argsort(v, kind="mergesort")
    ranks = np.empty(v.size, dtype=np.float64)
    ranks[order] = np.arange(v.size, dtype=np.float64)
    # Split ties so a block of identical returns cannot create a spurious rank.
    sv = v[order]
    i = 0
    while i < sv.size:
        j = i + 1
        while j < sv.size and sv[j] == sv[i]:
            j += 1
        if j - i > 1:
            ranks[order[i:j]] = 0.5 * (i + j - 1)
        i = j
    return ranks


@torch.no_grad()
def score_ic(models, a, t_list, device: str, account_state: bool = False) -> dict:
    """Rank IC of the ensemble score against the realised forward return.

    This is a diagnostic on the same rows the portfolio is read from: if the
    score stops ranking but the portfolio still works, the mapping is carrying
    the result; if the score ranks and the portfolio does not, the mapping is
    the bottleneck.
    """
    ics = []
    if a.with_prev_weight:
        # The score depends on the book each model is carrying, so every model
        # has to walk the dates on its own path before the ensemble is averaged.
        per_model = []
        for m in models:
            prev = np.zeros(len(a.symbols), dtype=np.float64)
            prev_cash, prev_hold = 0.0, 0.0
            zs = []
            for t in t_list:
                t = int(t)
                s = make_sample(a, t, slot=t // max(1, int(a.rb)),
                                cash_w=prev_cash if account_state else 0.0,
                                n_hold=prev_hold if account_state else 0.0,
                                prev_w=prev)
                x = torch.from_numpy(s.feat).to(device)
                zs.append(m.scores(x).cpu().numpy())
                w = m.weights(x).cpu().numpy()
                prev[s.ids] = w[:-1]
                if account_state:
                    prev_cash = float(w[-1])
                    prev_hold = float((w[:-1] > 1e-4).sum())
            per_model.append(zs)
        for i, t in enumerate(t_list):
            s = make_sample(a, int(t), slot=int(t) // max(1, int(a.rb)))
            r = np.asarray(s.returns, dtype=np.float64)
            z = np.mean([pm[i] for pm in per_model], axis=0)
            ok = np.isfinite(z) & np.isfinite(r)
            if int(ok.sum()) < 10:
                continue
            ics.append(float(np.corrcoef(_rankdata(z[ok]), _rankdata(r[ok]))[0, 1]))
    else:
        for t in t_list:
            s = make_sample(a, int(t), slot=int(t) // max(1, int(a.rb)))
            x = torch.from_numpy(s.feat).to(device)
            r = np.asarray(s.returns, dtype=np.float64)
            z = np.mean([m.scores(x).cpu().numpy() for m in models], axis=0)
            ok = np.isfinite(z) & np.isfinite(r)
            if int(ok.sum()) < 10:
                continue
            ics.append(float(np.corrcoef(_rankdata(z[ok]), _rankdata(r[ok]))[0, 1]))
    if not ics:
        return dict(n_dates=0)
    arr = np.asarray(ics, dtype=np.float64)
    return dict(n_dates=int(arr.size), ic_mean=float(arr.mean()),
                ic_std=float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
                ic_ir=float(arr.mean() / arr.std(ddof=1)) if arr.size > 1 and arr.std(ddof=1) > 1e-12 else 0.0,
                ic_t=float(arr.mean() / (arr.std(ddof=1) / np.sqrt(arr.size))) if arr.size > 1 and arr.std(ddof=1) > 1e-12 else 0.0,
                hit=float((arr > 0).mean()))




@torch.no_grad()
def save_scores(models, a, t_list, device, out_dir: Path, tag: str,
                fold_idx: int, seeds) -> None:
    """Persist per-seed scores, weights and realised returns for one window.

    The read-out layer needs the raw score, the model's own weight and the
    forward return on the same rows.  Saving all three keeps the final
    portfolio comparison free of a second graph assembly and, more importantly,
    free of any chance that the read-out sees a different sample than training
    did.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    t_arr = np.asarray([int(t) for t in t_list], dtype=np.int64)
    n_dates, n_sym = int(t_arr.size), int(len(a.symbols))
    for seed, model in zip(seeds, models):
        z = np.full((n_dates, n_sym), np.nan, dtype=np.float32)
        w = np.zeros((n_dates, n_sym), dtype=np.float32)
        rets = np.full((n_dates, n_sym), np.nan, dtype=np.float32)
        cash = np.zeros(n_dates, dtype=np.float64)
        for i, t in enumerate(t_arr):
            s = make_sample(a, int(t), slot=int(t) // max(1, int(a.rb)))
            x = torch.from_numpy(s.feat).to(device)
            z[i, s.ids] = model.scores(x).cpu().numpy()
            ww = model.weights(x).cpu().numpy()
            w[i, s.ids] = ww[:-1]
            cash[i] = float(ww[-1])
            rets[i, s.ids] = s.returns
        path = out_dir / ("%s_fold%d_seed%d_scores.npz" % (tag, fold_idx, seed))
        np.savez_compressed(path, z=z, w=w, rets=rets, t=t_arr, cash=cash)
        print("    saved scores %s (%d dates x %d symbols)"
              % (path.name, n_dates, n_sym), flush=True)


def select_top(grid: np.ndarray, t_list, top_names: int, cap: float) -> np.ndarray:
    """Apply the holding count and the cap to the seed-averaged targets.

    Averaging three sparse portfolios can produce a union wider than any one of
    them, so the selection is applied again to the average.  Without this the
    ensemble would quietly hold 3k names while each member held k, and the
    breadth claim in the report would be false.
    """
    for t in t_list:
        row = grid[int(t)]
        if top_names and int((row > 0).sum()) > top_names:
            keep = np.argpartition(-row, top_names - 1)[:top_names]
            mask = np.zeros_like(row, dtype=bool)
            mask[keep] = True
            row[~mask] = 0.0
        nz = np.flatnonzero(row > 0)
        if nz.size == 0:
            continue
        total = float(row[nz].sum())
        if top_names and total > 0:
            row[nz] = row[nz] / total
            total = 1.0
        row[nz] = wt.project_capped_simplex(
            row[nz], np.zeros(nz.size), np.full(nz.size, cap), total=total)[0]
    return grid


# -------------------------------------------------------------- execution ---

def engine_inputs(prep, a, carry, *, cap: float, capital: float, partial: float,
                  band: float, price_guard: float, strict: bool, band_rel: float = 0.0):
    """Feed the hard executor the same arrays the baselines use."""
    idx = pd.DatetimeIndex(a.dates)
    cols = pd.Index(a.symbols)
    close = pd.DataFrame(prep["close"], index=idx, columns=cols)
    open_ = pd.DataFrame(prep["open"], index=idx, columns=cols)
    adv = pd.DataFrame(prep["amount_lag"], index=idx, columns=cols)
    eligible = pd.DataFrame(a.universe, index=idx, columns=cols)
    limit_locked = pd.DataFrame(prep["limit_locked"], index=idx, columns=cols)
    tradable = eligible & (~limit_locked) & np.isfinite(open_) & (open_ > 0)
    locked = pd.DataFrame(False, index=idx, columns=cols)
    targets = pd.DataFrame(carry, index=idx, columns=cols)
    cfg = ex.ExecutionConfig(capital=capital, cap=cap, partial_adjustment=partial,
                             no_trade_weight=band, no_trade_relative=band_rel,
                             price_guard=price_guard, allow_stale=not strict)
    return ex.run_target_weight_engine(open_, close, targets, eligible, tradable,
                                       locked, adv, cfg)


def evaluate_targets(a, prep, fold, models, t_eval, args, device, seed_reports,
                    top_names: int, train_s, valid_s, tr_idx, va_t):
    """Read the trained ensemble out at one holding count and run the ledger.

    Separated from training so several holding counts can be compared without
    retraining.  Diagnostics are computed from the *final* targets: reading them
    off a per-seed intermediate once reported 927 names for a 30-name portfolio.
    """
    grids = [target_weights_for_dates(m, a, t_eval, args.cap, device,
                                      top_names=top_names,
                                      account_state=args.account_state)
             for m in models]
    grid = select_top(np.mean(grids, axis=0), t_eval, top_names, args.cap)
    carry = hold_forward(grid, t_eval, args.rebalance)

    led = engine_inputs(prep, a, carry, cap=args.cap, capital=args.capital,
                        partial=args.partial, band=args.band,
                        price_guard=args.price_guard, strict=args.strict)
    seg = slice_ledger(led, fold.test_start, fold.test_end)
    val_seg = slice_ledger(led, fold.valid_start, fold.valid_end)
    tr_seg = slice_ledger(led, pd.Timestamp(a.dates[int(tr_idx[0])]), fold.train_end)

    t_mask = np.zeros(len(a.dates), dtype=bool)
    t_mask[t_eval] = True
    in_test = t_mask & (a.dates >= np.datetime64(fold.test_start)) \
        & (a.dates <= np.datetime64(fold.test_end))
    names = (carry > 1e-4).sum(axis=1)[in_test].astype(float)
    deployed = carry.sum(axis=1)[in_test]
    max_w = carry[in_test].max(axis=1) if in_test.any() else np.zeros(0)
    diag = dict(
        n_decisions=int(names.size),
        mean_names_above_1e4=float(names.mean()) if names.size else 0.0,
        median_names_above_1e4=float(np.median(names)) if names.size else 0.0,
        p10_names_above_1e4=float(np.percentile(names, 10)) if names.size else 0.0,
        max_names_above_1e4=float(names.max()) if names.size else 0.0,
        mean_deployed=float(deployed.mean()) if deployed.size else 0.0,
        mean_cash=float(1.0 - deployed.mean()) if deployed.size else 0.0,
        max_single_weight=float(max_w.max()) if max_w.size else 0.0,
    )
    entry = dict(
        index=fold.index, label=fold.label(), top_names=int(top_names),
        contamination=("CONTAMINATED" if fold.contaminated else "untouched"),
        n_train=int(len(train_s)), n_valid=int(len(valid_s)),
        test_window=[str(pd.Timestamp(fold.test_start).date()),
                     str(pd.Timestamp(fold.test_end).date())],
        valid_window=[str(pd.Timestamp(va_t[0]).date()) if va_t.size else None,
                      str(pd.Timestamp(va_t[-1]).date()) if va_t.size else None],
        seeds=seed_reports, test_warmup=ledger_metrics(tr_seg),
        validation=ledger_metrics(val_seg), test=ledger_metrics(seg),
        target_diagnostics=diag,
    )
    return entry, carry, seg


# -------------------------------------------------------------------- main ---

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, nargs="*", default=[0])
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--rebalance", type=int, default=21)
    ap.add_argument("--first-year", type=int, default=2013)
    ap.add_argument("--holdout", action="store_true",
                    help="single final fit: train through --holdout-train-end, "
                         "validate the next two years, test --holdout-test-start..last")
    ap.add_argument("--holdout-train-end", type=int, default=2020)
    ap.add_argument("--holdout-test-start", type=int, default=2023)
    ap.add_argument("--hidden", type=int, nargs="*", default=[32, 16])
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--temp-end", type=float, default=0.1)
    ap.add_argument("--fixed-temperature", action="store_true",
                    help="do not anneal the temperature; use --temperature throughout")
    ap.add_argument("--selection", choices=("score", "growth", "ic", "icir", "last"),
                    default="score",
                    help="validation statistic used for early stopping; score is "
                         "ann_return - 0.5*ann_vol, ic/icir are cross-sectional rank IC")
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--top-g", type=int, default=0,
                    help="train-time softmax scope: only the G highest scores")
    ap.add_argument("--listnet-weight", type=float, default=0.0)
    ap.add_argument("--listnet-temp", type=float, default=1.0)
    ap.add_argument("--chronological", action="store_true",
                    help="visit training decisions in date order so the turnover "
                         "cost sees the book actually carried into the next date")
    ap.add_argument("--account-state", action="store_true",
                    help="feed the previous decision's cash weight and holding "
                         "count back into the 3 account-state inputs")
    ap.add_argument("--prev-weight-feature", action="store_true",
                    help="append each stock's own previous weight as a 71st input")
    ap.add_argument("--top-names", type=int, nargs="*", default=[0],
                    help="holding counts to read out; 0 keeps every name the "
                         "policy touches. Several values share one training run.")
    ap.add_argument("--cap", type=float, default=0.03)
    ap.add_argument("--l2", type=float, default=1e-3)
    ap.add_argument("--dropout", type=float, default=0.10)
    ap.add_argument("--cost-bp", type=float, default=25.0)
    ap.add_argument("--partial", type=float, default=0.5)
    ap.add_argument("--band", type=float, default=0.0025)
    ap.add_argument("--price-guard", type=float, default=0.0)
    ap.add_argument("--capital", type=float, default=10_000_000.0)
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--score-ic", action="store_true",
                    help="report rank IC of the ensemble score on the test window")
    ap.add_argument("--save-models", action="store_true",
                    help="write each seed's selected state dict next to the report")
    ap.add_argument("--save-scores", action="store_true",
                    help="persist per-seed scores/weights/returns for the read-out tool")
    ap.add_argument("--score-from", type=str, default="2022-01-01",
                    help="first decision date whose scores are persisted")
    ap.add_argument("--load-models", action="store_true",
                    help="skip training; load <tag>_fold<f>_seed<s>.pt and evaluate")
    ap.add_argument("--tag", type=str, default="policy")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args(argv)
    out_path = Path(args.out) if args.out else ART / ("%s_report.json" % args.tag)

    device = "cuda" if (torch.cuda.is_available() and not args.cpu) else "cpu"
    print("[run] device=%s torch=%s" % (device, torch.__version__), flush=True)
    prep = build_prep()
    t0 = time.time()
    a = assemble(prep, DatasetConfig(rebalance=args.rebalance,
                                     with_prev_weight=args.prev_weight_feature))
    print("[run] assembled in %.1fs: %d dates x %d symbols"
          % (time.time() - t0, len(a.dates), len(a.symbols)), flush=True)

    if args.holdout:
        folds = [holdout_fold(a.dates, first_year=args.first_year,
                              train_end=args.holdout_train_end,
                              valid_years=2,
                              test_start=args.holdout_test_start)]
        print("[run] HOLDOUT single split; --folds is ignored", flush=True)
    else:
        folds = expanding_folds(a.dates, train_years=6, valid_years=2, test_years=1,
                                first_year=args.first_year)
    cfg = PolicyConfig(n_own=int(a.own.shape[2]) + 1,
                       n_neighbour=int(a.peer.shape[2]),
                       n_market=int(a.mkt.shape[1]), n_state=3,
                       hidden=tuple(args.hidden), cap=args.cap,
                       cost_bp=args.cost_bp, l2=args.l2, dropout=args.dropout,
                       temperature=args.temperature, top_k=args.top_k,
                       top_g=args.top_g, n_prev=int(args.prev_weight_feature))
    if cfg.n_input != a.n_input:
        raise ValueError("policy expects %d inputs, dataset provides %d"
                         % (cfg.n_input, a.n_input))
    if args.save_scores and args.prev_weight_feature:
        raise ValueError("--save-scores does not walk the previous-weight path; "
                         "run the read-out from the model instead")
    n_par = int(sum(p.numel() for p in DirectWeightPolicy(cfg).parameters()))
    print("[run] n_input=%d trainable_params=%d" % (cfg.n_input, n_par), flush=True)
    print("[run] selection=%s fixed_temperature=%s start_temperature=%.2f"
          % (args.selection, args.fixed_temperature, args.temperature), flush=True)

    report = dict(tag=args.tag, config=_jsonable(vars(args)),
                  dataset=dict(n_dates=int(len(a.dates)), n_symbols=int(len(a.symbols)),
                               n_input=int(cfg.n_input), n_params=n_par,
                               warmup=int(a.warmup)),
                  folds=[])
    for fidx in args.folds:
        fold = folds[fidx]
        print("[fold %d] %s" % (fidx, fold.label()), flush=True)
        dates = pd.DatetimeIndex(a.dates)
        tr_idx = np.flatnonzero((dates >= fold.train_start) & (dates <= fold.train_end))
        va_idx = np.flatnonzero((dates >= fold.valid_start) & (dates <= fold.valid_end))
        te_idx = np.flatnonzero((dates >= fold.test_start) & (dates <= fold.test_end))
        a.state_scale = standardize_state(a.mkt,
                                          slice(int(tr_idx[0]), int(tr_idx[-1]) + 1))
        cache: dict = {}
        tr_t = a.decision_index(int(tr_idx[0]), int(tr_idx[-1]) + 1)
        va_t = a.decision_index(int(va_idx[0]), int(va_idx[-1]) + 1)
        train_s = build_samples(a, tr_t, cache)
        valid_s = build_samples(a, va_t, cache)
        print("[fold %d] train %d decisions | valid %d decisions"
              % (fidx, len(train_s), len(valid_s)), flush=True)
        if not train_s or not valid_s:
            print("[fold %d] insufficient decisions, skipped" % fidx, flush=True)
            continue

        models, seed_reports = [], []
        for seed in args.seeds:
            print("    seed %d" % seed, flush=True)
            if args.load_models:
                ckpt = ART / ("%s_fold%d_seed%d.pt" % (args.tag, fidx, seed))
                if not ckpt.exists():
                    raise FileNotFoundError(str(ckpt))
                model = DirectWeightPolicy(cfg).to(device)
                model.load_state_dict(torch.load(ckpt, map_location=device))
                model.eval()
                models.append(model)
                seed_reports.append(dict(seed=int(seed), loaded=True,
                                         checkpoint=str(ckpt.name),
                                         best_epoch=None, best_valid_score=None,
                                         best_valid_ir=None, seconds=0.0))
                print("    seed %d: loaded %s" % (seed, ckpt.name), flush=True)
                continue
            res = train_one_seed(train_s, valid_s, cfg, seed=int(seed),
                                 epochs=args.epochs, lr=args.lr, batch=args.batch,
                                 patience=args.patience,
                                 temp_end=(None if args.fixed_temperature
                                           else args.temp_end),
                                 rb=args.rebalance, n_symbols=len(a.symbols),
                                 device=device, listnet_weight=args.listnet_weight,
                                 listnet_temp=args.listnet_temp,
                                 chronological=args.chronological,
                                 account_state=args.account_state,
                                 prev_weight=args.prev_weight_feature,
                                 selection_metric=args.selection)
            models.append(res.pop("policy"))
            res.pop("history", None)
            seed_reports.append(res)
            print("    seed %d: best epoch %d, valid score %+.3f (IR %+.3f), %.1fs"
                  % (seed, res["best_epoch"], res["best_valid_score"],
                     res["best_valid_ir"], res["seconds"]), flush=True)

        if args.save_models and not args.load_models:
            for seed, model in zip(args.seeds, models):
                torch.save(model.state_dict(),
                           ART / ("%s_fold%d_seed%d.pt" % (args.tag, fidx, seed)))
        # The IC must sit on the same decision grid as the portfolio read-out.
        # Anchoring a second grid at the test start evaluates different dates
        # from the ones the executor trades, and on the 2023+ holdout that
        # changed the sign of the reported IC.
        ev_lo = int(tr_idx[0])
        ev_hi = int(min(int(te_idx[-1]), len(dates) - int(args.rebalance) - 1))
        t_eval = a.decision_index(ev_lo, ev_hi + 1)
        test_mask = ((np.asarray(a.dates)[t_eval] >= fold.test_start)
                     & (np.asarray(a.dates)[t_eval] <= fold.test_end))
        test_t = t_eval[test_mask]
        ic = score_ic(models, a, test_t, device, account_state=args.account_state) if args.score_ic else None
        if ic is not None and ic.get("n_dates", 0):
            print("    test score IC %+.4f  IR %+.2f  t %+.2f  hit %.2f  n=%d"
                  % (ic["ic_mean"], ic["ic_ir"], ic["ic_t"], ic["hit"],
                     ic["n_dates"]), flush=True)
        if args.save_scores:
            score_from = np.datetime64(pd.Timestamp(args.score_from))
            score_t = t_eval[np.asarray(a.dates)[t_eval] >= score_from]
            if score_t.size:
                save_scores(models, a, score_t, device, ART, args.tag, fidx, args.seeds)
            else:
                print("    no decisions on/after %s; scores not saved"
                      % args.score_from, flush=True)
        for top_names in args.top_names:
            entry, carry, seg = evaluate_targets(
                a, prep, fold, models, t_eval, args, device, seed_reports,
                top_names, train_s, valid_s, tr_idx, va_t)
            if ic is not None:
                entry["score_ic_test"] = ic
            report["folds"].append(entry)
            d, m = entry["target_diagnostics"], entry["test"]
            print("[fold %d k=%s] %s  cagr %+.3f  vol %.3f  sharpe %+.3f  maxdd %.3f"
                  % (fidx, top_names or "all", entry["contamination"], m["cagr"],
                     m["vol"], m["sharpe"], m["max_drawdown"]), flush=True)
            print("            names %.0f  deployed %.3f  cash %.3f  turnover %.0f%%"
                  "  cost %.0f bp"
                  % (d["mean_names_above_1e4"], d["mean_deployed"], d["mean_cash"],
                     m["annual_two_way_turnover"] * 100,
                     m["total_cost_bp_of_capital"]), flush=True)
            tag = "%s_fold%d_k%s" % (args.tag, fidx, top_names or "all")
            np.save(ART / (tag + "_carry.npy"), carry.astype(np.float32))
            seg["daily"].to_csv(ART / (tag + "_daily.csv"))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(_jsonable(report), indent=2), encoding="utf-8")
    print("[run] wrote " + str(out_path), flush=True)
    return report


if __name__ == "__main__":
    main()
