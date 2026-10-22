# Research project author: Gilmour Jia
"""Point-in-time dataset assembly for the direct-weight policy.

Design-matrix layout, written down once so nothing has to be inferred:

    own  ranked features        n_own
    own  missingness flags      n_own
    peer ranked, contemporary   n_own
    peer missing, contemporary  n_own
    peer ranked, lagged         n_own
    peer missing, lagged        n_own
    peer validity flag          1
    market state                n_market
    account state               3

so the encoder width is ``6*n_own + 1 + n_market + 3``.

Rules that decide whether a number is allowed to exist:

* features at date t use closes up to and including t;
* the neighbour graph at t is estimated on market-residual returns strictly
  before t, so no same-day return leaks into its own edges;
* the label of a decision taken at t starts at the *open of t + 1*, because
  that is when the engine can actually trade, and runs to the open of the next
  decision date.  The first holding period is never harvested from the
  formation-day close;
* the peer block is supplied twice.  The lagged copy is finished before the
  label window opens, so it cannot be circular by construction; the
  contemporaneous copy is the one with the strongest signal but the weakest
  claim.  Both are kept so the choice is pre-registered, not tuned;
* nothing is standardised or imputed with statistics from outside the training
  window.  Ranks are per date; the market block is fitted on the training
  slice and then frozen.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import gc

import numpy as np
import pandas as pd

from .features import build_market_state, build_stock_features
from .relations import (correlation_from_residuals, market_residual_returns,
                        topk_from_corr)

__all__ = ["DatasetConfig", "Sample", "Assembled", "assemble", "forward_returns",
           "daily_log_returns", "standardize_state", "make_sample"]


@dataclass(frozen=True)
class DatasetConfig:
    rebalance: int = 21          # trading days between decisions
    universe_top: int = 1000     # bookkeeping / reporting only
    lookback: int = 120          # neighbour-correlation estimation window
    min_obs: int = 80
    neighbours: int = 10
    warmup: int = 260            # first decision date index
    peer_lag: int = 20           # lag of the uncontroversial peer copy, days
    add_missing_flags: bool = True
    with_prev_weight: bool = False  # append the policy's own previous weight

    def key(self) -> str:
        return ("rb%d_lb%d_k%d_w%d_lag%d_mf%d_pw%d"
                % (self.rebalance, self.lookback, self.neighbours, self.warmup,
                   self.peer_lag, int(self.add_missing_flags),
                   int(self.with_prev_weight)))


@dataclass
class Sample:
    """One decision date, ready for the encoders."""
    slot: int                    # index of this decision within its window
    t: int                       # date-axis index of the formation date
    date: np.datetime64
    feat: np.ndarray             # (n, n_input) float32, already standardised
    returns: np.ndarray          # (n,) realised forward return, same order
    eligible: np.ndarray         # (n,) bool, tradable-and-in-universe on t
    ids: np.ndarray              # (n,) int32 positions on the stock axis

    def __len__(self) -> int:
        return int(self.feat.shape[0])


@dataclass
class Assembled:
    dates: np.ndarray
    symbols: np.ndarray
    open_: np.ndarray
    universe: np.ndarray
    tradable: np.ndarray
    own: np.ndarray              # (D, S, 2*n_own): ranked then missing flags
    peer: np.ndarray             # (D, S, 4*n_own): now/miss/lag/miss
    peer_valid: np.ndarray       # (D, S) bool
    mkt: np.ndarray              # (D, n_market)
    mkt_names: list
    own_names: list
    rb: int = 21
    warmup: int = 260
    state_scale: tuple = field(default=(None, None))
    with_prev_weight: bool = False
    _fwd_cache: dict = field(default_factory=dict, repr=False)

    @property
    def n_names(self) -> int:
        return len(self.own_names)

    @property
    def n_input(self) -> int:
        return int(self.own.shape[2] + self.peer.shape[2] + 1
                   + self.mkt.shape[1] + 3 + int(self.with_prev_weight))

    @property
    def n_own_block(self) -> int:
        """Width of the own block: the ranked features plus their flags."""
        return int(self.own.shape[2])

    @property
    def n_peer_block(self) -> int:
        """Width of one peer copy (contemporaneous or lagged)."""
        return int(self.peer.shape[2]) // 2

    def forward(self, k: int, start: int, end: int) -> np.ndarray:
        key = (int(k), int(start), int(end))
        hit = self._fwd_cache.get(key)
        if hit is None:
            hit = forward_returns(self.open_, int(start), int(end), int(k))
            if len(self._fwd_cache) > 32:
                self._fwd_cache.clear()
            self._fwd_cache[key] = hit
        return hit

    def build_state(self, slot: int, cash_w: float, n_hold: float) -> np.ndarray:
        """(3,) account-state block; constant across stocks on a date."""
        return np.array([float(cash_w),
                         float(slot) / max(1, int(self.rb)),
                         np.log1p(max(0.0, float(n_hold)))], dtype=np.float64)

    def decision_index(self, start: int, end: int) -> np.ndarray:
        """Date indices of decision points in [start, end), strictly causal.

        A decision at date index t needs a forward window ending at t + 1 + rb.
        The last usable decision is capped so its label is realised, never
        projected.
        """
        lo = max(int(start), int(self.warmup))
        hi = min(int(end), len(self.dates) - int(self.rb) - 1)
        if hi <= lo:
            return np.empty(0, dtype=np.int64)
        return np.arange(lo, hi, int(self.rb), dtype=np.int64)


def daily_log_returns(close: np.ndarray, shift: int = 1) -> np.ndarray:
    """Log return of close from shift rows earlier, no look-ahead.

    Missing quotes contribute a zero growth factor rather than a NaN so that a
    single halted day cannot destroy a whole window; the resulting label is a
    slight underestimate and never an overestimate of what was attainable.
    """
    logc = np.log(np.where(close > 0, close, np.nan))
    zero = np.zeros((int(shift), logc.shape[1]), dtype=logc.dtype)
    if int(shift) == 1:
        diff = np.diff(logc, axis=0)
    else:
        diff = logc[int(shift):] - logc[:-int(shift)]
    return np.nan_to_num(np.vstack([zero, diff]), nan=0.0, posinf=0.0, neginf=0.0)


def forward_returns(open_: np.ndarray, start: int, end: int, k: int) -> np.ndarray:
    """Realised return from the open of start+1 to the open of start+1+k.

    start is the formation date, end the next formation date.  When
    start + 1 + k overshoots end the extra days are not counted, which is what
    a walk-forward run actually experiences.
    """
    r = daily_log_returns(open_, shift=1)
    horizon = min(int(end) - int(start), int(k))
    if horizon <= 0:
        return np.zeros(open_.shape[1], dtype=np.float64)
    lo = int(start) + 1
    hi = min(lo + horizon, r.shape[0])
    if hi <= lo:
        return np.zeros(open_.shape[1], dtype=np.float64)
    return np.expm1(r[lo:hi].sum(axis=0))


def standardize_state(mkt: np.ndarray, ref: slice) -> tuple:
    """Time-series standardisation of the market block, fitted on ref only.

    A column with no usable observation in the fit window falls back to
    ``(0, 1)`` rather than NaN: a NaN scale would silently poison every sample
    built afterwards, which is exactly the kind of failure that looks like a bad
    model instead of a bad window.
    """
    sub = np.asarray(mkt[ref], dtype=np.float64)
    with np.errstate(invalid="ignore"):
        mu = np.nanmean(sub, axis=0) if sub.size else np.zeros(mkt.shape[1])
        sd = np.nanstd(sub, axis=0) if sub.size else np.ones(mkt.shape[1])
    mu = np.nan_to_num(mu, nan=0.0)
    sd = np.nan_to_num(sd, nan=1.0)
    sd = np.where(np.abs(sd) > 1e-12, sd, 1.0)
    return mu.astype(np.float64), sd.astype(np.float64)


def _peer_mean(block_t: np.ndarray, idx: np.ndarray, wts: np.ndarray,
               valid: np.ndarray) -> np.ndarray:
    """Edge-weighted mean of one per-stock block over each stock's partners.

    A partner whose own reading is missing is dropped from that stock's edge
    weights and the remaining weights are renormalised, so a halted partner
    dilutes the average instead of pulling it towards a value it never had.
    Stocks with no usable graph at all get zeros; the caller also carries a
    validity flag so the encoder can tell that apart from a flat reading.
    """
    out = np.zeros_like(block_t, dtype=np.float32)
    if idx.size == 0 or not valid.any():
        return out
    gathered = np.asarray(block_t[idx], dtype=np.float32)      # (n, k, f)
    hav = np.isfinite(gathered)
    gathered = np.where(hav, gathered, 0.0)
    w = np.where(valid[:, None], wts, 0.0).astype(np.float32)
    w = np.where(hav, w[:, :, None], 0.0)
    tot = w.sum(axis=1)
    safe = np.where(tot > 0, tot, 1.0)
    out = (gathered * w).sum(axis=1) / safe
    out[~valid] = 0.0
    return out.astype(np.float32)


def build_peer_blocks(ranked: np.ndarray, missing: np.ndarray, resid: pd.DataFrame,
                      universe: np.ndarray, cfg: DatasetConfig,
                      *, verbose: bool = True):
    """Peer means of the ranked features and of their missingness flags.

    Returns ``(peer, valid)`` with ``peer`` shaped ``(D, S, 4*n_own)`` ordered
    contemporary-ranked, contemporary-missing, lagged-ranked, lagged-missing.
    Every decision date rebuilds the graph from a strictly backward window and
    peers are restricted to that date's point-in-time universe.
    """
    if ranked.shape[:2] != missing.shape[:2] or ranked.shape[2] != missing.shape[2]:
        raise ValueError("ranked and missing blocks must be congruent")
    n_dates, n_sym, n_own = ranked.shape
    peer = np.zeros((n_dates, n_sym, n_own * 4), dtype=np.float32)
    valid_out = np.zeros((n_dates, n_sym), dtype=bool)
    k = int(cfg.neighbours)
    lag = int(cfg.peer_lag)
    # Missing readings stay NaN all the way to the correlation: the window has to
    # be centred on its own rows, and only that window knows which rows count.
    # float32 is enough here -- the correlation itself is accumulated in float64
    # inside correlation_from_residuals -- and holding this block as float64
    # doubled the largest temporary in the whole assembly for no measurable gain.
    filled = np.ascontiguousarray(resid.to_numpy(dtype=np.float32))

    cur_idx = np.zeros((n_sym, k), dtype=np.int32)
    cur_w = np.zeros((n_sym, k), dtype=np.float64)
    cur_valid = np.zeros(n_sym, dtype=bool)
    step = max(1, int(cfg.rebalance))
    for t in range(0, n_dates, step):
        if t >= int(cfg.min_obs):
            corr, enough = correlation_from_residuals(filled[:t + 1], cfg.lookback,
                                                      cfg.min_obs)
            elig = np.asarray(universe[t], dtype=bool).copy()
            cur_idx, cur_w, cur_valid = topk_from_corr(corr, elig, enough,
                                                       enough, k)
        valid_out[t] = cur_valid
        t_lag = t - lag if t >= lag else 0
        peer[t] = np.concatenate(
            [_peer_mean(ranked[t], cur_idx, cur_w, cur_valid),
             _peer_mean(missing[t], cur_idx, cur_w, cur_valid),
             _peer_mean(ranked[t_lag], cur_idx, cur_w, cur_valid),
             _peer_mean(missing[t_lag], cur_idx, cur_w, cur_valid)], axis=1)
        if verbose and (t // step) % 60 == 0:
            print("[dataset]   graph %d/%d valid=%.3f" % (t, n_dates, cur_valid.mean()),
                  flush=True)
    for t in range(n_dates):
        if t % step:
            peer[t] = peer[t - 1]
            valid_out[t] = valid_out[t - 1]
    return peer, valid_out


def assemble(prep: dict, cfg: DatasetConfig = DatasetConfig(), *,
             verbose: bool = True) -> Assembled:
    """Build every array the trainer needs, once, in point-in-time order."""
    cols = prep["symbols"]
    idx = prep["dates"]
    close = pd.DataFrame(prep["close"], index=idx, columns=cols)
    raw_close = pd.DataFrame(prep["raw_close"], index=idx, columns=cols)
    amount = pd.DataFrame(prep["amount_lag"], index=idx, columns=cols)
    universe = pd.DataFrame(prep["universe"], index=idx, columns=cols).fillna(False)
    suspended = pd.DataFrame(prep["suspended"], index=idx, columns=cols).fillna(False)
    tradable = universe & (~suspended)
    if prep.get("limit_locked") is not None:
        locked = pd.DataFrame(prep["limit_locked"], index=idx, columns=cols).fillna(False)
        tradable = tradable & (~locked)

    if verbose:
        print("[dataset] building stock features ...", flush=True)
    stock = build_stock_features(close, raw_close, amount, universe, rank=True)
    names = list(stock["features"].keys())
    ranked = np.stack([np.asarray(stock["features"][k], dtype=np.float32)
                       for k in names], axis=-1)
    missing = np.stack([np.asarray(stock["missing"][k + "_miss"], dtype=np.float32)
                        for k in names], axis=-1)
    del stock
    if ranked.ndim != 3 or missing.shape != ranked.shape:
        raise ValueError("stock blocks must be congruent (dates, symbols, features)")

    if verbose:
        print("[dataset] building market state ...", flush=True)
    mkt_df = build_market_state(close, universe, window=60)
    mkt = np.asarray(mkt_df, dtype=np.float32)
    if mkt.ndim != 2 or mkt.shape[0] != ranked.shape[0]:
        raise ValueError("market state must be (dates, features)")
    mkt_names = list(mkt_df.columns)
    del mkt_df

    if verbose:
        print("[dataset] residualising for the graph ...", flush=True)
    resid = market_residual_returns(close)
    del close, raw_close, amount, suspended
    peer, peer_valid = build_peer_blocks(ranked, missing, resid,
                                         universe.to_numpy(), cfg,
                                         verbose=verbose)
    del resid
    gc.collect()      # the graph leaves several hundred MB of garbage behind

    if cfg.add_missing_flags:
        own = np.concatenate([ranked, missing], axis=-1)
        del ranked
    else:
        own = ranked
    del missing

    out = Assembled(dates=np.asarray(idx), symbols=np.asarray(cols),
                    open_=np.asarray(prep["open"], dtype=np.float64),
                    universe=universe.to_numpy(), tradable=tradable.to_numpy(),
                    own=own.astype(np.float32), peer=peer, peer_valid=peer_valid,
                    mkt=mkt, mkt_names=mkt_names, own_names=names,
                    with_prev_weight=bool(cfg.with_prev_weight))
    out.rb = int(cfg.rebalance)
    out.warmup = int(cfg.warmup)
    out.state_scale = (np.zeros(int(mkt.shape[1])), np.ones(int(mkt.shape[1])))
    return out


def make_sample(a: Assembled, t: int, slot: int, end: int | None = None,
                cash_w: float = 0.0, n_hold: float = 0.0,
                prev_w: np.ndarray | None = None) -> Sample:
    """Design matrix for one decision date, in the layout documented above.

    With ``a.with_prev_weight`` the final column is the policy's own weight in
    each name at the previous decision.  It is zero unless the caller supplies
    ``prev_w``, which is what a chronological trainer or a sequential read-out
    does; an isolated decision has no book to look back on.
    """
    elig = a.tradable[t] & a.universe[t]
    ids = np.flatnonzero(elig)
    if ids.size == 0:
        raise ValueError("no tradable name on %s: the universe is empty there, so "
                         "there is no decision to make"
                         % pd.Timestamp(a.dates[t]).date())
    own = a.own[t][ids]
    pv = a.peer_valid[t][ids]
    peer = np.where(pv[:, None], a.peer[t][ids], 0.0)
    mu, sd = a.state_scale
    mkt = a.mkt[t]
    if mu is not None:
        if mu.shape[0] != a.mkt.shape[1]:
            raise ValueError("state_scale must cover the market columns")
        mkt = (mkt - mu) / sd
    state = np.concatenate([mkt, a.build_state(slot, cash_w, n_hold)])
    head = np.concatenate([own, peer, pv.astype(np.float32)[:, None]], axis=1)
    block = np.concatenate([head, np.repeat(state[None, :], ids.size, axis=0)],
                           axis=1)
    if a.with_prev_weight:
        prev = (np.zeros(ids.size, dtype=np.float64) if prev_w is None
                else np.asarray(prev_w, dtype=np.float64)[ids])
        block = np.concatenate([block, prev[:, None]], axis=1)
    block = block.astype(np.float32)
    if block.shape[1] != a.n_input:
        raise ValueError("sample width %d does not match declared n_input %d"
                         % (block.shape[1], a.n_input))
    stop = int(end) if end is not None else int(t) + int(a.rb)
    rets = a.forward(int(a.rb), int(t), stop)[ids].astype(np.float32)
    return Sample(slot=int(slot), t=int(t), date=np.datetime64(a.dates[t]),
                  feat=block, returns=rets, eligible=elig[ids].astype(bool),
                  ids=ids.astype(np.int32))
