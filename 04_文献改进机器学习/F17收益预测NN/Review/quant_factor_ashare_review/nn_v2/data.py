# Research project author: Gilmour Jia
"""Causal NN data contracts, factor preprocessing and versioned caches."""
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
import pandas as pd
from spec import FACTORS, FEATURES, SPEC, digest


def validate_keys(frame):
    if frame[["date", "symbol"]].isna().any().any():
        raise ValueError("Null date/security key")
    if frame.duplicated(["date", "symbol"]).any():
        raise ValueError("Duplicate date/security key; aggregation prohibited")


def canonical_union(sina, raw):
    """Retain raw-only securities, explicit source flags and source units.

    This table is an audit/repair intermediate, not accepted data. Raw-only rows
    still lack share capital/adjustment/status and are never silently dropped.
    """
    sina, raw = sina.copy(), raw.copy()
    if "ts_code" in raw:
        raw["symbol"] = raw.ts_code.str[7:].str.lower() + raw.ts_code.str[:6]
        raw["date"] = pd.to_datetime(raw.trade_date, format="%Y%m%d")
        raw["volume"] = raw["vol"] * 100.0
        raw["amount"] = raw["amount"] * 1000.0
    for f in (sina, raw):
        f["date"] = pd.to_datetime(f.date)
        validate_keys(f)
    sina = sina.rename(columns={c: "adjusted_" + c for c in ("open", "high", "low", "close")})
    raw = raw.rename(columns={c: "raw_" + c for c in ("open", "high", "low", "close", "pre_close", "volume", "amount")})
    result = sina.merge(raw, on=["date", "symbol"], how="outer", validate="one_to_one",
                        indicator="coverage_source")
    result["raw_close_source"] = np.where(result.raw_close.notna(), "tushare_daily", "missing")
    # A VWAP stays a VWAP. It cannot become a close, open or limit-price reference.
    result["sina_vwap"] = result["amount"] / result["volume"].where(result["volume"] > 0)
    return result.sort_values(["date", "symbol"]).reset_index(drop=True)


def validate_market(frame):
    validate_keys(frame)
    for prefix in ("raw_", "adjusted_"):
        cols = [prefix + c for c in ("open", "high", "low", "close")]
        if not set(cols).issubset(frame):
            raise ValueError(f"Missing {prefix} OHLC")
        p = frame[cols].to_numpy(float)
        finite = np.isfinite(p).all(axis=1)
        if (p[finite] <= 0).any():
            raise ValueError(f"Nonpositive {prefix} price")
        o, h, l, c = p.T
        if ((h[finite] + 1e-6 < np.maximum(o[finite], c[finite])) |
            (l[finite] - 1e-6 > np.minimum(o[finite], c[finite])) |
            (l[finite] > h[finite])).any():
            raise ValueError(f"Invalid {prefix} OHLC")
    if (frame[["raw_volume", "raw_amount"]] < 0).any().any():
        raise ValueError("Negative raw volume/amount")
    traded = frame.raw_volume > 0
    vwap = frame.raw_amount / frame.raw_volume.where(traded)
    # A unit check, not a replacement price. Accommodates auction/data rounding.
    bad = traded & ((vwap < frame.raw_low * 0.95) | (vwap > frame.raw_high * 1.05))
    if bad.any():
        raise ValueError(f"Amount/volume units inconsistent on {int(bad.sum())} rows")


def decision_indices(calendar):
    dates = pd.DatetimeIndex(calendar)
    if dates.has_duplicates or not dates.is_monotonic_increasing:
        raise ValueError("Calendar must be ordered and unique")
    anchor = pd.Timestamp(SPEC["calendar"]["anchor"])
    if not len(dates) or dates[0] != anchor:
        raise ValueError("Supply full global calendar from frozen anchor")
    return np.arange(0, len(dates), 21, dtype=int)


@dataclass
class DateSample:
    formation_time: np.datetime64
    entry_time: np.datetime64
    label_end_time: np.datetime64
    symbols: np.ndarray
    x: np.ndarray
    y: np.ndarray
    label_reason: np.ndarray
    f17_score: np.ndarray

    @property
    def valid(self):
        return (self.label_reason == "ok") & np.isfinite(self.y)


def load_samples(path):
    """Read a samples.npz package without importing the training stack."""
    with np.load(path, allow_pickle=False) as z:
        if z["features"].tolist() != FEATURES:
            raise ValueError("Feature ordering mismatch")
        if (len(z["offsets"]) != len(z["formation_time"])+1 or z["offsets"][0] != 0 or
                z["offsets"][-1] != len(z["y"]) or (np.diff(z["offsets"]) <= 0).any()):
            raise ValueError("Invalid sample offsets")
        samples = []
        for i in range(len(z["formation_time"])):
            sl = slice(int(z["offsets"][i]), int(z["offsets"][i+1]))
            s = DateSample(z["formation_time"][i], z["entry_time"][i], z["label_end_time"][i],
                           z["symbols"][sl], z["x"][sl], z["y"][sl], z["reason"][sl], z["f17_score"][sl])
            if not np.isfinite(s.x).all() or len(np.unique(s.symbols)) != len(s.symbols):
                raise ValueError("Invalid sample features/security keys")
            if ((s.label_reason == "ok") & ~np.isfinite(s.y)).any():
                raise ValueError("A valid label cannot be NaN")
            samples.append(s)
    if len({s.formation_time for s in samples}) != len(samples):
        raise ValueError("Duplicate formation dates")
    return sorted(samples, key=lambda s: s.formation_time)


def forward_label(adjusted_open, calendar, t, status, adjustment_verified, horizon=21):
    """Endpoints exactly t+1 and t+22. No chained NaN-to-zero returns.

    Missing interior quotes are allowed only for verified suspension.
    Delistings require a separate verified terminal-payoff label, never a zero.
    """
    prices = np.asarray(adjusted_open, float)
    status, verified = np.asarray(status), np.asarray(adjustment_verified, bool)
    if status.shape != prices.shape or verified.shape != prices.shape:
        raise ValueError("Label verification/status axes disagree")
    e, z = t + 1, t + 1 + horizon
    n = prices.shape[1]
    reason = np.full(n, "ok", dtype="U40")
    y = np.full(n, np.nan)
    dates = np.asarray(calendar, dtype="datetime64[ns]")
    entry = dates[e] if e < len(dates) else np.datetime64("NaT", "ns")
    end = dates[z] if z < len(dates) else np.datetime64("NaT", "ns")
    if z >= len(dates):
        reason[:] = "not_matured"
        return y, reason, entry, end
    p = prices[e:z+1]
    finite = np.isfinite(p) & (p > 0)
    unknown = (~finite & (status[e:z+1] != "suspended")).any(axis=0)
    reason[unknown] = "unexplained_price_gap"
    reason[~verified[e:z+1].all(axis=0)] = "unverified_adjustment"
    reason[~finite[0] | ~finite[-1]] = "missing_endpoint"
    reason[(status[e:z+1] == "delisted").any(axis=0)] = "delisting_unresolved"
    ok = reason == "ok"
    y[ok] = p[-1, ok] / p[0, ok] - 1
    return y, reason, entry, end


def allowed_sample(sample, start, formation_end, realized_cutoff, strict=False):
    f, e = pd.Timestamp(sample.formation_time), pd.Timestamp(sample.label_end_time)
    if pd.isna(e) or not (pd.Timestamp(start) <= f <= pd.Timestamp(formation_end)):
        return False
    return bool(e < pd.Timestamp(realized_cutoff) if strict else e <= pd.Timestamp(realized_cutoff))


def split_samples(samples, year):
    fold = next(f for f in SPEC["folds"] if f["year"] == year)
    outer = [s for s in samples if pd.Timestamp(s.formation_time).year == year]
    if not outer:
        raise ValueError(f"No prediction dates for {year}")
    first = min(s.formation_time for s in outer)
    train = [s for s in samples if allowed_sample(s, fold["train_start"],
              fold["train_end"], fold["train_end"])]
    valid = [s for s in samples if allowed_sample(s, fold["valid_start"],
              fold["valid_end"], first, strict=True)]
    final = [s for s in samples if allowed_sample(s, "2013-01-01",
              fold["valid_end"], first, strict=True)]
    for name, group in (("train", train), ("validation", valid), ("final", final)):
        if not group or any(not s.valid.any() for s in group):
            raise ValueError(f"{year} {name}: empty date/labels; inspect coverage")
    return train, valid, final, outer


def preprocess(raw, eligible, log_cap, industry):
    """One formation date only. Returns unfilled cleaned F17 + 34 NN inputs."""
    raw = np.asarray(raw, float)
    eligible = np.asarray(eligible, bool)
    cap = np.asarray(log_cap, float)
    ind = np.asarray(industry).astype(str)
    if raw.ndim != 2 or raw.shape[1] != 17:
        raise ValueError("Require all 17 named factors in frozen order")
    if eligible.sum() < 30:
        raise ValueError("Too few eligible stocks")
    clean = np.full_like(raw, np.nan)
    for j in range(17):
        m = eligible & np.isfinite(raw[:, j]) & np.isfinite(cap)
        if m.sum() < 30:
            raise ValueError(f"Factor {FACTORS[j]} has insufficient usable coverage")
        v, size, group = raw[m, j].copy(), cap[m].copy(), ind[m]
        median = np.median(v)
        mad = np.median(np.abs(v - median)) * 1.4826
        v = np.clip(v, median - 3 * mad, median + 3 * mad)
        # Match F17's documented small-industry fallback, in float64.
        vr, sr = v.copy(), size.copy()
        for key in np.unique(group):
            g = group == key
            vr[g] -= v[g].mean() if g.sum() >= 5 else v.mean()
            sr[g] -= size[g].mean() if g.sum() >= 5 else size.mean()
        denom = sr @ sr
        if denom > 1e-12:
            vr -= (vr @ sr / denom) * sr
        sd = vr.std(ddof=1)
        # Constant columns are defined zero signals, not missing observations.
        clean[m, j] = (vr - vr.mean()) / sd if sd > 1e-12 else 0.0
    missing = ~np.isfinite(clean)
    med = np.nanmedian(clean[eligible], axis=0)
    if not np.isfinite(med).all():
        raise ValueError("Whole factor missing; refuse training date")
    filled = np.where(missing, med, clean)
    x = np.concatenate([filled, missing.astype(float)], axis=1).astype(np.float32)
    x[~eligible] = np.nan
    count = np.isfinite(clean).sum(axis=1)
    score = np.divide(np.nansum(clean, axis=1), count,
                      out=np.full(len(raw), np.nan), where=count > 0)
    return clean, x, score


def cache_key(data_hash, code_version, parameters):
    return digest({"data": data_hash, "code": code_version, "parameters": parameters})


def save_cache(path, key, **arrays):
    if any(np.asarray(a).dtype.hasobject for a in arrays.values()):
        raise ValueError("Object arrays prohibited")
    np.savez_compressed(path, cache_key=np.asarray(key), **arrays)


def load_cache(path, expected_key):
    with np.load(path, allow_pickle=False) as z:
        if str(z["cache_key"]) != expected_key:
            raise ValueError("Cache provenance mismatch; rebuild")
        return {k: z[k].copy() for k in z.files if k != "cache_key"}
