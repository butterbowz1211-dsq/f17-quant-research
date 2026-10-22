# Research project author: Gilmour Jia
"""Point-in-time market data for the direct-weight policy research."""
from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = [
    "PROJECT_ROOT", "DATA_DIR", "ensure_deps_on_path",
    "load_panel", "price_matrices", "limit_threshold", "limit_locked",
    "trailing_amount", "build_universe", "build_prep",
]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

# The project's cached frames are pandas-3 / pyarrow pickles. The archive of
# pyarrow used when they were written lives beside the audit checkout; make it
# importable so the caches can be read without re-downloading anything.
_DEPS_CANDIDATES = [
    Path(r"workspace/review_workspace/quant_factor_ashare_review/.deps"),
    Path(r"workspace/review_workspace/residual_momentum_ashare/.deps"),
]

# The archived pyarrow wheel expects its VC runtime under delvewheel-mangled
# names (msvcp140-<hash>.dll).  Those files are absent from the archive, so the
# import fails with a bare "DLL load failed".  Stage copies of the system
# runtime under the expected names and put that directory on the Windows DLL
# search path.  Keeping the handles alive keeps the directory registered.
_DLL_HANDLES = []
_DLL_SHIM = Path(__file__).resolve().parents[1] / "tools" / "pyarrow_dlls"


def _add_dll_dir(path: Path) -> None:
    if os.name != "nt" or not path.is_dir():
        return
    try:
        _DLL_HANDLES.append(os.add_dll_directory(str(path)))
    except (AttributeError, OSError):
        pass


def _activate_pyarrow_for_pandas(pa) -> None:
    """pandas snapshots HAS_PYARROW at import time; refresh the gate."""
    try:
        import pyarrow.compute as pc
        import pandas.compat as compat
        compat.HAS_PYARROW = True
        import pandas.core.arrays.string_ as string_mod
        string_mod.HAS_PYARROW = True
        string_mod.pa = pa
        string_mod.pc = pc
        import pandas.core.dtypes.dtypes as dtypes_mod
        dtypes_mod.HAS_PYARROW = True
        dtypes_mod.pa = pa
        import pandas.core.arrays.arrow.array as arrow_mod
        from pandas.core.dtypes.dtypes import ArrowDtype
        from pandas.compat.pyarrow import _safe_fill_null
        arrow_mod.HAS_PYARROW = True
        arrow_mod.pa = pa
        arrow_mod.pc = pc
        arrow_mod.ArrowDtype = ArrowDtype
        arrow_mod._safe_fill_null = _safe_fill_null
        arrow_mod.pa_version_under16p0 = False
        arrow_mod.pa_version_under21p0 = False
        import pandas.core.arrays.string_arrow as string_arrow_mod
        string_arrow_mod.HAS_PYARROW = True
        string_arrow_mod.pa = pa
        string_arrow_mod.pc = pc
        string_arrow_mod.pa_version_under16p0 = False
    except Exception:
        pass


def ensure_deps_on_path() -> list:
    """Make a usable pyarrow importable; return the paths that were added."""
    added = []
    _add_dll_dir(_DLL_SHIM)
    try:
        import pyarrow
        _activate_pyarrow_for_pandas(pyarrow)
        return added
    except Exception:
        pass
    for cand in _DEPS_CANDIDATES:
        if cand.is_dir() and str(cand) not in sys.path:
            sys.path.append(str(cand))
            added.append(str(cand))
            _add_dll_dir(cand / "pyarrow")
            try:
                import pyarrow
                _activate_pyarrow_for_pandas(pyarrow)
                break
            except Exception:
                continue
    return added


@lru_cache(maxsize=1)
def load_panel() -> pd.DataFrame:
    """Load the daily panel once and normalise its keys."""
    ensure_deps_on_path()
    path = DATA_DIR / "panel.pkl"
    if not path.exists():
        raise FileNotFoundError(f"panel not found: {path}")
    df = pd.read_pickle(path)
    if "date" not in df.columns or "symbol" not in df.columns:
        raise ValueError("panel must carry symbol and date columns")
    df["date"] = pd.to_datetime(df["date"])
    df["symbol"] = df["symbol"].astype(str)
    return df


def _board_of(symbol: str) -> str:
    """Classify a symbol into the board whose price limit applies."""
    s = symbol.lower()
    if s.startswith("sh688") or s.startswith("sh689"):
        return "star"
    if s.startswith("sz300") or s.startswith("sz301"):
        return "chinext"
    if s.startswith("bj") or s.startswith("sz8") or s.startswith("sz4"):
        return "bse"
    return "main"


def price_matrices(panel: pd.DataFrame | None = None, *, value_cols=("open", "high", "low", "close", "raw_close", "amount", "volume", "outstanding_share")):
    """Pivot selected columns into date x symbol matrices on one shared axis."""
    df = load_panel() if panel is None else panel
    out = {}
    for col in value_cols:
        if col not in df.columns:
            continue
        out[col] = df.pivot_table(index="date", columns="symbol", values=col,
                                  aggfunc="last", observed=True).astype("float64")
    axes = next(iter(out.values())).index, next(iter(out.values())).columns
    for k, v in out.items():
        if not (v.index.equals(axes[0]) and v.columns.equals(axes[1])):
            raise ValueError(f"{k} is not on the shared axis")
    return out


def limit_threshold(symbols, dates, is_st: pd.DataFrame | None = None) -> pd.DataFrame:
    """Daily price-limit fraction per (date, symbol).

    Board rules change over time: ChiNext moved from 10% to 20% on 2020-08-24,
    STAR has been 20% since launch, and ST names carry a tighter 5% band. BSE
    names use 30%. The returned frame is the *fraction*, so a one-price bar at
    that distance from the previous close is a limit lock.
    """
    dates = pd.DatetimeIndex(dates)
    thr = pd.DataFrame(0.10, index=dates, columns=pd.Index(symbols))
    for col in thr.columns:
        board = _board_of(str(col))
        if board == "star":
            thr[col] = 0.20
        elif board == "chinext":
            thr[col] = np.where(dates >= pd.Timestamp("2020-08-24"), 0.20, 0.10)
        elif board == "bse":
            thr[col] = 0.30
    if is_st is not None:
        st = is_st.reindex(index=dates, columns=thr.columns).fillna(False).to_numpy(bool)
        thr = thr.mask(st, 0.05)
    return thr.astype("float64")


def limit_locked(open_, high, low, close, raw_close, threshold):
    """Bars where the price was locked at its limit and could not be filled.

    A one-price bar (open == high == low == close) whose move equals the
    regulatory band is treated as untradeable in both directions. This is a
    proxy: an exact answer needs the exchange-published limit price per day.
    """
    one_price = (open_ == high) & (high == low) & (low == close)
    prev = raw_close.shift(1)
    move = (raw_close - prev) / prev
    at_limit = move.abs() >= (threshold.reindex_like(raw_close) - 1e-4)
    locked = one_price & at_limit & prev.notna()
    return locked.fillna(False)


def trailing_amount(amount: pd.DataFrame, window: int = 60, min_periods: int | None = None) -> pd.DataFrame:
    """Trailing mean turnover in currency, to be lagged by the caller."""
    mp = min_periods if min_periods is not None else window
    return amount.rolling(window, min_periods=mp).mean()


def build_universe(amount_lag: pd.DataFrame, close: pd.DataFrame,
                   min_listing_days: int = 252, top_n: int = 1000,
                   st: pd.DataFrame | None = None, suspended: pd.DataFrame | None = None) -> pd.DataFrame:
    """Point-in-time universe: liquid enough, listed long enough, tradable."""
    listed = close.notna().cumsum() >= min_listing_days
    rank = amount_lag.rank(axis=1, ascending=False, method="first")
    liquid = rank <= top_n
    ok = liquid & listed & amount_lag.notna()
    if st is not None:
        ok = ok & ~st.reindex_like(ok).fillna(True)
    if suspended is not None:
        ok = ok & ~suspended.reindex_like(ok).fillna(True)
    return ok.fillna(False)

def _prep_cache_path() -> Path:
    return Path(__file__).resolve().parents[1] / "artifacts" / "prep_arrays.npz"


def build_prep(force: bool = False, *, min_listing_days: int = 252,
               top_n: int = 1000, adv_window: int = 60):
    """Build (and cache) the price axes plus the point-in-time rule masks.

    Pivoting the 13M-row panel takes about a minute, so the derived arrays are
    cached to a single npz. Everything cached here is pure data preparation:
    no labels and no model choices, so reusing it cannot leak information
    across a train/test boundary.
    """
    cache = _prep_cache_path()
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists() and not force:
        z = np.load(cache, allow_pickle=True)
        dates = pd.DatetimeIndex(z["dates"])
        symbols = pd.Index([str(s) for s in z["symbols"]])
        out = {}
        for k in z.files:
            if k in ("dates", "symbols"):
                continue
            v = z[k]
            out[k] = pd.DataFrame(v, index=dates, columns=symbols)
        # The axes ride along with the frames: the whole point of the cache is
        # that downstream code sees exactly the same object as a fresh build.
        out["dates"] = dates
        out["symbols"] = symbols
        return out

    panel = load_panel()
    mats = price_matrices(panel)
    close, raw = mats["close"], mats["raw_close"]
    st = (panel.pivot_table(index="date", columns="symbol", values="is_st",
                            aggfunc="last", observed=True)
            .reindex(index=close.index, columns=close.columns)
            .fillna(False).astype(bool))
    thr = limit_threshold(close.columns, close.index, st)
    locked = limit_locked(mats["open"], mats["high"], mats["low"], close, raw, thr)
    suspended = ~np.isfinite(mats["open"]) | ~np.isfinite(close)
    amt_lag = trailing_amount(mats["amount"], adv_window).shift(1)
    universe = build_universe(amt_lag, close, min_listing_days=min_listing_days,
                              top_n=top_n, st=st, suspended=suspended)
    out = dict(
        open=mats["open"], close=close, raw_close=raw, amount_lag=amt_lag,
        st=st, limit_locked=locked, suspended=suspended, universe=universe,
    )
    payload = {}
    for k, v in out.items():
        payload[k] = v.to_numpy() if isinstance(v, (pd.DataFrame, pd.Series)) else v
    payload["dates"] = close.index.to_numpy()
    payload["symbols"] = np.array([str(c) for c in close.columns], dtype=object)
    np.savez_compressed(cache, **payload)
    return out
