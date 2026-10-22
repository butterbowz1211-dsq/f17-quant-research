# Research project author: Gilmour Jia
import numpy as np
import pandas as pd


def metrics(returns, rf_annual=0.):
    r = np.asarray(returns, float)
    if len(r) < 2 or not np.isfinite(r).all() or (r <= -1).any():
        raise ValueError("Invalid return series; do not drop missing dates")
    rf_daily = (1+rf_annual)**(1/252)-1
    excess = r-rf_daily
    equity = np.r_[1., np.cumprod(1+r)]
    sd = r.std(ddof=1)
    return {"CAGR": float(equity[-1]**(252/len(r))-1),
            "period_return": float(equity[-1]-1), "days": len(r),
            "Sharpe": float(excess.mean()/excess.std(ddof=1)*np.sqrt(252)) if sd > 0 else None,
            "volatility": float(sd*np.sqrt(252)),
            "max_drawdown": float((equity/np.maximum.accumulate(equity)-1).min()),
            "rf_annual": rf_annual}


def newey_west(y, x, lags=21):
    y, x = np.asarray(y,float), np.asarray(x,float)
    X = np.column_stack([np.ones(len(y)), x])
    if not np.isfinite(X).all() or not np.isfinite(y).all() or len(y) <= X.shape[1]:
        raise ValueError("Invalid HAC input")
    bread = np.linalg.inv(X.T@X)
    beta = bread@X.T@y
    u = X*(y-X@beta)[:,None]
    meat = u.T@u
    for lag in range(1,min(lags,len(y)-1)+1):
        cross = u[lag:].T@u[:-lag]
        meat += (1-lag/(lags+1))*(cross+cross.T)
    # Summation convention: NO additional division by n.
    covariance = bread@meat@bread
    return beta, covariance


def rank_ic(pred, label):
    a, b = np.asarray(pred,float), np.asarray(label,float)
    m = np.isfinite(a)&np.isfinite(b)
    if m.sum() < 3:
        return None
    ar, br = pd.Series(a[m]).rank(), pd.Series(b[m]).rank()
    if ar.std() == 0 or br.std() == 0:
        return None
    return float(ar.corr(br))


def paired_interval(nn_returns, baseline, reps=2000, block=21, seed=42):
    a, b = np.asarray(nn_returns,float), np.asarray(baseline,float)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Require aligned, complete paired daily returns")
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(reps):
        starts = rng.integers(0,len(a),size=int(np.ceil(len(a)/block)))
        idx = ((starts[:,None]+np.arange(block)) % len(a)).ravel()[:len(a)]
        ma, mb = metrics(a[idx]), metrics(b[idx])
        if ma["Sharpe"] is None or mb["Sharpe"] is None:
            raise ValueError("Undefined bootstrap Sharpe")
        values.append([ma["CAGR"]-mb["CAGR"], ma["Sharpe"]-mb["Sharpe"],
                       float((a[idx]-b[idx]).mean()*252)])
    return dict(zip(["CAGR_difference","Sharpe_difference","annualized_mean_return_difference"],
                    np.quantile(values,[.025,.975],axis=0).T.tolist()))


def gates(nn_metrics, f17, data_pass):
    checks = {"CAGR": nn_metrics["CAGR"] > f17["CAGR"],
              "Sharpe": nn_metrics["Sharpe"] is not None and f17["Sharpe"] is not None and nn_metrics["Sharpe"] >= f17["Sharpe"],
              "volatility": nn_metrics["volatility"] <= f17["volatility"]*1.10,
              "drawdown": abs(nn_metrics["max_drawdown"]) <= abs(f17["max_drawdown"])+.03,
              "data_execution": bool(data_pass)}
    return {"checks": checks, "passed_historical_gate": all(checks.values()),
            "independent_future_confirmation": False}
