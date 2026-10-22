# Research project author: Gilmour Jia
"""Research primitives; no file I/O or source-project imports.

Event-driven long-only ledger in adjusted-price units. Fractional units, opening
auction liquidity proxy and adjusted-price corporate actions are approximations:
this is a research backtest, not a broker execution simulator.
"""
from dataclasses import dataclass
from statistics import NormalDist
import math
import numpy as np
import pandas as pd


def rank_ic(factor, label, min_n=50):
    """Pairwise complete observations BEFORE ranks, mean and normalization."""
    f, y = factor.align(label, join="inner")
    valid = np.isfinite(f) & np.isfinite(y)
    a, b = f.where(valid).rank(axis=1), y.where(valid).rank(axis=1)
    a, b = a.sub(a.mean(axis=1), axis=0), b.sub(b.mean(axis=1), axis=0)
    den = ((a*a).sum(axis=1) * (b*b).sum(axis=1))**0.5
    return ((a*b).sum(axis=1)/den.where(den > 0)).where(valid.sum(axis=1) >= min_n)


def forward_open_return(open_prices, horizon=21):
    """At close t, buy open t+1, exit open t+1+h; NaNs stay NaN."""
    if horizon < 1:
        raise ValueError("horizon must be positive")
    return open_prices.shift(-(horizon+1)) / open_prices.shift(-1) - 1


def purged_training_mask(dates, cutoff, horizon=21):
    """No training label may end after cutoff; use actual trading-index positions."""
    dates = pd.DatetimeIndex(dates)
    ends = pd.Series(dates, index=dates).shift(-(horizon+1))
    return (dates <= pd.Timestamp(cutoff)) & (ends <= pd.Timestamp(cutoff)).to_numpy()


def hac_mean(values, lags=None, horizon=1):
    """Bartlett HAC, biased autocovariance denominator T (PSD kernel).

For overlapping daily labels bandwidth is at least h-1. A bandwidth sensitivity
check is still needed: h-1 is not a universal sufficient bandwidth.
"""
    x = np.asarray(values, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 10:
        return dict(n=n, mean=None, se=None, t=None, p_two_sided=None, lags=None)
    automatic = int(np.floor(4*(n/100)**(2/9)))
    lag = min(n-1, max(horizon-1, automatic if lags is None else int(lags)))
    e = x-x.mean()
    lrv = float(e@e/n)
    for k in range(1, lag+1):
        lrv += 2*(1-k/(lag+1))*float(e[k:]@e[:-k]/n)
    se = math.sqrt(max(0, lrv)/n)
    t = float(x.mean()/se) if se > 0 else None
    return dict(n=n, mean=float(x.mean()), se=se, t=t,
                p_two_sided=math.erfc(abs(t)/math.sqrt(2)) if t is not None else None,
                lags=lag)


def fdr(pvalues, arbitrary_dependence=True):
    """BY by default (arbitrary dependence); BH available for justified dependence."""
    p = pd.Series(pvalues, dtype=float)
    valid = p.dropna().sort_values()
    if ((valid < 0) | (valid > 1)).any():
        raise ValueError("invalid p-value")
    n = len(valid)
    if not n:
        return p
    factor = sum(1/i for i in range(1,n+1)) if arbitrary_dependence else 1
    q = valid.to_numpy()*n*factor/np.arange(1,n+1)
    q = np.minimum.accumulate(q[::-1])[::-1].clip(0,1)
    return pd.Series(q,index=valid.index).reindex(p.index)


def performance(returns, rf_annual=0.02):
    x = np.asarray(returns, float)
    if len(x) < 2 or not np.isfinite(x).all() or (x <= -1).any():
        raise ValueError("need at least two finite non-bankrupt returns")
    rf = (1+rf_annual)**(1/252)-1
    nav = np.r_[1., np.cumprod(1+x)]
    dd = nav/np.maximum.accumulate(nav)-1
    vol = x.std(ddof=1)*np.sqrt(252)
    cagr = nav[-1]**(252/len(x))-1
    return dict(days=len(x), CAGR=float(cagr), AnnVol=float(vol),
                Sharpe=float((x.mean()-rf)*252/vol) if vol else None,
                MaxDD=float(dd.min()),
                Calmar=float(cagr/-dd.min()) if dd.min()<0 else None)


def stationary_indices(n, block, rng):
    if n < 1 or block < 1:
        raise ValueError("n and block must be positive")
    out = np.empty(n, dtype=int)
    i = 0
    while i<n:
        start = rng.integers(n)
        length = min(int(rng.geometric(1/block)), n-i)
        out[i:i+length] = (start+np.arange(length)) % n
        i += length
    return out


def bootstrap_sharpe(returns, other=None, n_boot=999, block=21, seed=731,
                     rf_annual=0.02):
    """Paired dates for differences. Percentile CI, NOT posterior P(SR>0)."""
    x = np.asarray(returns,float)
    y = None if other is None else np.asarray(other,float)
    if not np.isfinite(x).all() or (y is not None and
            (len(y)!=len(x) or not np.isfinite(y).all())):
        raise ValueError("use same finite dates")
    rng = np.random.default_rng(seed)
    out=[]
    for _ in range(n_boot):
        ix=stationary_indices(len(x),block,rng)
        s=performance(x[ix],rf_annual)["Sharpe"]
        if y is not None:
            s-=performance(y[ix],rf_annual)["Sharpe"]
        out.append(s)
    return dict(ci95=np.quantile(out,[.025,.975]).tolist(),block=block,n_boot=n_boot)


def monte_carlo_p(actual, null_values):
    a=np.asarray(null_values,float)
    if not len(a) or not np.isfinite(a).all() or not np.isfinite(actual):
        raise ValueError("finite nonempty null required")
    return float((1+np.count_nonzero(a>=actual))/(len(a)+1))


def ols_hac(y, factors, lags=6):
    """OLS intercept with Bartlett sandwich covariance, finite-sample correction."""
    data=pd.concat([pd.Series(y,name='y'),factors],axis=1).dropna()
    Y=data.pop('y').to_numpy(float)
    X=np.column_stack([np.ones(len(data)),data.to_numpy(float)])
    n,k=X.shape
    if n<=k+10 or np.linalg.matrix_rank(X)<k:
        raise ValueError('insufficient observations or rank-deficient factors')
    coef=np.linalg.lstsq(X,Y,rcond=None)[0]
    resid=Y-X@coef; Z=X*resid[:,None]
    meat=Z.T@Z
    for j in range(1,min(lags,n-1)+1):
        cross=Z[j:].T@Z[:-j]
        meat+=(1-j/(lags+1))*(cross+cross.T)
    bread=np.linalg.inv(X.T@X)
    covariance=bread@meat@bread*n/(n-k)
    se=np.sqrt(np.maximum(np.diag(covariance),0))
    return dict(n=n,alpha_monthly=float(coef[0]),alpha_annual_arithmetic=float(coef[0]*12),
                t_alpha=float(coef[0]/se[0]),
                betas=dict(zip(data.columns,coef[1:].tolist())),
                r_squared=float(1-resid@resid/np.sum((Y-Y.mean())**2)))


def monthly_alpha(daily_returns, formation_factors, rf_annual=.02, long_only=True):
    """Source factors labeled formation month -> return month by PeriodIndex + 1.

Drop partial last calendar month. Convert trading-month-end dates to periods
BEFORE alignment. Long-only y is excess; zero-cost spread y is already excess.
Input factor MKT must already be excess and SMB/VMG/PMO zero-investment spreads.
"""
    r=pd.Series(daily_returns).sort_index()
    monthly=(1+r).groupby(r.index.to_period('M')).prod()-1
    if r.index[-1].normalize()<r.index[-1].to_period('M').to_timestamp('M'):
        monthly=monthly.iloc[:-1]
    fac=formation_factors.copy()
    fac.index=fac.index.to_period('M')+1
    if fac.index.duplicated().any(): raise ValueError('duplicate formation months')
    if long_only: monthly=monthly-((1+rf_annual)**(1/12)-1)
    return ols_hac(monthly,fac,6)


def family_max_mean_test(differentials, n_boot=999, block=21, seed=731):
    """Conservative centered max-mean block bootstrap for a fixed candidate family.

Synchronized time resampling preserves cross-strategy dependence. All searched
candidates on identical dates must be supplied. Does not reconstruct missing
historical trials, and is not the Hansen studentized SPA statistic.
"""
    d=np.asarray(differentials,float)
    if d.ndim!=2 or len(d)<30 or d.shape[1]<1 or not np.isfinite(d).all():
        raise ValueError("T x K aligned finite return differences required")
    means=d.mean(axis=0)
    observed=max(0.,float(means.max()))*np.sqrt(len(d))
    centered=d-means
    rng=np.random.default_rng(seed)
    null=[max(0.,float(centered[stationary_indices(len(d),block,rng)].mean(axis=0).max()))
          *np.sqrt(len(d)) for _ in range(n_boot)]
    return dict(p=monte_carlo_p(observed,null), statistic=observed,
                trials=d.shape[1], block=block, n_boot=n_boot)


def dsr_iid(returns, trial_returns, rf_annual=.02):
    """Diagnostic IID DSR only: require same dates/window, never default variance=0."""
    if not returns.index.equals(trial_returns.index) or trial_returns.isna().any().any():
        raise ValueError("candidate returns must cover the identical evaluation dates")
    if trial_returns.shape[1]<2:
        raise ValueError("need >=2 actual recorded trials")
    srs=np.array([performance(trial_returns[c],rf_annual)["Sharpe"] for c in trial_returns])
    ntrial=len(srs)
    norm=NormalDist()
    sr0=srs.std(ddof=1)*((1-np.euler_gamma)*norm.inv_cdf(1-1/ntrial)
                       +np.euler_gamma*norm.inv_cdf(1-1/(ntrial*np.e)))
    sr=performance(returns,rf_annual)["Sharpe"]/np.sqrt(252)
    x=pd.Series(returns)
    den=1-x.skew()*sr+(x.kurt()+2)*sr*sr/4
    if den<=0:
        raise ValueError("invalid moment denominator")
    prob=norm.cdf((sr-sr0/np.sqrt(252))*np.sqrt(len(x)-1)/np.sqrt(den))
    return dict(DSR_iid=prob,SR0_annual=sr0,trials=ntrial,
                limitation="IID diagnostic; correlated trials and research selection remain")


@dataclass(frozen=True)
class ExecutionConfig:
    n: int=50
    rebalance: int=21
    buffer: float=.6
    cap: float=.03
    weight_band: float=0.0
    commission_bp: float=2.5
    slippage_bp: float=10.
    participation: float=.01
    capital: float=10_000_000.
    stale_limit: int=60
    allow_stale: bool=False


def stamp_rate(date):
    """Historical schedule for the project's 2010+ sample."""
    return .001 if pd.Timestamp(date)<pd.Timestamp("2023-08-28") else .0005


def event_backtest(open_prices, close_prices, score, eligible, can_buy, can_sell,
                   adv_lag, cfg=ExecutionConfig(), phase=0, audit_holdings=False):
    """Only decision t on grid executes at open t+1; no off-grid rebalancing.

Blocked trades are cancelled, existing assets remain owned, cash is explicit.
ADV is strictly lagged and only a capacity proxy, not proof of auction fills.
Missing valuations carry last close, never stale open. Default rejects prolonged
stale holdings. Price units must have consistent adjustment across OHLC.
"""
    frames=[close_prices,score,eligible,can_buy,can_sell,adv_lag]
    for f in frames:
        if not f.index.equals(open_prices.index) or not f.columns.equals(open_prices.columns):
            raise ValueError("all input axes must be identical")
    if not (0 <= phase < cfg.rebalance and 0 < cfg.cap <= 1 and cfg.n>0
            and cfg.rebalance>=1 and cfg.buffer>=0 and cfg.weight_band>=0
            and cfg.capital>0 and 0 < cfg.participation<=1
            and cfg.commission_bp>=0 and cfg.slippage_bp>=0):
        raise ValueError("invalid execution config")
    op,cl=open_prices.to_numpy(),close_prices.to_numpy()
    sc,el=score.to_numpy(),eligible.to_numpy(dtype=bool)
    cb,cs=can_buy.to_numpy(dtype=bool),can_sell.to_numpy(dtype=bool)
    adv=adv_lag.to_numpy()
    dates=open_prices.index
    n,m=op.shape
    value=np.zeros(m,float)
    mark=np.full(m,np.nan)
    stale=np.zeros(m,int)
    cash=cfg.capital
    last_nav=cash
    comm=cfg.commission_bp/1e4
    slip=cfg.slippage_bp/1e4
    daily=[]; events=[]; histories=[]
    for t,date in enumerate(dates):
        valid_op=np.isfinite(op[t]) & (op[t]>0)
        valid_cl=np.isfinite(cl[t]) & (cl[t]>0)
        open_mark=np.where(valid_op,op[t],mark)
        ratio=np.divide(open_mark,mark,out=np.ones(m),where=np.isfinite(mark)&(mark>0))
        value*=ratio
        nav_open=cash+value.sum()
        cost=turn=0.; blocked=0
        if t>0 and (t-1-phase)>=0 and (t-1-phase)%cfg.rebalance==0:
            s=sc[t-1]
            good=el[t-1]&np.isfinite(s)
            order=np.flatnonzero(good)
            order=order[np.argsort(-s[order],kind="stable")]
            band=set(order[:int(np.ceil(cfg.n*(1+cfg.buffer)))])
            held=set(np.flatnonzero(value>nav_open*1e-10))
            kept=[j for j in order if j in held and j in band][:cfg.n]
            selected=kept+[j for j in order if j not in kept][:max(0,cfg.n-len(kept))]
            target=np.zeros(m)
            # Cash remains when n*cap<1; never normalize past the cap.
            if selected:
                reserve=1-2*(comm+slip+stamp_rate(date))
                target[selected]=min(cfg.cap,1/len(selected))*nav_open*reserve
                if cfg.weight_band:
                    retain=np.array(kept,dtype=int)
                    near=(np.abs(value[retain]-target[retain])<=cfg.weight_band*target[retain])
                    target[retain[near]]=np.minimum(value[retain[near]],cfg.cap*nav_open*reserve)
            delta=target-value
            capacity=np.where(np.isfinite(adv[t])&(adv[t]>0),adv[t]*cfg.participation,0.)
            sellable=cs[t]&valid_op
            buyable=cb[t]&valid_op
            blocked=int(np.count_nonzero((delta<-.01)&~sellable)+np.count_nonzero((delta>.01)&~buyable))
            sell=np.minimum(np.maximum(-delta,0),capacity)*sellable
            sell_fee=sell.sum()*(comm+slip+stamp_rate(date))
            cash+=sell.sum()-sell_fee
            value-=sell
            buy=np.minimum(np.maximum(delta,0),capacity)*buyable
            expense=buy.sum()*(1+comm+slip)
            if expense>cash and expense>0:
                buy*=max(0,cash)/expense
            buy_fee=buy.sum()*(comm+slip)
            cash-=buy.sum()+buy_fee
            value+=buy
            cost=sell_fee+buy_fee
            turn=sell.sum()+buy.sum()
            events.append(dict(date=str(date.date()),decision=str(dates[t-1].date()),
                               selected=len(selected),buy=float(buy.sum()),sell=float(sell.sum()),
                               blocked=blocked,cost=float(cost)))
        if cash < -1e-6 or (value < -1e-6).any():
            raise AssertionError("negative cash or long-only holdings")
        end_mark=np.where(valid_cl,cl[t],open_mark)
        ratio=np.divide(end_mark,open_mark,out=np.ones(m),where=np.isfinite(open_mark)&(open_mark>0))
        value*=ratio
        mark=end_mark
        stale=np.where(valid_cl,0,stale+1)
        stale_value=float(value[stale>cfg.stale_limit].sum())
        if stale_value>1e-4 and not cfg.allow_stale:
            raise ValueError(f"{date}: held asset lacks valuation >{cfg.stale_limit} days; need delisting/valuation data")
        nav=cash+value.sum()
        if nav<=0 or not np.isfinite(nav):
            raise ValueError("invalid NAV")
        daily.append((nav/last_nav-1,nav,cash/nav,cost/last_nav,turn/nav_open,blocked,stale_value/nav))
        if audit_holdings:
            histories.append(value.copy()/nav)
        last_nav=nav
    result=pd.DataFrame(daily,index=dates,columns=["net","nav","cash_weight","cost","turnover","blocked","stale_weight"])
    return dict(daily=result,events=pd.DataFrame(events),
                holdings=pd.DataFrame(histories,index=dates,columns=open_prices.columns) if audit_holdings else None)
