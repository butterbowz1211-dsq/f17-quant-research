# Research project author: Gilmour Jia
"""PIT network signals and conservative group-score estimation.

These are research adapters, not claims of exact US-paper replication or A-share
profitability. Publication timestamps must be genuine first-available timestamps.
Daily-only timestamps should be shifted by the caller to the next decision date.
"""
import numpy as np
import pandas as pd


def analyst_edges(forecasts, asof):
    """Ali-Hirshleifer Eq(1): shared analyst count, trailing 12 calendar months.

    Input: analyst_id (stable person ID), symbol, issued_at, available_at, horizon.
    Repeat forecasts count once per analyst/stock. Brokerage != analyst identity.
    """
    cols = ['analyst_id', 'symbol', 'issued_at', 'available_at', 'horizon']
    if not set(cols).issubset(forecasts):
        raise ValueError(f'required: {cols}')
    f = forecasts.copy()
    for c in ('issued_at', 'available_at'):
        f[c] = pd.to_datetime(f[c], errors='raise')
    if f[cols].isna().any().any() or (f.available_at < f.issued_at).any():
        raise ValueError('missing fields or impossible publication timestamps')
    t = pd.Timestamp(asof)
    f = f.loc[(f.available_at <= t) & (f.issued_at <= t)
              & (f.issued_at > t-pd.DateOffset(months=12))
              & f.horizon.isin(['FY1', 'FY2']), ['analyst_id', 'symbol']].drop_duplicates()
    pairs = f.merge(f, on='analyst_id', suffixes=('_target', '_peer'))
    pairs = pairs.loc[pairs.symbol_target != pairs.symbol_peer]
    e = pairs.groupby(['symbol_target', 'symbol_peer']).size().rename('weight').reset_index()
    return e.rename(columns={'symbol_target':'target', 'symbol_peer':'peer'})


def snapshot_edges(records, asof):
    """Atomic source-level snapshots, including deletion of old relationships.

    Required: target, peer, weight, available_at, snapshot_id. All rows of a
    target/snapshot share a publication timestamp. Zero-weight sentinel rows
    allow an empty latest snapshot; missing peer allowed only on these rows.
    Caller must construct complete snapshots, never just changed edges.
    """
    req = {'target','peer','weight','available_at','snapshot_id'}
    if not req.issubset(records):
        raise ValueError(f'missing {req-set(records)}')
    x = records.copy()
    x['available_at'] = pd.to_datetime(x.available_at, errors='raise')
    if x[['target','weight','available_at','snapshot_id']].isna().any().any():
        raise ValueError('missing snapshot metadata')
    if not np.isfinite(x.weight).all() or (x.weight < 0).any():
        raise ValueError('weights must be finite and nonnegative')
    if x.loc[x.weight > 0, 'peer'].isna().any():
        raise ValueError('positive weight requires peer')
    if (x.groupby(['target','snapshot_id']).available_at.nunique() != 1).any():
        raise ValueError('snapshot is not atomic')
    headers = x[['target','snapshot_id','available_at']].drop_duplicates()
    if headers.duplicated(['target','available_at']).any():
        raise ValueError('ambiguous simultaneous snapshots')
    headers = headers.loc[headers.available_at <= pd.Timestamp(asof)]
    latest = headers.sort_values('available_at').drop_duplicates('target', keep='last')
    x = x.merge(latest, on=['target','snapshot_id','available_at'])
    x = x.loc[(x.weight > 0) & (x.target != x.peer), ['target','peer','weight']]
    if x.duplicated(['target','peer']).any():
        raise ValueError('duplicate edge in snapshot')
    return x


def network_signal(edges, peer_returns, min_coverage=.8):
    """Weighted peer return with explicit observed-weight coverage.

    Returns signal, coverage, peers; no edges -> missing (never zero alpha).
    min_coverage is an engineering choice, not a parameter from the papers.
    Industry identifiers can be peers for an input-output or segment network.
    """
    if not 0 < min_coverage <= 1 or not peer_returns.index.is_unique:
        raise ValueError('invalid coverage or return index')
    x = edges.copy()
    if not {'target','peer','weight'}.issubset(x):
        raise ValueError('target/peer/weight required')
    if not np.isfinite(x.weight).all() or (x.weight <= 0).any():
        raise ValueError('edge weights must be positive finite')
    if (x.target == x.peer).any() or x.duplicated(['target','peer']).any():
        raise ValueError('self links or duplicate links')
    x['ret'] = x.peer.map(peer_returns).replace([np.inf,-np.inf],np.nan)
    x['observed_weight'] = x.weight.where(x.ret.notna(),0)
    x['contribution'] = x.weight*x.ret.fillna(0)
    g = x.groupby('target')
    total = g.weight.sum()
    observed = g.observed_weight.sum()
    coverage = observed/total
    signal = (g.contribution.sum()/observed.replace(0,np.nan)).where(coverage >= min_coverage)
    return pd.DataFrame({'signal':signal,'coverage':coverage,'peers':g.ret.count()})


def segment_edges(segments, standalone, market_cap, asof):
    """Complicated-firm adapter: segment sales x standalone value weights.

    segments: latest complete PIT segment table produced upstream, with target,
    industry, sales, total_sales, available_at. standalone: symbol, industry,
    available_at; must have passed single-industry and >80% sales coverage rule.
    market_cap: Series known at formation. This adapter validates publication
    timing and conglomerate coverage but cannot verify upstream completeness.
    """
    s, p = segments.copy(), standalone.copy()
    t = pd.Timestamp(asof)
    for frame in (s,p):
        a = pd.to_datetime(frame.available_at, errors='raise')
        if a.isna().any() or (a > t).any():
            raise ValueError('inputs must be latest PIT snapshot available by asof')
    if p.symbol.duplicated().any() or s.duplicated(['target','industry']).any():
        raise ValueError('aggregate segments by industry; standalone symbols must be unique')
    if not np.isfinite(s[['sales','total_sales']]).all().all() or (s.sales<0).any() or (s.total_sales<=0).any():
        raise ValueError('invalid sales')
    if (s.groupby('target').total_sales.nunique()!=1).any():
        raise ValueError('inconsistent total sales')
    coverage = s.groupby('target').sales.sum()/s.groupby('target').total_sales.first()
    counts = s.loc[s.sales>0].groupby('target').industry.nunique()
    valid = coverage.index[(coverage>.8)&(coverage<=1.05)&(counts.reindex(coverage.index)>1)]
    s=s.loc[s.target.isin(valid)&(s.sales>0)]
    s['segment_weight']=s.sales/s.groupby('target').sales.transform('sum')
    p['cap']=p.symbol.map(market_cap)
    if p.cap.isna().any() or not np.isfinite(p.cap).all() or (p.cap<=0).any():
        raise ValueError('missing/invalid formation market caps')
    p['peer_weight']=p.cap/p.groupby('industry').cap.transform('sum')
    # Require all segments to have a standalone portfolio; do not silently
    # renormalize away an entire missing business segment.
    uncovered=set(s.loc[~s.industry.isin(p.industry),'target'])
    x=s.loc[~s.target.isin(uncovered)].merge(p,on='industry',suffixes=('_segment','_peer'))
    x['weight']=x.segment_weight*x.peer_weight
    x=x.rename(columns={'symbol':'peer'})
    if (x.target==x.peer).any():
        raise ValueError('conglomerate cannot also be standalone')
    return x[['target','peer','weight']].groupby(['target','peer'],as_index=False).weight.sum()


def simplex(x):
    """Euclidean projection to nonnegative weights summing to one."""
    x=np.asarray(x,dtype=float)
    u=np.sort(x)[::-1]; cs=np.cumsum(u)-1
    k=np.nonzero(u-cs/np.arange(1,len(x)+1)>0)[0][-1]
    return np.maximum(x-cs[k]/(k+1),0)


def fit_group_weights(gram, cross, ridge=1.):
    """Constrained rank-prediction loss + L2 shrink toward equal group weights.

    Inputs are averages of equal-date cross-sectional moments. No estimated
    expected-return inversion; not the Kozak-Nagel-Santosh SDF estimator.
    Fixed ridge is specified before evaluating candidates, not grid-searched.
    """
    gram=np.asarray(gram,float); cross=np.asarray(cross,float)
    k=len(cross); prior=np.ones(k)/k
    if gram.shape!=(k,k) or ridge<=0 or not np.isfinite(gram).all() or not np.isfinite(cross).all():
        raise ValueError('invalid moments')
    if not np.allclose(gram,gram.T) or np.linalg.eigvalsh(gram).min() < -1e-8:
        raise ValueError('gram must be positive semidefinite')
    h=gram+ridge*np.eye(k); b=cross+ridge*prior
    step=1/np.linalg.eigvalsh(h).max(); w=prior.copy()
    for _ in range(10000):
        nxt=simplex(w-step*(h@w-b))
        if np.max(np.abs(nxt-w))<1e-11:
            return nxt
        w=nxt
    raise RuntimeError('group optimizer did not converge')


def fit_asof(moments, cutoff, years=5, min_periods=36, ridge=1.):
    """Each tuple: decision timestamp, label exit timestamp, Gram, cross, nstocks.

    Exits equal to the cutoff are excluded: formation cannot see that open yet.
    """
    cutoff=pd.Timestamp(cutoff)
    hist=[m for m in moments if m[0]>=cutoff-pd.DateOffset(years=years) and m[0]<cutoff and m[1]<cutoff]
    if len(hist)<min_periods:raise ValueError('insufficient purged training dates')
    w=fit_group_weights(np.mean([m[2] for m in hist],axis=0),np.mean([m[3] for m in hist],axis=0),ridge)
    return w,len(hist),max(m[1] for m in hist)
