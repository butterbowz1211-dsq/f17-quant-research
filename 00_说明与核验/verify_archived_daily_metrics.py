# Research project author: Gilmour Jia
"""Portable verifier for archived ML daily returns; Python standard library only.

Run: python verify_archived_daily_metrics.py
Uses saved returns, never imports or executes strategy code. This verifies the
73 archived daily series, not periodic NN diagnostics or F17 execution revisions.
Conventions: annualization 252, rf=0, sample standard deviation, NAV starts at 1.
"""
from pathlib import Path
import csv
import json
import math
import statistics

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

def rows(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def metrics(returns, annual):
    assert len(returns) > 1 and all(math.isfinite(x) for x in returns)
    nav = peak = 1.0
    mdd = 0.0
    for value in returns:
        nav *= 1 + value
        peak = max(peak, nav)
        mdd = min(mdd, nav / peak - 1)
    cagr = nav ** (annual / len(returns)) - 1
    vol = statistics.stdev(returns) * math.sqrt(annual)
    return {'n': len(returns), 'cagr': cagr, 'volatility': vol,
            'sharpe_standard': statistics.mean(returns) * annual / vol,
            'max_drawdown': mdd, 'calmar': cagr / abs(mdd) if mdd < 0 else None}

def main():
    mapped = {r['source']: ROOT / r['relative_path'] for r in rows(HERE / 'source_manifest.csv')}
    expected = rows(HERE / '本次核验' / 'ml_daily_metrics_recomputed.csv')
    results, errors = [], []
    for e in expected:
        try:
            path = mapped[e['source']]
            data = rows(path)
            col = next(k for k in ('net', 'return', 'gross_return', 'ret') if k in data[0])
            actual = metrics([float(r[col]) for r in data], float(e['annual_periods']))
            assert int(e['n']) == actual['n']
            deltas = {k: abs(actual[k] - float(e[k])) for k in actual if k != 'n' and actual[k] is not None}
            passed = max(deltas.values()) < 1e-10
            result = {'model': e['model'], 'relative_path': path.relative_to(ROOT).as_posix(),
                      'n': actual['n'], 'passed': passed, 'max_abs_difference': max(deltas.values())}
            results.append(result)
            if not passed:
                errors.append(result)
        except Exception as exc:
            errors.append({'model': e['model'], 'error': str(exc)})
    report = {'expected_series': len(expected), 'verified_series': len(results),
              'failed': len(errors), 'errors': errors, 'results': results}
    (HERE / 'portable_metric_check.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'results'}, ensure_ascii=False, indent=2))
    raise SystemExit(1 if errors else 0)

if __name__ == '__main__':
    main()
