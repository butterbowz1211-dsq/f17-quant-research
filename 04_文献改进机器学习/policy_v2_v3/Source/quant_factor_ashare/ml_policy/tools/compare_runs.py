# Research project author: Gilmour Jia
"""Collect every finished run into one comparison table.

One row per (run, fold), read straight from the artefacts the runs wrote, so
the table cannot drift from the ledgers.  Contaminated folds are marked in
the output and must never be quoted as out-of-sample.

Usage:  python tools/compare_runs.py
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ART = HERE.parent / "artifacts"

RUNS = [
    ("p1 (T=annealed, full adj)", "p1_report.json"),
    ("t005 (T=0.05, partial+band)", "t005_report.json"),
    ("equal weight", "beq_report.json"),
    ("F17 top-50", "bf17_report.json"),
]

COLUMNS = [("cagr", "%+.3f"), ("vol", "%.3f"), ("sharpe", "%+.3f"),
           ("max_drawdown", "%.3f"), ("annual_two_way_turnover", "%.0f%%"),
           ("total_cost_bp_of_capital", "%.0f"), ("mean_cash_weight", "%.3f"),
           ("mean_holdings", "%.0f")]


def rows():
    out = []
    for name, fname in RUNS:
        path = ART / fname
        if not path.exists():
            continue
        rep = json.loads(path.read_text(encoding="utf-8"))
        for fold in rep.get("folds", []):
            m = fold.get("test") or {}
            if not m:
                continue
            out.append(dict(run=name, fold=fold.get("index"),
                            window=(fold.get("test_window") or
                                    [rep.get("config", {}).get("tag", "?"), ""])[0],
                            contamination=fold.get("contamination", "?"), **m))
    return out


def main():
    data = rows()
    if not data:
        print("no finished runs found in %s" % ART)
        return
    head = ["run", "fold", "start", "status"] + [c for c, _ in COLUMNS]
    widths = [max(len(str(h)), 14) for h in head]
    widths[0] = 30
    print("  ".join(str(h).ljust(w) for h, w in zip(head, widths)))
    print("-" * (sum(widths) + 2 * len(widths)))
    for r in data:
        cells = [r["run"], r["fold"], r["window"], r["contamination"]]
        cells += [fmt % r.get(key, float("nan")) if r.get(key) is not None else "-"
                  for key, fmt in COLUMNS]
        line = []
        for i, (c, w) in enumerate(zip(cells, widths)):
            text = str(c)
            if i >= 4 and isinstance(c, float):
                text = (COLUMNS[i - 4][1] % c)
            line.append(text.ljust(w))
        print("  ".join(line))
    print()
    print("Reminder: only folds marked untouched are out-of-sample.")


if __name__ == "__main__":
    main()
