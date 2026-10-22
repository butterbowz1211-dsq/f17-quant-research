# Research project author: Gilmour Jia
"""Two-year (2021+2022) beta decomposition by concatenating fold ledgers."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ART = ROOT / "artifacts"
sys.path.insert(0, str(HERE))

from beta_decompose import decompose, load_daily  # noqa: E402


def main():
    d0 = ART / "engine_daily_fold0"
    d1 = ART / "engine_daily_fold1"
    b = pd.concat([load_daily(d0 / "ew_all.csv"), load_daily(d1 / "ew_all.csv")])
    names = sorted(p.stem for p in d0.glob("*.csv") if p.stem != "ew_all")
    rows = []
    for n in names:
        if not (d1 / (n + ".csv")).exists():
            continue
        p = pd.concat([load_daily(d0 / (n + ".csv")), load_daily(d1 / (n + ".csv"))])
        rows.append(decompose(p, b, n))
    df = pd.DataFrame(rows).sort_values("alpha_ann", ascending=False)
    print("%-24s %7s %8s %8s %8s %8s %7s %7s %8s %8s"
          % ("rule", "beta", "b_down", "upCap", "dnCap", "alpha%", "a-t",
             "R2", "hCAGR", "hSharpe"))
    for _, r in df.iterrows():
        print("%-24s %7.3f %8.3f %8.2f %8.2f %+8.2f %7.2f %7.2f %+7.1f%% %8.2f"
              % (r["rule"], r["beta"], r["beta_dn"], r["up_capture"],
                 r["down_capture"], r["alpha_ann"] * 100, r["alpha_t"], r["r2"],
                 r["hedged_cagr"] * 100, r["hedged_sharpe"]))
    out = ART / "beta_decompose_2021_2022.json"
    out.write_text(json.dumps(dict(full=df.to_dict(orient="records")), indent=2),
                   encoding="utf-8")
    print("[beta] wrote %s" % out)
    return df


if __name__ == "__main__":
    main()
