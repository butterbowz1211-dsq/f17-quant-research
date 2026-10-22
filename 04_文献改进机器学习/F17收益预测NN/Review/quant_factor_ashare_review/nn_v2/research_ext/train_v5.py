# Research project author: Gilmour Jia
"""Train the 21-factor v5 experiment using the frozen v4 protocol implementation.

The top-level nn_v2 modules stay byte-identical. This wrapper patches only the
in-memory feature schema, network input width, and the two hard-coded 34/35
checks in model.py, then calls train.run unchanged.
"""
from pathlib import Path
import sys
import argparse
import json
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import runtime  # noqa: F401
import pyarrow  # noqa: E402
pyarrow._hotfix_installed = True

import spec  # noqa: E402
import model  # noqa: E402
import data  # noqa: E402
import train  # noqa: E402

V5_FEATURES = ["TURN","MAX5","INTRADAY","ILLIQ","COMMOVE","TURN_SHOCK","REV_1W","ISKEW",
               "IVOL","OVERNIGHT","LEAD_LAG","CO_SKEW","BETA","REV_LT","SIZE","HIGH52",
               "RES_MOM","EP","BP","SP","DIV"]
FEATURES42 = V5_FEATURES + [x + "_missing" for x in V5_FEATURES]


def load_v5(path):
    with np.load(path, allow_pickle=False) as z:
        # Materialize each array once: NpzFile.__getitem__ decompresses on every access.
        features = z["features"]
        x_all, y_all = z["x"], z["y"]
        sym_all, reason_all, f17_all = z["symbols"], z["reason"], z["f17_score"]
        ft, et, lt = z["formation_time"], z["entry_time"], z["label_end_time"]
        offsets = z["offsets"]
        if features.tolist() != V5_FEATURES:
            raise ValueError("v5 feature ordering mismatch")
        if len(x_all) and x_all.shape[1] != len(FEATURES42):
            raise ValueError("v5 input width mismatch")
        samples = []
        for i in range(len(ft)):
            sl = slice(int(offsets[i]), int(offsets[i+1]))
            s = data.DateSample(ft[i], et[i], lt[i], sym_all[sl], x_all[sl],
                                y_all[sl], reason_all[sl], f17_all[sl])
            if not np.isfinite(s.x).all() or len(np.unique(s.symbols)) != len(s.symbols):
                raise ValueError("Invalid v5 sample")
            samples.append(s)
    return sorted(samples, key=lambda s: s.formation_time)


@torch.no_grad()
def predict_v5(model_obj, x, scale, device="cpu"):
    import torch
    x = np.asarray(x, np.float32)
    expected = int(model_obj.config["input"])
    if x.shape[1] != expected or not np.isfinite(x).all():
        raise ValueError("Invalid v5 prediction inputs")
    model_obj.eval()
    return model_obj(torch.as_tensor(x, device=device)).cpu().numpy() * scale


def ridge_fit_v5(samples, alpha):
    scale = model.label_scale(samples)
    k = samples[0].x.shape[1] + 1
    xx = np.zeros((k, k))
    xy = np.zeros(k)
    for s in samples:
        m = s.valid
        x = np.column_stack([np.ones(m.sum()), s.x[m]]).astype(float)
        y = s.y[m] / scale
        w = 1/(len(samples)*m.sum())
        xx += x.T @ x * w
        xy += x.T @ y * w
    penalty = np.eye(k)*alpha
    penalty[0,0] = 0
    return np.linalg.solve(xx+penalty, xy), scale


def configure():
    spec.FEATURES = FEATURES42
    spec.SPEC["features"] = FEATURES42
    spec.SPEC["factors"] = V5_FEATURES
    spec.SPEC["network"] = dict(spec.SPEC["network"], input=len(FEATURES42))
    data.FEATURES = V5_FEATURES
    model.FEATURES = FEATURES42
    train.FEATURES = FEATURES42
    train.load_samples = load_v5
    train.predict = predict_v5
    train.ridge_fit = ridge_fit_v5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--package", default="derived/research_training_v5")
    ap.add_argument("--out", default="derived/research_model_v5")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--mode", default="research_with_limitations")
    ap.add_argument("--last-year", type=int, default=2025)
    args = ap.parse_args()

    configure()
    print(json.dumps({"v5_features": len(V5_FEATURES), "input": len(FEATURES42),
                      "package": args.package, "out": args.out}, ensure_ascii=False), flush=True)
    train.run(Path(args.package), Path(args.out), args.device, args.mode, str(args.last_year))


if __name__ == "__main__":
    main()
