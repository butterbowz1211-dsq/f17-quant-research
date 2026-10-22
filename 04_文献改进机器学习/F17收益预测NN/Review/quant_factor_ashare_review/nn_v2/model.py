# Research project author: Gilmour Jia
"""Fixed shallow return regression. Softmax is absent from the model/loss."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import copy
import random
import numpy as np
import torch
from torch import nn
from spec import FEATURES, SPEC


class ReturnNN(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = dict(config or SPEC["network"])
        c = self.config
        self.layers = nn.Sequential(nn.Linear(c["input"], c["hidden"][0]), nn.ReLU(),
                                    nn.Dropout(c["dropout"]),
                                    nn.Linear(c["hidden"][0], c["hidden"][1]), nn.ReLU(),
                                    nn.Dropout(c["dropout"]), nn.Linear(c["hidden"][1], 1))

    def forward(self, x):
        return self.layers(x).squeeze(-1)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(4)


def label_scale(samples):
    values, weights = [], []
    for s in samples:
        v = np.abs(s.y[s.valid])
        if not len(v):
            raise ValueError("Empty label date")
        values.append(v)
        weights.append(np.full(len(v), 1/len(v)))
    values, weights = np.concatenate(values), np.concatenate(weights)
    order = np.argsort(values, kind="stable")
    cdf = np.cumsum(weights[order])
    # Lower weighted median. Tolerance fixes floating summation at exactly 50%
    # so duplicating the securities of a date cannot change its scale.
    threshold = cdf[-1] * (0.5 - 1e-12)
    return max(float(values[order[np.searchsorted(cdf, threshold)]]), 1e-4)


def date_huber(model, sample, scale, device="cpu"):
    """A true mask; no NaN multiplied by zero."""
    m = sample.valid
    if not m.any():
        raise ValueError("No valid labels on formation date")
    x = torch.as_tensor(sample.x[m], dtype=torch.float32, device=device)
    y = torch.as_tensor(sample.y[m]/scale, dtype=torch.float32, device=device)
    if not torch.isfinite(x).all():
        raise ValueError("Nonfinite model features")
    return nn.functional.huber_loss(model(x), y, reduction="mean", delta=1.0)


@torch.no_grad()
def validation_loss(model, samples, scale, device="cpu"):
    model.eval()
    return float(np.mean([date_huber(model, s, scale, device).item() for s in samples]))


def fit(train, valid, lr, weight_decay, seed, epochs=None, device="cpu", smoke=False, progress=None):
    seed_all(seed)
    cfg = SPEC["training"]
    model = ReturnNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scale = label_scale(train)
    rng = np.random.default_rng(seed)
    best_loss, best_epoch, wait = float("inf"), 0, 0
    best_state, history = None, []
    maximum = epochs if epochs is not None else (3 if smoke else cfg["max_epochs"])
    for epoch in range(1, maximum+1):
        order = rng.permutation(len(train))
        model.train()
        total = 0.
        for b in range(0, len(order), cfg["batch_dates"]):
            group = order[b:b+cfg["batch_dates"]]
            optimizer.zero_grad(set_to_none=True)
            # Chunked gradients preserve date weights even with uneven stock counts.
            for i in group:
                s = train[i]
                idx = np.flatnonzero(s.valid)
                if not len(idx):
                    raise ValueError("No valid labels")
                for start in range(0, len(idx), cfg["stock_chunk"]):
                    take = idx[start:start+cfg["stock_chunk"]]
                    x = torch.as_tensor(s.x[take], dtype=torch.float32, device=device)
                    y = torch.as_tensor(s.y[take]/scale, dtype=torch.float32, device=device)
                    loss_sum = nn.functional.huber_loss(model(x), y, reduction="sum", delta=1.)
                    if not torch.isfinite(loss_sum):
                        raise ValueError("Nonfinite loss")
                    (loss_sum/(len(idx)*len(group))).backward()
                    total += loss_sum.item()/len(idx)
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise ValueError("Nonfinite gradient")
            optimizer.step()
        val = validation_loss(model, valid, scale, device) if valid is not None else None
        history.append({"epoch": epoch, "train_huber": total/len(train), "validation_huber": val})
        if progress is not None:
            progress(history[-1])
        if valid is None:
            best_epoch = epoch
        elif val < best_loss-cfg["min_delta"]:
            best_loss, best_epoch, wait = val, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            wait += 1
            if wait >= cfg["patience"]:
                break
    if valid is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, scale, best_epoch, (best_loss if valid is not None else None), history


def save_bundle(path, model, scale, seed, epoch, context):
    if not {"data_version", "code_version", "train_range", "spec_hash", "hyperparameters"} <= context.keys():
        raise ValueError("Incomplete model provenance")
    bundle = {
        "format_version": 1, "state_dict": {k: v.detach().cpu() for k,v in model.state_dict().items()},
        "network": model.config, "feature_schema": FEATURES,
        "scalers": {"feature": "identity_after_formation_cross_section_processing",
                    "label_scale": float(scale), "label_center": 0.0},
        "preprocessing": {**SPEC["preprocessing"], **context.get("data_contract", {}).get("preprocessing", {})},
        "label_contract": {**SPEC["label"], **context.get("data_contract", {}).get("label", {})},
        "seed": seed, "best_epoch": epoch, "readout": SPEC["portfolio"],
        "context": context,
    }
    torch.save(bundle, path)


def load_bundle(path, device="cpu"):
    b = torch.load(path, map_location=device, weights_only=True)
    if b["feature_schema"] != FEATURES or b["network"] != SPEC["network"]:
        raise ValueError("Bundle schema/config mismatch")
    model = ReturnNN(b["network"]).to(device)
    model.load_state_dict(b["state_dict"], strict=True)
    model.eval()
    return model, b


@torch.no_grad()
def predict(model, x, scale, device="cpu"):
    x = np.asarray(x, np.float32)
    if x.shape[1] != 34 or not np.isfinite(x).all():
        raise ValueError("Invalid prediction inputs")
    model.eval()
    return model(torch.as_tensor(x, device=device)).cpu().numpy() * scale


def ridge_fit(samples, alpha):
    scale = label_scale(samples)
    xx = np.zeros((35,35))
    xy = np.zeros(35)
    for s in samples:
        m = s.valid
        x = np.column_stack([np.ones(m.sum()), s.x[m]]).astype(float)
        y = s.y[m] / scale
        w = 1/(len(samples)*m.sum())
        xx += x.T @ x * w
        xy += x.T @ y * w
    penalty = np.eye(35)*alpha
    penalty[0,0] = 0
    return np.linalg.solve(xx+penalty, xy), scale


def ridge_predict(beta, x, scale):
    return (beta[0]+np.asarray(x)@beta[1:])*scale


def ridge_validation(beta, scale, samples):
    losses = []
    for s in samples:
        residual = (ridge_predict(beta, s.x[s.valid], scale)-s.y[s.valid])/scale
        a = np.abs(residual)
        losses.append(np.where(a <= 1, residual**2/2, a-0.5).mean())
    return float(np.mean(losses))
