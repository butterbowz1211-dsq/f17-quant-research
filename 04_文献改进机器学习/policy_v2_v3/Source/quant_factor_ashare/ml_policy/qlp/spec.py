# Research project author: Gilmour Jia
"""Freeze the exact specification of an experiment before it is run.

The point is to make "we chose this after seeing the result" impossible to
claim accidentally.  A spec records the code hash, the configuration, and the
sample windows; the runner writes the same file next to the report.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def code_fingerprint(files=None) -> dict:
    """Hash every module that can change a result."""
    if files is None:
        files = ["run_policy.py", "qlp/dataset.py", "qlp/policy.py",
                 "qlp/execution.py", "qlp/relations.py", "qlp/features.py",
                 "qlp/weights.py", "qlp/splits.py", "qlp/data.py", "qlp/baselines.py"]
    out = {}
    for rel in files:
        p = HERE / rel
        if p.exists():
            out[rel] = file_digest(p)
    return out


def git_revision() -> str | None:
    try:
        res = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             cwd=HERE, capture_output=True, text=True, timeout=10)
        if res.returncode == 0:
            return res.stdout.strip()
    except Exception:
        pass
    return None


def build_spec(tag: str, config: dict, folds: list, panel: dict) -> dict:
    return dict(
        tag=tag,
        code=code_fingerprint(),
        git=git_revision(),
        config=config,
        folds=[dict(index=f.index, train=[str(f.train_start.date()), str(f.train_end.date())],
                    valid=[str(f.valid_start.date()), str(f.valid_end.date())],
                    test=[str(f.test_start.date()), str(f.test_end.date())],
                    contaminated=bool(f.contaminated)) for f in folds],
        panel=panel,
    )


def write_spec(path, spec: dict) -> None:
    Path(path).write_text(json.dumps(spec, indent=2, ensure_ascii=False), encoding="utf-8")
