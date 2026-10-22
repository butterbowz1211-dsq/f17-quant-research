# Research project author: Gilmour Jia
"""Local, read-only dependency discovery. Never installs packages."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = Path("workspace/source_workspace/quant_factor_ashare")
_HANDLES = []


def bootstrap():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    deps = ROOT.parent / ".deps"
    if deps.is_dir() and str(deps) not in sys.path:
        sys.path.append(str(deps))
    if os.name == "nt":
        for p in (SOURCE / "ml_policy/tools/pyarrow_dlls", deps / "pyarrow"):
            if p.is_dir():
                _HANDLES.append(os.add_dll_directory(str(p)))
    # Import before pandas so pandas discovers Arrow without monkeypatching.
    import pyarrow  # noqa: F401


bootstrap()
