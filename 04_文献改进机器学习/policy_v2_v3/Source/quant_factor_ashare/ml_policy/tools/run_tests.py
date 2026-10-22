# Research project author: Gilmour Jia
"""Run the project's tests without pytest.

    python tools/run_tests.py            # every tests/test_*.py
    python tools/run_tests.py test_policy

Exits non-zero if anything fails, so it can gate a run.
"""
from __future__ import annotations

import importlib
import inspect
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import mini_pytest  # noqa: E402

sys.modules.setdefault("pytest", mini_pytest)


def collect(pattern: str):
    files = sorted((ROOT / "tests").glob("test_*.py"))
    if pattern:
        files = [f for f in files if pattern in f.stem]
    out = []
    for path in files:
        mod = importlib.import_module("tests.%s" % path.stem)
        for name, fn in vars(mod).items():
            if name.startswith("test_") and inspect.isfunction(fn):
                out.append((path.stem, name, fn))
    return out


def main(argv):
    pattern = argv[1] if len(argv) > 1 else ""
    cases = collect(pattern)
    if not cases:
        print("no tests matched %r" % pattern)
        return 1
    passed, failed, skipped = 0, [], 0
    for stem, name, fn in cases:
        t0 = time.time()
        try:
            fn()
            passed += 1
            print("  PASS  %-16s %-52s %5.1fs" % (stem, name, time.time() - t0))
        except mini_pytest._Skipped as exc:
            skipped += 1
            print("  SKIP  %-16s %-52s (%s)" % (stem, name, exc))
        except Exception:
            failed.append((stem, name))
            print("  FAIL  %-16s %s" % (stem, name))
            traceback.print_exc()
    print("\n%d passed, %d failed, %d skipped (of %d)"
          % (passed, len(failed), skipped, len(cases)))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))