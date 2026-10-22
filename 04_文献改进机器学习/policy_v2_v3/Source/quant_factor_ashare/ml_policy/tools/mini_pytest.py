# Research project author: Gilmour Jia
"""A minimal replacement for the two pytest features these tests use.

The research interpreter in this workspace has no pytest, and the tests are
plain functions, so nothing is gained by making them depend on a framework
that is not installed.  ``run_tests.py`` loads this as ``pytest``.
"""
from __future__ import annotations


class approx:
    """Numeric comparison with a relative and an absolute tolerance."""

    def __init__(self, expected, rel=1e-6, abs=1e-12):
        self.expected = expected
        self.rel = rel
        self.abs = abs

    def _close(self, other):
        import numpy as np
        e, o = np.asarray(self.expected, dtype=float), np.asarray(other, dtype=float)
        return bool(np.all(np.abs(e - o) <= self.abs + self.rel * np.abs(e)))

    def __eq__(self, other):
        return self._close(other)

    def __repr__(self):
        return "approx(%r, rel=%g, abs=%g)" % (self.expected, self.rel, self.abs)


class _Raises:
    def __init__(self, exc):
        self.exc = exc
        self.value = None

    def __enter__(self):
        return self

    def __exit__(self, et, ev, tb):
        if et is None:
            raise AssertionError("expected %s, nothing was raised" % self.exc)
        if not issubclass(et, self.exc):
            return False
        self.value = ev
        return True


def raises(exc):
    return _Raises(exc)


def skip(reason=""):
    raise _Skipped(reason)


class _Skipped(Exception):
    pass
