"""seqfeaturizer

A small, Python-only toolbox for featurizing DNA sequences and computing
feature deltas between two sequence sets.

The project is a clean reimplementation of a legacy pipeline that produced
per-module feature CSVs (5mer, polyA/polyT/GC, FIMO summary, DNAshape, DeepBind)
and then merged them into a consolidated table.

This package focuses on:
- clarity (simple, well-documented code)
- portability (paths/config passed explicitly)
- reproducibility (stable column ordering)
"""

from importlib.metadata import version as _version

__all__ = ["__version__"]

try:
    __version__ = _version("seqfeaturizer")
except Exception:  # pragma: no cover
    __version__ = "0.0.0"
