from __future__ import annotations

import itertools
from typing import Sequence

import pandas as pd

from ..io import SequenceRecord


def max_consecutive(seq: str, base: str, cap: int = 50) -> int:
    """Maximum length of a consecutive run of `base` in `seq`, capped."""
    runs = (sum(1 for _ in grp) for b, grp in itertools.groupby(seq) if b == base)
    m = max(runs, default=0)
    return min(cap, m)


def featurize_polyA_polyT_GC(records: Sequence[SequenceRecord], cap: int = 50) -> pd.DataFrame:
    """Compute simple sequence composition features.

    This matches the legacy pipeline's `polyA_polyT_GC` feature file.

    Output columns:
      - sequence_name
      - polyA: max run length of 'A' (capped)
      - polyT: max run length of 'T' (capped)
      - GC: count of 'G' or 'C'
    """
    rows = []
    for r in records:
        seq = r.sequence
        rows.append(
            {
                "sequence_name": r.name,
                "polyA": max_consecutive(seq, "A", cap=cap),
                "polyT": max_consecutive(seq, "T", cap=cap),
                "GC": sum(1 for b in seq if b in {"G", "C"}),
            }
        )
    return pd.DataFrame(rows)


def poly_small(poly_df: pd.DataFrame) -> pd.DataFrame:
    """Derived features used by the legacy summarizer: drop GC and rename runs."""
    out = poly_df.copy()
    if "GC" in out.columns:
        out = out.drop(columns=["GC"])
    rename = {}
    if "polyA" in out.columns:
        rename["polyA"] = "n.polyA"
    if "polyT" in out.columns:
        rename["polyT"] = "n.polyT"
    return out.rename(columns=rename)
