from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import pandas as pd
from rapidfuzz.distance import Levenshtein


@dataclass
class DiffResult:
    delta: pd.DataFrame


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    sep = "\t" if path.suffix.lower() in {".tsv", ".tab", ".txt"} else ","
    df = pd.read_csv(path, sep=sep)
    return df


def compute_delta(
    a: pd.DataFrame,
    b: pd.DataFrame,
    id_col: str = "sequence_name",
    *,
    pairs: Optional[pd.DataFrame] = None,
    a_id_col: str = "sequence_name_a",
    b_id_col: str = "sequence_name_b",
    subtract: str = "b_minus_a",
    fillna: float = 0.0,
    strict_columns: bool = True,
) -> DiffResult:
    """Compute feature deltas between two feature tables.

    Modes
    -----
    1) If `pairs` is None:
        Rows are matched by `id_col` (must exist in both a and b).
    2) If `pairs` is provided:
        `pairs` defines row pairing via columns `a_id_col` and `b_id_col`.

    The delta is computed column-wise on the intersection of numeric feature columns.

    Parameters
    ----------
    a, b:
        Feature tables.
    id_col:
        Column holding the sequence ID (default: 'sequence_name').
    pairs:
        Optional pairing DataFrame.
    subtract:
        'b_minus_a' or 'a_minus_b'.
    fillna:
        Value used for missing features/values (default: 0).
    strict_columns:
        If True, require that the two tables have identical feature column sets.
        If False, use the intersection.

    Returns
    -------
    DiffResult
        delta DataFrame contains id columns + delta features.
    """
    if id_col not in a.columns or id_col not in b.columns:
        raise ValueError(f"Both tables must contain id column '{id_col}'.")

    # Identify numeric feature columns (exclude id)
    a_feats = [c for c in a.columns if c != id_col]
    b_feats = [c for c in b.columns if c != id_col]

    if strict_columns and set(a_feats) != set(b_feats):
        missing_in_b = sorted(set(a_feats) - set(b_feats))
        missing_in_a = sorted(set(b_feats) - set(a_feats))
        raise ValueError(
            "Feature columns differ between A and B. "
            f"Missing in B: {missing_in_b[:10]}{'...' if len(missing_in_b)>10 else ''}; "
            f"Missing in A: {missing_in_a[:10]}{'...' if len(missing_in_a)>10 else ''}. "
            "Set strict_columns=False to use only the intersection."
        )

    feat_cols = sorted(set(a_feats).intersection(b_feats))

    # Convert to numeric
    a_num = a[[id_col] + feat_cols].copy()
    b_num = b[[id_col] + feat_cols].copy()
    for c in feat_cols:
        a_num[c] = pd.to_numeric(a_num[c], errors="coerce").fillna(fillna)
        b_num[c] = pd.to_numeric(b_num[c], errors="coerce").fillna(fillna)

    a_num = a_num.set_index(id_col)
    b_num = b_num.set_index(id_col)

    if pairs is None:
        common_ids = a_num.index.intersection(b_num.index)
        if len(common_ids) == 0:
            raise ValueError("No overlapping sequence IDs between A and B.")
        a_aligned = a_num.loc[common_ids]
        b_aligned = b_num.loc[common_ids]
        out_ids = common_ids
    else:
        if a_id_col not in pairs.columns or b_id_col not in pairs.columns:
            raise ValueError(f"pairs must have columns '{a_id_col}' and '{b_id_col}'")
        # Align using the provided pairing
        a_aligned = a_num.reindex(pairs[a_id_col].astype(str).values)
        b_aligned = b_num.reindex(pairs[b_id_col].astype(str).values)
        out_ids = pairs.index

    if subtract == "b_minus_a":
        delta = b_aligned[feat_cols].to_numpy() - a_aligned[feat_cols].to_numpy()
    elif subtract == "a_minus_b":
        delta = a_aligned[feat_cols].to_numpy() - b_aligned[feat_cols].to_numpy()
    else:
        raise ValueError("subtract must be 'b_minus_a' or 'a_minus_b'")

    delta_df = pd.DataFrame(delta, columns=feat_cols, index=out_ids)

    if pairs is None:
        delta_df = delta_df.reset_index().rename(columns={"index": id_col})
    else:
        delta_df = delta_df.reset_index(drop=True)
        delta_df.insert(0, a_id_col, pairs[a_id_col].astype(str).values)
        delta_df.insert(1, b_id_col, pairs[b_id_col].astype(str).values)

    return DiffResult(delta=delta_df)


def parse_pairs_file(
    path: str | Path,
    *,
    sep: Optional[str] = None,
    col: Optional[str] = None,
    split: str = "|",
    a_id_col: str = "sequence_name_a",
    b_id_col: str = "sequence_name_b",
) -> pd.DataFrame:
    """Read a pairs mapping file.

    Supports either:
      - Two-column file with columns already named a_id_col and b_id_col
      - One-column file, where each entry is "A|B" (or a custom `split`)

    Parameters
    ----------
    path:
        CSV/TSV file.
    sep:
        Override delimiter.
    col:
        If one-column mode, specify which column contains the pair string.
    split:
        Separator used inside the pair string.
    """
    path = Path(path)
    if sep is None:
        sep = "\t" if path.suffix.lower() in {".tsv", ".tab", ".txt"} else ","
    df = pd.read_csv(path, sep=sep)

    if a_id_col in df.columns and b_id_col in df.columns:
        return df[[a_id_col, b_id_col]].copy()

    if col is None:
        if df.shape[1] == 1:
            col = df.columns[0]
        else:
            # common legacy name
            for cand in ["Names", "name", "pair", "pairs"]:
                if cand in df.columns:
                    col = cand
                    break
    if col is None or col not in df.columns:
        raise ValueError(
            f"Pairs file must contain columns '{a_id_col}' and '{b_id_col}', or a single column to split. "
            f"Found columns: {list(df.columns)}"
        )

    pairs = df[col].astype(str).str.split(split, n=1, expand=True)
    if pairs.shape[1] != 2:
        raise ValueError(f"Could not split pairs using '{split}'")

    out = pd.DataFrame({a_id_col: pairs[0], b_id_col: pairs[1]})
    return out


def levenshtein_similarity(a_seq: str, b_seq: str) -> float:
    """Levenshtein similarity in [0,1], matching RecordLinkage::levenshteinSim behaviour."""
    a_seq = a_seq or ""
    b_seq = b_seq or ""
    if len(a_seq) == 0 and len(b_seq) == 0:
        return 1.0
    dist = Levenshtein.distance(a_seq, b_seq)
    denom = max(len(a_seq), len(b_seq), 1)
    return 1.0 - (dist / denom)
