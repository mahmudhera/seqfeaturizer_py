from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd


DROP_FIXED = {"sequence_name", "sequence", "name", "name.1", "index", "X", "V1"}
DROP_PREFIXES = ("name_meta",)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def which(cmd: str) -> Optional[str]:
    """Return full path of `cmd` if found on PATH, else None."""
    return shutil.which(cmd)


def run_cmd(
    cmd: List[str],
    cwd: Optional[str | Path] = None,
    check: bool = True,
    capture: bool = False,
) -> subprocess.CompletedProcess:
    """Run a subprocess with a nice error message."""
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd is not None else None,
        check=check,
        text=True,
        capture_output=capture,
    )


def standardize_sequence_name(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure the ID column is named `sequence_name`.

    Mirrors the intent of the legacy R code:
    - If sequence_name exists: do nothing
    - If one or more columns named 'name' exist: first becomes sequence_name, rest become name_meta1...
    - Else: first column becomes sequence_name
    """
    df = df.copy()

    if "sequence_name" in df.columns:
        return df

    name_cols = [c for c in df.columns if c == "name"]
    if name_cols:
        # Rename first 'name' column to 'sequence_name'
        first_idx = list(df.columns).index("name")
        cols = list(df.columns)
        cols[first_idx] = "sequence_name"
        # If more 'name' columns exist (rare in pandas, but possible after merges), rename them
        # Note: pandas does not allow duplicate column names in normal operations, so this is mostly
        # for parity with the legacy pipeline.
        df.columns = cols
        return df

    # Heuristic: if first column looks like an unnamed index column
    first = str(df.columns[0])
    if first in {"", "X", "V1"} or first.startswith("X."):
        cols = list(df.columns)
        cols[0] = "sequence_name"
        df.columns = cols
        return df

    # Fall back: rename first column
    cols = list(df.columns)
    cols[0] = "sequence_name"
    df.columns = cols
    return df


def drop_nonfeature_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Drop known metadata columns so they never enter summary/diff."""
    df = standardize_sequence_name(df)
    keep = []
    for c in df.columns:
        if c == "sequence_name":
            keep.append(c)
            continue
        if c in DROP_FIXED:
            continue
        if any(c.startswith(p) for p in DROP_PREFIXES):
            continue
        keep.append(c)
    return df.loc[:, keep]


def fill_numeric_na0(df: pd.DataFrame) -> pd.DataFrame:
    """Replace NA in numeric columns with 0."""
    out = df.copy()
    for c in out.columns:
        if c == "sequence_name":
            continue
        if pd.api.types.is_numeric_dtype(out[c]):
            out[c] = out[c].fillna(0)
    return out


def safe_full_join(dfs: Iterable[pd.DataFrame]) -> pd.DataFrame:
    """Full outer join on sequence_name across many DataFrames."""
    dfs = [standardize_sequence_name(df) for df in dfs if df is not None]
    if not dfs:
        raise ValueError("No dataframes to join")

    out = dfs[0]
    for df in dfs[1:]:
        out = out.merge(df, on="sequence_name", how="outer")
    return out


def to_numeric_matrix(df: pd.DataFrame, id_col: str = "sequence_name") -> pd.DataFrame:
    """Coerce all non-ID columns to numeric, replacing NA with 0."""
    df = standardize_sequence_name(df)
    df = drop_nonfeature_cols(df)

    out = df.set_index(id_col)
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.fillna(0.0)
    return out
