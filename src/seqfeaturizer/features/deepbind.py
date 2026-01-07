from __future__ import annotations

import io
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from ..io import SequenceRecord
from ..utils import ensure_dir, which


def featurize_deepbind(
    records: Sequence[SequenceRecord],
    ids_file: str | Path,
    deepbind_cmd: str = "deepbind",
    batch_size: int = 5000,
    out_dir: Optional[str | Path] = None,
) -> pd.DataFrame:
    """Run the DeepBind CLI to generate DeepBind features.

    This is an *optional* feature set because DeepBind is not easily installable
    via conda on modern systems.

    Requirements
    ------------
    - A working `deepbind` executable (or wrapper script) on PATH.
    - An ids/model list file understood by that executable.

    Notes
    -----
    The legacy pipeline invoked DeepBind like:
        cat locs.fasta | deepbind <IDS_FILE> > output.tab

    We reproduce that behaviour by piping a temporary FASTA into the deepbind
    command.

    Parameters
    ----------
    records:
        Input sequences.
    ids_file:
        Path to the DeepBind ids/model list.
    deepbind_cmd:
        Executable name or path.
    batch_size:
        Number of sequences per DeepBind call.
    out_dir:
        If provided, intermediate batch files are written here.

    Returns
    -------
    pd.DataFrame
        Wide table with `sequence_name` column + DeepBind model columns.
    """
    ids_file = Path(ids_file)
    if not ids_file.exists():
        raise FileNotFoundError(ids_file)

    if which(deepbind_cmd) is None and not Path(deepbind_cmd).exists():
        raise RuntimeError(
            f"DeepBind executable '{deepbind_cmd}' not found. "
            "Install DeepBind separately or skip --features deepbind."
        )

    if out_dir is not None:
        work = ensure_dir(out_dir)
    else:
        work = Path(tempfile.mkdtemp(prefix="deepbind_"))

    dfs = []
    for i in range(0, len(records), batch_size):
        batch = records[i : i + batch_size]
        fasta_text = "".join(f">{r.name}\n{r.sequence}\n" for r in batch)

        proc = subprocess.run(
            [deepbind_cmd, str(ids_file)],
            input=fasta_text,
            text=True,
            capture_output=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"DeepBind failed (exit {proc.returncode}).\nSTDERR:\n{proc.stderr}\n"
            )

        # Parse stdout as TSV
        df = pd.read_csv(io.StringIO(proc.stdout), sep="\t")
        if df.shape[0] != len(batch):
            # Some deepbind builds may include extra header/summary rows.
            # We attempt a fallback: if it has a name-like column, align by that.
            pass

        # Standardize ID
        if "sequence_name" in df.columns:
            df["sequence_name"] = df["sequence_name"].astype(str)
        elif "name" in df.columns:
            df = df.rename(columns={"name": "sequence_name"})
        else:
            # Assume output rows are in the same order as input
            df.insert(0, "sequence_name", [r.name for r in batch])

        dfs.append(df)

    out = pd.concat(dfs, ignore_index=True)
    return out


def deepbind_top_quantile(deepbind_df: pd.DataFrame, q: float = 0.9) -> pd.DataFrame:
    """Derived feature: number of DeepBind columns above a global quantile threshold.

    Matches the legacy feature_summary.R behaviour:
      cutoff = quantile(deepbind_mat, probs=q)
      n.deepbind_top = rowSums(deepbind_mat >= cutoff)
    """
    if deepbind_df is None or deepbind_df.empty:
        return pd.DataFrame(columns=["sequence_name", "n.deepbind_top"])

    df = deepbind_df.copy()
    if "sequence_name" not in df.columns:
        raise ValueError("deepbind_df must contain sequence_name")

    mat = df.drop(columns=["sequence_name"]).apply(pd.to_numeric, errors="coerce")
    mat = mat.fillna(0.0)

    # Global (flattened) quantile across *all* DeepBind scores
    values = mat.to_numpy(dtype=float).ravel()
    cutoff = float(np.quantile(values, q)) if values.size else 0.0

    n_top = (mat >= cutoff).sum(axis=1).astype(int)

    return pd.DataFrame(
        {
            "sequence_name": df["sequence_name"].astype(str).tolist(),
            "n.deepbind_top": n_top.tolist(),
        }
    )
