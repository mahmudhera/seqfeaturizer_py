from __future__ import annotations

import math
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
from tqdm import tqdm

from ..io import SequenceRecord
from ..utils import ensure_dir, run_cmd, which


def write_fasta(records: Sequence[SequenceRecord], out_fasta: str | Path) -> Path:
    out_fasta = Path(out_fasta)
    with out_fasta.open("w") as f:
        for r in records:
            f.write(f">{r.name}\n{r.sequence}\n")
    return out_fasta


def _find_max_window(starts: List[int], window: int) -> int:
    """Max number of motif start positions within any [pos, pos+window] interval."""
    if not starts:
        return 0
    starts = sorted(starts)
    m = 0
    j = 0
    for i in range(len(starts)):
        while j < len(starts) and starts[j] <= starts[i] + window:
            j += 1
        # interval includes i..(j-1)
        m = max(m, j - i)
    return m


def run_fimo(
    records: Sequence[SequenceRecord],
    motif_file: str | Path,
    out_dir: str | Path,
    label: str,
    thresh: float = 1e-4,
    keep_hits: bool = True,
) -> pd.DataFrame:
    """Run MEME-suite FIMO and return its hits as a DataFrame.

    Requirements
    ------------
    - `fimo` must be on PATH (provided by the `meme` conda package).

    Output files
    ------------
    If keep_hits=True, writes:
      <out_dir>/fimo_<label>/fimo.txt

    Returns
    -------
    pd.DataFrame
        Raw FIMO hits (may be empty).
    """
    if which("fimo") is None:
        raise RuntimeError(
            "fimo not found on PATH. Install MEME Suite via conda: conda install -c bioconda meme"
        )

    out_dir = ensure_dir(out_dir)
    motif_file = Path(motif_file)
    if not motif_file.exists():
        raise FileNotFoundError(motif_file)

    fimo_dir = ensure_dir(out_dir / f"fimo_{label}")

    # FIMO wants a FASTA file on disk
    fasta_path = fimo_dir / "locs.fasta"
    write_fasta(records, fasta_path)

    # Run FIMO
    cmd = [
        "fimo",
        "--thresh",
        str(thresh),
        "--verbosity",
        "1",
        "--oc",
        str(fimo_dir),
        str(motif_file),
        str(fasta_path),
    ]
    run_cmd(cmd, cwd=fimo_dir, check=True, capture=False)

    # FIMO outputs fimo.tsv in the output directory (newer versions). Some older versions use fimo.txt.
    tsv = fimo_dir / "fimo.tsv"
    txt = fimo_dir / "fimo.txt"
    hits_path = tsv if tsv.exists() else txt

    if not hits_path.exists():
        # No hits file produced (unexpected) — return empty.
        return pd.DataFrame()

    df = pd.read_csv(hits_path, sep="\t", comment="#")
    if df.empty:
        return df

    # Normalize column names
    df = df.rename(columns={"p-value": "p_value", "q-value": "q_value"})
    if keep_hits:
        df.to_csv(fimo_dir / "fimo.txt", sep="\t", index=False)

    return df


def featurize_fimo_summary(
    fimo_hits: pd.DataFrame,
    sequence_names: Sequence[str],
    label: str,
    p_cutoff: float = 1e-4,
    window: int = 20,
) -> pd.DataFrame:
    """Compute the compact FIMO summary features used by the legacy pipeline.

    For each sequence:
      1) For each motif_id, keep the hit with the smallest p-value.
      2) Take the start position of these "best" hits.
      3) Feature `<label>_motifs` = number of distinct motif_id with p_value < p_cutoff
      4) Feature `<label>_max_window` = max number of motif starts within any `window` bp interval.
    """
    # Handle no hits
    if fimo_hits is None or fimo_hits.empty:
        return pd.DataFrame(
            {
                "sequence_name": list(sequence_names),
                f"{label}_motifs": [0] * len(sequence_names),
                f"{label}_max_window": [0] * len(sequence_names),
            }
        )

    df = fimo_hits.copy()
    # Some FIMO versions still use 'p-value'
    if "p_value" not in df.columns and "p-value" in df.columns:
        df = df.rename(columns={"p-value": "p_value"})
    if "sequence_name" not in df.columns:
        raise ValueError("FIMO hits table missing 'sequence_name' column")
    if "motif_id" not in df.columns:
        raise ValueError("FIMO hits table missing 'motif_id' column")
    if "start" not in df.columns:
        raise ValueError("FIMO hits table missing 'start' column")
    if "p_value" not in df.columns:
        raise ValueError("FIMO hits table missing 'p_value' column")

    # Filter by p-value
    df = df[pd.to_numeric(df["p_value"], errors="coerce") < p_cutoff].copy()
    if df.empty:
        return pd.DataFrame(
            {
                "sequence_name": list(sequence_names),
                f"{label}_motifs": [0] * len(sequence_names),
                f"{label}_max_window": [0] * len(sequence_names),
            }
        )

    df["start"] = pd.to_numeric(df["start"], errors="coerce")
    df["p_value"] = pd.to_numeric(df["p_value"], errors="coerce")
    df = df.dropna(subset=["start", "p_value"]).copy()

    # Keep best hit per (sequence_name, motif_id)
    idx = df.groupby(["sequence_name", "motif_id"])["p_value"].idxmin()
    best = df.loc[idx, ["sequence_name", "motif_id", "start"]]

    # Aggregate per sequence
    feats = []
    for name, grp in best.groupby("sequence_name"):
        starts = grp["start"].astype(int).tolist()
        feats.append(
            {
                "sequence_name": name,
                f"{label}_motifs": int(len(starts)),
                f"{label}_max_window": int(_find_max_window(starts, window=window)),
            }
        )

    out = pd.DataFrame(feats)
    # Ensure every sequence appears
    out = out.set_index("sequence_name").reindex(sequence_names).fillna(0).reset_index()
    out["sequence_name"] = out["sequence_name"].astype(str)
    return out


def _symbol_from_motif_id(motif_id: str) -> str:
    # Legacy behavior: remove everything after the first underscore.
    return str(motif_id).split("_")[0]


def featurize_fimo_symbol_hits(fimo_hits: pd.DataFrame, label: str) -> pd.DataFrame:
    """Wide table: per-sequence hit counts per TF symbol.

    Column naming follows the legacy R script:
      fimo_<label>_target_gene_hits_<SYMBOL>
    """
    if fimo_hits is None or fimo_hits.empty:
        return pd.DataFrame(columns=["sequence_name"])

    df = fimo_hits.copy()
    if "sequence_name" not in df.columns or "motif_id" not in df.columns:
        raise ValueError("FIMO hits must contain sequence_name and motif_id")

    df["Symbol"] = df["motif_id"].map(_symbol_from_motif_id)

    counts = (
        df.groupby(["sequence_name", "Symbol"]).size().rename("n").reset_index()
    )

    wide = counts.pivot_table(
        index="sequence_name",
        columns="Symbol",
        values="n",
        fill_value=0,
        aggfunc="sum",
    )
    wide.columns = [f"fimo_{label}_target_gene_hits_{c}" for c in wide.columns]
    wide = wide.reset_index()
    return wide


def read_tfdb_family_map(tfdb_path: str | Path) -> pd.DataFrame:
    """Read an AnimalTFDB-style mapping file.

    Expected at least a 'Symbol' column and a family column. The legacy pipeline used 'Family.main'.
    """
    tfdb_path = Path(tfdb_path)
    df = pd.read_csv(tfdb_path, sep="\t")
    if "Symbol" not in df.columns:
        raise ValueError("TFDB file must contain a 'Symbol' column")
    if "Family.main" not in df.columns:
        # fall back to 'Family' if available
        if "Family" in df.columns:
            df = df.rename(columns={"Family": "Family.main"})
        else:
            raise ValueError("TFDB file must contain 'Family.main' (or 'Family')")
    return df[["Symbol", "Family.main"]].drop_duplicates()


def featurize_fimo_family_counts(
    fimo_hits: pd.DataFrame,
    tfdb_map: pd.DataFrame,
    label: str,
) -> pd.DataFrame:
    """Wide table: per-sequence counts by TF family.

    Column naming follows the legacy pipeline:
      n.<Family.main>_<label>
    """
    if fimo_hits is None or fimo_hits.empty:
        return pd.DataFrame(columns=["sequence_name"])

    df = fimo_hits.copy()
    df["Symbol"] = df["motif_id"].map(_symbol_from_motif_id)

    merged = df.merge(tfdb_map, on="Symbol", how="left")
    merged["Family.main"] = merged["Family.main"].fillna("unknown")

    counts = merged.groupby(["sequence_name", "Family.main"]).size().rename("n").reset_index()

    wide = counts.pivot_table(
        index="sequence_name",
        columns="Family.main",
        values="n",
        fill_value=0,
        aggfunc="sum",
    )
    # Prefix columns
    wide.columns = [f"n.{c}_{label}" for c in wide.columns]
    wide = wide.reset_index()
    return wide
