from __future__ import annotations

import itertools
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd

from ..io import SequenceRecord


def all_kmers(k: int) -> List[str]:
    """Return a stable list of all A/C/G/T k-mers.

    For compatibility with the legacy pipeline, we use the same ordering:
    itertools.product('ACGT', repeat=k) and reverse each tuple.

    This yields the same set of k-mers but with an ordering where the *first* base
    cycles fastest.
    """ 
    return ["".join(reversed(tup)) for tup in itertools.product("ACGT", repeat=k)]


def count_kmers(seq: str, k: int, kmer_to_index: Dict[str, int]) -> np.ndarray:
    """Count k-mers in a sequence.

    Non-ACGT characters (including N) are handled robustly: any window containing a non-ACGT
    base is skipped.
    """
    counts = np.zeros(len(kmer_to_index), dtype=np.int32)
    seq = seq.upper()
    for i in range(0, len(seq) - k + 1):
        kmer = seq[i : i + k]
        # Skip kmers with ambiguous bases
        if any(b not in "ACGT" for b in kmer):
            continue
        counts[kmer_to_index[kmer]] += 1
    return counts


def featurize_kmer(records: Sequence[SequenceRecord], k: int = 5, normalize: bool = False) -> pd.DataFrame:
    """Compute k-mer count features.

    Parameters
    ----------
    records:
        List of sequences.
    k:
        k-mer size (default: 5).
    normalize:
        If True, divide counts by the number of possible windows (len(seq)-k+1), per sequence.

    Returns
    -------
    pd.DataFrame
        Columns: sequence_name + 4**k k-mer columns.
    """
    kmers = all_kmers(k)
    kmer_to_index = {kmer: i for i, kmer in enumerate(kmers)}

    mat = np.zeros((len(records), len(kmers)), dtype=np.float32 if normalize else np.int32)
    for i, rec in enumerate(records):
        c = count_kmers(rec.sequence, k=k, kmer_to_index=kmer_to_index)
        if normalize:
            denom = max(len(rec.sequence) - k + 1, 1)
            mat[i, :] = c.astype(np.float32) / float(denom)
        else:
            mat[i, :] = c

    df = pd.DataFrame(mat, columns=kmers)
    df.insert(0, "sequence_name", [r.name for r in records])
    return df


def n_distinct_kmers(kmer_df: pd.DataFrame) -> pd.DataFrame:
    """Compute n.5mer-style derived feature: number of nonzero k-mers per sequence."""
    cols = [c for c in kmer_df.columns if c != "sequence_name"]
    x = (kmer_df[cols] != 0).sum(axis=1)
    return pd.DataFrame({"sequence_name": kmer_df["sequence_name"].astype(str), "n.5mer": x})
