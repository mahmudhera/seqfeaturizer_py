from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

from .features.deepbind import deepbind_top_quantile
from .features.kmer import n_distinct_kmers
from .features.poly import poly_small
from .utils import drop_nonfeature_cols, fill_numeric_na0, safe_full_join, standardize_sequence_name


@dataclass
class ConsolidatedResult:
    summary: pd.DataFrame
    feature_categories: pd.DataFrame


def _feature_list(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c != "sequence_name"]


def consolidate_features(
    sequence_names: Sequence[str],
    *,
    kmer_df: Optional[pd.DataFrame] = None,
    poly_df: Optional[pd.DataFrame] = None,
    fimo_summary_dfs: Optional[Dict[str, pd.DataFrame]] = None,
    fimo_symbol_hit_dfs: Optional[Dict[str, pd.DataFrame]] = None,
    fimo_family_count_dfs: Optional[Dict[str, pd.DataFrame]] = None,
    dnashape_df: Optional[pd.DataFrame] = None,
    deepbind_df: Optional[pd.DataFrame] = None,
    deepbind_top_q: float = 0.9,
) -> ConsolidatedResult:
    """Merge per-module features into a single consolidated table.

    This function is the python analogue of the legacy `feature_summary.R` script.

    Parameters
    ----------
    sequence_names:
        Canonical ordering of sequences.
    kmer_df:
        Wide k-mer counts (sequence_name + 4**k columns).
    poly_df:
        polyA/polyT/GC table.
    fimo_summary_dfs:
        Dict[label -> summary df], each with columns: sequence_name, <label>_motifs, <label>_max_window.
    fimo_symbol_hit_dfs:
        Optional dict[label -> wide hits by symbol].
    fimo_family_count_dfs:
        Optional dict[label -> wide family counts].
    dnashape_df:
        dna shape mean features.
    deepbind_df:
        wide deepbind predictions.
    deepbind_top_q:
        Quantile used to compute n.deepbind_top.

    Returns
    -------
    ConsolidatedResult
        - summary: merged feature table
        - feature_categories: mapping of each feature column to a category label
    """
    blocks: List[pd.DataFrame] = []
    categories: List[Tuple[str, str]] = []

    # --- polyA/polyT
    if poly_df is not None and not poly_df.empty:
        psmall = poly_small(standardize_sequence_name(poly_df))
        psmall = drop_nonfeature_cols(psmall)
        blocks.append(psmall)
        categories += [(c, "polyA_polyT_GC") for c in _feature_list(psmall)]

    # --- k-mer
    if kmer_df is not None and not kmer_df.empty:
        kmer_df = standardize_sequence_name(kmer_df)
        kmer_df = drop_nonfeature_cols(kmer_df)
        blocks.append(kmer_df)
        categories += [(c, "5mer") for c in _feature_list(kmer_df)]

        n5 = n_distinct_kmers(kmer_df)
        blocks.append(n5)
        categories += [(c, "5mer") for c in _feature_list(n5)]

    # --- dna shape
    if dnashape_df is not None and not dnashape_df.empty:
        ds = standardize_sequence_name(dnashape_df)
        ds = drop_nonfeature_cols(ds)
        blocks.append(ds)
        categories += [(c, "dna_shape") for c in _feature_list(ds)]

    # --- fimo summary
    if fimo_summary_dfs:
        for label, df in fimo_summary_dfs.items():
            df = standardize_sequence_name(df)
            df = drop_nonfeature_cols(df)
            blocks.append(df)
            categories += [(c, "fimo_summary") for c in _feature_list(df)]

    # --- fimo family counts and symbol hits
    if fimo_family_count_dfs:
        for label, df in fimo_family_count_dfs.items():
            df = standardize_sequence_name(df)
            df = drop_nonfeature_cols(df)
            if df.shape[1] > 1:
                blocks.append(df)
                categories += [(c, f"fimo_{label}") for c in _feature_list(df)]

    if fimo_symbol_hit_dfs:
        for label, df in fimo_symbol_hit_dfs.items():
            df = standardize_sequence_name(df)
            df = drop_nonfeature_cols(df)
            if df.shape[1] > 1:
                blocks.append(df)
                categories += [(c, f"fimo_{label}") for c in _feature_list(df)]

    # --- deepbind
    if deepbind_df is not None and not deepbind_df.empty:
        db = standardize_sequence_name(deepbind_df)
        db = drop_nonfeature_cols(db)
        db_top = deepbind_top_quantile(db, q=deepbind_top_q)

        blocks.append(db_top)
        categories += [(c, "deepbind") for c in _feature_list(db_top)]

        blocks.append(db)
        categories += [(c, "deepbind") for c in _feature_list(db)]

    if not blocks:
        raise ValueError("No features were provided/computed; cannot create summary")

    # Merge all blocks
    summary = safe_full_join(blocks)

    # Ensure canonical row order and fill numeric NA with 0
    summary = summary.set_index("sequence_name").reindex(sequence_names).reset_index()
    summary = fill_numeric_na0(summary)

    feature_categories = pd.DataFrame(categories, columns=["feature", "category"]).drop_duplicates()

    return ConsolidatedResult(summary=summary, feature_categories=feature_categories)
