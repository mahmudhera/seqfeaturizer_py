#!/usr/bin/env python3
"""Compute feature differences between two consolidated feature tables.

This script replaces the legacy R/python diff steps.

Typical usage:
  python scripts/compute_diff.py --a out_ref/summary.csv --b out_alt/summary.csv --out delta.csv

Paired IDs (e.g. a 'Names' column containing ref|alt):
  python scripts/compute_diff.py --a ref_summary.csv --b alt_summary.csv \
      --pairs pairs.tsv --pairs-col Names --pairs-split '|' --out delta.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from seqfeaturizer.diff import compute_delta, parse_pairs_file, read_table
from seqfeaturizer.io import read_fasta
from seqfeaturizer.diff import levenshtein_similarity


def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    ap.add_argument("--a", required=True, help="Feature table A (CSV/TSV) or a directory containing summary.csv")
    ap.add_argument("--b", required=True, help="Feature table B (CSV/TSV) or a directory containing summary.csv")
    ap.add_argument("--out", required=True, help="Output CSV/TSV path")

    ap.add_argument("--id-col", default="sequence_name", help="ID column name inside feature tables")
    ap.add_argument(
        "--strict-columns",
        action="store_true",
        help="Require identical feature columns in A and B (otherwise uses intersection)",
    )
    ap.add_argument(
        "--subtract",
        choices=["b_minus_a", "a_minus_b"],
        default="b_minus_a",
        help="Delta direction",
    )

    # Pairing support
    ap.add_argument("--pairs", default=None, help="Optional pairs file to align rows")
    ap.add_argument("--pairs-col", default=None, help="If one-column pairs file, which column contains 'A|B'")
    ap.add_argument("--pairs-split", default="|", help="Separator inside a pair string")
    ap.add_argument("--a-id-col", default="sequence_name_a", help="Column name for A IDs inside pairs")
    ap.add_argument("--b-id-col", default="sequence_name_b", help="Column name for B IDs inside pairs")

    # Optional Levenshtein similarity (requires FASTA inputs)
    ap.add_argument("--a-fasta", default=None, help="Optional FASTA for A sequences (for similarity)")
    ap.add_argument("--b-fasta", default=None, help="Optional FASTA for B sequences (for similarity)")

    args = ap.parse_args()

    def resolve_summary(path_like: str) -> Path:
        p = Path(path_like)
        if p.is_dir():
            return p / "summary.csv"
        return p

    a_path = resolve_summary(args.a)
    b_path = resolve_summary(args.b)

    a = read_table(a_path)
    b = read_table(b_path)

    pairs_df = None
    if args.pairs:
        pairs_df = parse_pairs_file(
            args.pairs,
            col=args.pairs_col,
            split=args.pairs_split,
            a_id_col=args.a_id_col,
            b_id_col=args.b_id_col,
        )

    res = compute_delta(
        a,
        b,
        id_col=args.id_col,
        pairs=pairs_df,
        a_id_col=args.a_id_col,
        b_id_col=args.b_id_col,
        subtract=args.subtract,
        strict_columns=args.strict_columns,
    )

    out_df = res.delta

    # Optional similarity computation
    if args.a_fasta and args.b_fasta and pairs_df is not None:
        a_seqs = {r.name: r.sequence for r in read_fasta(args.a_fasta)}
        b_seqs = {r.name: r.sequence for r in read_fasta(args.b_fasta)}

        sims = []
        for a_id, b_id in zip(pairs_df[args.a_id_col], pairs_df[args.b_id_col]):
            sims.append(levenshtein_similarity(a_seqs.get(str(a_id), ""), b_seqs.get(str(b_id), "")))

        out_df.insert(2, "levenshtein_similarity", sims)

    out_path = Path(args.out)
    sep = "\t" if out_path.suffix.lower() in {".tsv", ".tab", ".txt"} else ","
    out_df.to_csv(out_path, sep=sep, index=False)

    print(f"Wrote delta table to: {out_path}")


if __name__ == "__main__":
    main()
