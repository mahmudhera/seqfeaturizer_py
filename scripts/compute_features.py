#!/usr/bin/env python3
"""Compute consolidated sequence features.

This script is the python-only replacement for the legacy pipeline:
  1) Generate per-module feature CSVs
  2) Merge into a single summary table

It supports the five feature families the original project centered on:
  - 5mer
  - polyA_polyT_GC
  - fimo_summary (requires MEME-suite `fimo`)
  - dna_shape (optional; by default queries RohsLab DNAshape server)
  - deepbind (optional wrapper around DeepBind CLI)

Usage examples:
  # Basic (kmer + poly)
  python scripts/compute_features.py --fasta seqs.fasta --out-dir out

  # Add FIMO summary for one or more motif databases
  python scripts/compute_features.py --fasta seqs.fasta --out-dir out \
    --features 5mer polyA_polyT_GC fimo_summary \
    --motif-db encode=encode_motifs.meme --motif-db hg19=hg19_motifs.meme

  # Add DNAshape means (requires internet)
  python scripts/compute_features.py --fasta seqs.fasta --out-dir out --features dna_shape

  # Extract sequences from a genome + BED intervals
  python scripts/compute_features.py --bed intervals.bed --genome-fasta hg38.fa --out-dir out
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

from seqfeaturizer.consolidate import consolidate_features
from seqfeaturizer.io import (
    fetch_sequences_from_genome,
    read_bed,
    read_fasta,
    read_sequences_csv,
    read_vcf_like,
    records_to_dataframe,
    sequences_from_vcf_like,
)
from seqfeaturizer.features.deepbind import featurize_deepbind
from seqfeaturizer.features.dnashape import featurize_dna_shape_rohs_server
from seqfeaturizer.features.fimo import (
    featurize_fimo_family_counts,
    featurize_fimo_summary,
    featurize_fimo_symbol_hits,
    read_tfdb_family_map,
    run_fimo,
)
from seqfeaturizer.features.kmer import featurize_kmer
from seqfeaturizer.features.poly import featurize_polyA_polyT_GC
from seqfeaturizer.utils import ensure_dir


FEATURE_ALIASES = {
    "5mer": "5mer",
    "kmer": "5mer",
    "polyA_polyT_GC": "polyA_polyT_GC",
    "poly": "polyA_polyT_GC",
    "fimo_summary": "fimo_summary",
    "fimo": "fimo_summary",
    "dna_shape": "dna_shape",
    "dnashape": "dna_shape",
    "deepbind": "deepbind",
    "all": "all",
}


def parse_motif_db(arg: str) -> Tuple[str, str]:
    if "=" not in arg:
        raise argparse.ArgumentTypeError("--motif-db must be LABEL=PATH")
    label, path = arg.split("=", 1)
    label = label.strip()
    path = path.strip()
    if not label:
        raise argparse.ArgumentTypeError("Motif DB label cannot be empty")
    return label, path


def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Input options
    g_in = ap.add_argument_group("Input")
    g_in.add_argument("--fasta", type=str, help="FASTA file with sequences (headers become sequence_name)")
    g_in.add_argument("--csv", type=str, help="CSV/TSV with sequence_name + sequence columns")
    g_in.add_argument("--name-col", type=str, default="sequence_name", help="Name column for --csv")
    g_in.add_argument("--seq-col", type=str, default="sequence", help="Sequence column for --csv")

    g_in.add_argument("--bed", type=str, help="BED file (chr,start,end[,name])")
    g_in.add_argument("--vcf", type=str, help="Headerless VCF-like TSV (chr,pos,name,ref,alt)")
    g_in.add_argument("--genome-fasta", type=str, help="Genome FASTA for --bed/--vcf")
    g_in.add_argument("--vcf-window", type=int, default=200, help="Window length extracted around VCF position")
    g_in.add_argument("--vcf-allele", choices=["ref", "alt"], default="ref", help="Which allele sequence to generate for --vcf")

    # Output
    g_out = ap.add_argument_group("Output")
    g_out.add_argument("--out-dir", type=str, required=True, help="Directory to write outputs")
    g_out.add_argument(
        "--prefix",
        type=str,
        default="",
        help="Optional filename prefix (e.g., 'ref_' -> ref_summary.csv)",
    )

    # Feature selection
    g_feat = ap.add_argument_group("Features")
    g_feat.add_argument(
        "--features",
        nargs="+",
        default=["5mer", "polyA_polyT_GC"],
        help="Feature families to compute. Use 'all' to attempt everything.",
    )

    # 5-mer
    g_kmer = ap.add_argument_group("5mer options")
    g_kmer.add_argument("--kmer-k", type=int, default=5)
    g_kmer.add_argument("--kmer-normalize", action="store_true", help="Normalize k-mer counts by (L-k+1)")

    # poly
    g_poly = ap.add_argument_group("polyA/polyT/GC options")
    g_poly.add_argument("--poly-cap", type=int, default=50, help="Cap on max poly-run length")

    # FIMO
    g_fimo = ap.add_argument_group("FIMO options")
    g_fimo.add_argument(
        "--motif-db",
        type=parse_motif_db,
        action="append",
        default=[],
        help="Motif database in MEME format, as LABEL=PATH. Can be provided multiple times.",
    )
    g_fimo.add_argument("--fimo-p", type=float, default=1e-4, help="p-value threshold")
    g_fimo.add_argument("--fimo-window", type=int, default=20, help="window size for max_window feature")
    g_fimo.add_argument("--fimo-keep-hits", action="store_true", help="Keep raw fimo.txt in out-dir")
    g_fimo.add_argument(
        "--include-fimo-hits",
        action="store_true",
        help="Add per-TF hit count columns (can make summary very wide)",
    )
    g_fimo.add_argument(
        "--tfdb",
        type=str,
        default=None,
        help="Optional TFDB mapping file (tab-separated) with Symbol and Family.main (or Family).",
    )

    # DNAshape
    g_shape = ap.add_argument_group("DNA shape options")
    g_shape.add_argument(
        "--dnashape",
        choices=["rohs", "none"],
        default="none",
        help="How to compute DNA shape features.",
    )
    g_shape.add_argument("--dnashape-timeout", type=int, default=300)
    g_shape.add_argument("--dnashape-no-ssl-verify", action="store_true")

    # DeepBind
    g_db = ap.add_argument_group("DeepBind options")
    g_db.add_argument("--deepbind-ids", type=str, default=None, help="DeepBind ids/models file")
    g_db.add_argument("--deepbind-cmd", type=str, default="deepbind", help="DeepBind executable")
    g_db.add_argument("--deepbind-batch", type=int, default=5000)

    args = ap.parse_args()

    # Normalize feature names
    feats = []
    for f in args.features:
        key = FEATURE_ALIASES.get(f, None)
        if key is None:
            raise SystemExit(f"Unknown feature family: {f}. Choose from: {sorted(FEATURE_ALIASES)}")
        feats.append(key)
    if "all" in feats:
        feats = ["5mer", "polyA_polyT_GC", "fimo_summary", "dna_shape", "deepbind"]

    out_dir = ensure_dir(args.out_dir)
    prefix = args.prefix

    # --- Load sequences
    records = None
    if args.fasta:
        records = read_fasta(args.fasta)
    elif args.csv:
        records = read_sequences_csv(args.csv, name_col=args.name_col, seq_col=args.seq_col)
    elif args.bed:
        if not args.genome_fasta:
            raise SystemExit("--genome-fasta is required when using --bed")
        bed_df = read_bed(args.bed)
        records = fetch_sequences_from_genome(bed_df, args.genome_fasta)
    elif args.vcf:
        if not args.genome_fasta:
            raise SystemExit("--genome-fasta is required when using --vcf")
        vcf_df = read_vcf_like(args.vcf)
        records = sequences_from_vcf_like(vcf_df, args.genome_fasta, window=args.vcf_window, allele=args.vcf_allele)
    else:
        raise SystemExit("Provide one input source: --fasta, --csv, --bed, or --vcf")

    seq_names = [r.name for r in records]

    # Save the canonical sequence table (useful for auditing)
    seq_df = records_to_dataframe(records)
    seq_df.to_csv(out_dir / f"{prefix}sequences.csv", index=False)

    # --- Compute feature blocks
    kmer_df = None
    poly_df = None
    dnashape_df = None
    deepbind_df = None

    fimo_summary_dfs: Dict[str, pd.DataFrame] = {}
    fimo_symbol_hit_dfs: Dict[str, pd.DataFrame] = {}
    fimo_family_count_dfs: Dict[str, pd.DataFrame] = {}

    if "5mer" in feats:
        kmer_df = featurize_kmer(records, k=args.kmer_k, normalize=args.kmer_normalize)
        kmer_df.to_csv(out_dir / f"{prefix}5mer.csv", index=False)

    if "polyA_polyT_GC" in feats:
        poly_df = featurize_polyA_polyT_GC(records, cap=args.poly_cap)
        poly_df.to_csv(out_dir / f"{prefix}polyA_polyT_GC.csv", index=False)

    if "dna_shape" in feats:
        if args.dnashape == "none":
            raise SystemExit("dna_shape requested but --dnashape is 'none'. Use --dnashape rohs or remove dna_shape.")
        dnashape_df = featurize_dna_shape_rohs_server(
            records,
            out_dir=out_dir,
            timeout_s=args.dnashape_timeout,
            verify_ssl=not args.dnashape_no_ssl_verify,
        )
        dnashape_df.to_csv(out_dir / f"{prefix}dna_shape.csv", index=False)

    if "fimo_summary" in feats:
        if not args.motif_db:
            raise SystemExit(
                "fimo_summary requested but no --motif-db provided. Provide one or more LABEL=PATH motif files."
            )

        tfdb_map = read_tfdb_family_map(args.tfdb) if args.tfdb else None

        for label, motif_path in args.motif_db:
            hits = run_fimo(
                records,
                motif_file=motif_path,
                out_dir=out_dir,
                label=label,
                thresh=args.fimo_p,
                keep_hits=args.fimo_keep_hits,
            )

            summ = featurize_fimo_summary(
                hits,
                sequence_names=seq_names,
                label=label,
                p_cutoff=args.fimo_p,
                window=args.fimo_window,
            )
            fimo_summary_dfs[label] = summ

            if args.include_fimo_hits:
                fimo_symbol_hit_dfs[label] = featurize_fimo_symbol_hits(hits, label=label)

            if tfdb_map is not None:
                fimo_family_count_dfs[label] = featurize_fimo_family_counts(hits, tfdb_map=tfdb_map, label=label)

        # Also save a combined fimo_summary.csv for convenience
        combined = None
        for label, df in fimo_summary_dfs.items():
            combined = df if combined is None else combined.merge(df, on="sequence_name", how="outer")
        if combined is not None:
            combined = combined.set_index("sequence_name").reindex(seq_names).fillna(0).reset_index()
            combined.to_csv(out_dir / f"{prefix}fimo_summary.csv", index=False)

        # Save optional wide blocks
        for label, df in fimo_symbol_hit_dfs.items():
            df.to_csv(out_dir / f"{prefix}fimo_{label}_symbol_hits.csv", index=False)
        for label, df in fimo_family_count_dfs.items():
            df.to_csv(out_dir / f"{prefix}fimo_{label}_family_counts.csv", index=False)

    if "deepbind" in feats:
        if not args.deepbind_ids:
            raise SystemExit("deepbind requested but --deepbind-ids was not provided")
        deepbind_df = featurize_deepbind(
            records,
            ids_file=args.deepbind_ids,
            deepbind_cmd=args.deepbind_cmd,
            batch_size=args.deepbind_batch,
            out_dir=out_dir / "deepbind_batches",
        )
        deepbind_df.to_csv(out_dir / f"{prefix}deepbind.csv", index=False)

    # --- Consolidate
    consolidated = consolidate_features(
        seq_names,
        kmer_df=kmer_df,
        poly_df=poly_df,
        fimo_summary_dfs=fimo_summary_dfs or None,
        fimo_symbol_hit_dfs=fimo_symbol_hit_dfs or None,
        fimo_family_count_dfs=fimo_family_count_dfs or None,
        dnashape_df=dnashape_df,
        deepbind_df=deepbind_df,
    )

    consolidated.summary.to_csv(out_dir / f"{prefix}summary.csv", index=False)
    consolidated.feature_categories.to_csv(out_dir / f"{prefix}feature_categories.csv", index=False)

    print(f"Wrote consolidated summary to: {out_dir / f'{prefix}summary.csv'}")


if __name__ == "__main__":
    main()
