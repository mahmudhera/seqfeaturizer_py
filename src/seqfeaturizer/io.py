from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import pandas as pd
from Bio import SeqIO

try:
    from pyfaidx import Fasta  # type: ignore
except Exception:  # pragma: no cover
    Fasta = None  # type: ignore


DNA_ALPHABET = set("ACGTN")


@dataclass(frozen=True)
class SequenceRecord:
    """A simple sequence record.

    Attributes
    ----------
    name:
        Sequence identifier.
    sequence:
        Sequence string (uppercase).
    """

    name: str
    sequence: str


def normalize_sequence(seq: str) -> str:
    """Normalize a DNA sequence to uppercase A/C/G/T/N.

    Any character outside A/C/G/T is converted to 'N'.
    """
    seq = (seq or "").upper().replace("U", "T")
    return "".join((c if c in DNA_ALPHABET else "N") for c in seq)


def read_fasta(path: str | Path) -> List[SequenceRecord]:
    """Read a FASTA file into a list of SequenceRecord.

    Uses BioPython, so it supports multi-line sequences.
    """
    path = Path(path)
    out: List[SequenceRecord] = []
    for rec in SeqIO.parse(str(path), "fasta"):
        out.append(SequenceRecord(name=str(rec.id), sequence=normalize_sequence(str(rec.seq))))
    if not out:
        raise ValueError(f"No sequences found in FASTA: {path}")
    return out


def read_sequences_csv(path: str | Path, name_col: str = "sequence_name", seq_col: str = "sequence") -> List[SequenceRecord]:
    """Read sequences from a CSV/TSV with explicit name + sequence columns."""
    path = Path(path)
    sep = "\t" if path.suffix.lower() in {".tsv", ".tab", ".txt"} else ","
    df = pd.read_csv(path, sep=sep)
    if name_col not in df.columns or seq_col not in df.columns:
        raise ValueError(
            f"CSV must contain columns '{name_col}' and '{seq_col}'. Found: {list(df.columns)}"
        )
    return [SequenceRecord(str(n), normalize_sequence(str(s))) for n, s in zip(df[name_col], df[seq_col])]


def records_to_dataframe(records: List[SequenceRecord]) -> pd.DataFrame:
    """Convert SequenceRecord list to a canonical DataFrame."""
    return pd.DataFrame({"sequence_name": [r.name for r in records], "sequence": [r.sequence for r in records]})


def read_bed(path: str | Path) -> pd.DataFrame:
    """Read a BED (3 or 4 columns).

    Expected:
        chr  start  end  [name]

    start is 0-based, end is 1-based (standard BED).
    """
    path = Path(path)
    df = pd.read_csv(path, sep="\t", header=None, comment="#")
    if df.shape[1] < 3:
        raise ValueError("BED must have at least 3 columns (chr, start, end)")
    df = df.iloc[:, :4]
    df.columns = ["chr", "start", "end", "name"][: df.shape[1]]
    if "name" not in df.columns:
        # synthesize names as chr:start-end
        df["name"] = df.apply(lambda r: f"{r['chr']}:{int(r['start'])}-{int(r['end'])}", axis=1)
    return df


def read_vcf_like(path: str | Path) -> pd.DataFrame:
    """Read a simple, headerless VCF-like TSV.

    Expected columns:
        chr  pos  name  ref  alt

    - pos is 1-based.
    - name is an arbitrary identifier.

    This matches the original legacy pipeline.
    """
    path = Path(path)
    df = pd.read_csv(path, sep="\t", header=None, comment="#", dtype=str)
    if df.shape[1] < 5:
        raise ValueError("VCF-like file must have 5 columns: chr, pos, name, ref, alt")
    df = df.iloc[:, :5]
    df.columns = ["chr", "pos", "name", "ref", "alt"]
    df["pos"] = df["pos"].astype(int)
    return df


def fetch_sequences_from_genome(
    bed_df: pd.DataFrame,
    genome_fasta: str | Path,
    strand_col: Optional[str] = None,
) -> List[SequenceRecord]:
    """Extract sequences for intervals in a BED dataframe.

    Notes
    -----
    - Requires pyfaidx.
    - Does *not* reverse-complement by default (BED is usually strandless in this pipeline).

    Parameters
    ----------
    bed_df:
        DataFrame with chr/start/end/name.
    genome_fasta:
        Reference genome FASTA (indexed or indexable by pyfaidx).
    strand_col:
        If provided and the column exists, reverse-complement sequences where strand is '-'.
    """
    if Fasta is None:
        raise ImportError("pyfaidx is required for BED/VCF genome extraction. Install: conda install pyfaidx")

    genome_fasta = Path(genome_fasta)
    fa = Fasta(str(genome_fasta), as_raw=True, sequence_always_upper=True)

    def revcomp(seq: str) -> str:
        comp = str.maketrans("ACGTN", "TGCAN")
        return seq.translate(comp)[::-1]

    out: List[SequenceRecord] = []
    for _, row in bed_df.iterrows():
        chrom = str(row["chr"])
        start = int(row["start"])
        end = int(row["end"])
        name = str(row.get("name", f"{chrom}:{start}-{end}"))

        seq = str(fa[chrom][start:end])
        seq = normalize_sequence(seq)

        if strand_col and strand_col in bed_df.columns:
            strand = str(row[strand_col])
            if strand == "-":
                seq = revcomp(seq)

        out.append(SequenceRecord(name=name, sequence=seq))

    return out


def sequences_from_vcf_like(
    vcf_df: pd.DataFrame,
    genome_fasta: str | Path,
    window: int,
    allele: str,
) -> List[SequenceRecord]:
    """Generate sequences around variants, for either the REF or ALT allele.

    Parameters
    ----------
    vcf_df:
        DataFrame with chr,pos,name,ref,alt. pos is 1-based.
    genome_fasta:
        Reference genome FASTA.
    window:
        Total sequence length to extract (centered on the variant).
        Uses the same convention as the legacy pipeline: center is at the variant position;
        for even lengths, the center is just to the right of the middle.
    allele:
        'ref' or 'alt'.

    Returns
    -------
    List[SequenceRecord]
        One record per row in vcf_df.
    """
    if allele not in {"ref", "alt"}:
        raise ValueError("allele must be 'ref' or 'alt'")

    # Build a BED-like dataframe around the variant
    front = window // 2
    back = window - front

    bed = pd.DataFrame(
        {
            "chr": vcf_df["chr"],
            "start": vcf_df["pos"].astype(int) - 1 - front,  # 0-based
            "end": vcf_df["pos"].astype(int) - 1 + back,
            "name": vcf_df["name"],
        }
    )

    ref_records = fetch_sequences_from_genome(bed, genome_fasta)

    if allele == "ref":
        return ref_records

    # Substitute ALT into the reference sequence at the correct offset.
    out: List[SequenceRecord] = []
    for rec, (_, row) in zip(ref_records, vcf_df.iterrows()):
        ref = normalize_sequence(str(row["ref"]))
        alt = normalize_sequence(str(row["alt"]))
        # Variant is centered at index 'front' in the extracted window
        pos0 = front
        if rec.sequence[pos0 : pos0 + len(ref)] != ref:
            raise ValueError(
                f"REF allele mismatch for {rec.name}: expected '{ref}' at offset {pos0}, "
                f"found '{rec.sequence[pos0:pos0+len(ref)]}'.\n"
                "Check that the genome build matches the VCF-like coordinates."
            )
        seq_alt = rec.sequence[:pos0] + alt + rec.sequence[pos0 + len(ref) :]
        # Keep the same length if ref/alt length differs (indels): trim/pad with N
        if len(seq_alt) > window:
            seq_alt = seq_alt[:window]
        elif len(seq_alt) < window:
            seq_alt = seq_alt + "N" * (window - len(seq_alt))
        out.append(SequenceRecord(name=rec.name, sequence=seq_alt))

    return out
