# seqfeaturizer (python-only)

This is a clean, **python-only reimplementation** of the legacy `featurize_sequences` project you uploaded.

The original codebase produced several per-feature CSV files from a collection of DNA sequences:

- `deepbind`
- `5mer`
- `fimo_summary`
- `polyA_polyT_GC`
- `dna_shape`

…and then used R scripts to merge these into a single table and (optionally) compute diffs.

This reimplementation:

- removes the R dependency entirely (merging + diffing are in Python)
- is packaged as a standard python project (installable with conda)
- is modular and readable (each feature family lives in its own module)
- provides **two main scripts**:
  - `scripts/compute_features.py` → compute a consolidated feature table from sequences
  - `scripts/compute_diff.py` → compute feature deltas between two consolidated tables

---

## Installation (conda)

### Option A: Full environment (recommended; includes MEME-suite for FIMO)

```bash
# 1) create env
conda env create -f environment.yml
conda activate seqfeaturizer

# 2) sanity check
python -c "import seqfeaturizer; print(seqfeaturizer.__version__)"
```

### Option B: Minimal environment (no MEME-suite)

Use this if you do **not** need FIMO-based features:

```bash
conda env create -f environment.min.yml
conda activate seqfeaturizer
```

> **Note:** `deepbind` is not installed by conda in this project. If you want DeepBind features, you must install a working DeepBind CLI separately and pass `--deepbind-cmd` + `--deepbind-ids`.

---

## Inputs: how to prepare your data

### 1) FASTA (recommended)

A FASTA file where each record header becomes `sequence_name`:

```fasta
>seq1
ACGTACGTACGT
>seq2
TTTTTACGNNNN
```

### 2) CSV/TSV

A CSV/TSV with at least:

- `sequence_name`
- `sequence`

Example (`.csv`):

```csv
sequence_name,sequence
seq1,ACGTACGT
seq2,TTTTAAAA
```

### 3) BED + genome FASTA (optional)

If you have genomic intervals, you can extract sequences directly:

- BED columns: `chr start end [name]`
- Requires `--genome-fasta path/to/genome.fa`

### 4) Headerless VCF-like TSV + genome FASTA (optional)

Matches the legacy pipeline’s simple format:

```
chr  pos  name  ref  alt
```

- `pos` is **1-based**.
- Use `--vcf-allele ref|alt` and `--vcf-window N` to generate a centered sequence.

---

## Feature families (what gets computed)

### `5mer`
- 1024 columns (all A/C/G/T 5-mers)
- counts are robust to ambiguous bases: any window containing `N` is ignored
- optional normalization by window count

### `polyA_polyT_GC`
- `polyA`: max consecutive `A` run (capped; default 50)
- `polyT`: max consecutive `T` run (capped; default 50)
- `GC`: count of `G` or `C`

### `fimo_summary` (optional; requires MEME-suite `fimo`)
For each motif database you provide:

- `<label>_motifs`: number of distinct motifs with p-value < threshold
- `<label>_max_window`: max number of motif starts within any N-bp window (default 20)

You provide motif databases via repeated `--motif-db LABEL=PATH` arguments.

Optional expansions (can make the table very wide):

- per-TF hit counts: `--include-fimo-hits`
- per-TF-family counts: `--tfdb path/to/TFDB_mapping.tsv`

### `dna_shape` (optional)
Computes mean values of:

- HelT
- MGW
- ProT
- Roll

**Method:** RohsLab DNAshape webserver (internet required). Use:

- `--dnashape rohs`

### `deepbind` (optional)
Wraps an external DeepBind CLI. You must provide:

- `--deepbind-ids path/to/ids.txt`
- (optionally) `--deepbind-cmd /path/to/deepbind`

---

## Script 1: Compute consolidated features

### Minimal example (k-mer + poly runs)

```bash
python scripts/compute_features.py \
  --fasta seqs.fasta \
  --out-dir out_features
```

Outputs:

- `out_features/sequences.csv`
- `out_features/5mer.csv`
- `out_features/polyA_polyT_GC.csv`
- `out_features/summary.csv`  ✅ consolidated
- `out_features/feature_categories.csv`

### With FIMO summary

```bash
python scripts/compute_features.py \
  --fasta seqs.fasta \
  --out-dir out_features \
  --features 5mer polyA_polyT_GC fimo_summary \
  --motif-db encode=encode_motifs.meme \
  --motif-db hg19=hg19_motifs.meme \
  --fimo-p 1e-4 \
  --fimo-window 20 \
  --fimo-keep-hits
```

### With DNAshape means (internet required)

```bash
python scripts/compute_features.py \
  --fasta seqs.fasta \
  --out-dir out_features \
  --features 5mer polyA_polyT_GC dna_shape \
  --dnashape rohs
```

### With DeepBind (requires external DeepBind CLI)

```bash
python scripts/compute_features.py \
  --fasta seqs.fasta \
  --out-dir out_features \
  --features 5mer polyA_polyT_GC deepbind \
  --deepbind-ids ids.txt \
  --deepbind-cmd deepbind
```

---

## Script 2: Compute feature diffs

### A) Match by identical `sequence_name`

```bash
python scripts/compute_diff.py \
  --a out_ref/summary.csv \
  --b out_alt/summary.csv \
  --out delta.csv
```

### B) Pairing file (ref|alt style)

If your IDs differ between the two tables, provide a `pairs` file.

Example `pairs.tsv`:

```tsv
Names
ref1|alt1
ref2|alt2
```

Run:

```bash
python scripts/compute_diff.py \
  --a ref_summary.csv \
  --b alt_summary.csv \
  --pairs pairs.tsv \
  --pairs-col Names \
  --pairs-split '|' \
  --out delta.tsv
```

### Optional: add Levenshtein similarity for paired sequences

Provide FASTA files for both sets **and** a pairs file:

```bash
python scripts/compute_diff.py \
  --a ref_summary.csv \
  --b alt_summary.csv \
  --pairs pairs.tsv \
  --a-fasta ref.fasta \
  --b-fasta alt.fasta \
  --out delta_with_similarity.csv
```

---

## Notes on compatibility vs the legacy pipeline

- The original pipeline used an R script (`feature_summary.R`) to merge per-module features and add a few derived counts.
  This project performs the same merge logic in `seqfeaturizer/consolidate.py`.
- FIMO summary features follow the original python implementation: **one best hit per motif per sequence**, then a 20bp
  sliding window count.
- DNAshape defaults to **off** (`--dnashape none`) because it requires internet access.

---

## Project layout

```
seqfeaturizer_py/
  scripts/
    compute_features.py
    compute_diff.py
  src/seqfeaturizer/
    io.py
    consolidate.py
    diff.py
    features/
      kmer.py
      poly.py
      fimo.py
      dnashape.py
      deepbind.py
  environment.yml
  environment.min.yml
  README.md
```

---

## Troubleshooting

### `fimo not found on PATH`
Install MEME-suite:

```bash
conda install -c bioconda meme
```

### DNAshape server errors
If RohsLab changes their webserver output format, the DNAshape download link regex may need adjustment.
If you need offline DNAshape features, skip them and/or provide a precomputed `dna_shape.csv`.

---

## Citation

If you publish work using DNA shape features or DeepBind outputs, please cite the original tools:

- DNAshapeR / DNAshape method (Rohs Lab)
- DeepBind
- MEME-suite / FIMO

