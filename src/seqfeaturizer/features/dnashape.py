from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import pandas as pd
import requests

from ..io import SequenceRecord
from ..utils import ensure_dir


DNA_SHAPE_TYPES = ("HelT", "MGW", "ProT", "Roll")


def _parse_shape_fasta_like(path: Path) -> Dict[str, List[float]]:
    """Parse a DNAshape server output file.

    Files are FASTA-like:
      >seq_id
      v1,v2,v3,... (or whitespace-separated, sometimes includes 'NA')

    Returns
    -------
    dict: seq_id -> list of floats (NA removed)
    """
    out: Dict[str, List[float]] = {}
    cur_name: Optional[str] = None
    values: List[float] = []

    def flush() -> None:
        nonlocal cur_name, values
        if cur_name is None:
            return
        out[cur_name] = values
        cur_name = None
        values = []

    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                flush()
                cur_name = line[1:].strip().split()[0]
                continue
            # Values line
            parts = [p for p in re.split(r"[\s,]+", line) if p]
            for p in parts:
                if p.upper() == "NA":
                    continue
                try:
                    values.append(float(p))
                except ValueError:
                    # skip unexpected tokens
                    continue
    flush()
    return out


def featurize_dna_shape_rohs_server(
    records: Sequence[SequenceRecord],
    out_dir: str | Path,
    timeout_s: int = 300,
    verify_ssl: bool = True,
) -> pd.DataFrame:
    """Compute mean HelT/MGW/ProT/Roll using RohsLab's DNAshape webserver.

    This follows the legacy pipeline closely:
      - write locs.fasta
      - POST to .../serverBackend.php with fields seqfile + delimiter/entriesPerLine
      - scrape Download.php?filename=/tmp/<id>.zip
      - download + extract outputs
      - compute mean across non-NA values for each sequence

    Returns
    -------
    pd.DataFrame with columns: sequence_name, HelT, MGW, ProT, Roll
    """
    out_dir = ensure_dir(out_dir)
    work = ensure_dir(out_dir / "dna_shape")

    # 1) write FASTA
    fasta_path = work / "locs.fasta"
    with fasta_path.open("w") as f:
        for r in records:
            f.write(f">{r.name}\n{r.sequence}\n")

    # 2) If outputs already exist (cache), skip server call
    #    (mirrors legacy behavior: if all types exist, just parse)
    def _has_outputs() -> bool:
        files = list(work.rglob("*"))
        for t in DNA_SHAPE_TYPES:
            t_low = t.lower()
            if not any(p.is_file() and p.name.lower().endswith("." + t_low) for p in files):
                # allow token-in-name fallback too
                if not any(p.is_file() and t_low in p.name.lower() for p in files):
                    return False
        return True

    session = requests.Session()

    if not _has_outputs():
        # 3) POST to the legacy backend endpoint
        rohs_bases = [
            "https://rohslab.cmb.usc.edu/DNAshape/",
            "https://rohslab.usc.edu/DNAshape/",
        ]

        last_err: Exception | None = None
        html: str | None = None
        rohs_base_used: str | None = None

        for base in rohs_bases:
            try:
                post_url = base + "serverBackend.php"
                with fasta_path.open("rb") as fh:
                    resp = session.post(
                        post_url,
                        data={
                            "delimiter": "1",
                            "entriesPerLine": "20",
                            "submit_button": "Submit",
                        },
                        files={"seqfile": fh},  # legacy expects 'seqfile'
                        allow_redirects=True,
                        timeout=timeout_s,
                        verify=verify_ssl,
                    )
                resp.raise_for_status()
                html = resp.text
                rohs_base_used = base
                break
            except Exception as e:
                last_err = e

        if html is None or rohs_base_used is None:
            raise RuntimeError(
                "Failed to submit sequences to the DNAshape server at all known endpoints."
            ) from last_err

        # 4) Scrape the download link (legacy regex)
        m = re.search(r"(Download\.php\?filename=/tmp/\w+\.zip)", html)
        if not m:
            raise RuntimeError(
                "DNAshape server did not return a downloadable zip link. "
                "Try manually uploading locs.fasta to the DNAshape web UI to confirm format."
            )

        download_url = rohs_base_used + m.group(1)

        # 5) Download zip + extract
        zip_resp = session.get(download_url, timeout=timeout_s, verify=verify_ssl)
        zip_resp.raise_for_status()

        zip_path = work / "dna_shape.zip"
        zip_path.write_bytes(zip_resp.content)

        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(work)

    # 6) Locate shape files
    files = list(work.rglob("*"))
    type_to_file: Dict[str, Path] = {}
    for t in DNA_SHAPE_TYPES:
        t_low = t.lower()
        cand = [p for p in files if p.is_file() and p.name.lower().endswith("." + t_low)]
        if not cand:
            cand = [p for p in files if p.is_file() and t_low in p.name.lower()]
        if cand:
            type_to_file[t] = sorted(cand)[0]

    missing = [t for t in DNA_SHAPE_TYPES if t not in type_to_file]
    if missing:
        raise RuntimeError(
            f"Missing DNA shape output files for: {missing}. "
            "Inspect the extracted results in dna_shape/ to see what the server returned."
        )

    # 7) Parse and compute means (skip 'NA' as in legacy)
    means: Dict[str, Dict[str, float]] = {r.name: {} for r in records}
    for t in DNA_SHAPE_TYPES:
        parsed = _parse_shape_fasta_like(type_to_file[t])  # expected to drop 'NA'
        for r in records:
            vals = parsed.get(r.name, [])
            means[r.name][t] = float(pd.Series(vals).mean()) if vals else 0.0

    return pd.DataFrame(
        {
            "sequence_name": [r.name for r in records],
            **{t: [means[r.name][t] for r in records] for t in DNA_SHAPE_TYPES},
        }
    )
