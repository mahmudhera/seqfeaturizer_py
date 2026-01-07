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

    This reproduces the intent of the legacy pipeline's python implementation.

    Important
    ---------
    This feature requires internet access to RohsLab's DNAshape server.
    If you need fully offline featurization, skip this feature or provide your
    own dna_shape.csv.

    Returns
    -------
    pd.DataFrame with columns: sequence_name, HelT, MGW, ProT, Roll
    """
    out_dir = ensure_dir(out_dir)
    work = ensure_dir(out_dir / "dna_shape")

    fasta_path = work / "locs.fasta"
    with fasta_path.open("w") as f:
        for r in records:
            f.write(f">{r.name}\n{r.sequence}\n")

    session = requests.Session()

    # The legacy code posts to this endpoint.
    form_url = "https://rohslab.usc.edu/DNAshape/form_handler.php"
    with fasta_path.open("rb") as fh:
        resp = session.post(
            form_url,
            files={"seq_file": fh},
            allow_redirects=True,
            timeout=timeout_s,
            verify=verify_ssl,
        )
    resp.raise_for_status()

    # Find the generated zip download link in the HTML response.
    m = re.search(r"(Download\.php\?filename=[^\"\s>]+\.zip)", resp.text)
    if not m:
        # Some server versions use a slightly different path or param.
        m = re.search(r"(Download\.php\?filename=[^\"\s>]+)", resp.text)
    if not m:
        raise RuntimeError(
            "Could not find DNAshape download URL in server response. "
            "The DNAshape server output format may have changed."
        )

    download_rel = m.group(1)
    download_url = "https://rohslab.usc.edu/DNAshape/" + download_rel

    zip_resp = session.get(download_url, timeout=timeout_s, verify=verify_ssl)
    zip_resp.raise_for_status()

    zip_path = work / "dna_shape.zip"
    zip_path.write_bytes(zip_resp.content)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(work)

    # Locate the shape files
    files = list(work.rglob("*"))
    type_to_file: Dict[str, Path] = {}
    for t in DNA_SHAPE_TYPES:
        # match extension-like endings: .MGW, .ProT, .Roll, .HelT (case-insensitive)
        t_low = t.lower()
        cand = [p for p in files if p.is_file() and p.name.lower().endswith("." + t_low)]
        if not cand:
            # Some versions might use .txt suffix with token in name
            cand = [p for p in files if p.is_file() and t_low in p.name.lower()]
        if cand:
            type_to_file[t] = sorted(cand)[0]

    missing = [t for t in DNA_SHAPE_TYPES if t not in type_to_file]
    if missing:
        raise RuntimeError(
            f"Missing DNA shape output files for: {missing}. "
            "Inspect the extracted zip in dna_shape/ to see what the server returned."
        )

    # Parse and compute means
    means: Dict[str, Dict[str, float]] = {r.name: {} for r in records}
    for t in DNA_SHAPE_TYPES:
        parsed = _parse_shape_fasta_like(type_to_file[t])
        for name in means:
            vals = parsed.get(name, [])
            means[name][t] = float(pd.Series(vals).mean()) if vals else 0.0

    df = pd.DataFrame(
        {
            "sequence_name": [r.name for r in records],
            **{t: [means[r.name][t] for r in records] for t in DNA_SHAPE_TYPES},
        }
    )
    return df
