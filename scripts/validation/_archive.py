"""Loaders for the engine-output archive the re-solved cohort is built from."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
ARCHIVE = DATA / "reference" / "engine_archive"
STRUCTURES = DATA / "structures"


def residues_orig() -> pd.DataFrame:
    return pd.read_parquet(ARCHIVE / "residues_orig.parquet")


def residues_expand() -> pd.DataFrame:
    return pd.read_parquet(ARCHIVE / "residues_expand.parquet")


def pairs_orig() -> pd.DataFrame:
    return pd.read_parquet(ARCHIVE / "pairs_orig.parquet")


def pairs_expand() -> pd.DataFrame:
    return pd.read_parquet(ARCHIVE / "pairs_expand.parquet")


def uniques() -> pd.DataFrame:
    return pd.read_csv(DATA / "reference" / "reference_sets" / "uniqueRNAS.csv")


def metadata() -> pd.DataFrame:
    md = json.loads((DATA / "reference" / "metadata" / "metadata_cache.json").read_text())
    nt = md.get("nucleotide_counts", {})
    rows = []
    for pdb, m in md["validation_metrics"].items():
        if not isinstance(m, dict):
            continue
        res = m.get("Refinement_resolution")
        if res is None:
            res = m.get("EM Resolution (Å)")
        rows.append({"pdb_id": pdb, "resolution": res,
                     "method": m.get("Experimental_method"), "n_nt": nt.get(pdb)})
    df = pd.DataFrame(rows)
    df["resolution"] = pd.to_numeric(df.resolution, errors="coerce")
    df["n_nt"] = pd.to_numeric(df.n_nt, errors="coerce")
    return df
