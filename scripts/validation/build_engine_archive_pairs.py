"""Per-pair engine output behind the re-solved cohort (data/reference/engine_archive/)."""
import argparse
import json
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / "data" / "reference" / "engine_archive"
CACHE = Path(os.environ.get("PARSE_CACHE_DIR") or Path.home() / ".cache" / "parse")
DEFAULT_ENGINE = os.environ.get("PARSE_ENGINE") or shutil.which("parse") or "parse"
COLUMNS = ["pdb_id", "res_id1", "res_id2", "lw_class", "score", "tier"]


def structure_path(pdb):
    for ext in (".cif.gz", ".cif", ".pdb.gz", ".pdb"):
        p = CACHE / f"{pdb}{ext}"
        if p.exists():
            return p
    return None


def pairs(engine, pdb):
    path = structure_path(pdb)
    if path is None:
        return pdb, None
    run = subprocess.run([engine, str(path), "--no-download"], capture_output=True, text=True)
    if run.returncode != 0:
        return pdb, None
    rows = [(pdb, p["res_id1"], p["res_id2"], (p.get("classification") or {}).get("lw_class"),
             p.get("score"), p.get("tier")) for p in json.loads(run.stdout).get("pairs", [])]
    return pdb, rows


def build(engine, pdbs, workers):
    with ThreadPoolExecutor(workers) as ex:
        results = list(ex.map(lambda p: pairs(engine, p), pdbs))
    failed = [p for p, rows in results if rows is None]
    df = pd.DataFrame([r for _, rows in results if rows for r in rows], columns=COLUMNS)
    for c in ("pdb_id", "res_id1", "res_id2", "lw_class", "tier"):
        df[c] = df[c].astype("string")
    df["score"] = df["score"].astype("float64")
    return df, failed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default=DEFAULT_ENGINE)
    ap.add_argument("--out-dir", type=Path, default=ARCHIVE)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for part in ("orig", "expand"):
        pdbs = sorted(pd.read_parquet(ARCHIVE / f"residues_{part}.parquet",
                                      columns=["pdb_id"]).pdb_id.unique())
        if args.limit:
            pdbs = pdbs[: args.limit]
        df, failed = build(args.engine, pdbs, args.workers)
        df.to_parquet(args.out_dir / f"pairs_{part}.parquet", index=False)
        print(f"pairs_{part}: {len(df):,} pairs from {df.pdb_id.nunique():,} of {len(pdbs):,} "
              f"entries; {len(failed)} failed {failed[:10]}")


if __name__ == "__main__":
    main()
