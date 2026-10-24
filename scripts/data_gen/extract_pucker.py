"""Extract per-residue ribose pucker signals (delta and Pperp) for the high-quality reference set.

Runs `parse dump-pucker`, which prints one row per scored residue that has both a delta
torsion and a Pperp distance: res_id, delta, Pperp, and whether the residue is in a helical
canonical cWW pair. Used for the delta-vs-Pperp panel of Supplementary Fig. S5.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import subprocess
from pathlib import Path

import pandas as pd

import _engine
from _engine import structure_args

REPO = Path(__file__).resolve().parent.parent.parent
DEFAULT_REF = REPO / "data" / "reference" / "reference_sets" / "reference_set.json"
DEFAULT_OUT = REPO / "data" / "reference" / "high_quality_features" / "pucker.csv"
COLS = ["res_id", "delta", "pperp", "helical_wc"]

_ENGINE: str | None = None


def _init_worker(engine: str) -> None:
    global _ENGINE
    _ENGINE = engine


def _worker(pdb_id: str):
    try:
        src = structure_args(pdb_id)
        cmd = [_ENGINE, "dump-pucker", src[0], *src[1:]]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(r.stderr.strip()[:200])
        rows = []
        for line in r.stdout.splitlines():
            f = line.split("\t")
            if len(f) != 4:
                continue
            rows.append({"pdb_id": pdb_id, "res_id": f[0], "delta": float(f[1]),
                         "pperp": float(f[2]), "helical_wc": int(f[3])})
        return pdb_id, rows, None
    except Exception as e:  # noqa: BLE001
        return pdb_id, [], f"{type(e).__name__}: {e}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reference-set", type=Path, default=DEFAULT_REF)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--engine", default=None, help="path to `parse` (else PARSE_ENGINE / PATH)")
    ap.add_argument("--workers", type=int, default=0, help="0 = cpu_count")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    engine = _engine.find_engine(args.engine)
    with open(args.reference_set) as f:
        pdbs = list(json.load(f)["pdbs"])
    if args.limit:
        pdbs = pdbs[:args.limit]
    print(f"Engine: {engine}\nReference set: {len(pdbs)} structures")

    all_rows: list[dict] = []
    n_ok = n_fail = 0
    with mp.Pool(args.workers or mp.cpu_count(),
                 initializer=_init_worker, initargs=(engine,)) as pool:
        for i, (pdb_id, rows, err) in enumerate(pool.imap_unordered(_worker, pdbs), 1):
            if err:
                n_fail += 1
                print(f"  FAIL {pdb_id}: {err}")
            else:
                all_rows.extend(rows); n_ok += 1
            if i % 100 == 0 or i == len(pdbs):
                print(f"  {i}/{len(pdbs)}  ok={n_ok} fail={n_fail}  rows={len(all_rows):,}", flush=True)

    df = pd.DataFrame(all_rows, columns=["pdb_id", *COLS]).sort_values(["pdb_id", "res_id"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"\nWrote {args.output}")
    print(f"  {len(df):,} residues  |  {df.pdb_id.nunique()} structures")


if __name__ == "__main__":
    main()
