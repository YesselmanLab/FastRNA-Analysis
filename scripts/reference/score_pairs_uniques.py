"""Per-pair FastRNA scores over the non-redundant set."""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
UNIQUES = REPO / "data" / "reference" / "reference_sets" / "uniqueRNAS.csv"
STRUCTURES = REPO / "data" / "structures"
OUT = REPO / "data" / "reference" / "scores" / "pair_scores_uniques_parse.jsonl"
ENGINE = os.environ.get("PARSE_ENGINE") or shutil.which("parse") or "parse"


def distinct_ids(path: Path) -> list[str]:
    with open(path, newline="") as handle:
        return sorted({row["unique_pdb_id"].strip().upper()
                       for row in csv.DictReader(handle) if row["unique_pdb_id"].strip()})


def lw_of(pair: dict) -> tuple[str, bool]:
    c = pair.get("classification") or {}
    lw = c.get("lw_class") or ""
    if lw.startswith("ambiguous"):
        lw = lw.split("(", 1)[-1].rstrip(")")
    return lw, bool(c.get("is_ambiguous")) or "|" in lw


def score_one(pdb: str) -> tuple[str, str, list[dict]]:
    cif = STRUCTURES / f"{pdb}.cif"
    if not cif.exists():
        return pdb, "missing", []
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as fh:
        out = Path(fh.name)
    try:
        run = subprocess.run([ENGINE, str(cif), "--no-download", "--out", str(out)],
                             capture_output=True, text=True, timeout=600)
        if run.returncode != 0:
            return pdb, "failed", []
        d = json.loads(out.read_text())
    except (subprocess.TimeoutExpired, json.JSONDecodeError):
        return pdb, "failed", []
    finally:
        out.unlink(missing_ok=True)
    rows = []
    for pair in d.get("pairs", []):
        lw, amb = lw_of(pair)
        rows.append({
            "pdb_id": pdb, "lw_class": lw, "is_ambiguous": amb,
            "score": pair.get("score"), "sequence": pair.get("sequence") or "",
            "issues": {i["issue"]: round(i.get("severity", 0.0), 4)
                       for i in pair.get("issue_details") or [] if i.get("severity", 0.0) > 0},
        })
    return pdb, "ok", rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="first N entries only (0 = all)")
    args = ap.parse_args()

    ids = distinct_ids(UNIQUES)
    if args.limit:
        ids = ids[: args.limit]
    with ProcessPoolExecutor(args.workers) as ex:
        results = list(ex.map(score_one, ids, chunksize=4))

    status = {}
    n_rows = 0
    tmp = args.out.with_suffix(".jsonl.part")
    with open(tmp, "w") as out:
        for pdb, st, rows in results:
            status[st] = status.get(st, 0) + 1
            for row in rows:
                out.write(json.dumps(row) + "\n")
            n_rows += len(rows)
    tmp.replace(args.out)
    print(f"{len(ids):,} entries {status}; {n_rows:,} pair rows -> {args.out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
