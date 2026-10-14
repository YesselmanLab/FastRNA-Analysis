"""Build the curated and non-curated reference sets from the BGSU representative list."""

import argparse
import csv
import datetime
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SETS = REPO / "data" / "reference" / "reference_sets"
META = REPO / "data" / "reference" / "metadata" / "metadata_cache.json"

RMSZ, CLASH = 2.0, 20.0
XRAY_RES, XRAY_RFREE, XRAY_GAP = 3.0, 0.28, 0.05
EM_RES, EM_Q = 3.0, 0.5


def xray_resolution(m):
    return m.get("EM Diffraction Resolution (Å)") or m.get("Refinement_resolution")


def em_resolution(m):
    return m.get("EM Resolution (Å)")


def stereochemistry_ok(m):
    return all(m.get(k) is not None and m[k] <= RMSZ for k in ("bonds_rmsz", "angles_rmsz")) \
        and m.get("clashscore") is not None and m["clashscore"] <= CLASH


def curated(m):
    method = m.get("Experimental_method")
    if not stereochemistry_ok(m):
        return False
    if method == "X-ray":
        r = xray_resolution(m)
        rf, rw = m.get("R_free"), m.get("R_work")
        return (r is not None and r <= XRAY_RES and rf is not None and rw is not None
                and rf <= XRAY_RFREE and rf - rw <= XRAY_GAP)
    if method == "EM":
        r, q = em_resolution(m), m.get("em_q_score")
        return r is not None and r <= EM_RES and q is not None and q >= EM_Q
    return False


def entry(m, nt):
    method = m.get("Experimental_method")
    e = {"method": method, "nt_count": nt, "clashscore": m.get("clashscore"),
         "bonds_rmsz": m.get("bonds_rmsz"), "angles_rmsz": m.get("angles_rmsz"),
         "rnasuiteness": m.get("rnasuiteness")}
    if method == "EM":
        e.update(resolution=em_resolution(m), em_q_score=m.get("em_q_score"))
    else:
        rf, rw = m.get("R_free"), m.get("R_work")
        e.update(resolution=xray_resolution(m), R_free=rf, R_work=rw,
                 R_gap=round(rf - rw, 4) if rf is not None and rw is not None else None)
    return e


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--exclude", type=Path, default=None,
                    help="optional file of PDB ids (one per line) to leave out of both sets")
    args = ap.parse_args()

    meta = json.loads(META.read_text())
    vm, nc = meta["validation_metrics"], meta["nucleotide_counts"]
    with open(SETS / "uniqueRNAS.csv") as f:
        unique = sorted({r["unique_pdb_id"].strip().upper() for r in csv.DictReader(f)})
    excluded = set()
    if args.exclude:
        excluded = {l.strip().upper() for l in args.exclude.read_text().split() if l.strip()}

    good, poor, skipped = {}, {}, {"no_nucleotides": [], "no_metadata": [], "excluded": []}
    for p in unique:
        if p in excluded:
            skipped["excluded"].append(p)
            continue
        m = vm.get(p) if isinstance(vm.get(p), dict) else None
        nt = nc.get(p) or 0
        if nt == 0:
            skipped["no_nucleotides"].append(p)
            continue
        if not m:
            skipped["no_metadata"].append(p)
            continue
        (good if curated(m) else poor)[p] = entry(m, nt)

    today = datetime.date.today().isoformat()
    criteria = {"all_entries": f"bonds_rmsz <= {RMSZ}, angles_rmsz <= {RMSZ}, clashscore <= {CLASH}",
                "X-ray": f"resolution <= {XRAY_RES} A, R_free <= {XRAY_RFREE}, R_free - R_work <= {XRAY_GAP}",
                "EM": f"resolution <= {EM_RES} A, em_q_score >= {EM_Q}",
                "g_quadruplexes": "included"}

    def summary(d):
        by = {}
        for e in d.values():
            by[e["method"]] = by.get(e["method"], 0) + 1
        return {"total_pdbs": len(d), "total_nucleotides": sum(e["nt_count"] for e in d.values()),
                "by_method": by}

    (SETS / "reference_set.json").write_text(json.dumps(
        {"schema_version": "3.0", "generated_at": today, "criteria": criteria,
         "summary": summary(good), "pdbs": good}, indent=1))
    (SETS / "reference_set_poor.json").write_text(json.dumps(
        {"schema_version": "3.0", "generated_at": today,
         "definition": "representative entry with >= 1 nucleotide that is not in the curated set",
         "summary": summary(poor), "skipped": skipped, "pdbs": poor}, indent=1))
    print(f"representative entries {len(unique)} | curated {len(good)} {summary(good)['by_method']} "
          f"| non-curated {len(poor)} {summary(poor)['by_method']}")
    print("skipped:", {k: v for k, v in skipped.items() if v})


if __name__ == "__main__":
    main()
