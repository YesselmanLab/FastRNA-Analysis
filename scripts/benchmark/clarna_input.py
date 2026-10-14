"""Prepare ClaRNA's PDB-format inputs from the original mmCIF files."""
import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import statistics
import string
import time

import gemmi

ROOT = Path(__file__).resolve().parents[2]
ALIASES = string.ascii_uppercase + string.ascii_lowercase + string.digits
MAX_NUMBER = 9999


def pdb_lines(cif):
    st = gemmi.read_structure(str(cif))
    st.remove_alternative_conformations()
    chains = [(ch.name, [r for r in ch if r.find_atom("C1'", "*") is not None]) for ch in st[0]]
    chains = [(name, res) for name, res in chains if res]
    for name, res in chains:
        if len(res) > MAX_NUMBER:
            raise ValueError(f"chain {name} has {len(res)} nucleotides (> {MAX_NUMBER})")
    if len(chains) <= len(ALIASES):
        groups = [[c] for c in chains]
    else:
        groups, current, count = [], [], 0
        for chain in chains:
            if current and count + len(chain[1]) > MAX_NUMBER:
                groups.append(current)
                current, count = [], 0
            current.append(chain)
            count += len(chain[1])
        groups.append(current)
        if len(groups) > len(ALIASES):
            raise ValueError(f"{len(groups)} alias chains needed (> {len(ALIASES)})")
    lines, mapping, serial = [], {}, 0
    for alias, group in zip(ALIASES, groups):
        number = 0
        for name, residues in group:
            for r in residues:
                number += 1
                mapping[f"{alias}{number}"] = [name, f"{r.seqid.num}{r.seqid.icode.strip()}"]
                record = "HETATM" if r.het_flag == "H" else "ATOM"
                for a in r:
                    serial += 1
                    atom = a.name if len(a.name) == 4 else f" {a.name:<3}"
                    lines.append(f"{record:<6}{serial % 100000:5d} {atom:<4} {r.name[-3:]:>3} {alias}{number:4d}    "
                                 f"{a.pos.x:8.3f}{a.pos.y:8.3f}{a.pos.z:8.3f}{a.occ:6.2f}{a.b_iso:6.2f}"
                                 f"          {a.element.name.upper():>2}")
        lines.append("TER")
    lines.append("END")
    return "\n".join(lines) + "\n", mapping, len(chains), len(groups)


def prepare(job):
    pdb, cif, out_dir, runs = job
    cif.read_bytes()
    times = []
    for _ in range(runs):
        started = time.perf_counter()
        text, mapping, n_chains, n_aliases = pdb_lines(cif)
        (out_dir / f"{pdb}.pdb").write_text(text)
        times.append(time.perf_counter() - started)
    (out_dir / f"{pdb}.map.json").write_text(json.dumps(mapping))
    return {"pdb": pdb, "nucleotides_written": len(mapping), "chains": n_chains,
            "alias_chains": n_aliases, "conversion_seconds": statistics.median(times)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, default=ROOT / "data/structures")
    ap.add_argument("--cohort", type=Path, default=ROOT / "data/benchmark/dssr_PARSE_time_benchmark.csv")
    ap.add_argument("--output", type=Path, default=ROOT / ".runtime-tools/clarna-inputs")
    ap.add_argument("--pdbs")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    with args.cohort.open() as handle:
        pdbs = sorted(r["PDB"].upper() for r in csv.DictReader(handle))
    if args.pdbs:
        pdbs = [p for p in pdbs if p in set(args.pdbs.upper().split(","))]
    args.output.mkdir(parents=True, exist_ok=True)
    jobs = [(p, args.input / f"{p}.cif", args.output, args.runs) for p in pdbs]
    rows, failures = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for job, future in zip(jobs, [pool.submit(prepare, j) for j in jobs]):
            try:
                rows.append(future.result())
            except Exception as error:
                failures.append({"pdb": job[0], "error": f"{type(error).__name__}: {error}"})
    with (args.output / "conversion.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, ["pdb", "nucleotides_written", "chains", "alias_chains",
                                         "conversion_seconds", "error"])
        writer.writeheader()
        writer.writerows(sorted(rows + failures, key=lambda r: r["pdb"]))
    print(f"Prepared {len(rows)} ClaRNA inputs in {args.output}; {len(failures)} failed")
    for f in failures:
        print("  FAILED", f["pdb"], f["error"])


if __name__ == "__main__":
    main()
