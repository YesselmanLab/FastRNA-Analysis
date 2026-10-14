"""Count nucleotides per structure and write them into the PDB metadata cache."""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import gemmi

REPO = Path(__file__).resolve().parents[2]
STRUCTURES = REPO / "data" / "structures"
METADATA = REPO / "data" / "reference" / "metadata" / "metadata_cache.json"
NUCLEIC = {gemmi.PolymerType.Rna, gemmi.PolymerType.Dna, gemmi.PolymerType.DnaRnaHybrid}


def count(pdb: str) -> tuple[str, int | None]:
    path = STRUCTURES / f"{pdb}.cif"
    if not path.exists():
        return pdb, None
    try:
        st = gemmi.read_structure(str(path))
        st.setup_entities()
        seen = set()
        for chain in st[0]:
            for res in chain:
                if res.entity_type != gemmi.EntityType.Polymer or not res.subchain:
                    continue
                entity = st.get_entity_of(chain.get_subchain(res.subchain))
                if entity is not None and entity.polymer_type in NUCLEIC:
                    seen.add((chain.name, str(res.seqid)))
        return pdb, len(seen)
    except Exception:
        return pdb, None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="report changes, write nothing")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    meta = json.loads(METADATA.read_text())
    counts = meta["nucleotide_counts"]
    with ProcessPoolExecutor(args.workers) as ex:
        results = dict(ex.map(count, list(counts), chunksize=20))

    changed = {p: (counts[p], n) for p, n in results.items() if n is not None and counts.get(p) != n}
    counted = sum(n is not None for n in results.values())
    print(f"{counted:,} structures counted from local files "
          f"({len(counts) - counted:,} without a file keep their count)")
    print(f"{len(changed):,} counts changed; zeros before {sum(counts[p] == 0 for p in results if results[p] is not None)}"
          f" -> after {sum(n == 0 for n in results.values() if n is not None)}")
    for p, (old, new) in sorted(changed.items(), key=lambda kv: kv[1][1] - kv[1][0])[-5:]:
        print(f"  {p}: {old} -> {new}")
    if args.dry_run:
        return
    for p, (_, new) in changed.items():
        counts[p] = new
    METADATA.write_text(json.dumps(meta))
    print(f"wrote {METADATA.relative_to(REPO)}")


if __name__ == "__main__":
    main()
