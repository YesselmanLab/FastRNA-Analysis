"""Base identity of every residue in the tool-comparison structures."""
import json
import sys
from pathlib import Path

import gemmi

ROOT = Path(__file__).resolve().parents[2]
PAIRS = ROOT / "data" / "comparison" / "parse_pairs.json"
OUT = ROOT / "data" / "comparison" / "residue_names.json"
DEFAULT_STRUCTURES = ROOT / "data" / "structures"


def parent_base(name):
    if name in ("A", "C", "G", "U"):
        return name
    info = gemmi.find_tabulated_residue(name)
    code = info.one_letter_code.upper() if info is not None else ""
    return code if code in ("A", "C", "G", "U") else "?"


def residue_names(path):
    st = gemmi.read_structure(str(path))
    out = {}
    for chain in st[0]:
        for res in chain:
            base = parent_base(res.name)
            if base != "?":
                out[f"{chain.name}|{res.seqid.num}|{res.seqid.icode.strip()}"] = base
    return out


def main():
    structures = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_STRUCTURES
    pdb_ids = sorted(json.loads(PAIRS.read_text()))
    result, missing = {}, []
    for i, pdb in enumerate(pdb_ids, 1):
        path = next((p for p in (structures / f"{pdb}.cif", structures / f"{pdb.lower()}.cif")
                     if p.exists()), None)
        if path is None:
            missing.append(pdb)
            continue
        result[pdb] = residue_names(path)
        if i % 100 == 0:
            print(f"{i}/{len(pdb_ids)}", flush=True)
    if missing:
        sys.exit(f"no structure file for {len(missing)} entries, e.g. {missing[:5]}")
    OUT.write_text(json.dumps(result, separators=(",", ":")))
    print(f"wrote {OUT.relative_to(ROOT)}: {len(result)} structures, "
          f"{sum(map(len, result.values())):,} residues")


if __name__ == "__main__":
    main()
