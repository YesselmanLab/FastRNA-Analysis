"""Score the example base pairs shown in Figure 3 and record their flagged properties.

Runs `parse --details` on each structure and writes, for every example pair, its LW class,
score and the severity of each flagged property. Figure 3B reads the scores and Figure 3C
the severities (points lost = penalty weight x severity).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import _engine
from _engine import run_parse

REPO = Path(__file__).resolve().parent.parent.parent
DEFAULT_OUT = REPO / "data" / "reference" / "figure3" / "example_pairs.json"

# (panel, role, PDB entry, residue 1, residue 2)
EXAMPLES = [
    ("A", "opening -38.6", "6Y2L", "S2-A-1293", "S2-U-1306"),
    ("A", "opening -0.4", "9I78", "L1-A-2851", "L1-U-2868"),
    ("A", "opening +27.3", "6YSI", "1-A-2780", "1-U-2794"),
    ("B", "G-C ideal", "9AXU", "2-G-2456", "2-C-2467"),
    ("B", "G-C distorted", "4V2S", "Q-G-11", "Q-C-41"),
    ("B", "A-U ideal", "7UNW", "A-A-42", "A-U-428"),
    ("B", "A-U distorted", "7D6Z", "f-A-81", "f-U-88"),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--engine", default=None, help="path to `parse` (else PARSE_ENGINE / PATH)")
    args = ap.parse_args()

    engine = _engine.find_engine(args.engine)
    out = []
    for panel, role, pdb, r1, r2 in EXAMPLES:
        d = run_parse(pdb, engine)
        hit = [p for p in d.get("pairs", []) if {p["res_id1"], p["res_id2"]} == {r1, r2}]
        if len(hit) != 1:
            raise SystemExit(f"{pdb} {r1}-{r2}: expected one pair, found {len(hit)}")
        p = hit[0]
        out.append({"panel": panel, "role": role, "pdb_id": pdb, "res_id1": r1, "res_id2": r2,
                    "lw_class": (p.get("classification") or {}).get("lw_class"),
                    "score": p.get("score"),
                    "severity": {i["issue"]: i["severity"] for i in p.get("issue_details") or []}})
        print(f"{panel} {role:15s} {pdb} {r1}-{r2}: score {p.get('score')}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"engine": Path(engine).name, "pairs": out}, indent=1) + "\n")
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
