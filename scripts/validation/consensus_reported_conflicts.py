"""Referee consensus on FastRNA-DSSR class conflicts, restricted to pairs FastRNA actually reports."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import adjudicate_consensus as ac  # noqa: E402
from _vendor import annotators  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
PUBLISHED = ROOT / "data" / "validation" / "annotator_consensus.json"
OUT = ROOT / "data" / "validation" / "annotator_consensus_reported.json"


def summary(tallies, tools, dest):
    ac.OUT = dest
    ac.write_summary(tallies, tools)
    return json.loads(dest.read_text())


def main():
    fr3d = annotators.fr3d_pairs()
    tools = ac.load_available(list(ac.VOTING_TOOLS))
    scratch = OUT.with_suffix(".check.json")

    full = summary(ac.run(tools, fr3d, None), tools, scratch)
    scratch.unlink()
    published = json.loads(PUBLISHED.read_text())
    if {k: v for k, v in full.items() if k != "fr3d_local_vs_published"} != \
            {k: v for k, v in published.items() if k != "fr3d_local_vs_published"}:
        sys.exit("full run does not reproduce annotator_consensus.json; not writing")
    print("full run reproduces annotator_consensus.json")

    reported = annotators.parse_pairs()
    original = annotators.disagreement_records
    dropped = {"n": 0}

    def reported_only():
        for r in original():
            if (r.bucket == "class_disagreement" and r.pdb_id in reported
                    and not annotators.lookup(reported[r.pdb_id], r.res1, r.res2)):
                dropped["n"] += 1
                continue
            yield r

    published_sides = annotators.vote_sides

    def vote_sides(disputed, candidates):
        if len(candidates) > 1:
            dssr = {disputed, annotators.flip(disputed)}
            return dssr, {v for c in candidates for v in (c, annotators.flip(c))}
        return published_sides(disputed, candidates)

    annotators.disagreement_records = reported_only
    annotators.vote_sides = vote_sides
    try:
        out = summary(ac.run(tools, fr3d, None), tools, OUT)
    finally:
        annotators.disagreement_records = original
        annotators.vote_sides = published_sides

    out["provenance_note"] = (
        "same comparison inventory and referee files as annotator_consensus.json (the "
        "inventory was re-run with the current C++ engine on 2026-10-01 and is "
        "identical row for row); class conflicts restricted to pairs FastRNA reports "
        f"(present in comparison/parse_pairs.json): {dropped['n']:,} unreported-candidate "
        "records removed; ambiguous FastRNA calls credited when a referee reports any "
        "of FastRNA's candidates (a candidate shared with DSSR credits both); other "
        "buckets unchanged")
    out["class_disagreement_unreported_removed"] = dropped["n"]
    OUT.write_text(json.dumps(out, indent=1))
    pops = out["class_disagreement_populations"]
    for p in ("clean", "shared_candidate", "pooled"):
        print(f"  {p:17s} {pops[p]['verdicts']}")
    print(f"removed {dropped['n']:,} unreported class-conflict records; wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
