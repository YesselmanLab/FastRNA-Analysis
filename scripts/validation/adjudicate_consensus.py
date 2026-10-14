"""Consensus adjudication of PARSE-vs-DSSR disagreements across five annotators."""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _vendor import annotators  # noqa: E402
from _vendor.annotators import DisagreementRecord, flip  # noqa: E402

OUT = REPO / "data" / "validation" / "annotator_consensus.json"
VOTING_TOOLS = ("rnaview", "barnaba", "rnapolis", "bpnet")

LOO_DEFINITIONS = {
    "clean": (
        "single-candidate records whose PARSE candidate shares nothing with "
        "{C, flip(C)} — the same population as class_disagreement_populations.clean"
    ),
    "disjoint_candidates": (
        "EVERY PARSE candidate (however many) shares nothing with {C, flip(C)}; "
        "a wider population that also keeps multi-candidate records with no overlap"
    ),
}


def verdict_of(hits: dict) -> str:
    n_parse = sum(1 for hit_parse, _ in hits.values() if hit_parse)
    n_dssr = sum(1 for _, hit_dssr in hits.values() if hit_dssr)
    if not (n_parse or n_dssr):
        return "no_vote"
    if n_parse != n_dssr:
        return "parse" if n_parse > n_dssr else "dssr"
    return "tie"


class Tallies:
    def __init__(self, referees: tuple[str, ...] = ()) -> None:
        self.referees = referees
        self.n_records = 0
        self.degenerate = 0
        self.class_votes = Counter()
        self.loo_votes = Counter()
        self.reject_report = Counter()
        self.ponly_report = Counter()
        self.ponly_exact = Counter()
        self.per_tool = Counter()
        self.local_agree = Counter()

    def add(self, r: DisagreementRecord, found: dict) -> None:
        if r.bucket == "rejected_by_parse":
            n_rep = sum(1 for cs in found.values() if cs)
            self.reject_report[(len(found), n_rep)] += 1
            self._add_reporting("rejected_by_parse", found)
        elif r.bucket == "class_disagreement":
            self._add_class_votes(r, found)
        elif r.source == "python_only":
            self._add_python_only(r, found)

    def _add_reporting(self, bucket: str, found: dict) -> None:
        for tool, classes in found.items():
            self.per_tool[(bucket, tool, "covered")] += 1
            if classes:
                self.per_tool[(bucket, tool, "reports")] += 1

    def _add_class_votes(self, r: DisagreementRecord, found: dict) -> None:
        sides = annotators.vote_sides(r.disputed_class, r.parse_candidates)
        if sides is None:
            self.degenerate += 1
            return
        dssr_side, parse_side = sides
        clean = len(r.parse_candidates) == 1 and not set(r.parse_candidates) & dssr_side
        population = "clean" if clean else "shared_candidate"
        hits = {}
        for tool, classes in found.items():
            self.per_tool[("class_disagreement", tool, "covered")] += 1
            hit_parse, hit_dssr = bool(classes & parse_side), bool(classes & dssr_side)
            hits[tool] = (hit_parse, hit_dssr)
            if hit_parse:
                self.per_tool[("class_disagreement", tool, "votes_parse")] += 1
            if hit_dssr:
                self.per_tool[("class_disagreement", tool, "votes_dssr")] += 1
            if hit_parse and hit_dssr:
                self.per_tool[("class_disagreement", tool, "votes_both")] += 1
        verdict = verdict_of(hits)
        self.class_votes[(population, len(found), verdict)] += 1
        self._add_leave_one_out(r, dssr_side, clean, hits, verdict)

    def _add_leave_one_out(
        self, r: DisagreementRecord, dssr_side: set, clean: bool, hits: dict, verdict: str
    ) -> None:
        disjoint = not set(r.parse_candidates) & dssr_side
        for definition, member in (("clean", clean), ("disjoint_candidates", disjoint)):
            if not member:
                continue
            self.loo_votes[(definition, "none", verdict)] += 1
            for dropped in self.referees:
                remaining = {t: h for t, h in hits.items() if t != dropped}
                self.loo_votes[(definition, dropped, verdict_of(remaining))] += 1

    def _add_python_only(self, r: DisagreementRecord, found: dict) -> None:
        bucket = "novel" if r.bucket == "novel" else "dssr_partial_class"
        n_rep = sum(1 for cs in found.values() if cs)
        self.ponly_report[(bucket, len(found), n_rep)] += 1
        disputed = {r.disputed_class, flip(r.disputed_class)}
        n_exact = sum(1 for cs in found.values() if cs & disputed)
        self.ponly_exact[(bucket, n_exact)] += 1
        self._add_reporting(bucket, found)


def load_available(requested: list[str]) -> dict[str, dict]:
    out = {}
    for tool in requested:
        try:
            out[tool] = annotators.tool_pairs(tool)
        except FileNotFoundError:
            print(f"note: no output file for {tool}; skipped")
    return out


def real_classes(pairs: dict, r: DisagreementRecord) -> set:
    return {c for c in annotators.lookup(pairs, r.res1, r.res2) if not c.startswith("n")}


def fold(counter: Counter, keyfn) -> dict:
    grouped: dict = defaultdict(Counter)
    for key, n in counter.items():
        grouped[keyfn(key)][key[-1]] += n
    return grouped


def run(tools: dict[str, dict], fr3d: dict[str, dict], fr3d_local: dict | None) -> Tallies:
    tallies = Tallies(referees=("fr3d", *tools))
    for r in annotators.disagreement_records():
        tallies.n_records += 1
        panels = {}
        if r.pdb_id in fr3d:
            panels["fr3d"] = fr3d[r.pdb_id]
        for tool, data in tools.items():
            if r.pdb_id in data:
                panels[tool] = data[r.pdb_id]
        if not panels:
            continue
        found = {tool: real_classes(pairs, r) for tool, pairs in panels.items()}
        if fr3d_local is not None and "fr3d" in panels and r.pdb_id in fr3d_local:
            same = real_classes(fr3d_local[r.pdb_id], r) == found["fr3d"]
            tallies.local_agree["same" if same else "different"] += 1
        tallies.add(r, found)
    return tallies


def verdict_block(c: Counter) -> dict:
    return {"n": sum(c.values()), "verdicts": dict(c),
            "parse_dssr_ratio": round(c["parse"] / c["dssr"], 2) if c["dssr"] else None}


def population_blocks(class_votes: Counter) -> dict:
    by_pop = fold(class_votes, lambda k: k[0])
    blocks = {}
    for pop in ("clean", "shared_candidate", "pooled"):
        c = sum(by_pop.values(), Counter()) if pop == "pooled" else by_pop.get(pop, Counter())
        blocks[pop] = verdict_block(c)
    return blocks


def leave_one_out_blocks(loo_votes: Counter) -> dict:
    grouped = fold(loo_votes, lambda k: (k[0], k[1]))
    out = {}
    for definition, note in LOO_DEFINITIONS.items():
        blocks = {
            dropped: verdict_block(c) for (d, dropped), c in grouped.items() if d == definition
        }
        if not blocks:
            continue
        baseline = blocks.pop("none", None)
        ratios = [b["parse_dssr_ratio"] for b in blocks.values() if b["parse_dssr_ratio"]]
        out[definition] = {
            "definition": note,
            "baseline": baseline,
            "dropped": dict(sorted(blocks.items())),
            "parse_dssr_ratio_range": [min(ratios), max(ratios)] if ratios else None,
        }
    return out


def write_summary(tallies: Tallies, tools: dict) -> None:
    summary = {
        "n_records_deduped": tallies.n_records,
        "voting_tools": ["fr3d", *tools],
        "vote_direction_note": (
            "class votes are direction-agnostic: a referee class supports a side if it "
            "equals that side's label or its flip (comparison-file LW labels are not "
            "reliably in res_id1->res_id2 direction); a referee hit on the disputed class "
            "C or flip(C) counts for DSSR; PARSE candidates direction-equivalent to C are "
            "DROPPED from the PARSE side, so a PARSE vote requires a referee hit on a "
            "remaining candidate (a referee hitting both sides counts for both; records "
            "with no remaining candidate are degenerate and excluded)"
        ),
        "provenance_note": (
            "the differences inventory (data/comparison/{class}.json) is a July engine "
            "run kept as an archive-only input, while parse_pairs.json is an August "
            "engine run — two different PARSE snapshots: 24.9% of class-disagreement "
            "pairs are absent from the current engine output and 1,484 rejection records "
            "are now called by the current engine; consensus results here describe the "
            "July snapshot"
        ),
        "class_disagreement_degenerate_excluded": tallies.degenerate,
        "class_disagreement_majority": {
            f"panel_{p}": dict(c) for p, c in fold(tallies.class_votes, lambda k: k[1]).items()
        },
        "class_disagreement_populations": {
            "definition": (
                "clean = single-candidate records whose PARSE candidate shares nothing "
                "with {C, flip(C)}; shared_candidate = ambiguous X|Y candidate sets "
                "(these usually include C or flip(C)); pooled = both"
            ),
            **population_blocks(tallies.class_votes),
            "pooled_ratio_note": (
                "the pooled PARSE:DSSR ratio is a conservative lower bound: a referee "
                "hit on a candidate shared with DSSR's class is scored for DSSR"
            ),
        },
        "class_disagreement_leave_one_referee_out": {
            "note": (
                "robustness of the clean-population class-disagreement verdict to any "
                "single referee: the same tally re-run with each referee dropped from "
                "every panel. The whole roster is swept, so the denominator is identical "
                "across drops. Reported for both clean-population definitions; the "
                "'clean' block is the one the rest of this file uses"
            ),
            **leave_one_out_blocks(tallies.loo_votes),
        },
        "parse_rejections_n_annotators_reporting": {
            f"panel_{p}": dict(c) for p, c in fold(tallies.reject_report, lambda k: k[0]).items()
        },
        "python_only_n_annotators_reporting": {
            f"{b}_panel_{p}": dict(c)
            for (b, p), c in fold(tallies.ponly_report, lambda k: (k[0], k[1])).items()
        },
        "python_only_exact_class_confirmations": {
            b: dict(c) for b, c in fold(tallies.ponly_exact, lambda k: k[0]).items()
        },
        "per_annotator": {
            f"{b}|{t}|{w}": n for (b, t, w), n in sorted(tallies.per_tool.items())
        },
        "fr3d_local_vs_published": dict(tallies.local_agree) if tallies.local_agree else None,
    }
    OUT.write_text(json.dumps(summary, indent=1))


def print_report(tallies: Tallies) -> None:
    print(f"\nrecords: {tallies.n_records}")
    print(f"\nclass disagreements, majority of cast votes (by panel size; "
          f"{tallies.degenerate} degenerate records excluded):")
    for p, c in sorted(fold(tallies.class_votes, lambda k: k[1]).items()):
        tot = sum(c.values())
        cast = tot - c.get("no_vote", 0)
        print(f"  panel of {p} (n={tot}, {cast} with >=1 vote): "
              + ", ".join(f"{v}={n}" for v, n in c.most_common()))
    print("\nclass disagreements by population (clean vs shared-candidate):")
    for pop, block in population_blocks(tallies.class_votes).items():
        v = block["verdicts"]
        print(f"  {pop:<16} n={block['n']}: parse={v.get('parse', 0)}, "
              f"dssr={v.get('dssr', 0)}, ratio={block['parse_dssr_ratio']}")
    print("\nclass disagreements, leave-one-referee-out (PARSE:DSSR ratio):")
    for definition, block in leave_one_out_blocks(tallies.loo_votes).items():
        base = block["baseline"]
        print(f"  {definition} (n={base['n']}): baseline={base['parse_dssr_ratio']}, "
              + ", ".join(f"-{t}={b['parse_dssr_ratio']}" for t, b in block["dropped"].items())
              + f"  range={block['parse_dssr_ratio_range']}")
    print("\nPARSE rejections: how many annotators corroborate DSSR's pair:")
    for p, c in sorted(fold(tallies.reject_report, lambda k: k[0]).items()):
        tot = sum(c.values())
        print(f"  panel of {p} (n={tot}): "
              + ", ".join(f"{k} of {p}: {n} ({100*n/tot:.1f}%)" for k, n in sorted(c.items())))
    print("\npython_only: annotators reporting PARSE's pair:")
    for bp, c in sorted(fold(tallies.ponly_report, lambda k: (k[0], k[1])).items()):
        tot = sum(c.values())
        print(f"  {bp[0]}, panel of {bp[1]} (n={tot}): "
              + ", ".join(f"{k}: {n} ({100*n/tot:.1f}%)" for k, n in sorted(c.items())))
    _print_per_tool(tallies.per_tool)
    if tallies.local_agree:
        tot = sum(tallies.local_agree.values())
        same = tallies.local_agree["same"]
        print(f"\nfr3d_local vs published Hub (non-voting check): "
              f"{same}/{tot} records identical ({100*same/tot:.1f}%)")


def _print_per_tool(per_tool: Counter) -> None:
    print("\nper-annotator rates:")
    for bucket in sorted({b for (b, _t, _w) in per_tool}):
        print(f"  {bucket}:")
        for tool in ("fr3d", *VOTING_TOOLS):
            cov = per_tool.get((bucket, tool, "covered"), 0)
            if not cov:
                continue
            if bucket == "class_disagreement":
                vp = per_tool[(bucket, tool, "votes_parse")]
                vd = per_tool[(bucket, tool, "votes_dssr")]
                vb = per_tool.get((bucket, tool, "votes_both"), 0)
                print(f"    {tool:<8} covered {cov:>6}: votes PARSE {vp} ({100*vp/cov:.1f}%), "
                      f"votes DSSR {vd} ({100*vd/cov:.1f}%), both {vb}")
            else:
                rep = per_tool.get((bucket, tool, "reports"), 0)
                print(f"    {tool:<8} covered {cov:>6}: reports pair {rep} ({100*rep/cov:.1f}%)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tools", default=",".join(VOTING_TOOLS),
                    help="comma-separated voting tools (besides published fr3d)")
    args = ap.parse_args()

    fr3d = annotators.fr3d_pairs()
    tools = load_available([t for t in args.tools.split(",") if t])
    fr3d_local = load_available(["fr3d_local"]).get("fr3d_local")
    print({k: len(v) for k, v in tools.items()}, "structures annotated")

    tallies = run(tools, fr3d, fr3d_local)
    write_summary(tallies, tools)
    print_report(tallies)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
