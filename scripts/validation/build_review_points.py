"""Review-tier score and issue tables behind Figure 4 (data/reference/resolved_again/)."""
import argparse
import json
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "data_gen"))
import _engine  # noqa: E402

RESOLVED = ROOT / "data" / "reference" / "resolved_again"
REDO_PAIRS = ROOT / "data" / "validation" / "pdb_redo_pairs_comparison.json"
CUT = 80.0
ISSUES = ["hbond_angles", "distance", "shear", "opening", "stretch", "stagger", "buckle",
          "incorrect_hbond_count", "propeller", "clash"]


def redo_points(redo):
    pts = pd.read_csv(RESOLVED / "redo_per_structure_points.csv")
    review = redo[redo.orig_score < CUT]
    g = review.groupby("pdb_id").agg(n_review=("orig_score", "size"),
                                     score_o=("orig_score", "mean"),
                                     score_r=("redo_score", "mean"))
    pts = pts.drop(columns=["n_review", "score_o", "score_r"]).join(g, on="pdb_id")
    pts["n_review"] = pts.n_review.fillna(0).astype(int)
    pts[["score_o", "score_r"]] = pts[["score_o", "score_r"]].round(5)
    cols = ["pdb_id", "cluster", "n_res", "n_review", "score_o", "score_r", "suit_o", "suit_r",
            "bang_o", "bang_r", "puck_n_o", "puck_n_r"]
    return pts[cols], review[review.pdb_id.isin(set(pts.pdb_id[pts.n_review > 0]))]


def issue_rows(cohort, unit, before, after):
    return [(cohort, unit, issue, len(before), sum(issue in s for s in before),
             sum(issue in s for s in after)) for issue in ISSUES]


def scored_pairs(engine, pdb):
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as fh:
        out = Path(fh.name)
    run = subprocess.run([engine, *_engine.structure_args(pdb), "--details", "--out", str(out)],
                         capture_output=True, text=True)
    d = json.loads(out.read_text()) if run.returncode == 0 else {}
    out.unlink(missing_ok=True)
    return pdb, [p for p in d.get("pairs") or [] if p.get("score") is not None]


def index_by_chain(pairs, chain):
    idx = {}
    for p in pairs:
        c1, b1, n1 = p["res_id1"].rsplit("-", 2)
        c2, b2, n2 = p["res_id2"].rsplit("-", 2)
        if c1 != chain or c2 != chain:
            continue
        lw = (p.get("classification") or {}).get("lw_class")
        val = (float(p["score"]), {d["issue"] for d in p.get("issue_details") or []})
        idx.setdefault(tuple(sorted([(n1, b1), (n2, b2)])) + (lw,), val)
    return idx


def resolved_rows(engine, workers):
    strict = pd.read_csv(RESOLVED / "resolve_strict.csv")
    entries = sorted(set(strict.worst) | set(strict.best))
    with ThreadPoolExecutor(workers) as ex:
        pairs = dict(ex.map(lambda p: scored_pairs(engine, p), entries))
    rows = []
    for r in strict.itertuples():
        iw = index_by_chain(pairs[r.worst], str(r.cw))
        ib = index_by_chain(pairs[r.best], str(r.cb))
        matched = [(iw[k], ib[k]) for k in iw if k in ib and iw[k][0] < CUT]
        if matched:
            rows += issue_rows("resolved", r.worst, [w[1] for w, _ in matched],
                               [b[1] for _, b in matched])
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default=None)
    ap.add_argument("--out-dir", type=Path, default=RESOLVED)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    engine = _engine.find_engine(args.engine)

    redo = pd.DataFrame(json.loads(REDO_PAIRS.read_text())["pairs"])
    pts, review = redo_points(redo)
    rows = []
    for pdb, g in review.groupby("pdb_id"):
        names = lambda col: [{d["issue"] for d in x} for x in g[col]]
        rows += issue_rows("pdb_redo", pdb, names("orig_issues"), names("redo_issues"))
    rows += resolved_rows(engine, args.workers)
    rates = pd.DataFrame(rows, columns=["cohort", "unit", "issue", "n_review", "n_before", "n_after"])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pts.to_csv(args.out_dir / "redo_per_structure_points.csv", index=False)
    rates.to_csv(args.out_dir / "issue_rates_per_unit.csv", index=False)
    print(f"redo_per_structure_points: {len(pts)} structures, {(pts.n_review > 0).sum()} with "
          f"Review-tier pairs; issue_rates_per_unit: {rates[rates.cohort == 'pdb_redo'].unit.nunique()} "
          f"pdb_redo units, {rates[rates.cohort == 'resolved'].unit.nunique()} resolved units")


if __name__ == "__main__":
    main()
