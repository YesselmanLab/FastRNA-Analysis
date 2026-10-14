"""Time-reversal (regression-to-the-mean) control, recomputed at the paper's threshold."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _rtm import N_BOOT, SEED, paired_control, permutation_null  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def load_redo_pairs() -> pd.DataFrame:
    raw = json.loads((ROOT / "data/validation/pdb_redo_pairs_comparison.json").read_text())
    return pd.DataFrame(raw["pairs"])[["pdb_id", "orig_score", "redo_score"]].dropna()


if __name__ == "__main__":
    pairs = load_redo_pairs()
    out = {
        "cohort": "PDB-REDO (same measurements, re-refined)",
        "source": "data/validation/pdb_redo_pairs_comparison.json",
        "unit": "structure",
        "n_pairs_total": int(len(pairs)),
        "n_units_total": int(pairs.pdb_id.nunique()),
        "estimator": "per-unit means, paired on units present in both directions; "
                     f"bootstrap over units, {N_BOOT} resamples, seed {SEED}",
        "thresholds": {},
    }
    for CUT in (80.0, 75.0):
        blk = paired_control(pairs, "pdb_id", "orig_score", "redo_score", CUT)
        rng = np.random.default_rng(SEED)
        blk["pooled_pair_weighted"] = {
            "forward": float((pairs[pairs.orig_score < CUT].redo_score
                              - pairs[pairs.orig_score < CUT].orig_score).mean()),
            "reverse": float((pairs[pairs.redo_score < CUT].orig_score
                              - pairs[pairs.redo_score < CUT].redo_score).mean()),
        }
        blk["n_flagged_in_both_models"] = int(
            ((pairs.orig_score < CUT) & (pairs.redo_score < CUT)).sum())
        if CUT == 80.0:
            blk["permutation_null"] = permutation_null(pairs, "pdb_id", "orig_score",
                                                       "redo_score", CUT)
        out["thresholds"][f"{CUT:g}"] = blk
    dest = ROOT / "data/validation/time_reversal_controls.json"
    dest.write_text(json.dumps(out, indent=1) + "\n")
    for cut, blk in out["thresholds"].items():
        a = blk["asymmetry"]
        print(f"cut {cut:>2}  units {blk['n_units']:4d}  "
              f"forward {blk['forward']['mean']:+6.2f} "
              f"[{blk['forward']['ci95'][0]:+.2f}, {blk['forward']['ci95'][1]:+.2f}]  "
              f"reverse {blk['reverse']['mean']:+6.2f} "
              f"[{blk['reverse']['ci95'][0]:+.2f}, {blk['reverse']['ci95'][1]:+.2f}]  "
              f"asym {a['mean']:+6.2f} [{a['ci95'][0]:+.2f}, {a['ci95'][1]:+.2f}] P={a['p_le_0']:.3f}")
    print("wrote", dest.relative_to(ROOT))
