"""Build per-(bp_type, edge_type, parameter) ProSco lookup tables."""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
from _vendor.canonical import (  # noqa: E402
    SELF_RECIPROCAL_LW, SIGN_FLIPPING_PAIR_PARAMS,
)
from _vendor.hbond_distance_policy import (  # noqa: E402
    select_distance_rows, base_base_rows,
)

DEFAULT_FEATURES_DIR = REPO_ROOT / "data" / "reference" / "high_quality_features"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "reference" / "scoring_tables" / "prosco_distributions.json"

BANDWIDTH_FACTOR = 1.5
BIN_MIN = 50
BIN_MAX = 2048
SUPPORT_EXPANSION = 1.0

PARAMETERS = {
    "pairs":  ["shear", "stretch", "stagger", "buckle", "propeller", "opening"],
    "hbonds": ["distance", "donor_angle", "acceptor_angle"],
}


def canonicalize_dataframe(df: pd.DataFrame, source: str) -> pd.DataFrame:
    if "lw_class" not in df.columns or "bp_type" not in df.columns:
        return df
    df = df.copy()
    bp = df["bp_type"].astype(str)
    parts = bp.str.split("-", n=1, expand=True)
    needs_swap = (
        df["lw_class"].isin(SELF_RECIPROCAL_LW)
        & bp.str.contains("-", regex=False)
        & (parts[0] > parts[1])
    )
    n = int(needs_swap.sum())
    if n == 0:
        return df
    print(f"  [canonicalize/{source}] swapped {n:,} rows into canonical orientation")
    df.loc[needs_swap, "bp_type"] = parts.loc[needs_swap, 1] + "-" + parts.loc[needs_swap, 0]
    if source == "pairs":
        for p in SIGN_FLIPPING_PAIR_PARAMS:
            if p in df.columns:
                df.loc[needs_swap, p] = -df.loc[needs_swap, p]
    return df


def per_pdb_weights(pdb_ids: pd.Series) -> np.ndarray:
    counts = pdb_ids.value_counts()
    return np.array([1.0 / counts[p] for p in pdb_ids])


def build_prosco_cell(values: np.ndarray, weights: np.ndarray,
                      pdb_ids: pd.Series) -> dict:
    mask = ~np.isnan(values)
    values = values[mask]
    weights = weights[mask]
    pdb_ids_masked = pdb_ids[mask]

    base = {
        "n_observations": int(values.size),
        "n_effective": round(float(weights.sum()), 2) if values.size else 0.0,
        "n_pdbs": int(pdb_ids_masked.nunique()),
        "support_lo": None,
        "support_hi": None,
        "n_bins": None,
        "prosco_per_bin": None,
        "weighted_median": None,
    }
    if values.size < 10:
        return base

    if np.ptp(values) < 1e-9:
        constant = float(values[0])
        eps = 1e-6
        base["support_lo"] = round(constant - eps, 6)
        base["support_hi"] = round(constant + eps, 6)
        base["n_bins"] = BIN_MIN
        base["prosco_per_bin"] = [100.0] * BIN_MIN
        base["weighted_median"] = round(constant, 4)
        return base

    try:
        kde = gaussian_kde(values, weights=weights)
        kde.set_bandwidth(kde.factor * BANDWIDTH_FACTOR)
    except Exception as e:
        print(f"    KDE failed ({type(e).__name__}: {e}); leaving cell empty")
        return base

    vmin, vmax = float(values.min()), float(values.max())
    dr = vmax - vmin
    lo = vmin - SUPPORT_EXPANSION * dr
    hi = vmax + SUPPORT_EXPANSION * dr

    bandwidth = float(np.sqrt(kde.covariance.item()))
    support_range = hi - lo
    n_bins = int(np.ceil(2.0 * support_range / bandwidth))
    n_bins = max(BIN_MIN, min(BIN_MAX, n_bins))

    edges = np.linspace(lo, hi, n_bins + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    bin_width = edges[1] - edges[0]
    dens = kde(centres)
    mass = dens * bin_width

    order_desc = np.argsort(-dens)
    cum_through_higher = np.cumsum(mass[order_desc])
    cum_excluding_self = np.concatenate([[0.0], cum_through_higher[:-1]])
    prosco_sorted = 100.0 * (1.0 - cum_excluding_self)
    prosco_per_bin = np.empty_like(prosco_sorted)
    prosco_per_bin[order_desc] = prosco_sorted

    order = np.argsort(values)
    v, w = values[order], weights[order]
    cum = np.cumsum(w)
    median = float(np.interp(0.5 * cum[-1], cum - w / 2.0, v))

    base["support_lo"] = round(lo, 6)
    base["support_hi"] = round(hi, 6)
    base["n_bins"] = n_bins
    base["prosco_per_bin"] = [round(float(p), 2) for p in prosco_per_bin]
    base["weighted_median"] = round(median, 4)
    return base


def build_for_dataframe(df: pd.DataFrame, parameters: list[str]) -> dict:
    out: dict[str, dict[str, dict]] = {p: {} for p in parameters}

    print(f"  Primary cells (bp_type × lw_class) ...")
    n_cells = 0
    for (bp_type, lw_class), group in df.groupby(["bp_type", "lw_class"]):
        weights = per_pdb_weights(group["pdb_id"])
        key = f"{bp_type}/{lw_class}"
        for p in parameters:
            out[p][key] = build_prosco_cell(
                group[p].to_numpy(dtype=float),
                weights,
                group["pdb_id"].reset_index(drop=True),
            )
        n_cells += 1
    print(f"    {n_cells} primary cells")

    print(f"  _ANY/<edge_type> pools ...")
    for lw_class, group in df.groupby("lw_class"):
        weights = per_pdb_weights(group["pdb_id"])
        key = f"_ANY/{lw_class}"
        for p in parameters:
            out[p][key] = build_prosco_cell(
                group[p].to_numpy(dtype=float),
                weights,
                group["pdb_id"].reset_index(drop=True),
            )

    print(f"  _ANY/_ANY global pool ...")
    weights = per_pdb_weights(df["pdb_id"])
    for p in parameters:
        out[p]["_ANY/_ANY"] = build_prosco_cell(
            df[p].to_numpy(dtype=float),
            weights,
            df["pdb_id"].reset_index(drop=True),
        )

    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features-dir", type=Path, default=DEFAULT_FEATURES_DIR)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = ap.parse_args()

    print(f"Reading feature tables from {args.features_dir}/")
    pairs_df = pd.read_csv(args.features_dir / "pairs.csv")
    hbonds_df = pd.read_csv(args.features_dir / "hbonds.csv")
    print(f"  pairs:  {len(pairs_df):,} rows")
    print(f"  hbonds: {len(hbonds_df):,} rows")

    print("\nCanonicalizing self-reciprocal LW orientations...")
    pairs_df = canonicalize_dataframe(pairs_df, "pairs")
    hbonds_df = canonicalize_dataframe(hbonds_df, "hbonds")

    print("\nBuilding ProSco tables for pairs ...")
    pairs_prosco = build_for_dataframe(pairs_df, PARAMETERS["pairs"])
    print("\nBuilding ProSco tables for hbonds ...")
    dist_df = select_distance_rows(hbonds_df)
    angle_df = base_base_rows(hbonds_df)
    hbonds_prosco = build_for_dataframe(dist_df, ["distance"])
    hbonds_prosco.update(build_for_dataframe(angle_df, ["donor_angle", "acceptor_angle"]))

    output = {
        "schema_version": "1.0",
        "generated_at": datetime.date.today().isoformat(),
        "method": {
            "prosco": (
                "ProSco(x) = 100 × (1 − cum_mass_in_higher_density_bins(x)). "
                "Per cell: weighted KDE (Gaussian, bandwidth × 1.5), "
                "equal-width bins over [data_min − 1×range, data_max + 1×range]. "
                f"n_bins per cell = ⌈2 × support_range / bandwidth⌉ "
                f"clamped to [{BIN_MIN}, {BIN_MAX}] (Nyquist sampling rate; "
                "smallest discretisation that resolves smoothing-scale features). "
                "Per Černý et al. 2026 (NAR gkaf1335)."
            ),
            "weighting": (
                "Per-PDB downweighting: each observation weight = 1 / (same-PDB "
                "count in cell). n_effective ≤ number of distinct PDBs in the cell."
            ),
            "fallback_hierarchy": [
                "<bp_type>/<edge_type>", "_ANY/<edge_type>", "_ANY/_ANY",
            ],
        },
        "prosco_distributions": {
            "pairs":  pairs_prosco,
            "hbonds": hbonds_prosco,
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False)
    print(f"\nWrote {args.output}")
    size_mb = args.output.stat().st_size / 1e6
    print(f"  file size: {size_mb:.1f} MB")
    for source, params in output["prosco_distributions"].items():
        n_cells = len(next(iter(params.values()))) if params else 0
        print(f"  ProSco {source:8s}: {len(params)} params × {n_cells} cells")


if __name__ == "__main__":
    main()
