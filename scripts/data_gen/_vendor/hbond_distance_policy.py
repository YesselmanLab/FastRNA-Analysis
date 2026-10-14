"""Policy for which H-bond distances define a base pair's distance score."""

from __future__ import annotations

from collections.abc import Iterable

from .canonical import canonicalize_bp

WC_DISTANCE_CELLS: frozenset[tuple[str, str]] = frozenset({
    ("C-G", "cWW"), ("A-U", "cWW"),
})

MAX_STRONG_DISTANCE = 3.5

MIN_PHYSICAL_DISTANCE = 2.2

BASE_ATOMS: frozenset[str] = frozenset({
    "N1", "N2", "N3", "N4", "N6", "N7", "N9", "O2", "O4", "O6",
})


def is_wc_distance_cell(bp_type: str | None, lw_class: str | None) -> bool:
    cb, _ = canonicalize_bp(bp_type, lw_class)
    return (cb, lw_class) in WC_DISTANCE_CELLS


def distances_to_score(distances: Iterable[float | None],
                       bp_type: str | None, lw_class: str | None) -> list[float]:
    phys = [d for d in distances if d is not None and d >= MIN_PHYSICAL_DISTANCE]
    if not phys:
        return []
    if is_wc_distance_cell(bp_type, lw_class):
        return phys
    strong = [d for d in phys if d <= MAX_STRONG_DISTANCE]
    return [min(strong)] if strong else []


def select_distance_rows(hb_df):
    import pandas as pd

    df = hb_df[
        hb_df["donor_atom"].isin(BASE_ATOMS)
        & hb_df["acceptor_atom"].isin(BASE_ATOMS)
        & (hb_df["distance"] >= MIN_PHYSICAL_DISTANCE)
    ].copy()
    canon_bp = [canonicalize_bp(b, lw)[0] for b, lw in zip(df["bp_type"], df["lw_class"])]
    is_wc = pd.Series(
        [(cb, lw) in WC_DISTANCE_CELLS for cb, lw in zip(canon_bp, df["lw_class"])],
        index=df.index,
    )
    wc_rows = df[is_wc]
    other = df[~is_wc]
    strong_idx = other.groupby(["pdb_id", "pair_idx"])["distance"].idxmin()
    strong = other.loc[strong_idx]
    strong = strong[strong["distance"] <= MAX_STRONG_DISTANCE]
    return pd.concat([wc_rows, strong])


def base_base_rows(hb_df):
    return hb_df[
        hb_df["donor_atom"].isin(BASE_ATOMS)
        & hb_df["acceptor_atom"].isin(BASE_ATOMS)
    ]
