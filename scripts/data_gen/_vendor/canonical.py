"""Canonical orientation for self-reciprocal LW classes."""

from __future__ import annotations

SELF_RECIPROCAL_LW: frozenset[str] = frozenset({
    "cWW", "tWW", "cHH", "tHH", "cSS", "tSS",
})

SIGN_FLIPPING_PAIR_PARAMS: frozenset[str] = frozenset({
    "shear", "stagger", "buckle",
})


def canonicalize_bp(bp_type: str | None, lw_class: str | None) -> tuple[str | None, bool]:
    if bp_type is None or lw_class is None:
        return bp_type, False
    if lw_class not in SELF_RECIPROCAL_LW:
        return bp_type, False
    if "-" not in bp_type:
        return bp_type, False
    left, right = bp_type.split("-", 1)
    if left <= right:
        return bp_type, False
    return f"{right}-{left}", True


def maybe_flip(value: float | None, source: str, param: str, swapped: bool) -> float | None:
    if value is None or not swapped:
        return value
    if source == "pairs" and param in SIGN_FLIPPING_PAIR_PARAMS:
        return -value
    return value
