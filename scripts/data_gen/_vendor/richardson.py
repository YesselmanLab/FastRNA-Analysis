"""Richardson 2008 RNA backbone suite classification."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_SUITES_PATH = Path(__file__).resolve().parent / "richardson_suites.json"

OUTLIER_CONFORMER = "!!"

SUITE_ANGLE_NAMES = (
    "delta_prev", "epsilon_prev", "zeta_prev",
    "alpha", "beta", "gamma", "delta",
)


@dataclass
class SuiteResult:
    conformer: str
    suiteness: float
    bin: str | None
    is_outlier: bool
    distance: float | None
    issue: str | None


def _angle_diff(a: float, b: float) -> float:
    d = abs(a - b)
    return min(d, 360.0 - d)


def signed_angle_gap(value: float, center: float) -> float:
    return (value - center + 180.0) % 360.0 - 180.0


def _l3_distance(suite: np.ndarray, center: np.ndarray, widths: np.ndarray,
                 indices: range) -> float:
    total = 0.0
    for k in indices:
        d = _angle_diff(suite[k], center[k])
        total += (d / widths[k]) ** 3
    return total ** (1.0 / 3.0)


class RichardsonClassifier:
    def __init__(self, suites_path: Path | str = DEFAULT_SUITES_PATH):
        path = Path(suites_path)
        if not path.exists():
            raise FileNotFoundError(f"Richardson suites JSON not found: {path}")

        with open(path) as f:
            self._raw = json.load(f)

        self._by_bin: dict[str, list[dict]] = {}
        for conf in self._raw["conformers"]:
            self._by_bin.setdefault(conf["bin"], []).append(conf)

        self._normal_widths = np.array(self._raw["widths"]["normal"], dtype=float)
        self._satellite_widths = np.array(
            self._raw["widths"].get("satellite", self._raw["widths"]["normal"]),
            dtype=float,
        )
        self._sat_overrides: dict[str, dict] = self._raw.get("satellite_overrides", {})
        self._triage = self._raw["triage"]
        self._sieve = self._raw["sieve"]

        self._centers: dict[str, np.ndarray] = {
            conf["name"]: np.array(conf["angles"], dtype=float)
            for conf in self._raw["conformers"]
        }

    @property
    def normal_widths(self) -> np.ndarray:
        return self._normal_widths

    def center_for(self, conformer: str) -> np.ndarray | None:
        return self._centers.get(conformer)

    def nearest_conformer(self, suite_angles: np.ndarray | list[float]) -> str | None:
        suite = np.asarray(suite_angles, dtype=float)
        best_name, best_d = None, float("inf")
        for name, center in self._centers.items():
            d = sum(_angle_diff(suite[k], center[k]) for k in range(7))
            if d < best_d:
                best_d, best_name = d, name
        return best_name

    def _classify_pucker(self, delta: float) -> int:
        c3 = self._sieve["delta"]["C3_endo"]
        c2 = self._sieve["delta"]["C2_endo"]
        if c3[0] <= delta <= c3[1]:
            return 3
        if c2[0] <= delta <= c2[1]:
            return 2
        return 0

    def pucker(self, delta: float | None) -> int | None:
        if delta is None:
            return None
        return self._classify_pucker(float(delta) % 360)

    def _classify_gamma(self, gamma: float) -> str:
        for name, rng in self._sieve["gamma"].items():
            if rng[0] <= gamma <= rng[1]:
                return name
        return ""

    @staticmethod
    def _compose_widths(override: list[float], baseline: np.ndarray) -> np.ndarray:
        result = baseline.copy()
        for k, w in enumerate(override):
            if w > 0:
                result[k] = w
        return result

    def _widths_for_satellite(self, name: str) -> np.ndarray:
        ov = self._sat_overrides.get(name)
        if not ov:
            return self._satellite_widths
        return self._compose_widths(ov["sat_widths"], self._satellite_widths)

    def _widths_for_dominant(self, suite: np.ndarray,
                             dom_center: np.ndarray,
                             bin_confs: list[dict]) -> np.ndarray:
        best_d = float("inf")
        best_widths: np.ndarray | None = None
        for sib in bin_confs:
            if sib.get("dominance") != "sat":
                continue
            ov = self._sat_overrides.get(sib["name"])
            if not ov:
                continue
            widths = self._compose_widths(ov["dom_widths"], self._normal_widths)
            d = _l3_distance(suite, dom_center, widths, range(1, 5))
            if d < best_d:
                best_d = d
                best_widths = widths
        return best_widths if best_widths is not None else self._normal_widths

    def _resolve_widths(self, conf: dict, bin_confs: list[dict],
                        suite: np.ndarray) -> np.ndarray:
        dominance = conf.get("dominance", "ord")
        if dominance == "sat":
            return self._widths_for_satellite(conf["name"])
        if dominance == "dom":
            center = np.asarray(conf["angles"], dtype=float)
            return self._widths_for_dominant(suite, center, bin_confs)
        return self._normal_widths

    def classify(self, suite_angles: np.ndarray | list[float]) -> SuiteResult:
        suite = np.asarray(suite_angles, dtype=float)
        if suite.shape != (7,):
            raise ValueError(
                f"suite_angles must have shape (7,), got {suite.shape}"
            )

        for idx, name in [(1, "epsilon"), (3, "alpha"), (4, "beta"), (2, "zeta")]:
            lo, hi = self._triage[name]
            if not (lo <= suite[idx] <= hi):
                return SuiteResult(
                    conformer=OUTLIER_CONFORMER, suiteness=0.0, bin=None,
                    is_outlier=True, distance=None,
                    issue=f"{name}={suite[idx]:.1f} outside [{lo}, {hi}]",
                )

        pucker_dm = self._classify_pucker(suite[0])
        pucker_d = self._classify_pucker(suite[6])
        gamma_class = self._classify_gamma(suite[5])

        if pucker_dm == 0 or pucker_d == 0 or gamma_class == "":
            return SuiteResult(
                conformer=OUTLIER_CONFORMER, suiteness=0.0, bin=None,
                is_outlier=True, distance=None,
                issue=(f"sieve fail: pucker_dm={pucker_dm} pucker_d={pucker_d} "
                       f"gamma='{gamma_class}'"),
            )

        bin_key = f"{pucker_dm}{pucker_d}{gamma_class}"
        conformers = self._by_bin.get(bin_key, [])
        if not conformers:
            return SuiteResult(
                conformer=OUTLIER_CONFORMER, suiteness=0.0, bin=bin_key,
                is_outlier=True, distance=None,
                issue=f"no conformers in bin {bin_key}",
            )

        best_d4 = float("inf")
        best_conf = None
        best_widths: np.ndarray | None = None
        for conf in conformers:
            center = np.array(conf["angles"])
            widths = self._resolve_widths(conf, conformers, suite)
            d4 = _l3_distance(suite, center, widths, range(1, 5))
            if d4 < best_d4:
                best_d4 = d4
                best_conf = conf
                best_widths = widths

        if best_conf is None or best_d4 >= 1.0:
            return SuiteResult(
                conformer=OUTLIER_CONFORMER, suiteness=0.0, bin=bin_key,
                is_outlier=True, distance=None,
                issue=f"no 4D match (best d4={best_d4:.2f})",
            )

        center = np.array(best_conf["angles"])
        d7 = _l3_distance(suite, center, best_widths, range(7))
        if d7 > 1.0:
            return SuiteResult(
                conformer=OUTLIER_CONFORMER, suiteness=0.0, bin=bin_key,
                is_outlier=True, distance=round(d7, 3),
                issue=f"7D reject {best_conf['name']} (d7={d7:.2f})",
            )

        suiteness = (math.cos(math.pi * d7) + 1.0) / 2.0
        if suiteness < 0.01:
            suiteness = 0.01

        return SuiteResult(
            conformer=best_conf["name"],
            suiteness=round(suiteness, 3),
            bin=bin_key,
            is_outlier=False,
            distance=round(d7, 3),
            issue=None,
        )


def classify_chi(chi: float | None) -> str | None:
    if chi is None:
        return None
    if chi <= -90.0 or chi >= 90.0:
        return "anti"
    return "syn"
