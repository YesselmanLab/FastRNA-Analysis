"""Shared by the consensus scripts so they are self-contained."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from pathlib import Path as _Path
from types import SimpleNamespace as _NS

_DATA = _Path(__file__).resolve().parents[3] / "data"


def _data_file(rel: str) -> _Path:
    path = _DATA / rel
    if not path.exists():
        raise FileNotFoundError(path)
    return path


paths = _NS(data_file=_data_file)
loaders = _NS(LW_COMPARISON_CLASSES=("cWW", "tWW", "cWH", "tWH", "cWS", "tWS",
                                     "cHH", "tHH", "cHS", "tHS", "cSS", "tSS"))

UNCLASSIFIED = "?"

JSON_TOOLS = ("rnaview", "barnaba", "rnapolis", "bpnet", "fr3d_local")

REJECTION_REASONS = frozenset(
    {"no_strong_hbond", "stacking", "not_candidate", "not_selected", "adjacent"}
)

_NUMBER = re.compile(r"^(-?\d+)([A-Za-z]?)$")


@dataclass(frozen=True, order=True)
class ResidueId:
    chain: str
    number: int
    icode: str = ""


PairKey = tuple[ResidueId, ResidueId]

PairSets = dict[PairKey, set[str]]


def flip(lw: str) -> str:
    if lw.startswith("n"):
        return "n" + flip(lw[1:])
    if len(lw) != 3:
        return lw
    return lw[0] + lw[2] + lw[1]


def flip_label(lw: str) -> str:
    if lw == UNCLASSIFIED:
        return lw
    return "|".join(flip(part) for part in lw.split("|"))


def residue(chain: str, number: str, icode: str = "") -> ResidueId:
    match = _NUMBER.match(number)
    if not chain or match is None:
        raise ValueError(f"uncanonicalizable residue id: chain={chain!r} number={number!r}")
    appended = match.group(2)
    if appended and icode:
        raise ValueError(f"two insertion codes for chain={chain!r} number={number!r}: "
                         f"{appended!r} and {icode!r}")
    icode = icode or appended
    if icode and not icode.isalpha():
        raise ValueError(f"malformed insertion code {icode!r} for chain={chain!r} "
                         f"number={number!r}")
    return ResidueId(chain, int(match.group(1)), icode)


def parse_res_id(res_id: str) -> ResidueId:
    parts = res_id.split("-")
    if len(parts) < 3:
        raise ValueError(f"uncanonicalizable PARSE res_id: {res_id!r}")
    return residue(parts[0], "-".join(parts[2:]))


def number_string(rid: ResidueId) -> str:
    return f"{rid.number}{rid.icode}"


def pair_entry(res1: ResidueId, res2: ResidueId, label: str) -> tuple[PairKey, str]:
    if res2 < res1:
        return (res2, res1), flip_label(label)
    return (res1, res2), label


def lookup(pairs: PairSets, res1: ResidueId, res2: ResidueId) -> set[str]:
    if res2 < res1:
        return {flip_label(lw) for lw in pairs.get((res2, res1), set())}
    return set(pairs.get((res1, res2), set()))


def _ingest(rows: Iterator[tuple[ResidueId, ResidueId, str]]) -> PairSets:
    pairs: PairSets = defaultdict(set)
    for res1, res2, label in rows:
        key, canonical_label = pair_entry(res1, res2, label)
        pairs[key].add(canonical_label)
    return dict(pairs)


def fr3d_pairs() -> dict[str, PairSets]:
    out: dict[str, PairSets] = {}
    for csv in sorted(paths.data_file("comparison/fr3d").glob("[0-9A-Z]*.csv")):
        out[csv.stem] = _ingest(_hub_rows(csv.name, csv.read_text(errors="replace")))
    return out


def _hub_rows(name: str, text: str) -> Iterator[tuple[ResidueId, ResidueId, str]]:
    for line in text.splitlines():
        if not line.strip():
            continue
        fields = [part.strip().strip('"') for part in line.split('","')]
        if len(fields) != 3:
            raise ValueError(f"{name}: malformed Hub row {line!r}")
        res1, res2 = _hub_unit(name, fields[0]), _hub_unit(name, fields[2])
        if res1 is not None and res2 is not None:
            yield res1, res2, fields[1]


def _hub_unit(name: str, unit: str) -> ResidueId | None:
    fields = unit.split("|")
    if len(fields) < 5:
        raise ValueError(f"{name}: malformed Hub unit id {unit!r}")
    if fields[1] != "1":
        return None
    icode = fields[7] if len(fields) >= 8 else ""
    return residue(fields[2], fields[4], icode)


def tool_pairs(tool: str) -> dict[str, PairSets]:
    if tool not in JSON_TOOLS:
        raise ValueError(f"Unknown annotator {tool!r}; expected one of {JSON_TOOLS}")
    with paths.data_file(f"comparison/{tool}_pairs.json").open() as handle:
        raw = json.load(handle)
    out: dict[str, PairSets] = {}
    for pdb, record in raw.items():
        if record["status"] != "ok":
            continue
        out[pdb] = _ingest(
            (residue(c1, n1), residue(c2, n2), lw or UNCLASSIFIED)
            for c1, n1, c2, n2, lw in record["pairs"]
        )
    return out


def parse_pairs() -> dict[str, PairSets]:
    with paths.data_file("comparison/parse_pairs.json").open() as handle:
        raw = json.load(handle)
    out: dict[str, PairSets] = {}
    for pdb, record in raw.items():
        if record["status"] != "ok":
            continue
        out[pdb] = _ingest(_parse_rows(pdb, record["pairs"]))
    return out


def _parse_rows(pdb: str, rows: list[Any]) -> Iterator[tuple[ResidueId, ResidueId, str]]:
    for c1, n1, c2, n2, lw, ambiguous in rows:
        if bool(ambiguous) != ("|" in (lw or "")):
            raise ValueError(f"{pdb}: is_ambiguous={ambiguous!r} contradicts label {lw!r}")
        yield residue(c1, n1), residue(c2, n2), lw or UNCLASSIFIED


def vote_sides(disputed: str, candidates: tuple[str, ...]) -> tuple[set[str], set[str]] | None:
    dssr = {disputed, flip(disputed)}
    parse = {v for cand in candidates if cand not in dssr for v in (cand, flip(cand))}
    if not parse:
        return None
    return dssr, parse


@dataclass(frozen=True)
class DisagreementRecord:
    pdb_id: str
    res1: ResidueId
    res2: ResidueId
    source: str
    reason: str
    bucket: str
    disputed_class: str
    parse_candidates: tuple[str, ...]


def _bucket(reason: str) -> str:
    if reason in REJECTION_REASONS:
        return "rejected_by_parse"
    if reason.startswith("python_assigns:"):
        return "class_disagreement"
    if reason.startswith("dssr_assigns:"):
        return "dssr_partial_class"
    if reason == "novel":
        return "novel"
    return "other"


def disagreement_records() -> Iterator[DisagreementRecord]:
    seen: dict[tuple[str, frozenset[str], str], str] = {}
    for lw_class in loaders.LW_COMPARISON_CLASSES:
        with paths.data_file(f"comparison/{lw_class}.json").open() as handle:
            differences = json.load(handle)["differences"]
        for row in differences:
            key = (row["pdb_id"], frozenset((row["res_id1"], row["res_id2"])), row["source"])
            if key in seen:
                if seen[key] != lw_class:
                    raise ValueError(f"dedup collision across class files "
                                     f"{seen[key]}/{lw_class}: {key}")
                continue
            seen[key] = lw_class
            yield _record(row, lw_class)


def _record(row: dict[str, Any], lw_class: str) -> DisagreementRecord:
    reason = row.get("reason") or "?"
    candidates: tuple[str, ...] = ()
    if reason.startswith("python_assigns:"):
        candidates = tuple(reason.split(":", 1)[1].split("|"))
    return DisagreementRecord(
        pdb_id=row["pdb_id"].upper()[:4],
        res1=parse_res_id(row["res_id1"]),
        res2=parse_res_id(row["res_id2"]),
        source=row["source"],
        reason=reason,
        bucket=_bucket(reason),
        disputed_class=lw_class,
        parse_candidates=candidates,
    )
