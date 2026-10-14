"""How many of the other six annotators report each pair, for every annotator."""
import json
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
data_file = lambda rel: ROOT / "data" / rel
_json = lambda rel: json.loads(data_file(rel).read_text())
OUT = data_file("validation/tool_support_profile.json")

MATRIX_TOOLS = ["parse", "dssr", "fr3d", "rnaview", "barnaba", "rnapolis", "bpnet"]
MATRIX_LABELS = ["FastRNA", "DSSR", "FR3D", "RNAView", "barnaba", "RNApolis", "BPNET"]
JSON_PAIR_FILES = {
    "parse": "comparison/parse_pairs.json",
    "dssr": "comparison/dssr_pairs.json",
    "rnaview": "comparison/rnaview_pairs.json",
    "barnaba": "comparison/barnaba_pairs.json",
    "rnapolis": "comparison/rnapolis_pairs.json",
    "bpnet": "comparison/bpnet_pairs.json",
}
_NUM = re.compile(r"^(-?\d+)([A-Za-z]?)$")
UNCLASSIFIED = "?"
WATSON_CRICK = {frozenset("GC"), frozenset("AU")}


def _residue(chain, number, icode=""):
    m = _NUM.match(str(number))
    if not chain or m is None:
        raise ValueError(f"uncanonicalizable residue id: {chain!r} {number!r}")
    appended = m.group(2)
    if appended and icode:
        raise ValueError(f"two insertion codes: {chain!r} {number!r} {icode!r}")
    return (str(chain), int(m.group(1)), icode or appended)


def _flip(lw):
    if lw == UNCLASSIFIED:
        return lw
    if lw.startswith("n"):
        return "n" + _flip(lw[1:])
    if len(lw) != 3:
        return lw
    return lw[0] + lw[2] + lw[1]


def _flip_label(label):
    return "|".join(_flip(part) for part in label.split("|"))


def _pair_entry(res1, res2, label):
    if res2 < res1:
        return (res2, res1), _flip_label(label)
    return (res1, res2), label


def _adjacent(res1, res2):
    return res1[0] == res2[0] and abs(res1[1] - res2[1]) == 1


def _ingest(rows):
    pairs = {}
    for res1, res2, label in rows:
        if _adjacent(res1, res2):
            continue
        key, canonical_label = _pair_entry(res1, res2, label)
        pairs.setdefault(key, set()).add(canonical_label)
    return pairs


def _load_json_pairs(tool):
    out = {}
    for pdb, record in _json(JSON_PAIR_FILES[tool]).items():
        if record.get("status") != "ok":
            continue
        out[pdb] = _ingest((_residue(c1, n1), _residue(c2, n2), lw or UNCLASSIFIED)
                           for c1, n1, c2, n2, lw, *_ in record["pairs"])
    return out


def _hub_unit(unit):
    fields = unit.split("|")
    if len(fields) < 5 or fields[1] != "1":
        return None
    icode = fields[7] if len(fields) >= 8 else ""
    return _residue(fields[2], fields[4], icode)


def _load_fr3d_pairs():
    out = {}
    for csv_path in sorted(data_file("comparison/fr3d").glob("[0-9A-Z]*.csv")):
        rows = []
        for line in csv_path.read_text(errors="replace").splitlines():
            if not line.strip():
                continue
            fields = [part.strip().strip('"') for part in line.split('","')]
            if len(fields) != 3:
                continue
            res1, res2 = _hub_unit(fields[0]), _hub_unit(fields[2])
            if res1 is not None and res2 is not None:
                rows.append((res1, res2, fields[1]))
        out[csv_path.stem] = _ingest(rows)
    return out


def _load_all_pairsets():
    pairsets = {tool: _load_json_pairs(tool) for tool in JSON_PAIR_FILES}
    pairsets["fr3d"] = _load_fr3d_pairs()
    return pairsets


@lru_cache(maxsize=1)
def _residue_names():
    return _json("comparison/residue_names.json")


def _is_canonical(pdb, key, labels):
    names = _residue_names().get(pdb, {})
    bases = [names.get(f"{chain}|{num}|{icode}") for chain, num, icode in key]
    if None in bases or frozenset(bases) not in WATSON_CRICK:
        return False
    return any(part == "cWW" for lw in labels for part in lw.split("|"))


def main():
    pairsets = _load_all_pairsets()
    common = sorted(set.intersection(*(set(pairsets[t]) for t in MATRIX_TOOLS)))
    tools = {}
    for tool, label in zip(MATRIX_TOOLS, MATRIX_LABELS):
        others = [t for t in MATRIX_TOOLS if t != tool]
        counts = {"canonical": Counter(), "non-canonical": Counter()}
        for pdb in common:
            for key, labels in pairsets[tool][pdb].items():
                reporters = [t for t in others if key in pairsets[t][pdb]]
                seen = set(labels).union(*(pairsets[t][pdb][key] for t in reporters))
                kind = "canonical" if _is_canonical(pdb, key, seen) else "non-canonical"
                counts[kind][len(reporters)] += 1
        tools[tool] = {"label": label,
                       **{kind: {str(k): c[k] for k in range(len(others) + 1)}
                          for kind, c in counts.items()}}
    OUT.write_text(json.dumps({"n_structures": len(common), "tools": tools}, indent=1))
    print(f"{len(common):,} structures -> {OUT.relative_to(ROOT)}")
    for tool, d in tools.items():
        nc = d["non-canonical"]
        n = sum(nc.values())
        print(f"  {d['label']:9s} non-canonical {n:8,d}   unsupported {100 * nc['0'] / n:5.1f}%   "
              f">=3 others {100 * sum(nc[str(k)] for k in range(3, 7)) / n:5.1f}%")


if __name__ == "__main__":
    main()
