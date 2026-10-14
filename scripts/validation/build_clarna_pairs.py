"""Build data/comparison/clarna_pairs.json for the 1,183 disagreement structures."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/benchmark"))
from benchmark_direct import clarna_pairs  # noqa: E402
import benchmark_all as common  # noqa: E402


def flip(lw):
    return lw[0] + lw[2] + lw[1]


def mapped_records(raw, mapping):
    records = set()
    for a, b, lw in raw:
        (c1, n1), (c2, n2) = mapping[a], mapping[b]
        records.add((c1, n1, c2, n2, lw))
    out = set()
    for c1, n1, c2, n2, lw in records:
        if (c2, n2, c1, n1, flip(lw)) in out:
            continue
        out.add((c1, n1, c2, n2, lw))
    return sorted(map(list, out))


def run_long(pdb, args, config):
    with tempfile.TemporaryDirectory(prefix="rnac-", dir="/tmp") as temp:
        work = Path(temp)
        (work / f"{pdb}.pdb").symlink_to((args.clarna_input / f"{pdb}.pdb").resolve())
        command = [config["clarna_python"], config["clarna"], "-i", f"{pdb}.pdb",
                   "--normalize-graph", "--save-graph", "graph.json"]
        try:
            proc = subprocess.run(command, cwd=work, capture_output=True, text=True, timeout=args.timeout,
                                  env={**os.environ, **common.THREAD_ENV, "TMPDIR": str(work)})
        except subprocess.TimeoutExpired:
            return pdb, "timeout", None
        if proc.returncode or "find contacts .. DONE" not in proc.stdout:
            return pdb, "error", None
        return pdb, "ok", clarna_pairs(json.loads((work / "graph.json").read_text()))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmark", type=Path, default=ROOT / "data/benchmark/runtime_direct_clarna")
    ap.add_argument("--clarna-input", type=Path, default=ROOT / ".runtime-tools/clarna-inputs")
    ap.add_argument("--structures", type=Path, default=ROOT / "data/comparison/rnaview_pairs.json",
                    help="any *_pairs.json; its keys are the structure set")
    ap.add_argument("--output", type=Path, default=ROOT / "data/comparison/clarna_pairs.json")
    ap.add_argument("--config", type=Path, default=ROOT / ".runtime-tools/config.json")
    ap.add_argument("--run-missing", action="store_true",
                    help="annotate structures without a timed repeat-0 result, untimed")
    ap.add_argument("--timeout", type=float, default=6 * 3600)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    pdbs = sorted(json.loads(args.structures.read_text()))
    trials = common.read_trials(args.benchmark / "trials.jsonl")
    status = {r["pdb"]: r["status"] for r in trials if r["tool"] == "clarna" and r["repeat"] == 0}
    previous = json.loads(args.output.read_text()) if args.output.exists() else {}
    result, todo = {}, []
    for pdb in pdbs:
        raw_path = args.benchmark / "clarna_pairs" / f"{pdb}.json"
        mapping = json.loads((args.clarna_input / f"{pdb}.map.json").read_text())
        if raw_path.exists():
            result[pdb] = {"status": "ok", "pairs": mapped_records(json.loads(raw_path.read_text()), mapping)}
        elif previous.get(pdb, {}).get("via") == "untimed_long_run" and previous[pdb]["status"] == "ok":
            result[pdb] = previous[pdb]
        else:
            result[pdb] = {"status": status.get(pdb, "missing"), "pairs": []}
            todo.append(pdb)
    if args.run_missing and todo:
        config = json.loads(args.config.read_text())
        print(f"Annotating {len(todo)} structures without a timed result (limit {args.timeout:.0f} s)", flush=True)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for pdb, state, raw in pool.map(lambda p: run_long(p, args, config), todo):
                mapping = json.loads((args.clarna_input / f"{pdb}.map.json").read_text())
                result[pdb] = {"status": state, "via": "untimed_long_run",
                               "pairs": mapped_records(raw, mapping) if raw is not None else []}
                print(pdb, state, flush=True)
                common.write_json(args.output, result)
    common.write_json(args.output, result)
    print(args.output, dict(Counter(v["status"] for v in result.values())))


if __name__ == "__main__":
    main()
