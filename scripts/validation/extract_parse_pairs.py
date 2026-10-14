"""FastRNA's selected pairs for every structure of the tool comparison."""
import argparse
import json
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "comparison" / "parse_pairs.json"
DEFAULT_BINARY = Path(os.environ.get("PARSE_ENGINE") or shutil.which("parse") or "parse")
DEFAULT_STRUCTURES = ROOT / "data" / "structures"


def split_res(res_id):
    chain, _name, num = res_id.split("-", 2)
    return chain, num


def label(classification):
    lw = classification["lw_class"]
    if classification["is_ambiguous"]:
        return lw.removeprefix("ambiguous (").removesuffix(")")
    return lw


def selected(binary, path):
    run = subprocess.run([str(binary), str(path), "--no-download"],
                         capture_output=True, text=True)
    if run.returncode != 0:
        return {"status": "error", "pairs": []}
    rows = []
    for p in json.loads(run.stdout)["pairs"]:
        c = p["classification"]
        rows.append([*split_res(p["res_id1"]), *split_res(p["res_id2"]),
                     label(c), bool(c["is_ambiguous"])])
    return {"status": "ok", "pairs": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("binary", nargs="?", type=Path, default=DEFAULT_BINARY)
    ap.add_argument("structures", nargs="?", type=Path, default=DEFAULT_STRUCTURES)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    args = ap.parse_args()

    pdbs = sorted(json.loads(OUT.read_text()))
    with ThreadPoolExecutor(args.workers) as ex:
        records = ex.map(lambda p: selected(args.binary, args.structures / f"{p}.cif"), pdbs)
        result = dict(zip(pdbs, records))
    failed = [p for p, r in result.items() if r["status"] != "ok"]
    args.out.write_text(json.dumps(result))
    print(f"wrote {args.out}: {len(result) - len(failed)} ok, {len(failed)} failed {failed[:10]}")


if __name__ == "__main__":
    main()
