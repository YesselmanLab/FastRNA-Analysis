"""Time each tool's own command directly on unchanged local mmCIF files."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import random
import re
import signal
import subprocess
import tempfile
import threading
import time
import csv

import benchmark_all as common

ROOT = common.ROOT
HERE = Path(__file__).resolve().parent
STOP = threading.Event()
ACTIVE = {}
ACTIVE_LOCK = threading.Lock()


def public_path(value):
    path = Path(value)
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return path.name


def public_config(config):
    return {key: public_path(value) if isinstance(value, str) and value.startswith("/") else value
            for key, value in config.items()}


def public_requirement(line):
    name, sep, url = line.partition(" @ file://")
    return f"{name} @ {public_path(url)}" if sep else line


def public_command(command, work, config):
    python_bin = Path(config["python"]).parent
    names = {config.get(k): k for k in ("parse", "rnaview", "bpnet") if config.get(k)}
    names.update({config.get("dssr"): "x3dna-dssr", config["python"]: "python",
                  str(python_bin / "barnaba"): "barnaba", str(python_bin / "annotator"): "annotator"})
    if config.get("clarna"):
        names.update({config["clarna_python"]: "python2.7", config["clarna"]: "clarna.py"})
    shown = []
    for part in map(str, command):
        if part in names:
            shown.append(names[part])
        else:
            shown.append(part.replace(str(work), "WORK"))
    return shown


def command_for(tool, cif, work, config):
    python_bin = Path(config["python"]).parent
    if tool.startswith("parse"):
        command = [config["parse"], str(cif), "--no-download", "--cache-dir", str(work)]
        return command + (["--no-score"] if tool == "parse_noscore" else [])
    if tool == "dssr":
        return [config["dssr"], f"-i={cif}", "--pair-only"]
    if tool == "fr3d":
        return [config["python"], "-m", "fr3d.classifiers.NA_pairwise_interactions",
                "-c", "basepair", "-i", str(work), "-o", str(work), cif.name]
    if tool == "rnaview":
        return [config["rnaview"], "--cif", cif.name]
    if tool == "barnaba":
        return [str(python_bin / "barnaba"), "ANNOTATE", "--pdb", str(cif), "-o", "annotation"]
    if tool == "rnapolis":
        return [str(python_bin / "annotator"), str(cif), "--json", "annotation.json"]
    if tool == "bpnet":
        return [config["bpnet"], cif.name]
    if tool == "clarna":
        return [config["clarna_python"], config["clarna"], "-i", cif.name,
                "--normalize-graph", "--save-graph", "graph.json"]
    raise ValueError(tool)


CLARNA_LW = re.compile(r"^([WHS])([WHS])_(cis|tran)$")


def clarna_pairs(graph):
    names = [node["id"] for node in graph["nodes"]]
    pairs = []
    for link in graph["links"]:
        if link.get("type") != "contact" or link.get("is_alt"):
            continue
        match = CLARNA_LW.match(link.get("desc", ""))
        if match:
            lw = ("c" if match[3] == "cis" else "t") + match[1] + match[2]
            pairs.append([names[link["source"]], names[link["target"]], lw])
    return pairs


def validate_output(tool, cif, work, stdout, stderr):
    if tool.startswith("parse"):
        data = json.loads(stdout)
        pairs = data.get("pairs", [])
        if not isinstance(pairs, list) or len(pairs) != data.get("n_pairs"):
            raise ValueError("Incomplete PARSE pair output")
        return pairs
    if tool == "dssr":
        match = re.search(r"total number of base pairs:\s*(\d+)", stdout + stderr)
        if not match:
            raise ValueError("Missing DSSR pair count")
        pairs = [line.split()[1:] for line in stdout.splitlines() if re.match(r"^\s*\d+\s+\S+\s+\S+", line)]
        if len(pairs) != int(match[1]):
            raise ValueError("Incomplete DSSR pair table")
        return pairs
    if tool == "rnapolis":
        data = json.loads((work / "annotation.json").read_text())
        if not isinstance(data.get("base_pairs"), list):
            raise ValueError("Missing RNApolis base-pair list")
        return data["base_pairs"]
    if tool == "rnaview":
        from _annotators import _parse_rnaview_out
        output = (work / f"{cif.name}.out").read_text()
        if "END_base-pair" not in output:
            raise ValueError("Truncated RNAView pair table")
        return _parse_rnaview_out(output, None)
    if tool == "bpnet":
        from _annotators import _parse_bpnet_out
        output = (work / f"{cif.stem}.out").read_text()
        if not output.strip():
            raise ValueError("Empty BPNET output")
        return _parse_bpnet_out(output)
    if tool == "barnaba":
        output = (work / "annotation.ANNOTATE.pairing.out").read_text()
        if "# sequence" not in output or "RES1" not in output:
            raise ValueError("Incomplete barnaba annotation")
        return [line.split() for line in output.splitlines() if line.strip() and not line.startswith("#")]
    if tool == "fr3d":
        path = next((work / f"{cif.stem}_{suffix}.txt" for suffix in ("basepair", "basepairs")
                     if (work / f"{cif.stem}_{suffix}.txt").exists()), None)
        if path is None:
            raise ValueError("Missing FR3D annotation file")
        return [line.split("\t") for line in path.read_text().splitlines() if line.strip()]
    if tool == "clarna":
        if "find contacts .. DONE" not in stdout:
            raise ValueError("ClaRNA did not finish contact classification")
        return clarna_pairs(json.loads((work / "graph.json").read_text()))
    raise ValueError(tool)


def run_trial(tool, row, repeat, args, config, output, reference_dir, source_hash):
    with tempfile.TemporaryDirectory(prefix="rnad-", dir="/tmp") as temp:
        work = Path(temp)
        source = row["clarna_path"] if tool == "clarna" else row["source_path"]
        cif = work / Path(source).name
        cif.symlink_to(source)
        command = command_for(tool, cif, work, config)
        env = {**os.environ, **common.THREAD_ENV, "PYTHONHASHSEED": "0",
               "TMPDIR": str(work), "XDG_CACHE_HOME": str(work / "cache"),
               "MPLCONFIGDIR": str(output / "mpl-cache"),
               "RNAVIEW": str(reference_dir / "rv"),
               "NUCLEIC_ACID_DIR": str(reference_dir / "bp") + "/"}
        started = time.perf_counter()
        proc = subprocess.Popen(command, cwd=work, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, start_new_session=True)
        with ACTIVE_LOCK:
            ACTIVE[id(proc)] = proc
        try:
            stdout, stderr = proc.communicate(timeout=args.timeout)
            elapsed = time.perf_counter() - started
            status = "ok" if proc.returncode == 0 else "error"
        except subprocess.TimeoutExpired as error:
            common.terminate_tree(proc)
            elapsed = time.perf_counter() - started
            stdout, stderr = error.output or b"", error.stderr or b""
            status = "timeout"
        except BaseException:
            common.terminate_tree(proc)
            raise
        finally:
            with ACTIVE_LOCK:
                ACTIVE.pop(id(proc), None)
        if STOP.is_set():
            raise KeyboardInterrupt
        stdout = stdout.decode(errors="replace")
        stderr = stderr.decode(errors="replace")
        result = {"pdb": row["pdb"], "nucleotides": row["nucleotides"], "tool": tool,
                  "repeat": repeat, "warmup": repeat < 0, "wall_seconds": elapsed,
                  "status": status, "returncode": proc.returncode,
                  "command": public_command(command, work, config),
                  "source_sha256": source_hash}
        if status == "ok":
            try:
                pairs = validate_output(tool, cif, work, stdout, stderr)
                records = sorted(json.dumps(p, sort_keys=True, separators=(",", ":")) for p in pairs)
                result.update(pair_count=len(pairs), output_sha256=hashlib.sha256("\n".join(records).encode()).hexdigest())
                if tool == "clarna" and repeat == 0:
                    (output / "clarna_pairs").mkdir(exist_ok=True)
                    (output / "clarna_pairs" / f"{row['pdb']}.json").write_text(json.dumps(pairs))
            except Exception as error:
                result.update(status="invalid_output", error=f"{type(error).__name__}: {error}")
        log = output / "logs" / f"{row['pdb']}_{tool}_{repeat}.log"
        log.write_text("STDOUT\n" + stdout + "\nSTDERR\n" + stderr)
        result["log"] = str(log.relative_to(output))
        return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=ROOT / ".runtime-tools/config.json")
    ap.add_argument("--input", type=Path, default=ROOT / "data/structures",
                    help="folder of <ID>.cif files (scripts/fetch_structures.py)")
    ap.add_argument("--cohort", type=Path, default=ROOT / "data/benchmark/dssr_PARSE_time_benchmark.csv")
    ap.add_argument("--clarna-input", type=Path, default=ROOT / ".runtime-tools/clarna-inputs",
                    help="prepared <ID>.pdb files for ClaRNA (scripts/benchmark/clarna_input.py)")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--tools", default=",".join(common.TOOLS))
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--warmups", type=int, default=0,
                    help="untimed tool runs per structure (default 0: the input is pre-read instead)")
    ap.add_argument("--workers", type=int, default=1, help="structures benchmarked at once")
    ap.add_argument("--timeout", type=float, default=600)
    ap.add_argument("--skip-after-timeout", action="store_true",
                    help="after a tool times out on a structure, record its remaining repeats as skipped")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--pdbs")
    ap.add_argument("--seed", type=int, default=20261008)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    tools = args.tools.split(",")
    if not tools or len(set(tools)) != len(tools) or not set(tools) <= set(common.TOOLS):
        ap.error("Unknown or duplicate tools")
    if args.runs < 1 or args.warmups < 0 or args.limit < 0 or args.timeout <= 0 or args.workers < 1:
        ap.error("Invalid repeat, warm-up, worker, limit, or timeout")
    config = json.loads(args.config.read_text())
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()) and not args.resume:
        ap.error("Output exists; use --resume or a new directory")
    output.mkdir(parents=True, exist_ok=True)
    (output / "logs").mkdir(exist_ok=True)
    with args.cohort.open() as handle:
        rows = [{"pdb": r["PDB"].upper(), "nucleotides": int(r["Nucleotides"])} for r in csv.DictReader(handle)]
    if len({r["pdb"] for r in rows}) != len(rows):
        ap.error("Duplicate cohort IDs")
    if args.pdbs:
        wanted = set(args.pdbs.upper().split(","))
        rows = [r for r in rows if r["pdb"] in wanted]
        if {r["pdb"] for r in rows} != wanted:
            ap.error("Some --pdbs are absent from cohort")
    rows = common.choose_structures(rows, args.limit, args.seed)
    missing = [r["pdb"] for r in rows if not (args.input / f"{r['pdb']}.cif").is_file()]
    if missing:
        ap.error(f"{len(missing)} structures missing from {args.input}, e.g. {missing[:5]}; "
                 "run scripts/fetch_structures.py")
    for row in rows:
        source = (args.input / f"{row['pdb']}.cif").resolve()
        row.update(source_path=str(source), size=source.stat().st_size)
    if "clarna" in tools:
        missing = [r["pdb"] for r in rows if not (args.clarna_input / f"{r['pdb']}.pdb").is_file()]
        if missing:
            ap.error(f"{len(missing)} ClaRNA inputs missing from {args.clarna_input}, e.g. {missing[:5]}; "
                     "run scripts/benchmark/clarna_input.py")
        for row in rows:
            row["clarna_path"] = str((args.clarna_input / f"{row['pdb']}.pdb").resolve())
    manifest = [{"pdb": r["pdb"], "nucleotides": r["nucleotides"],
                 "input": public_path(r["source_path"]), "size": r["size"],
                 **({"clarna_input_sha256": common.sha(r["clarna_path"])} if "clarna_path" in r else {})}
                for r in rows]
    executables = {"python": config["python"]}
    for tool in tools:
        if tool.startswith("parse") or tool in ("dssr", "rnaview", "bpnet"):
            key = "parse" if tool.startswith("parse") else tool
            executables[key] = config[key]
        elif tool in ("barnaba", "rnapolis"):
            executables[tool] = str(Path(config["python"]).parent / ("annotator" if tool == "rnapolis" else tool))
        elif tool == "clarna":
            executables.update(clarna=config["clarna"], clarna_python=config["clarna_python"])
    metadata = {"protocol": "direct_cli_original_mmcif_v2", "tools": tools,
                "runs": args.runs, "warmups": args.warmups, "workers": args.workers,
                "warm_input": "each input is read in full (checksum), untimed, before its first trial",
                "timeout": args.timeout, "seed": args.seed, "limit": args.limit, "pdbs": args.pdbs,
                **({"skip_after_timeout": True} if args.skip_after_timeout else {}),
                "config": public_config(config), "hardware": common.hardware(),
                "platform": common.platform.platform(),
                "thread_environment": common.THREAD_ENV, "cohort_sha256": common.sha(args.cohort),
                "script_hashes": {name: common.sha(HERE / name) for name in
                                  ("benchmark_direct.py", "benchmark_all.py", "_annotators.py")
                                  + (("clarna_input.py",) if "clarna" in tools else ())},
                **({"clarna_input": "PDB format from clarna_input.py (model 1, nucleotides, aliased chains); "
                                    "conversion untimed here, see its conversion.csv",
                    "clarna_packages": subprocess.check_output(
                        [config["clarna_python"], "-m", "pip", "freeze"], text=True,
                        stderr=subprocess.DEVNULL).splitlines()} if "clarna" in tools else {}),
                "executable_hashes": {key: common.sha(value) for key, value in executables.items()},
                "packages": [public_requirement(line) for line in subprocess.check_output(
                    [config["python"], "-m", "pip", "freeze"], text=True,
                    env={**os.environ, "PIP_CACHE_DIR": str(ROOT / ".runtime-tools/pip-cache")}).splitlines()]}
    meta_path = output / "metadata.json"
    if meta_path.exists():
        if json.loads(meta_path.read_text()) != metadata:
            ap.error("Code, environment, or arguments changed; use a new output directory")
        if json.loads((output / "manifest.json").read_text()) != manifest:
            ap.error("Input files/cohort changed; use a new output directory")
    else:
        common.write_json(meta_path, metadata)
        common.write_json(output / "manifest.json", manifest)
    trials_path = output / "trials.jsonl"
    trials = common.read_trials(trials_path)
    done = {(r["pdb"], r["tool"], r["repeat"]) for r in trials}
    if len(done) != len(trials):
        ap.error("Duplicate trial records")
    prior_hashes = {r["pdb"]: r["source_sha256"] for r in trials}
    print(f"Direct CLI: {len(rows)} original mmCIFs; {len(tools)} modes; "
          f"{args.warmups} warm-up + {args.runs} repeats; {args.workers} structure(s) at once. "
          "No model filtering or conversion.", flush=True)
    subprocess.run([config["python"], "-c", "import matplotlib.font_manager"],
                   env={**os.environ, "MPLCONFIGDIR": str(output / "mpl-cache")}, check=False,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    def stopped(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stopped)
    lock = threading.Lock()

    def bench_structure(index, row, handle, reference):
        source_hash = common.sha(row["source_path"])
        if row["pdb"] in prior_hashes and prior_hashes[row["pdb"]] != source_hash:
            raise ValueError(f"Input changed: {row['pdb']}")
        rng = random.Random(f"{args.seed}:{row['pdb']}")
        for repeat in range(-args.warmups, args.runs):
            order = list(tools)
            rng.shuffle(order)
            for tool in order:
                if STOP.is_set():
                    return
                if (row["pdb"], tool, repeat) in done:
                    continue
                if args.skip_after_timeout and any(
                        r["pdb"] == row["pdb"] and r["tool"] == tool and r["status"] == "timeout" for r in trials):
                    result = {"pdb": row["pdb"], "nucleotides": row["nucleotides"], "tool": tool,
                              "repeat": repeat, "warmup": repeat < 0, "wall_seconds": 0.0,
                              "status": "skipped_after_timeout", "source_sha256": source_hash}
                else:
                    result = run_trial(tool, row, repeat, args, config, output, reference, source_hash)
                with lock:
                    handle.write(json.dumps(result, sort_keys=True) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                    trials.append(result)
                    print(f"[{index}/{len(rows)}] {row['pdb']} {tool} run={repeat} "
                          f"{result['status']} {result['wall_seconds']:.3f}s", flush=True)
        with lock:
            common.report(output, trials, manifest, tools, args.runs)

    try:
        with tempfile.TemporaryDirectory(prefix="rnar-", dir="/tmp") as reference, trials_path.open("a") as handle:
            reference = Path(reference)
            (reference / "rv").symlink_to(config["rnaview_home"], target_is_directory=True)
            (reference / "bp").symlink_to(config["bpnet_sysfiles"], target_is_directory=True)
            pool = ThreadPoolExecutor(max_workers=args.workers)
            try:
                futures = [pool.submit(bench_structure, index, row, handle, reference)
                           for index, row in enumerate(rows, 1)]
                for future in as_completed(futures):
                    future.result()
            except BaseException:
                STOP.set()
                with ACTIVE_LOCK:
                    for proc in list(ACTIVE.values()):
                        try:
                            os.killpg(proc.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                raise
            finally:
                pool.shutdown(wait=True, cancel_futures=True)
    finally:
        with lock:
            common.report(output, trials, manifest, tools, args.runs)
    print("Finished:", output, dict(Counter(r["status"] for r in trials)), flush=True)


if __name__ == "__main__":
    main()
