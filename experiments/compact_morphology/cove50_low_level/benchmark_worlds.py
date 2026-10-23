"""Run bounded scratch PPO throughput trials on the actual Cove velocity task.

Each world count uses the same body/task and a fresh actor. Results measure
capacity and training speed, not whether a useful policy was learned.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import time


ROOT = Path(__file__).resolve().parents[3]
TRAIN = ROOT / "experiments/compact_morphology/cove50_policy_transfer/train_velocity.py"
BODY = ROOT / "artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml"
ANSI = re.compile(r"\x1b\[[0-9;]*m")
ITERATION = re.compile(r"Learning iteration\s+(\d+)/\d+")
METRIC = re.compile(r"(Steps per second|Collection time|Learning time):\s+([0-9.]+)")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def gpu_reading() -> tuple[float, float] | None:
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu",
             "--format=csv,noheader,nounits"], text=True, timeout=5,
            stderr=subprocess.DEVNULL).splitlines()[0]
        memory_mib, utilization = (float(value.strip()) for value in output.split(",")[:2])
        return memory_mib, utilization
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired,
            IndexError, ValueError):
        return None


def parse_iterations(raw: str) -> list[dict]:
    text = ANSI.sub("", raw)
    rows = []
    current = None
    for line in text.splitlines():
        match = ITERATION.search(line)
        if match:
            current = {"iteration": int(match.group(1))}
            rows.append(current)
            continue
        match = METRIC.search(line)
        if match and current is not None:
            key = {"Steps per second": "steps_per_second",
                   "Collection time": "collection_seconds",
                   "Learning time": "update_seconds"}[match.group(1)]
            current[key] = float(match.group(2))
    return rows


def run_one(args: argparse.Namespace, worlds: int, output: Path) -> dict:
    run_name = f"throughput-{worlds}-{int(time.time())}"
    command = [str(args.python), str(TRAIN), "--body-xml", str(args.body),
               "--profile", args.profile, "--worlds", str(worlds),
               "--updates", str(args.updates), "--run-name", run_name]
    log = output / f"{worlds}-worlds.log"
    row = {"worlds": worlds, "run_name": run_name, "command": command,
           "log": str(log), "started_utc": utc_now(), "status": "running"}
    if args.dry_run:
        row["status"] = "dry_run"
        return row
    started = time.monotonic()
    max_memory_mib = 0.0
    utilization_samples = []
    with log.open("w") as stream:
        process = subprocess.Popen(command, cwd=ROOT, stdout=stream,
                                   stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        try:
            while process.poll() is None:
                reading = gpu_reading()
                if reading is not None:
                    memory_mib, utilization = reading
                    max_memory_mib = max(max_memory_mib, memory_mib)
                    utilization_samples.append(utilization)
                if time.monotonic() - started >= args.timeout_minutes * 60:
                    process.terminate()
                    try:
                        process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    row["status"] = "timeout"
                    break
                time.sleep(2)
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait()
    row["exit_code"] = process.returncode
    row["finished_utc"] = utc_now()
    row["wall_seconds"] = round(time.monotonic() - started, 3)
    row["peak_gpu_memory_mib"] = max_memory_mib or None
    row["mean_sampled_gpu_utilization_percent"] = (
        round(sum(utilization_samples) / len(utilization_samples), 2)
        if utilization_samples else None)
    iterations = parse_iterations(log.read_text(errors="replace"))
    measured = [item for item in iterations if item["iteration"] >= args.warmup
                and all(key in item for key in ("steps_per_second", "collection_seconds",
                                                 "update_seconds"))]
    row["completed_iterations"] = len(iterations)
    row["warmup_iterations_excluded"] = args.warmup
    row["measured_iterations"] = len(measured)
    row["rollout_steps_per_world"] = 24
    row["measured_environment_steps"] = len(measured) * worlds * 24
    if measured:
        total = sum(item["collection_seconds"] + item["update_seconds"] for item in measured)
        row["steady_steps_per_second"] = round(row["measured_environment_steps"] / total, 2)
        row["mean_collection_seconds"] = round(sum(item["collection_seconds"] for item in measured) / len(measured), 3)
        row["mean_update_seconds"] = round(sum(item["update_seconds"] for item in measured) / len(measured), 3)
    if row["status"] == "running":
        row["status"] = "measured" if process.returncode == 0 and len(measured) == args.updates - args.warmup else "failed"
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, default=Path("/root/microduck_rl/.venv/bin/python"))
    parser.add_argument("--body", type=Path, default=BODY)
    parser.add_argument("--profile", choices=("forward", "backward", "planar", "turn", "stop"), default="forward")
    parser.add_argument("--worlds", type=int, nargs="+", default=[256, 512, 1024, 2048, 4096])
    parser.add_argument("--updates", type=int, default=13)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--timeout-minutes", type=float, default=20)
    parser.add_argument("--total-minutes", type=float, default=70)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.updates <= args.warmup or min(args.worlds) <= 0:
        parser.error("updates must exceed warmup and worlds must be positive")
    args.body = args.body.resolve()
    if not args.body.is_file():
        parser.error(f"missing body: {args.body}")
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"schema": "robot-learning-lab.cove50-gpu-throughput.v1",
              "scope": "fresh scratch PPO capacity/speed, not skill evaluation",
              "body": str(args.body), "profile": args.profile,
              "started_utc": utc_now(), "requested_worlds": args.worlds,
              "updates_per_count": args.updates, "warmup_updates": args.warmup,
              "trials": []}
    started = time.monotonic()
    for worlds in args.worlds:
        if not args.dry_run and time.monotonic() - started >= args.total_minutes * 60:
            report["stop_reason"] = "total_time_cap"
            break
        row = run_one(args, worlds, args.output)
        report["trials"].append(row)
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        if row["status"] not in ("measured", "dry_run"):
            report["stop_reason"] = row["status"]
            break
    report["finished_utc"] = utc_now()
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
