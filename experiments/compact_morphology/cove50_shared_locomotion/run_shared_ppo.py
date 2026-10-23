"""Benchmark two many-world capacities, then run one bounded shared-gait PPO fit."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import torch


ROOT = Path(__file__).resolve().parents[3]
ART = ROOT / "artifacts/ape-low-level-training-20261015/shared-forward-turn"
METHOD = ART / "method.json"
BASE = Path("/workspace/robot-learning-lab/ape-low-level-training-20261015/shared-forward-turn/bc/bc-best.pt")
BC_REPORT = BASE.parent / "report.json"
OUT = Path("/workspace/robot-learning-lab/ape-low-level-training-20261015/shared-forward-turn/ppo")
TRAINER = ROOT / "experiments/compact_morphology/cove50_policy_transfer/train_velocity.py"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_branch(label: str, worlds: int, updates: int, body: Path) -> dict:
    source_run = ROOT / "logs/rsl_rl/ape_cove50_velocity" / f"shared-forward-turn-v1-{label}-source"
    source_run.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(BASE, source_run / "model_844.pt")
    parent = source_run.parent
    before = {p.name for p in parent.iterdir() if p.is_dir()}
    command = [sys.executable, str(TRAINER), "--body-xml", str(body),
               "--profile", "shared_forward_turn", "--worlds", str(worlds),
               "--updates", str(updates), "--save-interval", str(updates),
               "--run-name", f"shared-forward-turn-v1-{label}",
               "--resume-run", source_run.name, "--resume-checkpoint", "model_844.pt",
               "--ankle-target-margin-rad", "0.02"]
    start = time.monotonic()
    log = OUT / f"{label}.log"
    with log.open("x") as stream:
        proc = subprocess.run(command, cwd=ROOT, stdout=stream,
                              stderr=subprocess.STDOUT, timeout=3600, check=False)
    elapsed = time.monotonic() - start
    new = [p for p in parent.iterdir() if p.is_dir() and p.name not in before]
    checkpoint = None
    if proc.returncode == 0 and len(new) == 1:
        candidates = sorted(new[0].glob("model_*.pt"),
                            key=lambda p: int(p.stem.split("_")[-1]))
        if candidates:
            checkpoint = candidates[-1]
    result = {"label": label, "worlds": worlds, "requested_updates": updates,
              "exit_code": proc.returncode, "elapsed_s": round(elapsed, 3),
              "nominal_transitions": worlds * 24 * updates,
              "nominal_transitions_per_s": round(worlds * 24 * updates / elapsed, 1),
              "log": str(log), "log_sha256": sha(log),
              "checkpoint": str(checkpoint) if checkpoint else None,
              "checkpoint_sha256": sha(checkpoint) if checkpoint else None}
    if checkpoint:
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        result["saved_iteration"] = saved.get("iter")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("benchmark", "train"), required=True)
    args = parser.parse_args()
    method = json.loads(METHOD.read_text())
    for name, item in method["sources"].items():
        path = ROOT / item["path"]
        if not path.is_file() or sha(path) != item["sha256"]:
            raise RuntimeError(f"Frozen source mismatch: {name}")
    if sha(TRAINER) != method["sources"]["ppo_trainer"]["sha256"]:
        raise RuntimeError("Shared trainer changed")
    bc = json.loads(BC_REPORT.read_text())
    if (bc["method_sha256"] != sha(METHOD)
            or bc["candidate_checkpoint_sha256"] != sha(BASE)):
        raise RuntimeError("BC warm start identity mismatch")
    body = ROOT / method["sources"]["body"]["path"]
    phase_dir = OUT / args.phase
    if phase_dir.exists():
        raise RuntimeError(f"Original PPO phase already exists: {phase_dir}")
    OUT.mkdir(parents=True, exist_ok=True)
    phase_dir.mkdir()
    if args.phase == "benchmark":
        rows = []
        for worlds in method["ppo_world_benchmark"]:
            row = run_branch(f"bench-{worlds}", worlds, 1, body)
            rows.append(row)
            (phase_dir / "partial.json").write_text(json.dumps(rows, indent=2) + "\n")
            print(json.dumps(row), flush=True)
        passed = [row for row in rows
                  if row["exit_code"] == 0 and row.get("saved_iteration", 0) >= 845]
        if not passed:
            chosen = None
        else:
            chosen = max(passed, key=lambda row: row["nominal_transitions_per_s"])["worlds"]
        result = {"schema": "robot-learning-lab.shared-forward-turn.ppo-benchmark.v1",
                  "method_sha256": sha(METHOD), "bc_checkpoint_sha256": sha(BASE),
                  "rows": rows, "chosen_worlds": chosen,
                  "claim_limit": "One-update CUDA throughput/capacity checks, not skill training evidence."}
        (phase_dir / "report.json").write_text(json.dumps(result, indent=2) + "\n")
        if chosen is None:
            raise RuntimeError("Neither many-world capacity passed")
    else:
        benchmark = json.loads((OUT / "benchmark/report.json").read_text())
        if (benchmark["method_sha256"] != sha(METHOD)
                or benchmark["bc_checkpoint_sha256"] != sha(BASE)
                or benchmark["chosen_worlds"] not in method["ppo_world_benchmark"]):
            raise RuntimeError("Benchmark identity or choice mismatch")
        row = run_branch("learn", benchmark["chosen_worlds"],
                         method["ppo_updates"], body)
        if row["exit_code"] or row.get("saved_iteration", 0) < 844 + method["ppo_updates"]:
            (phase_dir / "failed.json").write_text(json.dumps(row, indent=2) + "\n")
            raise RuntimeError("Bounded PPO fit did not produce requested checkpoint")
        candidate = phase_dir / "candidate-final.pt"
        shutil.copyfile(row["checkpoint"], candidate)
        result = {"schema": "robot-learning-lab.shared-forward-turn.ppo-fit.v1",
                  "at_utc": datetime.now(timezone.utc).isoformat(),
                  "method_sha256": sha(METHOD), "bc_checkpoint_sha256": sha(BASE),
                  "worlds": row["worlds"], "optimizer_updates": method["ppo_updates"],
                  "nominal_transitions": row["nominal_transitions"],
                  "elapsed_s": row["elapsed_s"], "saved_iteration": row["saved_iteration"],
                  "candidate_checkpoint_sha256": sha(candidate), "source_row": row,
                  "claim_limit": "One bounded shared-command PPO fit; fresh native physical replay still required."}
        (phase_dir / "report.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({"worlds": row["worlds"], "updates": method["ppo_updates"],
                          "checkpoint_sha256": sha(candidate)}), flush=True)


if __name__ == "__main__":
    main()
