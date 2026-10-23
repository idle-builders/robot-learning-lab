"""One unscored CUDA stop visit and one disposable PPO update for V4."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[3]
ART = ROOT / "artifacts/ape-low-level-training-20261014/locomotion-fast-v4"
METHOD = ART / "method.json"
EVIDENCE = Path("/workspace/robot-learning-lab/ape-low-level-training-20261014/locomotion-fast-v4/sentinel")
CHECKER = ROOT / "experiments/compact_morphology/cove50_low_level/check_locomotion_fast_v4.py"
TRAINER = ROOT / "experiments/compact_morphology/cove50_policy_transfer/train_velocity.py"
EVALUATOR = ROOT / "experiments/compact_morphology/mjlab_ape/evaluate.py"
SOURCE = ROOT / "artifacts/ape-cove50-recovery-20261008/ankleonly100-model844.pt"
BODY = ROOT / "artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml"
SENTINEL = 48097


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run(label: str, command: list[str], limit_s: int) -> None:
    log = EVIDENCE / f"{label}.log"
    with log.open("x") as output:
        result = subprocess.run(command, cwd=ROOT, stdout=output,
                                stderr=subprocess.STDOUT, timeout=limit_s, check=False)
    if result.returncode:
        raise RuntimeError(f"{label} exited {result.returncode}; inspect {log}")


def main() -> None:
    if not Path("/workspace").is_mount() or EVIDENCE.exists():
        raise RuntimeError("V4 sentinel needs fresh mounted persistent evidence")
    EVIDENCE.mkdir(parents=True)
    method = json.loads(METHOD.read_text())
    if SENTINEL in {row["seed"] for row in method["development_cases"]}:
        raise RuntimeError("Sentinel overlaps V4 development")
    run("source-runtime-check", [sys.executable, str(CHECKER), "--runtime"], 90)
    report = EVIDENCE / "stop-visit.json"
    trace = EVIDENCE / "stop-visit.npz"
    run("stop-visit", [sys.executable, str(EVALUATOR), str(SOURCE),
                       "--task", "Mjlab-Velocity-Flat-Ape-Elbows",
                       "--body-xml", str(BODY), "--physics-model", "frame50",
                       "--initial-root-z", "0.231", "--yaw-only-head", "--arms-down",
                       "--worlds", "1", "--seed", str(SENTINEL),
                       "--velocity-x", "0.35", "--velocity-y", "0", "--angular-z", "0",
                       "--motor-rail-v", "12", "--command-delay-ms", "0",
                       "--ankle-target-margin-rad", "0.02", "--physics-dt", "0.002",
                       "--stop-at-s", "4.0", "--duration-s", "4.04",
                       "--out", str(report), "--diagnostic-trace", str(trace)], 300)
    visit = json.loads(report.read_text())
    world = visit["worlds"][0]
    if (visit["seed"] != SENTINEL or visit["body_sha256"] != sha(BODY)
            or visit["stop_at_s"] != 4.0 or not 4 <= world["duration_s"] <= 4.04
            or world.get("stop", {}).get("command_time_s") != 4.0
            or len(visit["worlds"]) != 1 or not trace.is_file()):
        raise RuntimeError("CUDA stop visit or trace differs from sentinel contract")
    import numpy as np
    with np.load(trace, allow_pickle=False) as saved:
        native_steps = len(saved["time_s"])
        if (native_steps < 200 or not np.isfinite(saved["raw_action"]).all()
                or not np.isfinite(saved["actuator_force"]).all()):
            raise RuntimeError("Sentinel trace nonfinite or ended before stop")
    resume_dir = ROOT / "logs/rsl_rl/ape_cove50_velocity/v4-sentinel-source"
    resume_dir.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(SOURCE, resume_dir / "model_844.pt")
    before = {p.name for p in resume_dir.parent.iterdir() if p.is_dir()}
    run("one-update", [sys.executable, str(TRAINER), "--body-xml", str(BODY),
                       "--profile", "stop", "--worlds", "512", "--updates", "1",
                       "--save-interval", "1", "--run-name", "v4-disposable-one-update",
                       "--resume-run", "v4-sentinel-source", "--resume-checkpoint", "model_844.pt",
                       "--ankle-target-margin-rad", "0.02",
                       "--walking-hip-roll-pose-std", "0.05"], 900)
    new = [p for p in resume_dir.parent.iterdir() if p.is_dir() and p.name not in before]
    if len(new) != 1 or not (new[0] / "model_844.pt").is_file():
        raise RuntimeError("One-update checkpoint missing or ambiguous")
    checkpoint = new[0] / "model_844.pt"
    import torch
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if state.get("iter") != 844 or sha(checkpoint) == sha(SOURCE):
        raise RuntimeError("One-update saved model did not advance from source")
    target = EVIDENCE / "one-update-model_844.pt"
    shutil.copyfile(checkpoint, target)
    result = {"schema": "robot-learning-lab.locomotion-fast-v4.cuda-sentinel.v1",
              "at_utc": datetime.now(timezone.utc).isoformat(),
              "method_sha256": sha(METHOD), "source_checkpoint_sha256": sha(SOURCE),
              "body_sha256": sha(BODY), "sentinel_seed": SENTINEL,
              "native_steps": native_steps,
              "stop_command_time_s": world["stop"]["command_time_s"],
              "stop_visit_report_sha256": sha(report), "stop_visit_trace_sha256": sha(trace),
              "disposable_optimizer_updates": 1,
              "disposable_checkpoint_sha256": sha(target),
              "candidate_optimizer_updates": 0,
              "claim_limit": "One unscored CUDA stop visit and one disposable PPO update; no candidate training or score."}
    (EVIDENCE / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
