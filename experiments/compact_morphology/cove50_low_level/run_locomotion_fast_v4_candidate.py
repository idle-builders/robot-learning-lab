"""One bounded V4 stop-profile continuation and paired development replays."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[3]
ART = ROOT / "artifacts/ape-low-level-training-20261014/locomotion-fast-v4"
OUT = Path("/workspace/robot-learning-lab/ape-low-level-training-20261014/locomotion-fast-v4/candidate")
SOURCE = ROOT / "artifacts/ape-cove50-recovery-20261008/ankleonly100-model844.pt"
BODY = ROOT / "artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml"
TRAINER = ROOT / "experiments/compact_morphology/cove50_policy_transfer/train_velocity.py"
EVALUATOR = ROOT / "experiments/compact_morphology/mjlab_ape/evaluate.py"
CHECKER = ROOT / "experiments/compact_morphology/cove50_low_level/check_locomotion_fast_v4.py"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def run(name: str, args: list[str], seconds: int) -> None:
    with (OUT / f"{name}.log").open("x") as log:
        proc = subprocess.run(args, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                              timeout=seconds, check=False)
    if proc.returncode:
        raise RuntimeError(f"{name} exited {proc.returncode}")


def main() -> None:
    if not Path("/workspace").is_mount() or OUT.exists():
        raise RuntimeError("Fresh persistent V4 candidate path required")
    method = json.loads((ART / "method.json").read_text())
    if (method["schema"] != "robot-learning-lab.cove50-locomotion-fast-v4.method.v1"
            or len(method["development_cases"]) != 8
            or method["learner"]["additional_updates"] != 24):
        raise RuntimeError("V4 frozen candidate method differs")
    OUT.mkdir(parents=True)
    run("source-runtime-check", [sys.executable, str(CHECKER), "--runtime"], 90)
    if sha(SOURCE) != method["source"]["source_checkpoint"]["sha256"]:
        raise RuntimeError("Source checkpoint changed")
    source_run = ROOT / "logs/rsl_rl/ape_cove50_velocity/v4-candidate-source"
    source_run.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(SOURCE, source_run / "model_844.pt")
    before = {p.name for p in source_run.parent.iterdir() if p.is_dir()}
    run("train", [sys.executable, str(TRAINER), "--body-xml", str(BODY),
                  "--profile", "stop", "--worlds", "512", "--updates", "24",
                  "--save-interval", "8", "--run-name", "v4-candidate-24update",
                  "--resume-run", "v4-candidate-source", "--resume-checkpoint", "model_844.pt",
                  "--ankle-target-margin-rad", "0.02",
                  "--walking-hip-roll-pose-std", "0.05"], 1200)
    new = [p for p in source_run.parent.iterdir() if p.is_dir() and p.name not in before]
    if len(new) != 1:
        raise RuntimeError("Candidate run directory missing or ambiguous")
    checkpoints = sorted(new[0].glob("model_*.pt"),
                         key=lambda p: int(p.stem.split("_")[-1]))
    if not checkpoints or int(checkpoints[-1].stem.split("_")[-1]) < 867:
        raise RuntimeError("No final 24-update checkpoint")
    candidate = OUT / "candidate-final.pt"
    shutil.copyfile(checkpoints[-1], candidate)
    import torch
    saved = torch.load(candidate, map_location="cpu", weights_only=False)
    if saved.get("iter", 0) < 867 or sha(candidate) == sha(SOURCE):
        raise RuntimeError("Candidate optimizer lineage did not advance 24 updates")
    for case in method["development_cases"]:
        for arm, checkpoint in (("baseline", SOURCE), ("candidate", candidate)):
            label = f"seed{case['seed']}-{arm}"
            report = OUT / f"{label}.json"
            trace = OUT / f"{label}.npz"
            run(label, [sys.executable, str(EVALUATOR), str(checkpoint),
                        "--task", "Mjlab-Velocity-Flat-Ape-Elbows",
                        "--body-xml", str(BODY), "--physics-model", "frame50",
                        "--initial-root-z", str(case["initial_root_z_m"]),
                        "--yaw-only-head", "--arms-down", "--worlds", "1",
                        "--seed", str(case["seed"]),
                        "--velocity-x", str(case["velocity_x_m_s"]),
                        "--velocity-y", "0", "--angular-z", "0",
                        "--motor-rail-v", str(case["motor_rail_v"]),
                        "--command-delay-ms", str(case["command_delay_ms"]),
                        "--ankle-target-margin-rad", "0.02", "--physics-dt", "0.002",
                        "--stop-at-s", str(case["stop_at_s"]),
                        "--duration-s", str(case["duration_s"]),
                        "--out", str(report), "--diagnostic-trace", str(trace)], 300)
            opened = json.loads(report.read_text())
            if (opened["seed"] != case["seed"] or opened["body_sha256"] != sha(BODY)
                    or opened["stop_at_s"] != case["stop_at_s"]
                    or len(opened["worlds"]) != 1 or not trace.is_file()):
                raise RuntimeError(f"Source identity mismatch: {label}")
    result = {"schema": "robot-learning-lab.locomotion-fast-v4.candidate.v1",
              "at_utc": datetime.now(timezone.utc).isoformat(),
              "method_sha256": sha(ART / "method.json"),
              "source_checkpoint_sha256": sha(SOURCE),
              "candidate_checkpoint_sha256": sha(candidate),
              "candidate_saved_iteration": saved["iter"],
              "worlds": 512, "optimizer_updates": 24,
              "environment_steps": 512 * 24 * 24,
              "development_cases": len(method["development_cases"]),
              "native_visits": 16,
              "files": {p.name: sha(p) for p in sorted(OUT.iterdir()) if p.is_file()},
              "claim_limit": "Short development continuation and eight paired native starts; not an independent score or physical transfer."}
    (OUT / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
