"""One capped V3 PPO continuation and eight matched two-arm movement visits."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
ART = ROOT / "artifacts/ape-low-level-training-20261015/cardinal-v3"
OUT = Path("/workspace/robot-learning-lab/ape-low-level-training-20261015/cardinal-v3/candidate")
BASE = ROOT / "artifacts/ape-low-level-training-20261014/cardinal-v2/candidate/candidate-final.pt"
BODY = ROOT / "artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml"
TRAINER = ROOT / "experiments/compact_morphology/cove50_policy_transfer/train_velocity.py"
EVALUATOR = ROOT / "experiments/compact_morphology/mjlab_ape/evaluate.py"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(label: str, command: list[str], timeout_s: int) -> None:
    with (OUT / f"{label}.log").open("x") as log:
        proc = subprocess.run(command, cwd=ROOT, stdout=log,
                              stderr=subprocess.STDOUT, timeout=timeout_s, check=False)
    if proc.returncode:
        raise RuntimeError(f"{label} exited {proc.returncode}")


def main() -> None:
    if not Path("/workspace").is_mount() or OUT.exists():
        raise RuntimeError("Fresh mounted cardinal candidate path required")
    method = json.loads((ART / "method.json").read_text())
    sentinel_path = Path("/workspace/robot-learning-lab/ape-low-level-training-20261015/cardinal-v3/sentinel/result.json")
    sentinel = json.loads(sentinel_path.read_text())
    if (sentinel["method_sha256"] != sha(ART / "method.json")
            or sentinel["worlds"] != 1024
            or sentinel["worlds_with_stable_nonzero_command_per_episode_all_400_steps"] != 1024
            or sentinel["zero_command_world_steps"]
            or sentinel["disposable_optimizer_updates"] != 1
            or sha(BASE) != method["source"]["v2_checkpoint"]["sha256"]):
        raise RuntimeError("CUDA cardinal sentinel or method identity differs")
    OUT.mkdir(parents=True)
    source_run = ROOT / "logs/rsl_rl/ape_cove50_velocity/cardinal-v3-candidate-source"
    source_run.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(BASE, source_run / "model_883.pt")
    before = {p.name for p in source_run.parent.iterdir() if p.is_dir()}
    run("train", [sys.executable, str(TRAINER), "--body-xml", str(BODY),
        "--profile", "cardinal_move", "--worlds", "1024", "--updates", "24",
        "--save-interval", "12", "--run-name", "cardinal-v3-24update",
        "--resume-run", "cardinal-v3-candidate-source", "--resume-checkpoint", "model_883.pt",
        "--ankle-target-margin-rad", ".02", "--directional-progress-weight", "2.0",
        "--walking-hip-roll-pose-std", ".2"], 2700)
    new = [p for p in source_run.parent.iterdir() if p.is_dir() and p.name not in before]
    if len(new) != 1:
        raise RuntimeError("Cardinal candidate run directory missing or ambiguous")
    checkpoints = sorted(new[0].glob("model_*.pt"),
                         key=lambda p: int(p.stem.split("_")[-1]))
    if not checkpoints or int(checkpoints[-1].stem.split("_")[-1]) < 906:
        raise RuntimeError("Cardinal V3 24-update final checkpoint missing")
    candidate = OUT / "candidate-final.pt"
    shutil.copyfile(checkpoints[-1], candidate)
    import torch
    saved = torch.load(candidate, map_location="cpu", weights_only=False)
    if saved.get("iter", 0) < 906 or sha(candidate) == sha(BASE):
        raise RuntimeError("Cardinal candidate optimizer lineage did not advance")
    for case in method["development_cases"]:
        for arm, checkpoint in (("baseline", BASE), ("candidate", candidate)):
            label = f"seed{case['seed']}-{arm}"
            report = OUT / f"{label}.json"
            trace = OUT / f"{label}.npz"
            video = OUT / f"{label}.mp4"
            run(label, [sys.executable, str(EVALUATOR), str(checkpoint),
                "--task", "Mjlab-Velocity-Flat-Ape-Elbows",
                "--body-xml", str(BODY), "--physics-model", "frame50",
                "--initial-root-z", str(case["initial_root_z_m"]),
                "--yaw-only-head", "--arms-down", "--worlds", "1",
                "--seed", str(case["seed"]),
                "--velocity-x", str(case["velocity_x_m_s"]),
                "--velocity-y", str(case["velocity_y_m_s"]), "--angular-z", "0",
                "--motor-rail-v", str(case["motor_rail_v"]),
                "--command-delay-ms", str(case["command_delay_ms"]),
                "--ankle-target-margin-rad", ".02", "--physics-dt", ".002",
                "--duration-s", str(case["duration_s"]),
                "--out", str(report), "--diagnostic-trace", str(trace),
                "--video", str(video)], 300)
            opened = json.loads(report.read_text())
            if (opened["seed"] != case["seed"] or opened["body_sha256"] != sha(BODY)
                    or opened["stop_at_s"] is not None
                    or len(opened["worlds"]) != 1 or not trace.is_file() or not video.is_file()):
                raise RuntimeError(f"Native report or video identity differs: {label}")
    result = {
        "schema": "robot-learning-lab.cardinal-v3.candidate.v1",
        "at_utc": datetime.now(timezone.utc).isoformat(),
        "method_sha256": sha(ART / "method.json"),
        "sentinel_sha256": sha(sentinel_path),
        "source_checkpoint_sha256": sha(BASE),
        "candidate_checkpoint_sha256": sha(candidate),
        "candidate_saved_iteration": saved["iter"],
        "worlds": 1024, "optimizer_updates": 24,
        "environment_steps": 589824,
        "development_cases": len(method["development_cases"]),
        "native_visits": 2 * len(method["development_cases"]),
        "files": {p.name: sha(p) for p in sorted(OUT.iterdir()) if p.is_file()},
        "claim_limit": "One development PPO continuation plus paired native Direct Frame visits; no independent promotion or physical transfer.",
    }
    (OUT / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
