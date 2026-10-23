"""Run frozen V19 matched native CUDA development starts, preserving originals."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[3]
TRAINER = Path(__file__).with_name("train_release_residual_v19.py")
POSE = ROOT / "artifacts/ape-cove50-support-recovery-20261009"
BODY = ROOT / "artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    method = json.loads(args.method.read_text())
    if method["schema"] != "robot-learning-lab.cove50-release-v19.nominal-pilot.v1":
        raise RuntimeError("Wrong frozen method")
    for relative, expected in method["source_sha256"].items():
        if sha(ROOT / relative) != expected:
            raise RuntimeError(f"Source changed: {relative}")
    if not args.checkpoint.is_file():
        raise RuntimeError("Checkpoint missing")
    args.out.mkdir(parents=True, exist_ok=True)
    checkpoint_sha = sha(args.checkpoint)
    for family in method["development_families"]:
        pair = []
        for controller in ("reference", "checkpoint"):
            stem = f"seed{family['seed']}-{controller}"
            report = args.out / f"{stem}.json"
            trace = report.with_suffix(".npz")
            log = args.out / f"{stem}.log"
            if not report.exists():
                cmd = [sys.executable, str(TRAINER),
                       "--body-xml", str(BODY),
                       "--right-pose-json", str(POSE / "right-low-swing-endpoint.json"),
                       "--left-pose-json", str(POSE / "left-low-swing-endpoint.json"),
                       "--lesson", "release_transition", "--reference-residual",
                       "--preflight-only", "--preflight-reward-controller",
                       "reference" if controller == "reference" else "ppo",
                       "--preflight-side", family["side"],
                       "--seed", str(family["seed"]), "--eval-device", "cuda:0",
                       "--run-name", "support-release-v19-eval", "--preflight-out", str(report)]
                if controller == "checkpoint":
                    cmd += ["--eval-checkpoint", str(args.checkpoint)]
                with log.open("wb") as output:
                    done = subprocess.run(cmd, cwd=ROOT, stdout=output,
                                          stderr=subprocess.STDOUT, timeout=240)
                if done.returncode:
                    raise RuntimeError(f"Original visit {stem} failed; retained log")
            if not report.is_file() or not trace.is_file():
                raise RuntimeError(f"Original visit {stem} incomplete")
            episode = json.loads(report.read_text())["reward_episode_preflight"]
            if (episode["seed"] != family["seed"] or episode["side"] != family["side"]
                    or episode["eval_device"] != "cuda:0"
                    or episode["controller"] != ("reference" if controller == "reference" else "ppo")
                    or episode["checkpoint_iteration"] != (None if controller == "reference" else 3)
                    or episode["checkpoint_sha256"] !=
                    (checkpoint_sha if controller == "checkpoint" else None)):
                raise RuntimeError(f"Original visit {stem} identity mismatch")
            pair.append(episode)
        if (pair[0]["initial_state_sha256"] != pair[1]["initial_state_sha256"]
                or pair[0]["initial_domain_sha256"] != pair[1]["initial_domain_sha256"]):
            raise RuntimeError(f"Original pair {family['seed']} start differs")
        print(json.dumps({"seed": family["seed"], "side": family["side"],
                          "source_steps": pair[0]["steps"],
                          "checkpoint_steps": pair[1]["steps"],
                          "valid_pair": True}), flush=True)


if __name__ == "__main__":
    main()
