"""Run frozen V18 source/checkpoint CUDA pairs and retain every original visit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
EVALUATOR = HERE / "evaluate_support_residual_v18.py"
BODY = ROOT / "artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml"
POSE = ROOT / "artifacts/ape-cove50-support-recovery-20261009"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    method = json.loads(args.method.read_text())
    source_matches = all(sha(ROOT / relative) == expected
                         for relative, expected in method["source_sha256"].items())
    if (method["schema"] != "robot-learning-lab.support-residual-v18.evaluation-method.v5"
            or method["evaluator_sha256"] != sha(EVALUATOR)
            or method["runner_sha256"] != sha(Path(__file__))
            or method["source_body_sha256"] != sha(BODY)
            or not source_matches
            or not args.checkpoint.is_file()
            or method["checkpoint_sha256"] != sha(args.checkpoint)):
        raise RuntimeError("Frozen evaluation source, body or checkpoint unavailable")
    args.out.mkdir(parents=True, exist_ok=False)
    checkpoint_sha = sha(args.checkpoint)
    completed = []
    for family in method["fresh_families"]:
        seed, side = family["seed"], family["side"]
        pair = []
        for controller in ("reference", "checkpoint"):
            stem = f"seed{seed}-{controller}"
            report = args.out / f"{stem}.json"
            video = args.out / f"{stem}.mp4"
            log = args.out / f"{stem}.log"
            command = [sys.executable, str(EVALUATOR),
                "--body-xml", str(BODY),
                "--right-pose-json", str(POSE / "right-low-swing-endpoint.json"),
                "--left-pose-json", str(POSE / "left-low-swing-endpoint.json"),
                "--lesson", "load_transfer", "--reference-residual", "--domain-variation",
                "--preflight-only", "--preflight-reward-controller",
                "reference" if controller == "reference" else "ppo",
                "--preflight-side", side, "--seed", str(seed),
                "--eval-fixed-lag-steps", str(family["fixed_lag_steps"]),
                "--eval-device", "cuda:0", "--run-name", "support-residual-v18-eval",
                "--preflight-out", str(report), "--eval-video", str(video)]
            if controller == "checkpoint":
                command += ["--eval-checkpoint", str(args.checkpoint)]
            with log.open("wb") as stream:
                done = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                                      cwd=ROOT, timeout=300, check=False)
            trace = report.with_suffix(".npz")
            if done.returncode or not all(path.is_file() for path in (report, trace, video)):
                raise RuntimeError(f"Original {stem} failed; retained logs and partial evidence")
            episode = json.loads(report.read_text())["reward_episode_preflight"]
            if (episode["side"] != side or episode["seed"] != seed
                    or episode["steps"] != len(episode["lag_trace"])
                    or episode["eval_fixed_lag_steps"] != family["fixed_lag_steps"]
                    or not episode["eval_device"].startswith("cuda")
                    or not episode["eval_cuda_device_name"]
                    or episode["checkpoint_iteration"] != (31 if controller == "checkpoint" else None)
                    or episode["checkpoint_sha256"] != (checkpoint_sha if controller == "checkpoint" else None)):
                raise RuntimeError(f"Original {stem} identity differs")
            pair.append(episode)
            completed.append(stem)
        source, candidate = pair
        if (source["initial_state_sha256"] != candidate["initial_state_sha256"]
                or source["initial_domain_sha256"] != candidate["initial_domain_sha256"]
                or source["eval_cuda_device_name"] != candidate["eval_cuda_device_name"]
                or source["eval_fixed_lag_steps"] != candidate["eval_fixed_lag_steps"]):
            raise RuntimeError(f"Original seed {seed} pair start/domain/lag mismatch; no replacement")
        print(json.dumps({"seed": seed, "side": side, "valid_pair": True,
                          "source_steps": source["steps"],
                          "candidate_steps": candidate["steps"]}), flush=True)
    files = {path.name: sha(path) for path in args.out.iterdir() if path.is_file()}
    result = {
        "schema": "robot-learning-lab.support-residual-v18.cuda-eval-source.v1",
        "evaluation_method_sha256": sha(args.method),
        "evaluator_sha256": sha(EVALUATOR),
        "checkpoint_sha256": checkpoint_sha,
        "valid_pairs": len(method["fresh_families"]),
        "completed_visits": completed,
        "files": files,
        "claim_limit": "Original CUDA visits and matched videos only; independent physical scorer decides the development gate.",
    }
    target = args.out / "source-result.json"
    with target.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"valid_pairs": result["valid_pairs"],
                      "visits": len(completed), "checkpoint_sha256": checkpoint_sha}), flush=True)


if __name__ == "__main__":
    main()
