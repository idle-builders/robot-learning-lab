"""Run the frozen local signed-action support authority screen."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[3]
TRAINER = Path(__file__).with_name("train_support_cycle.py")
BODY = ROOT / "artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml"
RIGHT = ROOT / "artifacts/ape-cove50-support-recovery-20261009/right-low-swing-endpoint.json"
LEFT = ROOT / "artifacts/ape-cove50-support-recovery-20261009/left-low-swing-endpoint.json"
OUT = ROOT / "artifacts/ape-low-level-training-20261014/support-pulse-authority-v17"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--side", choices=("left", "right"), required=True)
    args = p.parse_args()
    method = json.loads((OUT / "method.json").read_text())
    if sha(TRAINER) != method["trainer_sha256"] or sha(BODY) != method["body_sha256"]:
        raise RuntimeError("Frozen support pulse method source changed")
    starts = [("baseline", None, None)]
    for joint in method["joint_names"]:
        for value in method["pulse_raw_values"]:
            starts.append((joint + ("-minus" if value < 0 else "-plus"), joint, value))
    initial_hash = None
    for name, joint, value in starts:
        stem = f"{args.side}-{name}"
        report, trace, log = (OUT / f"{stem}.{extension}" for extension in ("json", "npz", "log"))
        if any(path.exists() for path in (report, trace, log)):
            raise RuntimeError(f"Refusing to overwrite {stem}")
        cmd = [sys.executable, str(TRAINER),
               "--body-xml", str(BODY), "--right-pose-json", str(RIGHT),
               "--left-pose-json", str(LEFT), "--lesson", "load_transfer",
               "--reference-residual", "--preflight-only",
               "--preflight-reward-controller", "reference",
               "--preflight-side", args.side,
               "--seed", str(method["seeds"][args.side]),
               "--run-name", "support-pulse-authority-v17",
               "--preflight-out", str(report)]
        if joint is not None:
            cmd += ["--preflight-pulse-joint", joint,
                    "--preflight-pulse-raw", str(value)]
        with log.open("wb") as handle:
            done = subprocess.run(cmd, cwd=ROOT, stdout=handle,
                                  stderr=subprocess.STDOUT, timeout=100, check=False)
        if done.returncode or not report.is_file() or not trace.is_file():
            raise RuntimeError(f"{stem} failed with return code {done.returncode}; see {log}")
        episode = json.loads(report.read_text())["reward_episode_preflight"]
        assert episode["side"] == args.side and episode["seed"] == method["seeds"][args.side]
        if initial_hash is None:
            initial_hash = episode["initial_state_sha256"]
        assert episode["initial_state_sha256"] == initial_hash
        print(json.dumps({"stem": stem, "steps": episode["steps"],
                          "initial_state_sha256": initial_hash,
                          "reward_sum": episode["reward_sum"]}), flush=True)


if __name__ == "__main__":
    main()
