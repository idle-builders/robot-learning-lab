"""Advance frozen V97 native retention only while each completed gate passes."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010/v97"
MODULE = "experiments.compact_morphology.cove50_policy_transfer.reach_image_goal.run_v97_native_retention"
STAGES = ("variance16", "controls4", "routes63", "prior75")


def wait_for_gate(stage: str) -> None:
    path = OUT / f"native-{stage}-retention/matrix.json"
    while True:
        if path.exists():
            matrix = json.loads(path.read_text())
            if matrix["complete"]:
                print(f"{stage}: {matrix['pass_count']}/{matrix['case_count']} full visits; "
                      f"screen_gate_met={matrix['screen_gate_met']}", flush=True)
                if not matrix["screen_gate_met"]:
                    raise RuntimeError(f"Stopped after failed {stage} gate: {path}")
                return
        time.sleep(30)


def main() -> None:
    os.chdir(ROOT)
    env = {**os.environ, "MUJOCO_GL": "egl", "PYTHONPATH": str(ROOT)}
    wait_for_gate(STAGES[0])
    for stage in STAGES[1:]:
        command = [sys.executable, "-u", "-m", MODULE, stage]
        subprocess.run(command + ["--freeze-only"], env=env, check=True)
        print(f"starting {stage}", flush=True)
        subprocess.run(command + ["--workers", "4"], env=env, check=True)
        wait_for_gate(stage)
    print("V97 opened retention ladder cleared; fresh scoring remains unopened.", flush=True)


if __name__ == "__main__":
    main()
