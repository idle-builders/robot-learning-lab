"""Launch the frozen V99 score only when the live V97 ladder clears every gate."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal import (
    run_v99_disjoint_reach as v99,
)


def ladder_live(pid: int) -> bool:
    result = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)],
                            capture_output=True, text=True, check=False)
    return result.returncode == 0 and bool(result.stdout.strip()) and not (
        result.stdout.strip().startswith("Z"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ladder-pid", type=int, required=True)
    parser.add_argument("--poll-s", type=int, default=45)
    args = parser.parse_args()
    os.chdir(ROOT)
    while True:
        try:
            v99.require_retention()
            break
        except (ValueError, FileNotFoundError) as error:
            if not ladder_live(args.ladder_pid):
                print(f"V99 remains sealed: V97 ladder ended before all gates: {error}",
                      flush=True)
                return
        time.sleep(args.poll_s)
    print("V97 four-stage retention passed; starting untouched V99 score", flush=True)
    env = {**os.environ, "MUJOCO_GL": "egl", "PYTHONPATH": str(ROOT)}
    subprocess.run([sys.executable, "-u", "-m",
                    "experiments.compact_morphology.cove50_policy_transfer."
                    "reach_image_goal.run_v99_disjoint_reach", "--workers", "4"],
                   env=env, check=True)


if __name__ == "__main__":
    main()
