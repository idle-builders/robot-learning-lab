"""Start the frozen varied-entry scan after its fixed-condition predecessor exits."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
V98_PLAN = BASE / "v98/entry-exposure-plan.json"
V98_MATRIX = BASE / "v98/native-entry-exposure/matrix.json"
V105_PLAN = BASE / "v105/varied-entry-exposure-plan.json"
V98_PLAN_SHA = "ea22d254c99f9eb5d568870802c2276282ad09612ef2a0b030b8ec0ec88773b9"
V105_PLAN_SHA = "a60172d1b927351f88f740bba70cf1e6439f193f49974f05cc1dcfae1577fadb"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def running_v98(pid: int) -> bool:
    cmdline = Path(f"/proc/{pid}/cmdline")
    return cmdline.is_file() and b"run_v98_entry_exposure" in cmdline.read_bytes()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v98-pid", type=int, required=True)
    args = parser.parse_args()
    if sha(V98_PLAN) != V98_PLAN_SHA or sha(V105_PLAN) != V105_PLAN_SHA:
        raise ValueError("A frozen V98 or V105 plan changed")
    while True:
        if V98_MATRIX.exists():
            matrix = json.loads(V98_MATRIX.read_text())
            if matrix.get("complete"):
                if (matrix.get("plan_sha256") != V98_PLAN_SHA or
                        matrix.get("case_count") != 64):
                    raise ValueError("V98 complete matrix failed frozen identity")
                break
        if not running_v98(args.v98_pid):
            raise RuntimeError("V98 stopped before completing its frozen 64 entries")
        time.sleep(30)
    print("V98 fixed-condition scan complete; starting frozen V105 varied development entries", flush=True)
    env = os.environ.copy()
    env["MUJOCO_GL"] = "egl"
    env["PYTHONPATH"] = str(ROOT)
    subprocess.run([sys.executable, "-u", "-m",
                    "experiments.compact_morphology.cove50_policy_transfer."
                    "reach_image_goal.run_v105_varied_entry_exposure",
                    "--workers", "1"], cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
