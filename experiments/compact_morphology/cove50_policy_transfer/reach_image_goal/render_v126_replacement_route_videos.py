"""Render the source-matched V126 opened route comparison for the inspector."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal import (
    render_v122_near_route_videos as renderer,
)


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010/v126"


def main() -> None:
    renderer.BASE = BASE
    renderer.PLAN = BASE / "replacement-route-pairs-plan.json"
    renderer.MATRIX = BASE / "native-replacement-route-pairs/matrix.json"
    renderer.INSPECTOR = BASE / "pair-inspector-matrix.json"
    renderer.main()
    display = json.loads(renderer.INSPECTOR.read_text())
    display["schema"] = "robot-learning-lab.cove50-v126-inspector.v1"
    display["parent_v122_plan_sha256"] = (
        "cf9fac8fbf677893ac580016388d173ab21c42d170520654b8ef67366b1e31a0"
    )
    renderer.INSPECTOR.write_text(json.dumps(display, indent=2) + "\n")
    subprocess.run([sys.executable, str(renderer.PUBLISHER)], cwd=ROOT,
                   check=True, stdout=subprocess.DEVNULL)


if __name__ == "__main__":
    main()
