"""V92: rerun V91 with actor-predicted self depth removed from sensor rays."""

from __future__ import annotations

import json
from pathlib import Path

from audit_actor_side_continuation_v91 import MIN_SIDE_PROGRESS_M, replay
from robot_geometry import RobotGeometry
from run import H, ROOT, W, Room, sha
from self_depth_mask import SelfDepthMask


SOURCE = Path(__file__)
SEARCH_CODE = SOURCE.with_name("audit_actor_side_continuation_v91.py")
MASK_CODE = SOURCE.with_name("self_depth_mask.py")
V91 = ROOT / "artifacts/ape-closed-loop-cove50-v91/actor-side-continuation-audit.json"
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v92"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    room = Room()
    self_mask = SelfDepthMask(room.model, camera="lower_rgb",
                              width=W, height=H)
    try:
        robot = RobotGeometry(room.model)
        rows = [replay(seed, room, robot, self_mask=self_mask)
                for seed in (85, 88)]
    finally:
        self_mask.close()
        room.close()
    baseline = {int(row["seed"]): row for row in
                json.loads(V91.read_text())["rows"]}
    for row in rows:
        old = baseline[row["seed"]]
        row["unmasked_occupied_voxels"] = old["actor_voxel_occupied"]
        row["unmasked_expanded_cells"] = old["expanded_cells"]
    gate = all(row["actor_goal_cell"] is not None and
               row["actor_side_progress_m"] >= MIN_SIDE_PROGRESS_M and
               row["first_edge_evaluator_min_clearance_m"] >= .05
               for row in rows)
    result = {"scope": "Actor model-predicted pixel self-depth mask on all saved V90 lower-head RGB-D frames, then frozen V91 local outward graph search. Evaluator geometry only labels chosen first edge after search; no action executed. Physical model/calibration unqualified.",
              "source_sha256": sha(SOURCE),
              "search_code_sha256": sha(SEARCH_CODE),
              "mask_code_sha256": sha(MASK_CODE),
              "v91_baseline_audit_sha256": sha(V91),
              "selected_design_sha256": sha(ROOT / "robot/ape-current-design.json"),
              "rows": rows,
              "self_mask_side_continuation_screen_pass": gate}
    (OUT / "self-depth-mask-audit.json").write_text(
        json.dumps(result, indent=2) + "\n")
    (OUT / "audit_self_depth_mask_v92_source.py").write_bytes(
        SOURCE.read_bytes())
    (OUT / "audit_actor_side_continuation_v91_imported_source.py").write_bytes(
        SEARCH_CODE.read_bytes())
    (OUT / "self_depth_mask_source.py").write_bytes(MASK_CODE.read_bytes())
    for row in rows:
        print(row["seed"], "masked", row["actor_self_depth_mask_totals"],
              "occupied", row["unmasked_occupied_voxels"], "->",
              row["actor_voxel_occupied"], "expanded",
              row["unmasked_expanded_cells"], "->", row["expanded_cells"],
              "goal", row["actor_goal_cell"], "first gap",
              row["first_edge_evaluator_min_clearance_m"], flush=True)
    print("self mask side continuation screen:", gate, flush=True)


if __name__ == "__main__":
    main()
