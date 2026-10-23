"""V99: same V98 graph, but continue the actor-committed side."""

from __future__ import annotations

import json
from pathlib import Path

from audit_post_detour_viewpoint_v98 import (MIN_COURSE_PROGRESS_M, replay)
from robot_geometry import RobotGeometry
from run import H, ROOT, W, Room, sha
from self_depth_mask import SelfDepthMask


SOURCE = Path(__file__)
SEARCH_CODE = SOURCE.with_name("audit_post_detour_viewpoint_v98.py")
V98 = ROOT / "artifacts/ape-closed-loop-cove50-v98/post-detour-viewpoint-audit.json"
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v99"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    room = Room()
    mask = SelfDepthMask(room.model, camera="lower_rgb",
                         width=W, height=H)
    try:
        robot = RobotGeometry(room.model)
        row = replay(85, room, robot, mask, search_direction="side")
    finally:
        mask.close()
        room.close()
    gate = (row["status"] == "screened" and
            row["actor_goal_cell"] is not None and
            row["actor_course_progress_m"] >= MIN_COURSE_PROGRESS_M and
            row["first_edge_evaluator_min_clearance_m"] >= .05)
    result = {"scope": "Actor-only side-extension information goal after V98 forward-course search failed from the V97 seed85 target-not-reacquired stop. Same 4/8/11cm scans, V95 self-masked near RGB-D, 4cm graph,6.5cm selected-body point rule and visited penalty. Evaluator labels first edge only after choice; no action executed.",
              "source_sha256": sha(SOURCE),
              "search_code_sha256": sha(SEARCH_CODE),
              "v98_failed_audit_sha256": sha(V98),
              "selected_design_sha256": sha(ROOT / "robot/ape-current-design.json"),
              "row": row,
              "side_extension_view_screen_pass": gate}
    (OUT / "side-extension-view-audit.json").write_text(
        json.dumps(result, indent=2) + "\n")
    (OUT / "audit_side_extension_view_v99_source.py").write_bytes(
        SOURCE.read_bytes())
    (OUT / "audit_post_detour_viewpoint_v98_imported_source.py").write_bytes(
        SEARCH_CODE.read_bytes())
    print("goal", row["actor_goal_cell"], "path", row["actor_path_cells"],
          "side progress", row["actor_course_progress_m"],
          "target memory gap", row["actor_goal_target_memory_gap_m"],
          "first true clearance", row[
              "first_edge_evaluator_min_clearance_m"], flush=True)
    print("side extension view screen:", gate, flush=True)


if __name__ == "__main__":
    main()
