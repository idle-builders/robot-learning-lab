"""V95: near masked depth retains V94 occluder and V91 route viability."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from audit_actor_side_continuation_v91 import (MIN_SIDE_PROGRESS_M, replay)
from audit_self_occluder_v94 import occupied_upper
from depth_voxel import DepthVoxelMap, VOXEL_M
from robot_geometry import RobotGeometry
from run import H, ROOT, W, Room, sha
from self_depth_mask import SelfDepthMask


SOURCE = Path(__file__)
VOXEL_CODE = SOURCE.with_name("depth_voxel.py")
SEARCH_CODE = SOURCE.with_name("audit_actor_side_continuation_v91.py")
MASK_CODE = SOURCE.with_name("self_depth_mask.py")
V90 = ROOT / "artifacts/ape-closed-loop-cove50-v90"
V92 = ROOT / "artifacts/ape-closed-loop-cove50-v92/self-depth-mask-audit.json"
V94 = ROOT / "artifacts/ape-closed-loop-cove50-v94/near-arm-occluder-audit.json"
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v95"


def occluder_replay(seed: int, mask: SelfDepthMask,
                    fovy: float, v94_row: dict) -> dict:
    baseline_path = V90 / f"seed{seed}-actor/frame-13.npz"
    changed_path = ROOT / (f"artifacts/ape-closed-loop-cove50-v94/"
                           f"seed{seed}/near-arm-actor-input.npz")
    with np.load(baseline_path) as baseline:
        qpos = baseline["actor_qpos"].copy()
        stamp = float(baseline["capture_time_s"])
        original_depth = baseline["depth_m"].copy()
        camera = (baseline["camera_origin_m"].copy(),
                  baseline["camera_rotation"].copy())
    with np.load(changed_path) as changed:
        changed_depth = changed["depth_m"].copy()
    before_filtered, before_mask = mask.apply(original_depth, qpos)
    after_filtered, after_mask = mask.apply(changed_depth, qpos)
    maps = []
    for depth in (before_filtered, after_filtered):
        voxel = DepthVoxelMap()
        voxel.observe(depth, camera, stamp, fovy_deg=fovy,
                      self_masked=True)
        maps.append(voxel)
    added = occupied_upper(maps[1], stamp) - occupied_upper(maps[0], stamp)
    center = np.asarray(v94_row["occluder_center_evaluator_only_m"])
    near = {cell for cell in added if np.linalg.norm(
        (np.asarray(cell[:2], dtype=float) + .5) * VOXEL_M -
        center[:2]) <= .12}
    return {"seed": seed,
            "baseline_frame_sha256": sha(baseline_path),
            "changed_actor_input_sha256": sha(changed_path),
            "before_mask": before_mask,
            "after_mask": after_mask,
            "added_upper_cells_anywhere": len(added),
            "added_upper_cells_near_box_evaluator_only": len(near),
            "near_box_cell_centers_m_evaluator_only": [
                [round((cell[0] + .5) * VOXEL_M, 4),
                 round((cell[1] + .5) * VOXEL_M, 4),
                 round((cell[2] + .5) * VOXEL_M, 4)]
                for cell in sorted(near)]}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    v94_rows = {int(row["seed"]): row for row in
                json.loads(V94.read_text())["rows"]}
    v92_rows = {int(row["seed"]): row for row in
                json.loads(V92.read_text())["rows"]}
    room = Room()
    mask = SelfDepthMask(room.model, camera="lower_rgb",
                         width=W, height=H)
    try:
        robot = RobotGeometry(room.model)
        fovy = float(room.model.cam_fovy[room.model.camera("lower_rgb").id])
        occluders = [occluder_replay(seed, mask, fovy, v94_rows[seed])
                     for seed in (85, 88)]
        routes = [replay(seed, room, robot, self_mask=mask,
                         allow_near_masked_depth=True)
                  for seed in (85, 88)]
    finally:
        mask.close()
        room.close()
    for row in routes:
        row["v92_goal_cell"] = v92_rows[row["seed"]]["actor_goal_cell"]
        row["v92_expanded_cells"] = v92_rows[row["seed"]]["expanded_cells"]
    gate = (all(row["added_upper_cells_near_box_evaluator_only"] > 0
                for row in occluders) and
            all(row["actor_goal_cell"] is not None and
                row["actor_side_progress_m"] >= MIN_SIDE_PROGRESS_M and
                row["first_edge_evaluator_min_clearance_m"] >= .05
                for row in routes))
    result = {"scope": "Actor CAD/shell-predicted pixel self mask followed by near-depth voxel ray admission on saved V94 rendered near-arm occluders and V90 ordinary endpoint frames. No simulator obstacle geometry enters actor; evaluator box location scores near map cell and selected first-edge body clearance only after planning. No action executed.",
              "source_sha256": sha(SOURCE),
              "voxel_code_sha256": sha(VOXEL_CODE),
              "search_code_sha256": sha(SEARCH_CODE),
              "mask_code_sha256": sha(MASK_CODE),
              "v94_audit_sha256": sha(V94),
              "v92_audit_sha256": sha(V92),
              "selected_design_sha256": sha(ROOT / "robot/ape-current-design.json"),
              "occluder_rows": occluders,
              "ordinary_route_rows": routes,
              "near_depth_after_self_screen_pass": gate}
    (OUT / "near-depth-after-self-audit.json").write_text(
        json.dumps(result, indent=2) + "\n")
    (OUT / "audit_near_depth_after_self_v95_source.py").write_bytes(
        SOURCE.read_bytes())
    (OUT / "depth_voxel_imported_source.py").write_bytes(VOXEL_CODE.read_bytes())
    (OUT / "audit_actor_side_continuation_v91_imported_source.py").write_bytes(
        SEARCH_CODE.read_bytes())
    (OUT / "self_depth_mask_imported_source.py").write_bytes(
        MASK_CODE.read_bytes())
    for near, route in zip(occluders, routes):
        print(near["seed"], "near cube cells",
              near["added_upper_cells_near_box_evaluator_only"],
              "goal", route["actor_goal_cell"], "expanded",
              route["expanded_cells"], "first clearance",
              route["first_edge_evaluator_min_clearance_m"], flush=True)
    print("near depth after self screen:", gate, flush=True)


if __name__ == "__main__":
    main()
