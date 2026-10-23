"""V203: fourth split route strata and near-target absent scene validity."""

from __future__ import annotations

import json
from pathlib import Path

import mujoco

from audit_heldout_rooms_v175 import audit_scene
from audit_heldout_feasible_v176 import props
from run_moved_props_v134 import MovedPropRoom
from run import ROOT, Room, sha


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v203"
SEEDS = tuple(range(107, 115))


def absent_challenge(seed: int, target_xy: list[float]) -> dict:
    positions = props(seed)
    positions["Parmesan"] = tuple(target_xy)
    room = MovedPropRoom(positions)
    try:
        room.reset()
        room.layout(seed, absent=True, furnished=True)
        room.model.body_pos[room.model.body("loop_table").id] = (
            .40, (1 if seed % 2 else -1) * .46, 0.)
        room.set_table_height(.37)
        room._forward()
        moved_id = room.model.geom("loop_Parmesan_geom").id
        others = [gid for gid in range(room.model.ngeom)
                  if room.model.geom(gid).name.startswith("loop_")
                  and room.model.geom(gid).name != "loop_rug_visual"
                  and gid != moved_id]
        gap, closest = min((float(mujoco.mj_geomDistance(
            room.model, room.data, moved_id, other, 10., None)),
            room.model.geom(other).name) for other in others)
        start_body_gap = room._current_obstacle_clearance()[0]
        return {"parmesan_xy_m": list(target_xy),
                "cream_cheese_xy_m": list(positions["CreamCheese"]),
                "minimum_scene_gap_m": round(gap, 5),
                "closest_scene_geom": closest,
                "start_body_clearance_m": round(start_body_gap, 5),
                "valid_absent": gap >= .03 and start_body_gap >= .05}
    finally:
        room.close()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    room = Room()
    try:
        for seed in SEEDS:
            audits = {str(margin): audit_scene(
                room, seed, prop_overrides=props(seed), output_dir=OUT,
                model_margin_m=margin) for margin in (.065, .12)}
            narrow, generous = audits["0.065"], audits["0.12"]
            assert narrow["object_scene_gap_m"] == generous["object_scene_gap_m"]
            absent = absent_challenge(seed, narrow["positions_xy_m"]["TomatoSauce"])
            if narrow["object_scene_gap_m"] < .03:
                label = "invalid_present_object_separation"
            elif not absent["valid_absent"]:
                label = "invalid_absent_scene"
            elif generous["fixed_yaw_route_found"]:
                label = "generous_kinematic"
            elif narrow["fixed_yaw_route_found"]:
                label = "tight_kinematic"
            else:
                label = "blocked_fixed_yaw"
            rows.append({"seed": seed, "label": label,
                         "absent_challenge": absent, "audits": audits})
            print(seed, label, "object gap", narrow["object_scene_gap_m"],
                  "absent gap", absent["minimum_scene_gap_m"],
                  "12cm route", generous["fixed_yaw_route_found"], flush=True)
    finally:
        room.close()
    result = {"scope": "Predeclared seeds107-114, V176 parity-fixed moved props, V175 evaluator-only all-solid present separation and fixed-yaw 4cm/21-sample witness path at6.5/12cm, paired target-absent Parmesan at true present target XY with >=3cm scene gap and >=5cm initial body gap. No actor sensor/control input.",
              "source_sha256": sha(SOURCE),
              "v175_source_sha256": sha(ROOT / "experiments/ape_high_level/closed_loop/audit_heldout_rooms_v175.py"),
              "v176_source_sha256": sha(ROOT / "experiments/ape_high_level/closed_loop/audit_heldout_feasible_v176.py"),
              "selected_design_sha256": sha(ROOT / "robot/ape-current-design.json"),
              "seeds": list(SEEDS), "rows": rows}
    (OUT / "fourth-split-geometry-audit.json").write_text(json.dumps(result, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())


if __name__ == "__main__":
    main()
