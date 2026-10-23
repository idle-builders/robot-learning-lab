"""Screen low ToF view self-occlusion on historical Cove-50 poses.

The body used by V70 is rejected as a selected design. This is a negative
geometry stress test for the provisional Isaac ToF layout, not mount approval.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "closed_loop"))
from audit_low_range_height_transfer_v70 import SCENES  # noqa: E402
from run import ROOT, Room, sha  # noqa: E402
from near_body_tof import MOUNTS, FOV_DEG, ZONES_PER_AXIS  # noqa: E402


SOURCE = Path(__file__)
TOF_SOURCE = HERE / "near_body_tof.py"
V70 = ROOT / "artifacts/ape-closed-loop-cove50-v70/low-range-height-transfer.json"
OUT = ROOT / "artifacts/ape-realistic-people/low-tof-historical-self-occlusion.json"
ROOM_GROUP = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)
SELF_GROUP = np.array([0, 1, 0, 0, 1, 0], dtype=np.uint8)
AUDIT_MOUNTS = {**MOUNTS,
    "side_left": {"forward_m": 0., "lateral_m": -.14,
                  "height_m": .12, "yaw_deg": -90., "pitch_deg": -10.},
    "side_right": {"forward_m": 0., "lateral_m": .14,
                   "height_m": .12, "yaw_deg": 90., "pitch_deg": -10.},
    "side_left_18cm": {"forward_m": 0., "lateral_m": -.18,
                       "height_m": .12, "yaw_deg": -90., "pitch_deg": -10.},
    "side_right_18cm": {"forward_m": 0., "lateral_m": .18,
                        "height_m": .12, "yaw_deg": 90., "pitch_deg": -10.},
    "side_left_22cm": {"forward_m": 0., "lateral_m": -.22,
                       "height_m": .12, "yaw_deg": -90., "pitch_deg": -10.},
    "side_right_22cm": {"forward_m": 0., "lateral_m": .22,
                        "height_m": .12, "yaw_deg": 90., "pitch_deg": -10.},
    "front_flank_left": {"forward_m": .14, "lateral_m": -.14,
                         "height_m": .12, "yaw_deg": -90., "pitch_deg": -10.},
    "front_flank_right": {"forward_m": .14, "lateral_m": .14,
                          "height_m": .12, "yaw_deg": 90., "pitch_deg": -10.}}


def screen_mount(room: Room, pose: np.ndarray, mount: dict,
                 cube_id: int) -> dict:
    c, s = math.cos(pose[2]), math.sin(pose[2])
    forward, lateral = mount["forward_m"], mount["lateral_m"]
    origin = np.array([pose[0] + c * forward - s * lateral,
                       pose[1] + s * forward + c * lateral,
                       mount["height_m"]], dtype=float)
    count = {"self_first": 0, "room_hits": 0, "cube_hits": 0,
             "blind_hits": 0, "no_return": 0}
    for row in range(ZONES_PER_AXIS):
        pitch = math.radians(mount["pitch_deg"] +
            (row - (ZONES_PER_AXIS - 1) / 2) * FOV_DEG / ZONES_PER_AXIS)
        for column in range(ZONES_PER_AXIS):
            bearing = pose[2] + math.radians(mount["yaw_deg"] +
                (column - (ZONES_PER_AXIS - 1) / 2) * FOV_DEG / ZONES_PER_AXIS)
            direction = np.array([math.cos(pitch) * math.cos(bearing),
                                  math.cos(pitch) * math.sin(bearing),
                                  math.sin(pitch)])
            room_id = np.array([-1], dtype=np.int32)
            self_id = np.array([-1], dtype=np.int32)
            room_distance = mujoco.mj_ray(room.model, room.data, origin,
                                          direction, ROOM_GROUP, True, -1,
                                          room_id)
            self_distance = mujoco.mj_ray(room.model, room.data, origin,
                                          direction, SELF_GROUP, True, -1,
                                          self_id)
            if self_distance >= 0 and (room_distance < 0 or
                                       self_distance <= room_distance):
                count["self_first"] += 1
            elif room_distance < 0 or room_distance > 4.:
                count["no_return"] += 1
            elif room_distance < .02:
                count["blind_hits"] += 1
            else:
                count["room_hits"] += 1
                if room_id[0] == cube_id:
                    count["cube_hits"] += 1
    return {"origin_xyz_m": origin.tolist(), **count}


def main() -> None:
    source = json.loads(V70.read_text())
    assert source["transfer_interpretable"]
    cases = [case for case in source["cases"] if case["valid_start"]]
    assert len(cases) == 74
    unsafe_cases = {row["case"] for row in source["rows"]
                    if not row["evaluator_safe_5cm"]}
    specs = {spec[0]: spec for spec in SCENES}
    room = Room()
    rows = []
    try:
        for case in cases:
            seed = case["seed"]
            _, crate_x, crate_y_rel, half_x, half_y, half_z = specs[seed]
            side = 1 if seed % 2 else -1
            room.reset()
            room.layout(seed, absent=False, furnished=True)
            cube_id = room.model.geom("loop_crate_geom").id
            room.model.geom_size[cube_id] = (half_x, half_y, half_z)
            room.model.body_pos[room.model.body("loop_crate").id] = (
                crate_x, side * crate_y_rel, half_z)
            pose = np.asarray(case["pose_xyyaw"], dtype=float)
            room.actual[:] = pose
            room.estimated[:] = pose
            room.head = 0.
            room._forward()
            rows.append({"case": case["case"], "seed": seed,
                "cube_height_m": 2 * half_z,
                "mounts": {name: screen_mount(room, pose, mount, cube_id)
                           for name, mount in AUDIT_MOUNTS.items()}})
    finally:
        room.close()
    stats = {name: {"mean_self_first_zones": round(float(np.mean([
                row["mounts"][name]["self_first"] for row in rows])), 2),
            "cases_with_cube_return": sum(row["mounts"][name]["cube_hits"] > 0
                for row in rows),
            "unsafe_cases_with_cube_return": sum(row["case"] in unsafe_cases
                and row["mounts"][name]["cube_hits"] > 0 for row in rows),
            "cases_with_no_room_return": sum(row["mounts"][name]["room_hits"] == 0
                for row in rows),
            "mean_cube_hit_zones": round(float(np.mean([
                row["mounts"][name]["cube_hits"] for row in rows])), 2)}
        for name in AUDIT_MOUNTS}
    base_names = tuple(MOUNTS)
    centered_side_names = base_names + ("side_left", "side_right")
    outside_18_names = base_names + ("side_left_18cm", "side_right_18cm")
    outside_22_names = base_names + ("side_left_22cm", "side_right_22cm")
    front_flank_names = base_names + ("front_flank_left", "front_flank_right")
    def unsafe_covered(names):
        return sum(row["case"] in unsafe_cases and any(
            row["mounts"][name]["cube_hits"] > 0 for name in names)
            for row in rows)
    report = {"scope": "static ray-first self-occlusion screen on 74 historical V70 valid poses; rejected Direct Frame geometry, no live control, no mount fit, no optical physics",
        "source_sha256": sha(SOURCE), "tof_geometry_sha256": sha(TOF_SOURCE),
        "v70_sha256": sha(V70),
        "nominal_sensor": "VL53L5CX-like 8x8 center ray, 45x45 degrees, 2–400 cm",
        "mounts_body_relative": AUDIT_MOUNTS, "case_count": len(rows),
        "unsafe_crate_cases": len(unsafe_cases),
        "unsafe_cases_with_cube_return_base_four": unsafe_covered(base_names),
        "unsafe_cases_with_cube_return_centered_sides": unsafe_covered(centered_side_names),
        "unsafe_cases_with_cube_return_outside_18cm": unsafe_covered(outside_18_names),
        "unsafe_cases_with_cube_return_outside_22cm": unsafe_covered(outside_22_names),
        "unsafe_cases_with_cube_return_front_flanks": unsafe_covered(front_flank_names),
        "unsafe_crate_cases_without_cube_return_outside_22cm": [row["case"] for row in rows
            if row["case"] in unsafe_cases and not any(
                row["mounts"][name]["cube_hits"] > 0 for name in outside_22_names)],
        "stats": stats, "rows": rows}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
