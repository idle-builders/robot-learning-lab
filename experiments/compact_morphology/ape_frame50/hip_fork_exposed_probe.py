"""Bare-frame recheck of the earlier Cove fork hip with an exposed pitch drive.

This deliberately reuses the sourced bearing, ST3250 and Gates geometry from
swarm_cove_fork_hip. It is a local joint coupon, not a whole ape assembly.
The cosmetic skin is not a collision boundary in this phase.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import build123d as bd
import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiments.compact_morphology.ape_design.swarm_cove_fork_hip import build as source


OUT = ROOT / "artifacts/ape-frame50-hip-fork-exposed"
PITCH_MOTOR_SHIFT_Y_MM = 18.0
YAW_MOTOR_DROP_Z_MM = 34.0
PITCH_BELT_PART = "159-3MGT-15"
PITCH_BELT_SOURCE = "https://www.gates.com/content/dam/documents-library/catalogs/power-transmission-catalog.pdf"


def moved_hardware():
    groups = source.right_hardware()
    fixed = groups["fixed"]
    fixed["yaw_ST3250_case"] = bd.Pos(0, 0, -YAW_MOTOR_DROP_Z_MM) * fixed["yaw_ST3250_case"]
    # Keep the belt at the source plane while extending the motor's vertical
    # output coupler. A separate bearing support must still be mounted.
    fixed["yaw_motor_output"] = source.cyl(3, 25 + YAW_MOTOR_DROP_Z_MM,
                                            (27.5, 2, 245 - YAW_MOTOR_DROP_Z_MM))
    roll = groups["roll"]
    move = bd.Pos(0, PITCH_MOTOR_SHIFT_Y_MM, 0)
    for name in ("pitch_ST3250_case", "pitch_motor_output", "pitch_drive_20T"):
        roll[name] = move * roll[name]
    roll["pitch_belt_159_3MGT_15"] = source.belt((87, 2, 219), (87, 51.5, 219), "x")
    del roll["pitch_belt_123_3MGT_15"]
    # A roll-carried exposed saddle needs room for actual clamp screws and a
    # verified connection to the fork. These two rails are only first stock.
    for side, x in (("medial", 43), ("lateral", 75)):
        roll[f"pitch_motor_{side}_stock"] = source.box((x, 35, 250), (3, 58, 5))
    # A U-shaped clamp *envelope* runs over, under and behind the servo with
    # nominal 0.5-0.9 mm case clearance. Its right/output side remains open.
    # Bolt holes, tabs and preload still need a print coupon and maker drawing.
    roll["pitch_motor_cage_top"] = source.box((59, 52.4, 246), (42, 49.2, 4))
    roll["pitch_motor_cage_bottom"] = source.box((53, 52.4, 216.3), (30, 49.2, 4))
    roll["pitch_motor_cage_medial"] = source.box((39.75, 51.5, 232.15), (2.5, 30, 33))
    roll["pitch_motor_cage_front"] = source.box((57, 27.3, 232.15), (22, 2, 33))
    roll["pitch_motor_cage_rear"] = source.box((53, 76, 232.15), (30, 2.5, 33))
    return groups


def volume(a, b):
    return max(source.both(a, b)[0])


def main():
    groups = moved_hardware()
    flat = {name: shape for group in groups.values() for name, shape in group.items()}
    # Only pairs that must be separated are screened. Bearings intentionally
    # meet seats and shafts; belt material intentionally engages pulleys.
    pairs = (
        ("yaw_ST3250_case", "pitch_ST3250_case"),
        ("roll_ST3250_case", "pitch_ST3250_case"),
        ("yaw_ST3250_case", "roll_ST3250_case"),
        ("pitch_ST3250_case", "pitch_fork_medial"),
        ("pitch_ST3250_case", "pitch_fork_lateral"),
        ("pitch_belt_159_3MGT_15", "yaw_ST3250_case"),
        ("pitch_belt_159_3MGT_15", "roll_ST3250_case"),
        ("pitch_belt_159_3MGT_15", "pitch_fork_medial"),
        ("pitch_belt_159_3MGT_15", "pitch_fork_lateral"),
        ("pitch_motor_medial_stock", "yaw_ST3250_case"),
        ("pitch_motor_lateral_stock", "yaw_ST3250_case"),
        ("pitch_motor_medial_stock", "roll_ST3250_case"),
        ("pitch_motor_lateral_stock", "roll_ST3250_case"),
        ("pitch_motor_medial_stock", "pitch_belt_159_3MGT_15"),
        ("pitch_motor_lateral_stock", "pitch_belt_159_3MGT_15"),
        *(("pitch_ST3250_case", f"pitch_motor_cage_{part}")
          for part in ("top", "bottom", "medial", "front", "rear")),
        *(("pitch_belt_159_3MGT_15", f"pitch_motor_cage_{part}")
          for part in ("top", "bottom", "medial", "front", "rear")),
        *(("yaw_ST3250_case", f"pitch_motor_cage_{part}")
          for part in ("top", "bottom", "medial", "front", "rear")),
        *(("roll_ST3250_case", f"pitch_motor_cage_{part}")
          for part in ("top", "bottom", "medial", "front", "rear")),
    )
    results = []
    for a, b in pairs:
        ab, error = source.both(flat[a], flat[b])
        results.append({"a": a, "b": b,
                        "overlap_mm3": None if error else round(max(ab), 3),
                        "error": error})
    # The 18 mm shift clears the drive cases in neutral; this verifies only a
    # possible bracket connection, not printed threads or bearing alignment.
    rails = {}
    for side in ("medial", "lateral"):
        rail = flat[f"pitch_motor_{side}_stock"]
        rails[side] = {
            "rail_to_pitch_fork_mm3": round(volume(rail, flat[f"pitch_fork_{side}"]), 3),
            "rail_to_motor_cage_top_mm3": round(volume(rail, flat["pitch_motor_cage_top"]), 3),
        }
    cage = {f"{part}_to_top_mm3": round(volume(flat[f"pitch_motor_cage_{part}"],
                                            flat["pitch_motor_cage_top"]), 3)
            for part in ("medial", "front", "rear")}
    cage["medial_to_bottom_mm3"] = round(volume(flat["pitch_motor_cage_medial"],
                                                 flat["pitch_motor_cage_bottom"]), 3)
    pose_screen = {}
    for label, (yaw, roll, pitch) in source.POSES.items():
        sides = {}
        for side, tag in ((-1, "left"), (1, "right")):
            moved = source.frame(groups, side, yaw, roll, pitch)
            g = {name: shape for group in moved.values() for name, shape in group.items()}
            hits = []
            for a, b in pairs:
                ab, error = source.both(g[a], g[b])
                if error or max(ab) > .05:
                    hits.append({"a": a, "b": b, "overlap_mm3": None if error else round(max(ab), 3),
                                 "error": error})
            sides[tag] = {"nonmating_hard_hits": hits,
                          "motor_case": {key: g[key] for key in
                                         ("yaw_ST3250_case", "roll_ST3250_case", "pitch_ST3250_case")}}
        bilateral = []
        for key in ("yaw_ST3250_case", "roll_ST3250_case", "pitch_ST3250_case"):
            ab, error = source.both(sides["left"]["motor_case"][key],
                                    sides["right"]["motor_case"][key])
            if error or max(ab) > .05:
                bilateral.append({"motor": key, "overlap_mm3": None if error else round(max(ab), 3),
                                  "error": error})
        pose_screen[label] = {tag: {"nonmating_hard_hits": sides[tag]["nonmating_hard_hits"]}
                              for tag in ("left", "right")}
        pose_screen[label]["bilateral_motor_hits"] = bilateral
    OUT.mkdir(parents=True, exist_ok=True)
    report = {
        "stage": "exposed right-hip hard-package static probe; NOT assembled or motion-qualified",
        "source": "experiments/compact_morphology/ape_design/swarm_cove_fork_hip/build.py",
        "source_pivots_mm": {"yaw_roll": source.PIVOT, "pitch": source.PITCH},
        "target_contract_hip_m": [0, -0.059, 0.231],
        "geometry_mismatch_note": "Source coupon yaw/roll z247 mm and pitch z219 mm; target layout currently uses z231 mm for all three. Do not promote without one shared joint manifest and body re-export.",
        "pitch_motor_shift_y_mm": PITCH_MOTOR_SHIFT_Y_MM,
        "yaw_motor_drop_z_mm": YAW_MOTOR_DROP_Z_MM,
        "pitch_belt_candidate": PITCH_BELT_PART,
        "pitch_belt_source": PITCH_BELT_SOURCE,
        "nonmating_pairs": results,
        "rail_interfaces": rails,
        "cage_interfaces": cage,
        "sampled_pose_screen": pose_screen,
        "caveats": ["only neutral pose", "no mounting screws", "no clamp or tensioner",
                    "no belt tooth or spline adapter", "no bilateral shared-frame sweep",
                    "no material strength or thermal qualification"],
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    neutral = []
    stage_colors = {"fixed": (100, 113, 119, 255), "yaw": (212, 166, 76, 255),
                    "roll": (83, 152, 145, 255), "thigh": (193, 125, 99, 255)}
    for pose_name, (yaw, roll, pitch) in source.POSES.items():
        scene = trimesh.Scene()
        for side, tag in ((-1, "left"), (1, "right")):
            for stage, group in source.frame(groups, side, yaw, roll, pitch).items():
                for name, shape in group.items():
                    full_name = f"{tag}_{stage}_{name}"
                    if pose_name == "stand":
                        neutral.append(shape)
                    vertices, faces = shape.tessellate(0.35, 0.4)
                    tri = trimesh.Trimesh(vertices=np.array([[v.X, v.Y, v.Z] for v in vertices]),
                                          faces=np.asarray(faces), process=False)
                    tri.visual.vertex_colors = ((68, 103, 138, 255) if "ST3250_case" in name
                                                else (45, 50, 51, 255) if "belt" in name
                                                else stage_colors[stage])
                    scene.add_geometry(tri, geom_name=full_name, node_name=full_name)
        scene.export(OUT / f"hip-fork-{pose_name}.glb")
    bd.export_step(bd.Compound(children=neutral), str(OUT / "hip-fork-neutral.step"))
    print(json.dumps({"nonmating_penetrations": [r for r in results if r["overlap_mm3"] and r["overlap_mm3"] > .05],
                      "unknown": [r for r in results if r["error"]],
                      "rail_interfaces": rails, "cage_interfaces": cage,
                      "pose_hard_hits": {name: sum(len(pose[tag]["nonmating_hard_hits"])
                                                     for tag in ("left", "right")) +
                                               len(pose["bilateral_motor_hits"])
                                         for name, pose in pose_screen.items()}}, indent=2))


if __name__ == "__main__":
    main()
