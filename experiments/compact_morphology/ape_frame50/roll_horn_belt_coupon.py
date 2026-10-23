"""Small hip-roll drive coupon: leave room for a real 25T servo horn.

The 20 mm horn is a conservative catalog-size envelope, not a selected horn
or a spline/bolt drawing. This only checks hard stock, not belt capacity.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import build123d as bd
import trimesh

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiments.compact_morphology.ape_design.swarm_cove_fork_hip import build as hip
from experiments.compact_morphology.ape_frame50.pelvis_tube_candidate import (
    candidate, intersection, mesh,
)

OUT = ROOT / "artifacts/ape-frame50-roll-drive-coupon"


def main():
    hw, _ = candidate()
    yaw, roll = hw["yaw"], hw["roll"]
    # Feetech specifies a 25T/5.9 mm output and M3x6 axial screw. A
    # Feetech-distributed 25T round-horn example is 20x20x4.5 mm and 5 g;
    # its exact spline/bolt compatibility with ST-3250 remains to verify.
    horn = hip.cyl(10, 4.5, (59, -43, 278.5), (0, 1, 0))
    old_belt = hip.belt((59, -43, 247), (59, -43, 278.5), "y", width=15)
    old_drive = hip.cyl(hip.R, 15, (59, -43, 278.5), (0, 1, 0))
    old = {
        "horn_vs_15mm_belt_mm3": intersection(horn, old_belt),
        "horn_vs_drive_pulley_mm3": intersection(horn, old_drive),
    }
    # Gates catalogs a 123-3MGT-09 belt. Shift its two 20T pulleys and
    # 9 mm belt 5.5 mm toward the fork; this leaves 1 mm horn-to-belt
    # axial separation while keeping the same 31.5 mm centers and ratio.
    # Reconstruct this historical horn-only comparison explicitly. The
    # current whole-body candidate has since moved to a 25T/6 mm coupler,
    # flanged 14 mm pulleys and a farther-forward roll motor.
    yaw["roll_belt_123_3MGT_09_coupon"] = hip.belt(
        (59, -37.5, 247), (59, -37.5, 278.5), "y", width=9)
    yaw["roll_horn_20mm_coupon"] = horn
    yaw["roll_drive_20T"] = hip.cyl(hip.R, 9, (59, -37.5, 278.5), (0, 1, 0))
    roll["roll_driven_20T"] = hip.cyl(hip.R, 9, (59, -37.5, 247), (0, 1, 0))
    pairs = [
        ("yaw", "roll_horn_20mm_coupon", "yaw", "roll_belt_123_3MGT_09_coupon"),
        ("yaw", "roll_horn_20mm_coupon", "yaw", "roll_drive_20T"),
        ("yaw", "roll_belt_123_3MGT_09_coupon", "yaw", "yaw_fork_front"),
        ("yaw", "roll_belt_123_3MGT_09_coupon", "roll", "roll_spindle_front"),
        ("yaw", "roll_belt_123_3MGT_09_coupon", "roll", "pitch_fork_bridge_front"),
        ("yaw", "roll_belt_123_3MGT_09_coupon", "roll", "roll_driven_20T"),
        ("yaw", "roll_horn_20mm_coupon", "fixed", "yaw_housing_outboard_rib"),
    ]
    pose_hits = {}
    for pose, angles in hip.POSES.items():
        side_hits = {}
        for side, sign in (("left", -1), ("right", 1)):
            stages = hip.frame(hw, sign, *angles)
            side_hits[side] = [
                {"a": f"{a_stage}_{a}", "b": f"{b_stage}_{b}",
                 "overlap_mm3": round(intersection(stages[a_stage][a],
                                                    stages[b_stage][b]), 3)}
                for a_stage, a, b_stage, b in pairs
                if intersection(stages[a_stage][a], stages[b_stage][b]) > 0.05
                and (a, b) != ("roll_belt_123_3MGT_09_coupon", "roll_driven_20T")
            ]
        pose_hits[pose] = side_hits
    OUT.mkdir(parents=True, exist_ok=True)
    bd.export_step(bd.Compound(children=[horn, yaw["roll_drive_20T"],
                                         yaw["roll_belt_123_3MGT_09_coupon"],
                                         roll["roll_driven_20T"]]),
                   str(OUT / "roll-horn-belt-coupon.step"))
    for label, parts in (
        ("old", (("horn", horn, (223, 142, 86, 255)),
                  ("15mm_belt", old_belt, (180, 62, 57, 255)),
                  ("old_drive", old_drive, (80, 118, 153, 255)))),
        ("new", (("horn", horn, (223, 142, 86, 255)),
                  ("9mm_belt", yaw["roll_belt_123_3MGT_09_coupon"], (66, 152, 123, 255)),
                  ("new_drive", yaw["roll_drive_20T"], (80, 118, 153, 255)),
                  ("new_driven", roll["roll_driven_20T"], (80, 118, 153, 255)))),
    ):
        scene = trimesh.Scene()
        for name, shape, color in parts:
            scene.add_geometry(mesh(shape, color), geom_name=name, node_name=name)
        scene.export(OUT / f"roll-horn-belt-{label}.glb")
    (OUT / "index.html").write_text(Path(__file__).with_name(
        "roll_horn_belt_viewer.html").read_text())
    result = {
        "status": "unselected 25T horn envelope and roll-belt fit study only",
        "source_feetech": "https://www.feetechrc.com/en/562636.html",
        "horn_size_source": "https://www.mantech.co.za/Datasheets/Products/FK-RP-001-181020A.pdf",
        "belt_source": "https://gates.com/content/dam/documents-library/catalogs/power-transmission-catalog.pdf",
        "old_15mm_plane": old,
        "candidate": {"belt": "Gates 123-3MGT-09 envelope", "width_mm": 9,
                      "pulley_y_start_mm": -37.5,
                      "horn_to_belt_axial_gap_mm": 1.0,
                      "center_distance_mm": 31.5, "ratio": "20T:20T"},
        "pose_nonmating_hits": pose_hits,
        "open": ["FK-RP-001 spline compatibility with ST-3250 and exact hole pattern",
                 "M3 screw and 4-hole adapter from horn to 6 mm shaft",
                 "belt tension/torque capacity and pulley side flanges",
                 "continuous motion, fits, service and actual print strength"],
    }
    (OUT / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"old": old,
                      "candidate_pose_hits": {p: sum(len(v) for v in s.values())
                                              for p, s in pose_hits.items()}}, indent=2))


if __name__ == "__main__":
    main()
