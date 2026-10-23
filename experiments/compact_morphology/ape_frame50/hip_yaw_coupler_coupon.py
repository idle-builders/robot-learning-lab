"""Fit a source 25T-to-6 mm coupler under the supported hip-yaw shaft.

This is a removable coupon on the current ape anatomy. It resolves the
servo-spline/long-shaft geometric gap and screens catalog bearing static load.
It does not claim spline fit, shaft strength, housing fits, or bearing life.
"""

from __future__ import annotations

import hashlib
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
from experiments.compact_morphology.ape_frame50.roll_coupler_package import (
    STEP, STEP_SHA, URL, exact_intersection,
)

OUT = ROOT / "artifacts/ape-frame50-hip-yaw-coupler"
POSES = ROOT / "experiments/compact_morphology/ape_frame50/coupled_motion_probes.json"


def coupon():
    return candidate()


def overlaps(a: bd.Shape, b: bd.Shape) -> bool:
    aa, bb = a.bounding_box(), b.bounding_box()
    return all(getattr(aa.min, axis) < getattr(bb.max, axis) - .05 and
               getattr(bb.min, axis) < getattr(aa.max, axis) - .05
               for axis in "XYZ")


def screen_coordinated_poses() -> None:
    """Recheck only changed parts against the 77 existing coordinated poses."""
    OUT.mkdir(parents=True, exist_ok=True)
    hw, shared = coupon()
    names = (
        "yaw_motor_coupler_4001", "yaw_motor_output",
        "yaw_motor_axial_DIN471_6_ring", "yaw_motor_lower_washer",
        "yaw_motor_inner_spacer", "yaw_motor_bearing_low_F606_ZZ",
        "yaw_motor_bearing_high_F606_ZZ", "yaw_motor_housing_tube")
    rows = []
    for entry in json.loads(POSES.read_text())["poses"]:
        for side, sign in (("left", 1), ("right", -1)):
            q = entry[side]
            stages = hip.frame(hw, sign, q["hip_yaw"] / sign,
                               -q["hip_roll"], q["hip_pitch"])
            hits = []
            for new_name in names:
                fixed = stages["fixed"][new_name]
                for stage in ("yaw", "roll", "thigh"):
                    for name, moving in stages[stage].items():
                        if overlaps(fixed, moving):
                            amount = exact_intersection(fixed, moving)
                            if amount > .05:
                                hits.append([new_name, stage, name, round(amount, 3)])
            spindle = stages["thigh"]["pitch_spindle"]
            for name, fixed in {**shared, **stages["fixed"]}.items():
                if name in names or not overlaps(spindle, fixed):
                    continue
                amount = exact_intersection(spindle, fixed)
                if amount > .05:
                    hits.append(["shortened_pitch_spindle", "fixed", name,
                                 round(amount, 3)])
            rows.append({"pose": entry["name"], "side": side, "hits": hits})
    report = {"scope": "Changed coupon fixed parts versus hip yaw/roll/thigh and shortened pitch spindle versus unchanged fixed/shared; 77 authored poses per side; not lower-limb or continuous motion",
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "pose_file": str(POSES.relative_to(ROOT)),
              "pose_sha256": hashlib.sha256(POSES.read_bytes()).hexdigest(),
              "pose_sides": len(rows),
              "hard_hits": sum(len(row["hits"]) for row in rows),
              "reports": rows}
    (OUT / "coordinated-final-changed-parts.json").write_text(
        json.dumps(report, indent=2) + "\n")
    print(json.dumps({"pose_sides": len(rows), "hard_hits": report["hard_hits"]}))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    hw, shared = coupon()
    fixed = hw["fixed"]
    coupler = fixed["yaw_motor_coupler_4001"]
    bbox = coupler.bounding_box()
    avoid = {"yaw_motor_output", "yaw_motor_coupler_4001"}
    fixed_hits = {name: round(volume, 3) for name, solid in fixed.items()
                  if name not in avoid
                  and (volume := exact_intersection(coupler, solid)) > .05}
    shared_hits = {name: round(volume, 3) for name, solid in shared.items()
                   if (volume := exact_intersection(coupler, solid)) > .05}
    pose_hits = {}
    package_pose_hits = {}
    new_package = (
        "yaw_motor_coupler_4001", "yaw_motor_output",
        "yaw_motor_axial_DIN471_6_ring", "yaw_motor_lower_washer",
        "yaw_motor_inner_spacer", "yaw_motor_bearing_low_F606_ZZ",
        "yaw_motor_bearing_high_F606_ZZ", "yaw_motor_housing_tube")
    for label, angles in hip.POSES.items():
        sides = {}
        package_sides = {}
        for side, sign in (("left", -1), ("right", 1)):
            stages = hip.frame(hw, sign, *angles)
            static = stages["fixed"]["yaw_motor_coupler_4001"]
            hits = []
            package_hits = []
            for stage in ("yaw", "roll", "thigh"):
                for name, solid in stages[stage].items():
                    volume = exact_intersection(static, solid)
                    if volume > .05:
                        hits.append([stage, name, round(volume, 3)])
                    for package_name in new_package:
                        part = stages["fixed"][package_name]
                        if package_name == "yaw_motor_coupler_4001" or not overlaps(part, solid):
                            continue
                        amount = exact_intersection(part, solid)
                        if amount > .05:
                            package_hits.append([package_name, stage, name,
                                                 round(amount, 3)])
            sides[side] = hits
            package_sides[side] = package_hits
        pose_hits[label] = sides
        package_pose_hits[label] = package_sides
    axial_pair_hits = {}
    for a, b in (("yaw_motor_axial_DIN471_6_ring", "yaw_motor_output"),
                 ("yaw_motor_axial_DIN471_6_ring", "yaw_motor_coupler_4001"),
                 ("yaw_motor_lower_washer", "yaw_motor_output"),
                 ("yaw_motor_lower_washer", "yaw_motor_bearing_low_F606_ZZ"),
                 ("yaw_motor_inner_spacer", "yaw_motor_output"),
                 ("yaw_motor_inner_spacer", "yaw_motor_housing_tube"),
                 ("yaw_motor_bearing_high_F606_ZZ", "yaw_drive_20T")):
        axial_pair_hits[f"{a}/{b}"] = round(exact_intersection(fixed[a], fixed[b]), 3)
    show = {
        "yaw_ST3250_case": (fixed["yaw_ST3250_case"], (77, 114, 142, 255)),
        "source_25T_to_6mm_coupler": (coupler, (221, 145, 78, 255)),
        "candidate_6mm_shaft": (fixed["yaw_motor_output"], (170, 177, 177, 255)),
        "DIN471_lower_clip": (fixed["yaw_motor_axial_DIN471_6_ring"], (229, 167, 74, 255)),
        "lower_inner_washer": (fixed["yaw_motor_lower_washer"], (193, 190, 176, 255)),
        "inner_race_spacer": (fixed["yaw_motor_inner_spacer"], (193, 190, 176, 255)),
        "lower_F606_ZZ_bearing": (fixed["yaw_motor_bearing_low_F606_ZZ"], (198, 202, 197, 255)),
        "upper_F606_ZZ_bearing": (fixed["yaw_motor_bearing_high_F606_ZZ"], (198, 202, 197, 255)),
        "shortened_housing_tube": (fixed["yaw_motor_housing_tube"], (143, 154, 149, 255)),
        "housing_arm": (fixed["yaw_motor_housing_arm"], (143, 154, 149, 255)),
        "drive_20T_pulley": (fixed["yaw_drive_20T"], (72, 145, 168, 255)),
        "yaw_belt": (fixed["yaw_belt_123_3MGT_15"], (91, 172, 134, 255)),
    }
    scene = trimesh.Scene()
    for name, (solid, color) in show.items():
        scene.add_geometry(mesh(solid, color), geom_name=name, node_name=name)
    scene.export(OUT / "hip-yaw-coupler.glb")
    bd.export_step(bd.Compound(children=[solid for solid, _ in show.values()]),
                   str(OUT / "hip-yaw-coupler.step"))
    (OUT / "index.html").write_text(Path(__file__).with_name(
        "hip_yaw_coupler_viewer.html").read_text())
    # Elementary overhung two-support beam screen. Pulley radial force is
    # not determined by transmitted torque alone: T/r is only the belt's
    # tension difference. Apply 1x/2x total-radial-force sensitivities.
    rated_torque_nm, stall_torque_nm = 1.569064, 4.903325
    pulley_pitch_radius_m = (20 * 3 / (2 * 3.141592653589793)) / 1000
    low_z, high_z, pulley_z = 221., 249., 262.5
    arm_ratio = (pulley_z - low_z) / (high_z - low_z)
    loads = {}
    for label, torque in (("rated", rated_torque_nm), ("stall", stall_torque_nm)):
        delta = torque / pulley_pitch_radius_m
        loads[label] = {"belt_tension_difference_n": round(delta, 2),
                        "upper_reaction_n_at_1x_delta": round(delta * arm_ratio, 2),
                        "upper_reaction_n_at_2x_delta": round(2 * delta * arm_ratio, 2),
                        "lower_reaction_n_at_2x_delta": round(
                            2 * delta * (arm_ratio - 1), 2)}
    report = {
        "stage": "source-step hip-yaw servo-to-supported-shaft coupon",
        "source_page": "https://www.gobilda.com/4001-series-clamping-servo-to-shaft-coupler-25-tooth-spline-to-6mm-round-bore/",
        "source_step_url": URL, "source_step_sha256": STEP_SHA,
        "source_step_file": str(STEP.relative_to(ROOT)),
        "source_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "coupler_bounds_mm": [[bbox.min.X, bbox.min.Y, bbox.min.Z],
                              [bbox.max.X, bbox.max.Y, bbox.max.Z]],
        "coupler_to_case_axial_gap_mm": .5,
        "coupler_to_lower_bearing_axial_gap_mm": 3.,
        "coupler_to_lower_retaining_ring_axial_gap_mm": 1.1,
        "coupler_to_housing_tube_axial_gap_mm": 4.2,
        "shaft_in_coupler_axial_overlap_mm": 12.,
        "shaft_to_coupler_hard_overlap_mm3": round(exact_intersection(
            coupler, fixed["yaw_motor_output"]), 3),
        "coupler_fixed_hard_hits_mm3": fixed_hits,
        "coupler_shared_hard_hits_mm3": shared_hits,
        "coupler_moving_hard_hits_mm3": pose_hits,
        "new_package_moving_hard_hits_mm3": package_pose_hits,
        "axial_nonmating_pair_hard_overlap_mm3": axial_pair_hits,
        "bearing_source_page": "https://koyo.jtekt.co.jp/en/products/detail/print.php?pno=F606+ZZ",
        "bearing_catalog_basic_static_radial_n": 840,
        "bearing_station_centers_z_mm": [low_z, high_z],
        "bearing_station_spacing_mm": high_z - low_z,
        "pulley_center_z_mm": pulley_z,
        "pulley_overhang_past_upper_center_mm": pulley_z - high_z,
        "pulley_pitch_radius_m": pulley_pitch_radius_m,
        "beam_screen_assumptions": "Point radial belt load at pulley center; simple two-bearing reaction; 1x and 2x transmitted-tension-difference sensitivities; no axial/shock/fatigue or seat deflection; static rating is not permissible working load",
        "beam_screen": loads,
        "axial_stack_z_mm": {"groove_and_DIN471": [216.1, 216.8],
                              "washer": [217., 217.8],
                              "lower_bearing": [218., 224.],
                              "inner_spacer": [224., 246.],
                              "upper_bearing": [246., 252.],
                              "shaft_shoulder": [252., 253.],
                              "pulley": [255., 270.]},
        "axial_stop_source": "https://www.beneri.com/en/prodotto/din-471",
        "housing_tube_to_arm_overlap_mm3": round(intersection(
            fixed["yaw_motor_housing_tube"], fixed["yaw_motor_housing_arm"]), 3),
        "open": ["Feetech ST3250/goBILDA H25T spline and clamp torque unverified",
                 "6 mm shaft stock, fits, bending/fatigue and 20T set-screw duty unverified",
                 "0.2-4.2 mm axial gaps are nominal and need tolerance/service checks",
                 "bearing fits, DIN 471 groove/ring thrust capacity and printed housing load path not qualified",
                 "stall at 2x belt tension difference exceeds catalog basic static radial rating",
                 "continuous multi-axis and tolerance clearance unqualified; see separate sampled-gait and coordinated-pose reports"],
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"fixed_hits": fixed_hits, "shared_hits": shared_hits,
                      "pose_hits": pose_hits,
                      "housing_join_mm3": report["housing_tube_to_arm_overlap_mm3"]}, indent=2))


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--screen-coordinated":
        screen_coordinated_poses()
    else:
        main()
