"""Screen elbow case, moving arm skin and closed-hand/gripper contradiction.

Published motor envelopes are used at absolute size; the sketch-06 exterior and
the retained v4 opposed-jaw MJCF supply the geometry being compared. This is a
coupon, not a completed gripper or proof of floor-grasp performance.
"""

from __future__ import annotations

import json
from pathlib import Path

import build123d as bd
import trimesh

import sketch_shell as shell
from hip_joint_coupon import add, case, cyl

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/ape-elbow-hand-coupon-20261002"
HEIGHTS = (350, 500, 600)
FLEXES = (0, 60, 120)
ST3215 = (45.2, 24.7, 35.0)
XL330 = (20.0, 34.0, 26.0)


def rotated(shape: bd.Shape, scale: float, angle: int) -> bd.Shape:
    pivot = (0, 96 * scale, 200 * scale)
    return bd.Pos(*pivot) * bd.Rot(0, -angle, 0) * bd.Pos(*(-v for v in pivot)) * shape.scale(scale)


def overlap(a: bd.Shape, b: bd.Shape) -> float:
    result = a.intersect(b)
    return round(sum(item.volume for item in result), 1) if result else 0.0


def screen(height: int) -> dict:
    s = height / 350
    upper = next(p.shape for p in shell.PARTS if p.name == "left_upper_arm_shell").scale(s)
    forearm = next(p.shape for p in shell.PARTS if p.name == "left_forearm_shell")
    direct = case("direct_elbow_ST3215", (0, 96 * s, 200 * s), ST3215,
                  ("left_upper_arm_shell", "left_forearm_shell"), s)
    proximal = case("upper_arm_ST3215", (-3 * s, 88 * s, 237 * s), ST3215,
                    ("left_upper_arm_shell",), s)
    gripper_motor = case("hand_XL330", (9 * s, 108 * s, 103 * s), XL330,
                         ("left_hand_shell",), s)
    # MJCF v4: opposed pads have centres y +/-15 mm and z=-90 mm from
    # the elbow, with 8x6x18 mm boxes at 35 cm. Their centres sit within
    # the smooth, closed hand shell, which has no aperture at any height.
    jaw_centres = [(0, (108 + sign * 15) * s, 110 * s)
                   for sign in (-1, 1)]
    jaw_inside_closed_hand = [shell.INTERIORS["left_hand_shell"].is_inside(
        bd.Vector(*(v / s for v in centre))) for centre in jaw_centres]
    poses = [{"elbow_flex_deg": a,
              "original_rigid_upper_forearm_overlap_mm3": overlap(upper, rotated(forearm, s, a))}
             for a in FLEXES]
    # Sketch-06 hand shell ends at z=78mm. Here we place the retained v4
    # opposed jaw tips with the coupon's elbow at z=200mm: centre z=110mm,
    # box half-height 9mm. Do not mix the v4 link lengths with CAD anchors.
    jaw_lowest_z = 101 * s
    return {
        "height_mm": height,
        "source_elbow": "Feetech ST3215-C018 45.2x24.7x35 mm, 55 g; 0.981 Nm rated",
        "source_gripper": "ROBOTIS XL330-M288-T 20x34x26 mm, 18 g; 0.52 Nm stall, no published continuous torque (vendor estimates ~0.10 Nm)",
        "direct_case": direct,
        "proximal_case": proximal,
        "gripper_case": gripper_motor,
        "original_closed_hand_contains_both_opposed_jaw_centres": all(jaw_inside_closed_hand),
        "jaw_centres_inside_closed_hand": jaw_inside_closed_hand,
        "gripper_force_at_25mm_ideal_N": {"vendor_estimated_continuous": 4.0,
                                           "stall_only": 20.8},
        "neutral_white_hand_shell_lowest_z_mm": round(78*s, 1),
        "neutral_simulated_jaw_pad_lowest_z_mm": round(jaw_lowest_z, 1),
        "poses": poses,
        "warning": "Neutral CAD/jaw heights are not a floor-grasp reach proof; squat, shoulder pose, payload, joint load, support shaft and arm-body collision not validated. The white hand is sealed around the current simulated jaw tips.",
    }


def render(report: dict, mode: str, flex: int) -> None:
    s = report["height_mm"] / 350
    scene = trimesh.Scene()
    upper = next(p.shape for p in shell.PARTS if p.name == "left_upper_arm_shell")
    lower = next(p.shape for p in shell.PARTS if p.name == "left_forearm_shell")
    hand = next(p.shape for p in shell.PARTS if p.name == "left_hand_shell")
    add(scene, "approved_upper_arm_skin", upper.scale(s), "shell")
    add(scene, "approved_forearm_skin", rotated(lower, s, flex), "shell")
    add(scene, "approved_closed_hand_skin", rotated(hand, s, flex), "shell")
    selected = report["direct_case" if mode == "direct" else "proximal_case"]
    motor = bd.Pos(*selected["center_mm"]) * bd.Box(*ST3215)
    if mode == "direct":
        motor = rotated(motor, 1, flex) if s == 1 else (
            bd.Pos(0, 96*s, 200*s) * bd.Rot(0, -flex, 0) *
            bd.Pos(0, -96*s, -200*s) * motor)
    add(scene, f"{mode}_ST3215_case", motor, "motor")
    add(scene, "elbow_supported_axis_concept", cyl(3, 45*s, (0, 96*s, 200*s), "y"), "shaft")
    gm = report["gripper_case"]
    add(scene, "source_XL330_in_closed_hand", rotated(
        bd.Pos(*gm["center_mm"]) * bd.Box(*XL330), 1, flex) if s == 1 else
        bd.Pos(0, 96*s, 200*s) * bd.Rot(0, -flex, 0) *
        bd.Pos(0, -96*s, -200*s) * bd.Pos(*gm["center_mm"]) * bd.Box(*XL330), "motor")
    for i, sign in enumerate((-1, 1)):
        jaw = bd.Pos(0, (108 + sign*15)*s, 110*s) * bd.Box(16*s, 6*s, 18*s)
        if flex:
            jaw = bd.Pos(0, 96*s, 200*s) * bd.Rot(0, -flex, 0) * bd.Pos(0, -96*s, -200*s) * jaw
        add(scene, f"trapped_jaw_{i}_from_retained_MJCF", jaw, "gear")
    scene.export(OUT / f"elbow-{report['height_mm']}-{mode}-{flex}.glb")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    shell.build()
    reports = [screen(h) for h in HEIGHTS]
    for report in reports:
        for mode in ("direct", "proximal"):
            for flex in FLEXES:
                render(report, mode, flex)
        print(report["height_mm"], "mm: motor case failures direct/proximal/gripper",
              *(len(report[k]["outside_corners"]) for k in
                ("direct_case", "proximal_case", "gripper_case")),
              "shell overlap at 120deg",
              report["poses"][-1]["original_rigid_upper_forearm_overlap_mm3"], flush=True)
    (OUT / "screen.json").write_text(json.dumps(reports, indent=2) + "\n")
    (OUT / "index.html").write_text(Path(__file__).with_name("elbow_hand_viewer.html").read_text())


if __name__ == "__main__":
    main()
