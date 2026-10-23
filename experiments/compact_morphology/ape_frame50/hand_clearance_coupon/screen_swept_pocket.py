"""Source-shape CAD coupon for a distal Cove opposed-jaw clearance pocket."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import build123d as bd

from experiments.compact_morphology.ape_frame50 import direct_skeleton_candidate as direct


ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "artifacts/ape-cove50-hand-clearance-coupon/swept-pocket-v1"
JAW_ANGLES_DEG = tuple(28.64788975654116 * i / 20 for i in range(21))  # 0..0.5 rad


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def side_box(side_sign: int, center: tuple[float, float, float],
             size: tuple[float, float, float]):
    return direct.box((center[0], side_sign * center[1], center[2]), size)


def source_shapes(side: str):
    raw = dict(direct.stage_frames())
    forearm = raw[f"{side}_forearm"]
    frame = next(iter(forearm.values()))
    for shape in list(forearm.values())[1:]:
        frame = frame + shape
    drive = next(d for d in direct.drives() if d.name == f"{side}_gripper")
    frame = frame + direct.motor_cradle(drive)
    frame = frame - direct.box(drive.center, tuple(v + 1 for v in drive.size))
    jaw = raw[f"{side}_moving_jaw"]["jaw_frame"]
    case = direct.box(drive.center, drive.size)
    return frame, jaw, case, drive


def candidate_shapes(side: str, pocket_z_min: float = 118,
                     high_bridge: bool = False, narrow_bridge: bool = False):
    sign = 1 if side == "left" else -1
    source_frame, source_jaw, case, drive = source_shapes(side)
    pocket = side_box(sign, (-3, 163, (pocket_z_min + 155) / 2),
                      (18, 42, 155 - pocket_z_min))
    frame = source_frame - pocket
    palm_root = side_box(sign, (-1, 124, 132), (10, 24, 12))
    palm_rail = side_box(sign, (-18, 118, 132), (40, 6, 12))
    palm_pad = side_box(sign, (-35, 118, 132), (12, 6, 26))
    frame = frame + palm_root + palm_rail + palm_pad
    jaw_rail = side_box(sign, (-21, 165, 132), (28, 8, 8))
    jaw_bridge = side_box(sign, (-35, 157 if narrow_bridge else 154,
                                 147 if high_bridge else 132),
                          (10, 18 if narrow_bridge else 24,
                           6 if high_bridge else 10))
    jaw_riser = (side_box(sign, (-35, 165, 140), (10, 8, 20))
                 if high_bridge else None)
    jaw_pad = side_box(sign, (-35, 146, 132), (12, 6, 26))
    jaw = source_jaw + jaw_rail + jaw_bridge + jaw_pad
    if jaw_riser is not None:
        jaw = jaw + jaw_riser
    cube = side_box(sign, (-35, 132, 132), (20, 20, 20))
    return {
        "source_frame": source_frame, "source_jaw": source_jaw,
        "frame": frame, "jaw": jaw, "case": case, "cube": cube,
        "pocket": pocket, "drive": drive,
        "palm_root": palm_root, "palm_rail": palm_rail, "palm_pad": palm_pad,
        "jaw_rail": jaw_rail, "jaw_bridge": jaw_bridge, "jaw_pad": jaw_pad,
        "jaw_riser": jaw_riser,
    }


def intersection_mm3(a, b) -> float:
    return round(float((a & b).volume), 3)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--pocket-z-min-mm", type=float, default=118)
    parser.add_argument("--high-bridge", action="store_true")
    parser.add_argument("--narrow-bridge", action="store_true")
    parser.add_argument("--left-axis-sign", type=int, choices=(-1, 1), default=-1)
    args = parser.parse_args()
    if args.pocket_z_min_mm not in (114, 118):
        raise ValueError("Only predeclared v1 and v2 pocket floors are supported")
    out = args.out.resolve()
    expected_name = ("swept-pocket-v4" if args.narrow_bridge else
                     "swept-pocket-v3" if args.high_bridge else
                     "swept-pocket-v1" if args.pocket_z_min_mm == 118 else
                     "swept-pocket-v2")
    if (out.name != expected_name or
            (args.high_bridge and args.narrow_bridge) or
            ((args.high_bridge or args.narrow_bridge) and args.pocket_z_min_mm != 114) or
            (args.left_axis_sign == 1 and not (args.high_bridge or args.narrow_bridge))):
        raise ValueError("Pocket floor and artifact version mismatch")
    rows = []
    parts = {}
    for side in ("left", "right"):
        shapes = candidate_shapes(side, args.pocket_z_min_mm,
                                  args.high_bridge, args.narrow_bridge)
        drive = shapes["drive"]
        frame = shapes["frame"]
        jaw = shapes["jaw"]
        case = shapes["case"]
        cube = shapes["cube"]
        axis = bd.Axis(drive.pivot,
                       (args.left_axis_sign if side == "left" else -1, 0, 0))
        angles = []
        for angle in JAW_ANGLES_DEG:
            moved = jaw.rotate(axis, angle)
            moved_bridge = shapes["jaw_bridge"].rotate(axis, angle)
            moved_pad = shapes["jaw_pad"].rotate(axis, angle)
            moved_riser = (shapes["jaw_riser"].rotate(axis, angle)
                           if shapes["jaw_riser"] is not None else None)
            angles.append({
                "jaw_rad": round(angle / 57.29577951308232, 5),
                "jaw_frame_overlap_mm3": intersection_mm3(moved, frame),
                "motor_case_overlap_mm3": intersection_mm3(moved, case),
                "cube_overlap_mm3": intersection_mm3(moved, cube),
                "bridge_cube_overlap_mm3": intersection_mm3(moved_bridge, cube),
                "pad_cube_overlap_mm3": intersection_mm3(moved_pad, cube),
                "riser_cube_overlap_mm3": (intersection_mm3(moved_riser, cube)
                                            if moved_riser is not None else 0.0),
            })
        attachment = {
            "palm_root_to_source_mm3": intersection_mm3(
                shapes["palm_root"], shapes["source_frame"]),
            "palm_rail_to_root_mm3": intersection_mm3(
                shapes["palm_rail"], shapes["palm_root"]),
            "palm_pad_to_rail_mm3": intersection_mm3(
                shapes["palm_pad"], shapes["palm_rail"]),
            "jaw_rail_to_source_mm3": intersection_mm3(
                shapes["jaw_rail"], shapes["source_jaw"]),
            "jaw_bridge_to_rail_mm3": intersection_mm3(
                shapes["jaw_bridge"], shapes["jaw_rail"]),
            "jaw_pad_to_bridge_mm3": intersection_mm3(
                shapes["jaw_pad"], shapes["jaw_bridge"]),
        }
        if args.high_bridge:
            attachment.pop("jaw_bridge_to_rail_mm3")
            attachment["jaw_riser_to_rail_mm3"] = intersection_mm3(
                shapes["jaw_riser"], shapes["jaw_rail"])
            attachment["jaw_bridge_to_riser_mm3"] = intersection_mm3(
                shapes["jaw_bridge"], shapes["jaw_riser"])
        row = {
            "side": side,
            "source_frame_volume_mm3": round(float(shapes["source_frame"].volume), 3),
            "candidate_frame_volume_mm3": round(float(frame.volume), 3),
            "source_jaw_frame_volume_mm3": round(float(shapes["source_jaw"].volume), 3),
            "candidate_jaw_frame_volume_mm3": round(float(jaw.volume), 3),
            "petg_hf_stock_mass_delta_g_at_1p28_g_cm3": round(
                (float(frame.volume) + float(jaw.volume) -
                 float(shapes["source_frame"].volume) -
                 float(shapes["source_jaw"].volume)) * 1.28 / 1000, 3),
            "source_frame_solids": len(shapes["source_frame"].solids()),
            "candidate_frame_solids": len(frame.solids()),
            "candidate_jaw_solids": len(jaw.solids()),
            "frame_motor_case_overlap_mm3": intersection_mm3(frame, case),
            "frame_open_cube_overlap_mm3": intersection_mm3(frame, cube),
            "jaw_open_cube_overlap_mm3": angles[0]["cube_overlap_mm3"],
            "attachment_overlap_mm3": attachment,
            "maximum_jaw_frame_overlap_mm3": max(
                a["jaw_frame_overlap_mm3"] for a in angles),
            "maximum_motor_case_overlap_mm3": max(
                a["motor_case_overlap_mm3"] for a in angles),
            "maximum_bridge_cube_overlap_mm3": max(
                a["bridge_cube_overlap_mm3"] for a in angles),
            "maximum_riser_cube_overlap_mm3": max(
                a["riser_cube_overlap_mm3"] for a in angles),
            "angles": angles,
        }
        row["screen_pass"] = (
            row["candidate_frame_solids"] == 1 and row["candidate_jaw_solids"] == 1 and
            row["frame_motor_case_overlap_mm3"] == 0 and
            row["frame_open_cube_overlap_mm3"] == 0 and
            row["jaw_open_cube_overlap_mm3"] == 0 and
            row["maximum_jaw_frame_overlap_mm3"] == 0 and
            row["maximum_motor_case_overlap_mm3"] == 0 and
            all(v > 0 for v in attachment.values()) and
            (not (args.high_bridge or args.narrow_bridge) or
             (row["maximum_bridge_cube_overlap_mm3"] == 0 and
              row["maximum_riser_cube_overlap_mm3"] == 0)))
        rows.append(row)
        parts[side] = (frame, jaw)
    report = {
        "schema": "robot-learning-lab.cove50-swept-pocket-cad-screen.v1",
        "selected_pointer_sha256": sha(ROOT / "robot/ape-current-design.json"),
        "selected_generator_sha256": sha(Path(direct.__file__)),
        "selected_visual_body_sha256": sha(
            ROOT / "artifacts/ape-frame50-direct-skeleton/ape-frame50-direct-visual.xml"),
        "plan_sha256": sha(out / "plan.json"),
        "pocket_z_min_mm": args.pocket_z_min_mm,
        "high_bridge": args.high_bridge,
        "narrow_bridge": args.narrow_bridge,
        "left_axis_sign": args.left_axis_sign,
        "angles_per_side": len(JAW_ANGLES_DEG),
        "stock_density_g_cm3_assumption": 1.28,
        "sides_passing": sum(row["screen_pass"] for row in rows),
        "rows": rows,
        "limits": [
            "This is a source-shape CAD coupon. Motor-case placement and 21-axis kinematics are retained, but frame strength, bearing support and fastening are unverified.",
            "The pocket is a conservative box around sampled jaw motion; only 21 joint angles were intersected, not an exact continuous swept volume.",
            "A 20 mm cube at the proposed open jaw is a static local geometry check, not a floor-entry trajectory, force, lift or possession result.",
            "No selected design pointer or policy checkpoint was changed.",
        ],
    }
    out.mkdir(parents=True, exist_ok=True)
    for side, (frame, jaw) in parts.items():
        bd.export_step(frame, out / f"{side}-forearm-frame.step")
        bd.export_step(jaw, out / f"{side}-moving-jaw-frame.step")
    report["step_sha256"] = {
        p.name: sha(p) for p in sorted(out.glob("*.step"))}
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (out / "summary-matrix.json").write_text(json.dumps({
        "schema": "robot-learning-lab.cove50-swept-pocket-summary.v1",
        "cases": [{k: v for k, v in row.items() if k != "angles"} for row in rows]},
        indent=2) + "\n")
    print(json.dumps({"sides_passing": report["sides_passing"], "rows": [
        {k: v for k, v in row.items() if k != "angles"} for row in rows]}, indent=2))


if __name__ == "__main__":
    main()
