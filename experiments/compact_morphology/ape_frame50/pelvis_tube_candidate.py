"""A slimmer, bearing-supported shared pelvis for the exposed 50 cm hip.

This retains source ST3250 cases and the existing fork stages, but replaces
the colliding fixed yaw plates with two narrow bearing housings on each side.
It screens sampled rigid motion only; fasteners, belt engagement, manufacture
and loaded duty remain future work. Coordinates are Cove millimetres.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys

import build123d as bd
import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiments.compact_morphology.ape_frame50.hip_fork_exposed_probe import moved_hardware
from experiments.compact_morphology.ape_design.swarm_cove_fork_hip import build as hip

OUT = ROOT / "artifacts/ape-frame50-pelvis-tube"


def roll_pulley(z):
    """MISUMI GPA20GT3090-A-P6 conservative flanged, 6 mm-bore envelope.

    20 teeth, 3GT, 9 mm belt: OD 18.34, flange OD 23, width 14 mm.
    The pitch/tooth profile and M4 set-screw holes are not solid-modeled.
    """
    axis = (0, 1, 0)
    core = hip.ring(9.17, 3, 10.3, (59, -37.15, z), axis)
    front = hip.ring(11.5, 3, 1.85, (59, -39, z), axis)
    rear = hip.ring(11.5, 3, 1.85, (59, -26.85, z), axis)
    return core + front + rear


def intersection(a, b):
    aa, bb = a.bounding_box(), b.bounding_box()
    if any(getattr(aa.max, key) < getattr(bb.min, key) or
           getattr(bb.max, key) < getattr(aa.min, key) for key in "XYZ"):
        return 0.0
    # The imported goBILDA STEP contains two separate solids. OCC's direct
    # compound boolean can report zero against a real overlap; screen each
    # solid so the main frame and the joint coupon use the same BRep result.
    if len(a.solids()) > 1 or len(b.solids()) > 1:
        return sum(intersection(x, y) for x in a.solids() for y in b.solids())
    both, error = hip.both(a, b)
    if error:
        raise RuntimeError(error)
    return max(both)


def candidate():
    hw = moved_hardware()
    fixed, yaw = hw["fixed"], hw["yaw"]
    for name in list(fixed):
        if name.startswith(("yaw_bearing_", "yaw_seat_", "yaw_bulkhead_")):
            del fixed[name]
    # JTEKT F606 ZZ is a catalog 6x17x6 mm bearing, with a 19x1.2 mm
    # flange on its outer ring. Install the pair from opposite housing ends:
    # the 4.8 mm barrels enter the 17 mm bore and the flanges locate the
    # outer rings axially against the tube ends. B includes flange thickness.
    fixed["yaw_bearing_low_F606_ZZ"] = (
        hip.ring(8.5, 3, 6, (59, 2, 226.3)) +
        hip.ring(9.5, 8.5, 1.2, (59, 2, 226.3)))
    fixed["yaw_bearing_high_F606_ZZ"] = (
        hip.ring(8.5, 3, 6, (59, 2, 237.7)) +
        hip.ring(9.5, 8.5, 1.2, (59, 2, 242.5)))
    fixed["yaw_housing_tube"] = hip.ring(10.5, 8.5, 15, (59, 2, 227.5))
    fixed["yaw_housing_arm"] = (hip.box((59, -15, 225), (12, 34, 4)) -
                                hip.cyl(6.5, 6, (59, 2, 222)))
    # The raised tube no longer touches the old arm. This rib joins their
    # outboard stock and stays outside the observed medial pitch-seat sweep.
    fixed["yaw_housing_outboard_rib"] = hip.box((65, -7, 231), (8, 10, 14))
    # The lower flange occupies Z=226.3..227.5; the clip and washer extend
    # below it to Z=224.8. Relieve their pocket in arm/rib while retaining
    # the higher tube-to-frame structural overlap.
    for stock in ("yaw_housing_arm", "yaw_housing_outboard_rib"):
        fixed[stock] -= hip.cyl(9.6, 3.3, (59, 2, 224.6))
    fixed["yaw_housing_outboard_rib"] -= hip.cyl(8.6, 4.8, (59, 2, 227.5))
    # The learned walk came within 1.5 degrees of its pitch seat entering
    # the inboard underside of this arm. Relieve only that local corner;
    # the arm's rib and pelvis-crossbeam joins remain intact.
    fixed["yaw_housing_arm"] -= hip.box((54.5, -2.5, 223.75), (3, 7, 2.5))
    # A stepped 6/8 mm shaft bears on the high inner ring. A 5.4 mm inner
    # spacer reaches the low inner ring. A 0.8 mm washer and DIN 471-6
    # retaining ring close the stack below, with a catalog 5.7 mm groove.
    # This keeps the yaw shaft integral with its upper fork and permits
    # insertion from above; selected washer/shim and ring-groove strength
    # still need a loaded assembly check.
    yaw["yaw_spindle"] = (hip.cyl(3, 48.2, (59, 2, 224.8)) +
                           hip.cyl(4, 29.3, (59, 2, 243.7)) -
                           hip.ring(3, 2.85, 0.7, (59, 2, 224.8)))
    yaw["yaw_inner_spacer"] = hip.ring(4, 3, 5.4, (59, 2, 232.3))
    yaw["yaw_lower_washer"] = hip.ring(4, 3, 0.8, (59, 2, 225.5))
    yaw["yaw_axial_DIN471_6_ring"] = (hip.ring(6, 2.85, 0.7, (59, 2, 224.8)) -
                                      hip.box((59, -4.5, 225.15), (4, 4, 1)))
    # The original full-length roll spindle physically crossed the yaw
    # spindle. Two coaxial journals and a pair of x-bridges transmit roll
    # through the pitch-fork side plates while leaving the yaw axis open.
    # Finish the fork bores through both 3 mm cheeks: the source coupon's
    # cut stopped 0.5 mm short of the inner faces.
    for cheek, y0 in (("yaw_fork_front", -24), ("yaw_fork_rear", 20)):
        yaw[cheek] -= hip.cyl(6.5, 6, (59, y0, 247), (0, 1, 0))
    roll = hw["roll"]
    del roll["roll_spindle"]
    del roll["pitch_fork_bridge"]
    roll["roll_spindle_front"] = hip.cyl(3, 33, (59, -45, 247), (0, 1, 0))
    roll["roll_spindle_rear"] = hip.cyl(3, 15, (59, 16, 247), (0, 1, 0))
    # The original horn and 15 mm belt occupied the same space. A goBILDA
    # 4001-0025-0006 is an actual 25T spline-to-6 mm clamping coupler:
    # its manufacturer STEP is 17 mm long, with a 14 x 16 mm cross section.
    # Shift only the yaw-carried roll motor 17 mm forward, so the coupler
    # fits from Y=-60..-43 and a separate 6 mm shaft reaches a pair of
    # MISUMI 20T 3GT, 9 mm belt, 6 mm-bore flanged pulleys at -39..-25.
    # The clamp is asymmetric around its spline axis. Its source-STEP bounds
    # are X=51.5..65.5, Y=-60..-43 and Z=272..288 after placement;
    # a 0.1 mm-per-face rectangular envelope covers that entire outline.
    yaw["roll_clamping_coupler_4001"] = hip.box(
        (58.5, -51.5, 280), (14.2, 17.2, 16.2))
    yaw["roll_6mm_output_shaft"] = hip.cyl(3, 29, (59, -54, 278.5), (0, 1, 0))
    yaw["roll_drive_20T"] = roll_pulley(278.5)
    del yaw["roll_belt_123_3MGT_15"]
    yaw["roll_belt_123_3MGT_09"] = hip.belt(
        (59, -36.5, 247), (59, -36.5, 278.5), "y", width=9)
    roll["roll_driven_20T"] = roll_pulley(247)
    roll["pitch_fork_bridge_front"] = hip.box((59, -13.5, 247), (35, 4, 4))
    roll["pitch_fork_bridge_rear"] = hip.box((59, 17.5, 247), (35, 4, 4))
    # Source drawing p6: both ST3250 outputs leave the 35 mm case-depth
    # direction, not the 45.22 mm long side. Put the yaw output on Z and the
    # roll output on Y; the old axis/case pairs could not be assembled.
    # Move the yaw case below the pitch fork, just above the shared tray.
    # A 6 mm adapter and two external bearing stations span the rise to the
    # existing 20T belt plane. Their fatigue and horn attachment remain open.
    fixed["yaw_ST3250_case"] = hip.box((27.5, 2+10.11, 180), (24.72,45.22,35))
    fixed["yaw_motor_output"] = hip.cyl(3, 72, (27.5, 2, 198))
    fixed["yaw_motor_bearing_low_W618_6"] = hip.ring(6.5, 3, 3.5, (27.5, 2, 211))
    fixed["yaw_motor_bearing_high_W618_6"] = hip.ring(6.5, 3, 3.5, (27.5, 2, 238))
    fixed["yaw_motor_housing_tube"] = hip.ring(8.5, 6.5, 36, (27.5, 2, 208))
    fixed["yaw_motor_housing_arm"] = (hip.box((27.5, -15, 227.5), (12, 34, 4)) -
                                      hip.cyl(6.5, 6, (27.5, 2, 224.5)))
    fixed["yaw_motor_tray_pad"] = hip.box((27.5,2+10.11,161.7),(20,30,1.8))
    yaw["roll_ST3250_case"] = hip.box((59-10.11,-77.5,278.5), (45.22,35,24.72))
    yaw["roll_motor_output"] = hip.cyl(2.95, 3, (59,-60,278.5), (0,1,0))
    yaw["roll_motor_standoff"] = hip.cyl(3, float(np.linalg.norm((0,42,-4))),
                                        (59,-61,267), (0,42,-4))
    # The +18 mm pitch-motor relocation in moved_hardware also moves its
    # output, pulley and 159 mm belt to Y=51.5. Retain that alignment when
    # correcting case height; Y=33.5 overlapped the yaw fork and detached
    # the case from its drive proxy.
    roll["pitch_ST3250_case"] = hip.box((59, 51.5, 219 + 10.11),
                                         (35, 24.72, 45.22))
    roll["pitch_motor_cage_front"] = hip.box((57, 38, 232.15), (22, 2, 33))
    # Keep the thigh root on the spindle but outside the pitch-belt plane.
    hw["thigh"]["thigh_root_pad"] = hip.box((107, 2, 218), (9, 25, 14))
    shared = {
        "pelvis_front_crossbeam": hip.box((0, -32, 226), (120, 6, 6)),
        "front_spine": hip.box((0, -32, 272), (12, 6, 98)),
        # This tray lies below the pair of yaw cases, leaving case removal
        # and motor-clamp fasteners for the next assembly operation.
        "yaw_motor_tray": hip.box((0, 1, 159), (94, 61, 4)),
        "tray_to_front_spine": hip.box((0, -29.5, 193), (12, 7, 66)),
    }
    # Keep the motor-side yaw coupler/bearings and the shortened pitch
    # spindle in the one source used by CAD, mass ledger and simulator.
    from experiments.compact_morphology.ape_frame50.hip_yaw_motor_support import apply
    apply(hw)
    return hw, shared


def mesh(shape, color):
    vertices, faces = shape.tessellate(0.35, 0.4)
    tri = trimesh.Trimesh(vertices=np.asarray([[v.X, v.Y, v.Z] for v in vertices]),
                          faces=np.asarray(faces), process=False)
    tri.visual.vertex_colors = color
    return tri


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    hw, shared = candidate()
    fixed = hw["fixed"]
    yaw = hw["yaw"]
    joins = []
    for a, b in (("yaw_housing_outboard_rib", "yaw_housing_tube"),
                 ("yaw_housing_outboard_rib", "yaw_housing_arm"),
                 ("yaw_motor_housing_tube", "yaw_motor_housing_arm")):
        joins.append({"pair": [a, b], "overlap_mm3": round(intersection(fixed[a], fixed[b]), 3)})
    for side in ("left", "right"):
        sign = -1 if side == "left" else 1
        for arm in ("yaw_housing_arm", "yaw_motor_housing_arm"):
            shape = fixed[arm] if sign == 1 else fixed[arm].mirror(bd.Plane.YZ)
            joins.append({"pair": [f"{side}_{arm}", "pelvis_front_crossbeam"],
                          "overlap_mm3": round(intersection(shape, shared["pelvis_front_crossbeam"]), 3)})
    for a, b in (("pelvis_front_crossbeam", "front_spine"),
                 ("pelvis_front_crossbeam", "tray_to_front_spine"),
                 ("yaw_motor_tray", "tray_to_front_spine")):
        joins.append({"pair": [a, b], "overlap_mm3": round(intersection(shared[a], shared[b]), 3)})
    joins.append({"pair": ["right_yaw_motor_tray_pad", "yaw_motor_tray"],
                  "overlap_mm3": round(intersection(fixed["yaw_motor_tray_pad"],
                                                    shared["yaw_motor_tray"]),3)})
    joins.append({"pair": ["right_roll_motor_standoff", "yaw_fork_front"],
                  "overlap_mm3": round(intersection(yaw["roll_motor_standoff"],
                                                    yaw["yaw_fork_front"]),3)})
    roll = hw["roll"]
    for a, b in (("roll_spindle_front", "pitch_fork_bridge_front"),
                 ("roll_spindle_rear", "pitch_fork_bridge_rear"),
                 ("pitch_fork_bridge_front", "pitch_fork_medial"),
                 ("pitch_fork_bridge_rear", "pitch_fork_lateral"),
                 ("thigh_root_pad", "pitch_spindle")):
        group = hw["thigh"] if a == "thigh_root_pad" else roll
        joins.append({"pair": [a, b],
                      "overlap_mm3": round(intersection(group[a],
                                                       hw["thigh"][b] if b == "pitch_spindle" else roll[b]), 3)})

    pose_hits = {}
    adjacent_stage_hits = {}
    for pose, angles in hip.POSES.items():
        scene = trimesh.Scene()
        for name, solid in shared.items():
            scene.add_geometry(mesh(solid, (92, 102, 103, 255)), geom_name=name, node_name=name)
        hits = []
        cross_hits = []
        for side, sign in (("left", -1), ("right", 1)):
            stages = hip.frame(hw, sign, *angles)
            stationary = {**stages["fixed"], **shared}
            for stage, group in stages.items():
                for name, solid in group.items():
                    full = f"{side}_{stage}_{name}"
                    color = ((74, 118, 150, 255) if "ST3250_case" in name else
                             (205, 154, 82, 255) if stage != "fixed" else
                             (116, 122, 121, 255))
                    scene.add_geometry(mesh(solid, color), geom_name=full, node_name=full)
                    if stage == "fixed":
                        continue
                    for fixed_name, fixed_shape in stationary.items():
                        # The vertical yaw spindle turns inside its bearing
                        # pair; the output pulley deliberately engages belt.
                        if name == "yaw_spindle" and fixed_name in (
                                "yaw_bearing_low_F606_ZZ", "yaw_bearing_high_F606_ZZ", "yaw_housing_tube"):
                            continue
                        if name == "yaw_driven_20T" and fixed_name == "yaw_belt_123_3MGT_15":
                            continue
                        amount = intersection(solid, fixed_shape)
                        if amount > .05:
                            hits.append({"side": side, "moving": f"{stage}_{name}",
                                         "fixed": fixed_name, "overlap_mm3": round(amount, 3)})
            for parent, child in (("yaw", "roll"), ("yaw", "thigh"),
                                  ("roll", "thigh")):
                for parent_name, parent_part in stages[parent].items():
                    for child_name, child_part in stages[child].items():
                        amount = intersection(parent_part, child_part)
                        if amount > .05:
                            cross_hits.append({"side": side,
                                               "parent": f"{parent}_{parent_name}",
                                               "child": f"{child}_{child_name}",
                                               "overlap_mm3": round(amount, 3)})
        scene.export(OUT / f"pelvis-tube-{pose}.glb")
        pose_hits[pose] = hits
        adjacent_stage_hits[pose] = cross_hits
    neutral = list(shared.values())
    for sign in (-1, 1):
        neutral.extend(s for group in hip.frame(hw, sign, 0, 0, 0).values() for s in group.values())
    bd.export_step(bd.Compound(children=neutral), str(OUT / "pelvis-tube-neutral.step"))
    gross = sum(s.volume for s in shared.values())
    report = {
        "stage": "bearing-supported bilateral hip and shared pelvis candidate; NOT full-body assembly",
        "body": "approved 500 mm Cove ape, cosmetic shell excluded from clearance",
        "source": "experiments/compact_morphology/ape_design/swarm_cove_fork_hip/build.py",
        "axes_right_mm": {"yaw_roll": hip.PIVOT, "pitch": hip.PITCH},
        "bearings": "JTEKT F606 ZZ 6x17x6 mm, 19x1.2 mm flange and 840 N catalog basic static radial rating, at both hip-yaw output and remote yaw-motor shaft; two stations per location and side",
        "yaw_bearing_source": "https://koyo.jtekt.co.jp/en/products/detail/print.php?pno=F606+ZZ",
        "yaw_bearing_spacing_mm": 11.4,
        "roll_drive": {"belt": "Gates 123-3MGT-09 catalog width/length envelope",
                       "source": "https://gates.com/content/dam/documents-library/catalogs/power-transmission-catalog.pdf",
                       "pulley": "MISUMI GPA20GT3090-A-P6, 20T/3GT, 9 mm belt, 6 mm bore, 14 mm full width, M4 set screws",
                       "pulley_source": "https://uk.misumi-ec.com/pdf/fa/2014/P1_1383-1384_F19_EN.pdf",
                       "pulley_front_y_mm": -39, "belt_start_y_mm": -36.5,
                       "coupler_to_pulley_gap_mm": 4.0,
                       "coupler": "goBILDA 4001-0025-0006 H25T-to-6 mm clamp, 17 mm axial STEP length, 6 g catalog mass",
                       "coupler_source": "https://www.gobilda.com/4001-series-clamping-servo-to-shaft-coupler-25-tooth-spline-to-6mm-round-bore/",
                       "coupler_to_Feetech_spline_fit": "nominal 25T/5.9 mm; physical fit unverified"},
        "yaw_flange_projection_gap_to_unsplit_roll_axis_envelope_mm": 0.3,
        "yaw_axial_location_candidate": "Opposed outer-ring flanges bear on the housing tube ends; a stepped shaft bears on the high inner ring, a 5.4 mm inner spacer reaches the low inner ring, and a 0.8 mm washer/DIN 471-6 retaining-ring envelope stops it below. Insert bearings from opposite ends and yaw carrier from above, then washer and expandable ring from below; washer/shims, groove duty, fits and bearing axial play remain unspecified.",
        "yaw_retaining_ring_source": "https://www.beneri.com/en/prodotto/din-471",
        "yaw_bearing_screen": {"assumed_single_support_mass_kg": 2.75,
                               "assumed_transverse_lever_mm": 50,
                               "assumed_vertical_multiplier": 2,
                               "assumed_axial_force_n": 53.96,
                               "estimated_bearing_radial_reaction_n": 236.5,
                               "estimated_static_equivalent_n": 236.5,
                               "catalog_basic_static_radial_rating_n": 840,
                               "ratio_catalog_static_to_screen_reaction": 3.55,
                               "static_equivalent_rule_source": "https://koyo.jtekt.co.jp/en/support/catalog-download/uploads/catbs004en_b.pdf",
                               "qualification": "illustrative 2g radial/axial statics only; no measured force, shock spectrum, ring-groove strength, misalignment, tolerance/preload or bearing-life proof"},
        "motor_case_orientation": "ST3250 output normal to 45.22 x 24.72 mm face, source drawing page 6",
        "yaw_motor_center_mm": [27.5,12.11,180],
        "yaw_motor_coupler": "goBILDA 4001-0025-0006 exact STEP, 25T spline to 6 mm clamp; Feetech mating and torque unverified",
        "yaw_motor_shaft_candidate": "67 mm long 6 mm nominal shaft with 8 mm shoulder and DIN 471-6 groove; custom turning, stock/strength/fit unqualified",
        "yaw_motor_bearing_spacing_mm": 28,
        "yaw_motor_radial_screen_report": "artifacts/ape-frame50-hip-yaw-coupler/report.json",
        "intentional_fixed_joins": joins,
        "moving_vs_fixed_hits": pose_hits,
        "adjacent_stage_nonmating_hits": adjacent_stage_hits,
        "shared_frame_gross_volume_mm3": round(gross, 2),
        "shared_frame_gross_petg_hf_mass_kg": round(gross * 1.28 / 1e6, 4),
        "open": ["F606 ZZ opposed flange and stepped shafts give candidate axial paths, not duty/life proof; the motor-side upper bearing exceeds its 840 N basic static radial rating in the assumed 2x tension-difference stall case (1522 N), so this is not a qualified stall/load limit", "0.3 mm is only a projection to the former continuous roll-axis envelope; both bearing fits, inner-spacer shims, DIN 471 groove/ring duty, custom shaft material and printed-seat strength need qualification", "Verify goBILDA H25T spline fit to Feetech ST3250, shaft insertion/clamp torque, MISUMI M4 set-screw duty, belt tension/tooth load and supported 6 mm shaft deflection",
                 "motor case clamps and mounting screws", "yaw tray and battery service",
                 "pose continuum and tolerance gap", "stage-to-stage nonmating interference",
                 "knee/ankle and shoulder/elbow link attachment", "mass/inertia and loaded duty"],
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    shutil.copy2(Path(__file__).with_name("pelvis_tube_viewer.html"), OUT / "index.html")
    print(json.dumps({"joins": joins,
                      "hit_counts": {name: len(v) for name, v in pose_hits.items()},
                      "adjacent_stage_hit_counts": {name: len(v) for name, v in adjacent_stage_hits.items()},
                      "gross_shared_frame_mass_kg": report["shared_frame_gross_petg_hf_mass_kg"]}, indent=2))


if __name__ == "__main__":
    main()
