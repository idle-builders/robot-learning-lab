"""Compact FT-3322-C001 direct-drive knee fit coupon on the exact 350 mm shell.

Run: .venv/bin/python experiments/compact_morphology/ape_design/swarm_compact_direct_knee_coupon.py
X lateral, Y rearward, Z upward; units mm. Geometric study only.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import build123d as bd
import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "artifacts/sketch-ape-shell/exports/350mm/ape_shell_assembly.step"
OUT = ROOT / "artifacts/swarm-compact-direct-knee-20261004"
AXIS = (42.0, 12.0, 98.0)
BASE_AXIS = (42.0, 3.0, 86.0)
QUICK_MODE = os.environ.get("SWARM_COMPACT_QUICK", "")
QUICK = bool(QUICK_MODE)
ANGLES = (np.array((0, 30, 60, 90, 120), dtype=float)
          if QUICK_MODE == "1" else np.arange(0.0, 120.0001, 2.5))
VIEW_ANGLES = (0, 30, 60, 90, 120)
PDF = "https://www.feetechrc.com/Data/feetechrc/upload/file/20260623/6391782196214275119734720.pdf"


def box(center, size):
    return bd.Pos(*center) * bd.Box(*size)


def clip(x0, x1, y0, y1, z0, z1):
    return bd.Solid.make_box(x1-x0, y1-y0, z1-z0,
                             bd.Plane(origin=(x0, y0, z0)))


def ellipsoid(center, radii):
    return bd.Solid.make_sphere(1).scale(tuple(radii)).translate(center)


def cyl_x(radius, x0, x1, y=AXIS[1], z=AXIS[2]):
    return bd.Solid.make_cylinder(
        radius, x1-x0, bd.Plane(origin=(x0, y, z), z_dir=(1, 0, 0)))


def rod(a, b, radius):
    vec = bd.Vector(b)-bd.Vector(a)
    return bd.Solid.make_cylinder(
        radius, vec.length, bd.Plane(origin=a, z_dir=vec.normalized()))


def comp(shapes):
    return bd.Compound(children=[solid for shape in shapes
                                 for solid in shape.solids()])


def turn(shape, deg):
    return bd.Pos(*AXIS) * bd.Rot(float(deg), 0, 0) * \
           bd.Pos(*(-v for v in AXIS)) * shape


def aabb_gap(a, b):
    return math.sqrt(sum(max(u-v, w-t, 0)**2 for u, v, w, t in (
        (b.min.X, a.max.X, a.min.X, b.max.X),
        (b.min.Y, a.max.Y, a.min.Y, b.max.Y),
        (b.min.Z, a.max.Z, a.min.Z, b.max.Z))))


def collision_gap(left, right, measure_distance=True):
    """Return exact overlap and either distance or conservative AABB lower bound."""
    smallest = 3.0
    overlap = 0.0
    bb = [(s, s.bounding_box(optimal=False)) for s in right.solids()]
    for a in left.solids():
        ab = a.bounding_box(optimal=False)
        for b, bounds in bb:
            coarse = aabb_gap(ab, bounds)
            if coarse >= smallest:
                continue
            if coarse < 1e-8:
                hit = a & b
                overlap += sum(v.volume for v in hit.solids()) if hit else 0
            if measure_distance and coarse < smallest:
                smallest = min(smallest, a.distance_to(b))
            elif not measure_distance:
                smallest = min(smallest, coarse)
    return smallest, overlap


def metrics(shape):
    bounds = shape.bounding_box()
    return {
        "bounds_xyz_mm": [[round(v, 3) for v in (bounds.min.X, bounds.min.Y, bounds.min.Z)],
                          [round(v, 3) for v in (bounds.max.X, bounds.max.Y, bounds.max.Z)]],
        "volume_mm3": round(sum(v.volume for v in shape.solids()), 3),
    }


def add(scene, name, shape, rgba):
    for index, solid in enumerate(shape.solids()):
        if solid.volume < 1e-6:
            continue
        vertices, faces = solid.tessellate(.4, .32)
        mesh = trimesh.Trimesh(
            vertices=np.asarray([(v.X, v.Y, v.Z) for v in vertices]),
            faces=np.asarray(faces), process=False)
        mesh.visual.vertex_colors = rgba
        node = f"{name}_{index}"
        scene.add_geometry(mesh, geom_name=node, node_name=node)


def main():
    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)
    OUT.mkdir(parents=True, exist_ok=True)
    part = {p.label: p for p in bd.import_step(SOURCE).children}
    source_thigh = comp([part["R_thigh_front"], part["R_thigh_back"]])
    source_cuff = part["R_knee_cuff"]
    original_shin = comp([part["R_shin_front"], part["R_shin_back"]])
    shin = comp([part[f"R_shin_{side}"] & clip(-50, 140, -80, 80, -10, 70)
                 for side in ("front", "back")])
    foot = comp([part[f"R_{name}"] for name in
                 ("foot_dorsal", "toe_1", "toe_2", "toe_3", "sole")])

    # Manufacturer page 5/7: 30 breadth, 12 thickness, 35 axial case. The
    # output center is 6 mm from one 30-mm edge and central in the 12 mm edge.
    # Rotate the offset 30-mm breadth upward: z=92..122, y=6..18.
    case = bd.Pos(*AXIS) * bd.Rot(90, 0, 0) * bd.Pos(0, 9, 0) * \
           bd.Box(35.0, 30.0, 12.0)
    # A one-mm access envelope relieves the exact shell. It is not a proven
    # mounting/fastener clearance; the drawn ±0.1 mm case tolerance is smaller.
    service = bd.Pos(*AXIS) * bd.Rot(90, 0, 0) * bd.Pos(0, 9, 0) * \
              bd.Box(37.0, 32.0, 14.0)
    relieved_thigh = comp([p - service for p in source_thigh.solids()])

    # Frontal rounded seam retains a portion of the exact cuff. A shallow
    # upper lozenge bridges it toward the raised pivot. Both stay frontal
    # (negative Y) so the calf can fold into the rear.
    front_cuff = source_cuff & clip(-50, 140, -60, 3, 83, 140)
    upper_outer = ellipsoid((42, 5, 94), (22, 14, 11))
    upper_inner = ellipsoid((42, 5, 94), (19.8, 11.8, 8.8))
    upper_lip = (upper_outer - upper_inner) & clip(-50, 140, -50, 3.5, 84, 140)
    fixed_skin = comp([relieved_thigh, front_cuff, upper_lip])

    # Separate opposed bearing stations are radially outside the case ends.
    # Dimensions are package envelopes, not a selected bearing rating.
    left_bearing = cyl_x(6.5, 15.5, 20.5) - cyl_x(2.6, 15.4, 20.6)
    right_bearing = cyl_x(6.5, 67.5, 72.5) - cyl_x(2.6, 67.4, 72.6)
    bearings = comp([left_bearing, right_bearing])
    left_housing = cyl_x(9, 14.5, 21.5) - cyl_x(6.7, 14.4, 21.6)
    right_housing = cyl_x(9, 66.5, 73.5) - cyl_x(6.7, 66.4, 73.6)
    housings = comp([left_housing, right_housing])
    # The two side struts visibly terminate inside the imported thigh skin,
    # rather than leaving the bearing housings as floating rings.
    supports = comp([
        rod((18, 12, 104.8), (23.5, 12, 116), 2.6),
        rod((70, 12, 104.8), (62.5, 12, 116), 2.6),
    ])

    # The motor's left spline is a drive interface only. Two axle stubs run
    # through separate bearings and join the side arms through small hubs.
    # The right stub starts beyond the drawn opposite 6 mm projection.
    stubs = comp([cyl_x(2.45, 15.8, 23.6),
                  cyl_x(2.45, 65.6, 72.2)])
    hubs = comp([cyl_x(7.5, 21.6, 23.6),
                 cyl_x(7.5, 65.6, 66.4)])
    arms = comp([
        rod((21, 12, 89.5), (25.5, 3, 68), 2.0),
        rod((67, 12, 89.5), (63, 3, 68), 2.0),
    ])
    bridge = rod((25.5, 3, 68), (63, 3, 68), 2.2)
    yoke = comp([stubs, hubs, arms, bridge])
    # A low moving gusset covers the z70->83 neutral-pose opening. A front
    # source-like lozenge keeps the calf looking joined while the rear opens.
    guard = (ellipsoid((42, 3, 75), (18, 18, 8)) -
             ellipsoid((42, 3, 75), (15.8, 15.8, 5.8))) & \
            clip(-50, 140, -50, 30, 67, 81.5)
    moving_skin = comp([shin, foot, guard])

    fixed_obstacles = {
        "exact_thigh_after_service_relief": relieved_thigh,
        "frontal_source_cuff": front_cuff,
        "raised_front_lip": upper_lip,
        "manufacturer_size_case": case,
        "bearing_housings": housings,
        "bearing_supports": supports,
    }
    static = {
        "case_vs_original_thigh_mm3": round(collision_gap(case, source_thigh)[1], 4),
        "case_vs_relieved_thigh_mm3": round(collision_gap(case, relieved_thigh)[1], 4),
        "service_relief_removed_thigh_mm3": round(
            metrics(source_thigh)["volume_mm3"] -
            metrics(relieved_thigh)["volume_mm3"], 3),
        "service_relief_removed_thigh_pct": round(100 * (
            metrics(source_thigh)["volume_mm3"] -
            metrics(relieved_thigh)["volume_mm3"]) /
            metrics(source_thigh)["volume_mm3"], 3),
        "relieved_thigh_solid_count": len(relieved_thigh.solids()),
        "original_thigh_solid_count": len(source_thigh.solids()),
        "relieved_thigh_all_valid": all(s.is_valid for s in relieved_thigh.solids()),
        "support_to_exact_thigh_overlap_mm3": round(collision_gap(supports, source_thigh)[1], 4),
        "front_cuff_retained_fraction": round(
            front_cuff.volume / source_cuff.volume, 4),
        "original_shin_trimmed_top_mm": 83,
        "coupon_shin_top_mm": 70,
        "neutral_skin_opening_before_guard_mm": 13,
        "neutral_guard_max_z_mm": round(guard.bounding_box().max.Z, 3),
        "fixed_cuff_min_z_mm": round(front_cuff.bounding_box().min.Z, 3),
        "neutral_guard_to_source_cuff_exact_gap_mm": round(
            collision_gap(guard, front_cuff)[0], 4),
        "relief_wall_assessment": "Both imported thigh shell halves remain one valid solid; the relief opens local wall. Minimum residual wall and mount stiffness are unverified.",
    }

    samples = []
    for deg in ANGLES:
        posed_skin = turn(moving_skin, deg)
        posed_yoke = turn(yoke, deg)
        pair_data = {}
        for moving_name, moving_shape in (
            ("shin_guard_foot", posed_skin), ("yoke", posed_yoke)):
            for fixed_name, fixed_shape in fixed_obstacles.items():
                clearance, overlap = collision_gap(
                    moving_shape, fixed_shape, measure_distance=False)
                pair_data[f"{moving_name}__{fixed_name}"] = {
                    "aabb_lower_bound_gap_mm": round(clearance, 5),
                    "overlap_mm3": round(overlap, 5)}
        # The axle/bore is a designed mating pair; report radial 0.15 mm
        # separately. The motor spline/horn adapter and fasteners are absent.
        min_gap = min(v["aabb_lower_bound_gap_mm"] for v in pair_data.values())
        max_overlap = max(v["overlap_mm3"] for v in pair_data.values())
        samples.append({
            "flex_deg": float(deg),
            "split": "design_grid" if deg % 5 == 0 else "held_out",
            "minimum_aabb_lower_bound_gap_mm": round(min_gap, 5),
            "maximum_nonmating_overlap_mm3": round(max_overlap, 5),
            "pairs": pair_data})
        if not QUICK and round(deg) in VIEW_ANGLES and abs(deg-round(deg)) < 1e-8:
            scene = trimesh.Scene()
            for name, shape, color in (
                ("exact_thigh_relief", relieved_thigh, (226, 218, 202, 255)),
                ("source_front_cuff", front_cuff, (85, 87, 76, 255)),
                ("raised_front_lip", upper_lip, (85, 87, 76, 255)),
                ("FT3322_case", case, (53, 108, 139, 230)),
                ("bearings", bearings, (113, 119, 119, 255)),
                ("bearing_housings", housings, (73, 77, 75, 255)),
                ("supports", supports, (87, 92, 87, 255)),
                ("source_shin_trimmed", turn(shin, deg), (238, 228, 207, 255)),
                ("source_foot_full", turn(foot, deg), (230, 221, 203, 255)),
                ("moving_gusset", turn(guard, deg), (104, 98, 82, 255)),
                ("moving_yoke", posed_yoke, (196, 107, 60, 255)),
            ):
                add(scene, name, shape, color)
            scene.export(OUT / f"swarm_compact_direct_knee_{int(deg):03d}.glb")

    if QUICK:
        print(json.dumps({"static": static, "samples": [
            {"deg": r["flex_deg"],
             "hits": {k:v for k,v in r["pairs"].items() if v["overlap_mm3"] > 1e-5}}
            for r in samples]}, indent=2), flush=True)
        return

    report = {
        "source_step": str(SOURCE.relative_to(ROOT)),
        "coordinates": "X lateral, Y rearward, Z upward; mm",
        "motor": {
            "model": "Feetech FT-3322-C001",
            "manufacturer_page": "https://www.feetechrc.com/728213",
            "manufacturer_pdf": PDF,
            "body_xyz_mm": [35, 12, 30],
            "body_orientation": "35 axial X; 12 fore-aft Y; 30 vertical Z",
            "output_axis_from_30mm_edge_mm": 6,
            "output_axis_from_12mm_edge_mm": 6,
            "case_mass_g": "32.6 ± 2",
            "rated_torque_7v4_Nm": 0.745,
            "rated_torque_source": "PDF section 5-8: 7.6 kg cm at 7.4 V",
            "stall_torque_7v4_Nm": 2.245,
            "no_load_speed_7v4_sec_per_60deg": 0.093,
            "travel_deg": "180 ± 5 per PDF control specification",
            "life_test": ">200000 cycles at rated 7.6 kg cm, 7.4 V; 60 deg each way in 0.25 s with 0.5 s rests",
        },
        "source_knee_axis_xyz_mm": BASE_AXIS,
        "coupon_axis_xyz_mm": AXIS,
        "axis_shift_mm": {"rearward": 9, "upward": 12},
        "bearing_scheme": "Two separate 5 mm bore / 13 mm OD / 5 mm width radial bearing envelopes. Independent moving yoke stubs enter opposite housings; radial load path reaches imported thigh via visible side supports. Selection, retention and strength unproved.",
        "appearance": {
            "source_cuff_bounds": metrics(source_cuff)["bounds_xyz_mm"],
            "coupon_bearing_housing_bounds": metrics(housings)["bounds_xyz_mm"],
            "source_cuff_x_width_mm": round(
                source_cuff.bounding_box().max.X - source_cuff.bounding_box().min.X, 3),
            "housing_stack_x_width_mm": round(
                housings.bounding_box().max.X - housings.bounding_box().min.X, 3),
            "housing_added_total_width_mm": round(
                housings.bounding_box().max.X - housings.bounding_box().min.X -
                (source_cuff.bounding_box().max.X - source_cuff.bounding_box().min.X), 3),
            "honest_axial_stack_lower_bound_mm": 57.8,
            "axial_stack_basis": "35 mm case + 3.8 mm output projection + 6 mm opposite projection + two 5 mm bearing stations beyond the projections + 1.5 mm housing wall at each end = 57.8 mm. Actual model spans 59 mm.",
            "source_shin_bounds": metrics(original_shin)["bounds_xyz_mm"],
            "trimmed_shin_bounds": metrics(shin)["bounds_xyz_mm"],
            "assessment": "Fails the requested small sympathetic cuff widening: 59 mm bearing-housing stack nearly doubles the exact 30 mm sketch cuff. Frontal source cuff and moving lozenge preserve a rounded seam, but the side pods and exposed yoke break the compact silhouette.",
        },
        "static_fit": static,
        "component_metrics": {name: metrics(value) for name, value in {
            "case": case, "source_thigh": source_thigh, "relieved_thigh": relieved_thigh,
            "front_cuff": front_cuff, "upper_lip": upper_lip, "shin": shin,
            "foot": foot, "guard": guard, "bearings": bearings,
            "bearing_housings": housings, "supports": supports, "yoke": yoke}.items()},
        "sweep": {
            "range_deg": [0, 120],
            "step_deg": 2.5,
            "held_out_angles_deg": [float(r["flex_deg"]) for r in samples
                                    if r["split"] == "held_out"],
            "minimum_aabb_lower_bound_gap_mm": min(
                r["minimum_aabb_lower_bound_gap_mm"] for r in samples),
            "maximum_nonmating_overlap_mm3": max(
                r["maximum_nonmating_overlap_mm3"] for r in samples),
            "positive_gap_on_sampled_sweep": False,
            "all_sampled_poses_volume_clear": all(
                r["maximum_nonmating_overlap_mm3"] < 1e-5 for r in samples),
            "clearance_interpretation": "BRep Boolean intersections are zero at all listed poses. The AABB lower bound can be zero even when surfaces are apart, so no positive minimum surface gap is established.",
            "samples": samples,
        },
        "capacity_warning": "This is a package hypothesis. The earlier 1.569 Nm all-leg actuator cap is a simulation assumption for another motor and cannot be carried over. Retrain and evaluate the policy with a 0.745 Nm rated envelope, torque-speed curve, thermal duty and full-body contacts before selecting this servo.",
        "unmodeled": [
            "Real motor end-cap contour, mounting tabs, screw heads, cable exit and routing",
            "Spline-to-yoke adapter, opposing stub manufacture, actual bearing catalog/rating and retention",
            "Fatigue, strength, shock load, print tolerances, shell fastening and ground/other-leg contact",
            "Continuous-angle minimum clearance between the 2.5-degree poses",
            "Local residual thigh wall thickness and structural effect of the service opening",
        ],
    }
    verdict = ("VOLUME-CLEAR SAMPLED MOTION; COMPACT CUFF FAILS; POSITIVE GAP UNPROVEN"
               if report["sweep"]["all_sampled_poses_volume_clear"] else
               "HARD-PART CLASH IN SAMPLED SWEEP")
    report["verdict"] = verdict
    (OUT / "swarm_compact_direct_knee_report.json").write_text(
        json.dumps(report, indent=2) + "\n")
    for name, shape in (
        ("fixed", comp([fixed_skin, case, bearings, housings, supports])),
        ("moving_standing", comp([moving_skin, yoke])),
        ("source_reference", comp([source_thigh, source_cuff, original_shin, foot])),
    ):
        bd.export_step(shape, OUT / f"swarm_compact_direct_knee_{name}.step")
    (OUT / "swarm_compact_direct_knee_index.html").write_text(
        Path(__file__).with_name("swarm_compact_direct_knee_viewer.html").read_text())
    print(json.dumps({
        "verdict": verdict, "artifacts": str(OUT),
        "static_fit": static,
        "positive_gap_verified": False,
        "sweep_max_overlap_mm3": report["sweep"]["maximum_nonmating_overlap_mm3"],
        "worst_angles": [r["flex_deg"] for r in samples
                         if r["maximum_nonmating_overlap_mm3"] > 1e-5],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
