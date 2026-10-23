"""One bounded torso-mounted cable shoulder layout on the 500 mm Cove BRep.

Millimetres; X right, Y rear, Z up. This is a geometric falsification coupon,
not a motor-duty qualification. Run from any directory with the repo .venv.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import build123d as b
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "experiments/compact_morphology/ape_design/swarm_shell_v2"))
sys.path.insert(0, str(ROOT / "experiments/compact_morphology/ape_design/swarm_cove_shoulder"))
from build import VARIANTS, build, ellipsoid  # noqa: E402
from cove_shoulder import (  # noqa: E402
    PACK_CENTER, PACK_SIZE, SERVICE_CENTER, SERVICE_SIZE, POSES,
    box, checked, cylinder, ring, rod, pose, scene, contact_sheet,
)

CASE_SIZE = (20., 34., 26.)  # XL330 official body size, axis-aligned in this trial
CASE_CENTERS = {"A": (53., 0., 307.), "B": (59., 0., 338.), "C": (48., 0., 369.)}
PIVOTS = {"yaw": (78., 1., 335.), "roll": (108., 1., 326.), "pitch": (121., 1., 312.)}
CAPSTAN_RADIUS = 8.
OUTPUT_RADIUS = 8.
IDLER_RADIUS = 8.
CABLE_RADIUS = .5
MIN_BEND = 8.


def rounded_centerline(corners, requested_radius=MIN_BEND):
    """Circularly blend a polyline and report any radius forced below 8 mm.

    The sampled rods are a conservative visualization/clearance envelope. The
    explicit arcs, rather than sharp vertices, define the intended cable path.
    """
    points = [np.asarray(p, float) for p in corners]
    output = [points[0]]
    radii = []
    for i in range(1, len(points)-1):
        p0, p, p1 = points[i-1:i+2]
        d0 = (p-p0)/np.linalg.norm(p-p0)
        d1 = (p1-p)/np.linalg.norm(p1-p)
        delta = math.acos(float(np.clip(d0 @ d1, -1, 1)))
        if delta < 1e-5:
            output.append(p)
            continue
        if delta > math.pi-1e-5:
            output.append(p)
            radii.append(0.)
            continue
        tangent = math.tan(delta/2)
        radius = min(requested_radius, .45*np.linalg.norm(p-p0)/tangent,
                     .45*np.linalg.norm(p1-p)/tangent)
        radii.append(radius)
        entry = p-d0*radius*tangent
        exit_ = p+d1*radius*tangent
        inward = (d1-(d0@d1)*d0)/math.sin(delta)
        center = entry+radius*inward
        radial = entry-center
        normal = np.cross(d0, d1)/math.sin(delta)
        output.append(entry)
        for j in range(1, 13):
            angle = delta*j/12
            v = radial*math.cos(angle)+np.cross(normal,radial)*math.sin(angle)
            output.append(center+v)
        output.append(exit_)
    output.append(points[-1])
    return [tuple(map(float, p)) for p in output], [round(r, 4) for r in radii]


def cable_rods(points):
    return [rod(a, z, CABLE_RADIUS) for a, z in zip(points[:-1], points[1:])
            if np.linalg.norm(np.asarray(z)-np.asarray(a)) > .01]


def hardware_and_skin(source):
    # One local side relief and a moving overlapped root. The fixed top roof is
    # the original Cove loft; no outboard drive case is allowed.
    torso = source["torso"] - ellipsoid((97., 1., 336.), (18., 23., 18.))
    root = ellipsoid((111., 2., 341.), (20., 22., 17.)) - \
           ellipsoid((111., 2., 341.), (18., 20., 15.))
    skin_parts = {"R_upper_arm": source["R_upper_arm"].fuse(root),
                  "L_upper_arm": source["L_upper_arm"]}
    hardware = {
        "single_shared_4S850_pack": ("fixed", box(PACK_CENTER, PACK_SIZE)),
        "CAN_5V_service_reserve": ("fixed", box(SERVICE_CENTER, SERVICE_SIZE)),
    }
    for motor, center in CASE_CENTERS.items():
        hardware[f"R_XL330_{motor}_case"] = ("fixed", box(center, CASE_SIZE))
        hardware[f"R_{motor}_capstan"] = ("fixed", cylinder("y", CAPSTAN_RADIUS, 3.,
                                                             (center[0], -20., center[2])))
        hardware[f"R_{motor}_output_stub"] = ("fixed", cylinder("y", 3., 8.,
                                                                 (center[0], -18., center[2])))
    # External shafts and two separated radial bearings per output; transverse
    # loads go to yoke/bulkhead, never to an XL330 horn.
    hardware["R_yaw_bearing_low"] = ("fixed", ring("z", (78., 1., 324.)))
    hardware["R_yaw_bearing_high"] = ("fixed", ring("z", (78., 1., 344.)))
    hardware["R_yaw_carrier"] = ("fixed", box((78., 1., 334.), (17., 17., 28.))-
                                  cylinder("z", 6.7, 29., (78., 1., 334.)))
    hardware["R_yaw_shaft"] = ("yaw", cylinder("z", 3., 30., PIVOTS["yaw"]))
    hardware["R_yaw_output_pulley"] = ("yaw", cylinder("z", OUTPUT_RADIUS, 3.,
                                                       (78., 1., 348.)))
    hardware["R_swing_link"] = ("yaw", rod((78., 1., 331.), (108., 1., 326.), 3.5))
    hardware["R_roll_bearing_inner"] = ("yaw", ring("x", (99., 1., 326.)))
    hardware["R_roll_bearing_outer"] = ("yaw", ring("x", (117., 1., 326.)))
    hardware["R_roll_yoke"] = ("yaw", box((108., 1., 326.), (27., 17., 17.))-
                                cylinder("x", 6.7, 28., (108., 1., 326.)))
    hardware["R_roll_shaft"] = ("roll", cylinder("x", 3., 29., PIVOTS["roll"]))
    hardware["R_roll_output_pulley"] = ("roll", cylinder("x", OUTPUT_RADIUS, 3.,
                                                         (120., 1., 326.)))
    hardware["R_pitch_bearing_front"] = ("roll", ring("y", (121., -14., 312.)))
    hardware["R_pitch_bearing_rear"] = ("roll", ring("y", (121., 17., 312.)))
    hardware["R_pitch_yoke"] = ("roll", box((121., 1.5, 312.), (17., 40., 17.))-
                                 cylinder("y", 6.7, 41., (121., 1.5, 312.)))
    hardware["R_pitch_shaft"] = ("pitch", cylinder("y", 3., 38., PIVOTS["pitch"]))
    hardware["R_pitch_output_pulley"] = ("pitch", cylinder("y", OUTPUT_RADIUS, 3.,
                                                           (121., -20., 312.)))
    # Every line below runs from a named capstan to an output via a distinct
    # outer guide lane. A/B are crossed opposed differential legs; C drives yaw.
    # The centerline is rounded at each waypoint and measured below.
    paths = {
        "A_roll_plus": [(53,-20,307),(54,-31,307),(70,-34,318),(91,-31,326),(120,-7,326)],
        "A_pitch_minus": [(53,-20,307),(62,-35,303),(84,-37,302),(108,-31,305),(121,-20,320)],
        "B_roll_plus": [(59,-20,338),(66,-32,341),(84,-35,341),(104,-28,334),(120,-7,326)],
        "B_pitch_plus": [(59,-20,338),(65,-39,334),(82,-41,319),(103,-35,315),(121,-20,320)],
        "C_yaw_plus": [(48,-20,369),(55,-32,371),(67,-36,363),(78,-29,350),(78,-7,348)],
        "C_yaw_minus": [(48,-20,369),(40,-32,367),(53,-40,353),(67,-32,345),(78,-7,348)],
    }
    # The two jackscrew sleeves are modelled as serviceable guide cartridges.
    hardware["R_AB_tensioner_access"] = ("fixed", box((68., -34., 322.), (12., 5., 12.)))
    hardware["R_C_tensioner_access"] = ("fixed", box((57., -34., 357.), (12., 5., 12.)))
    hardware["R_anchor_access"] = ("yaw", box((101., -23., 329.), (14., 7., 18.)))
    for label, center in {"A":(70.,-34.,318.), "B":(84.,-35.,341.),
                          "C":(67.,-36.,363.)}.items():
        hardware[f"R_{label}_idler"] = ("fixed", cylinder("y", IDLER_RADIUS, 3., center))
    route_data = {}
    cables = {}
    for name, corners in paths.items():
        points, radii = rounded_centerline(corners)
        route_data[name] = {"corners_mm": corners, "blend_radii_mm": radii,
                            "minimum_bend_mm": min(radii) if radii else None,
                            "length_mm": round(sum(np.linalg.norm(np.asarray(z)-np.asarray(a))
                                                   for a,z in zip(points[:-1],points[1:])), 3)}
        cables[name] = cable_rods(points)
    return torso, skin_parts, hardware, cables, route_data


def check_many(name, shapes, target):
    # A compound keeps the individual swept rod BReps while doing one Boolean
    # per operand order; the route length/arc data remains per branch above.
    return checked(name, b.Compound(children=shapes), target)


def evaluate(source, torso, skin_parts, hardware, cables):
    static = []
    pack = hardware["single_shared_4S850_pack"][1]
    service = hardware["CAN_5V_service_reserve"][1]
    for letter in CASE_CENTERS:
        case = hardware[f"R_XL330_{letter}_case"][1]
        static += [checked(f"XL330 {letter} case / fixed roof", case, torso),
                   checked(f"XL330 {letter} case / shared pack", case, pack),
                   checked(f"XL330 {letter} case / CAN reserve", case, service)]
    for a,b_ in (("A","B"),("A","C"),("B","C")):
        static.append(checked(f"XL330 {a} case / XL330 {b_} case",
                              hardware[f"R_XL330_{a}_case"][1],
                              hardware[f"R_XL330_{b_}_case"][1]))
    for name in ("R_yaw_carrier","R_yaw_bearing_low","R_yaw_bearing_high",
                 "R_AB_tensioner_access","R_C_tensioner_access"):
        static.append(checked(f"{name} / fixed roof", hardware[name][1], torso))
        static.append(checked(f"{name} / shared pack", hardware[name][1], pack))
    # A case/support hard failure rejects this bounded coupon before expensive
    # cable/roof Booleans. Keep the actual rounded paths and report this gate.
    package_pass = all(row["pass_0p5_mm"] for row in static)
    if package_pass:
        for name, segments in cables.items():
            static.append(check_many(f"cable {name} / fixed roof", segments, torso))
            static.append(check_many(f"cable {name} / shared pack", segments, pack))
    # Rearward battery service sweep, same common datum as Cove shoulder.
    withdrawal = box((0., PACK_CENTER[1]+50., PACK_CENTER[2]),
                     (PACK_SIZE[0], PACK_SIZE[1]+100., PACK_SIZE[2]))
    static.append(checked("100 mm rear pack withdrawal / fixed roof", withdrawal, torso))
    for letter in CASE_CENTERS:
        static.append(checked(f"100 mm rear pack withdrawal / XL330 {letter} case",
                              withdrawal, hardware[f"R_XL330_{letter}_case"][1]))
    poses = {}
    for pose_name, angles in POSES.items():
        arm = pose(skin_parts["R_upper_arm"], "R", angles)
        checks = [checked("moving arm / fixed roof", arm, torso)]
        for name in ("R_swing_link", "R_roll_yoke", "R_pitch_yoke",
                     "R_roll_output_pulley", "R_pitch_output_pulley", "R_anchor_access"):
            level, sh = hardware[name]
            moving = pose(sh, "R", angles, level)
            checks.append(checked(f"{name} / fixed roof", moving, torso))
            if name in ("R_roll_yoke","R_pitch_yoke","R_anchor_access"):
                checks.append(checked(f"{name} / moving arm skin", moving, arm))
        poses[pose_name] = checks
    return static, poses


def views(source, torso, skin_parts, hardware):
    reference = {name: scene(source, torso, skin_parts, hardware, angle, False)
                 for name, angle in POSES.items()}
    candidate = {name: scene(source, torso, skin_parts, hardware, angle, True)
                 for name, angle in POSES.items()}
    contact_sheet(reference, candidate, HERE/"matched-opaque.png",
                  poses=("arm_down","yaw_forward_30","yaw_rear_30"))
    contact_sheet(reference, candidate, HERE/"shoulder-crop-opaque.png",
                  poses=("arm_down","yaw_forward_30","yaw_rear_30"), crop=True)
    contact_sheet(reference, candidate, HERE/"reach-opaque.png",
                  poses=("arm_out_35","forward_reach","floor_reach_proxy"), crop=True)


def main():
    parts, _ = build(VARIANTS[0])
    source = {p.name:p.shape for p in parts}
    torso, skin_parts, hardware, cables, routes = hardware_and_skin(source)
    if "--render-only" in sys.argv:
        views(source, torso, skin_parts, hardware)
        return
    static, poses = evaluate(source, torso, skin_parts, hardware, cables)
    failures = [{"phase":"static",**c} for c in static if not c["pass_0p5_mm"]]
    failures += [{"phase":"pose","pose":name,**c} for name, checks in poses.items()
                 for c in checks if not c["pass_0p5_mm"]]
    J = np.array([[0.,0.,1.],[.5,.5,0.],[.5,-.5,0.]])
    report = {
        "source":"500 mm 2 mm-wall Cove BRep from swarm_shell_v2.build(VARIANTS[0])",
        "candidate":"one right shoulder, fixed torso XL330 A/B differential for roll/pitch, C yaw",
        "motor_cases_mm":{"size":CASE_SIZE,"centers":CASE_CENTERS,"total_mass_g":54},
        "supported_outputs":{"pivots_mm":PIVOTS,"shaft_diameter_mm":6,
           "bearings":"two W618/6 6x13x3.5 per axis, stations yaw 20 / roll 18 / pitch 31 mm apart",
           "path":"fixed torso bulkhead -> yaw bearing pair / carrier -> swing link -> roll bearing pair / yoke -> pitch bearing pair -> arm"},
        "cable":{"diameter_mm":2*CABLE_RADIUS,"bend_gate_mm":MIN_BEND,
                 "capstan_radius_mm":CAPSTAN_RADIUS,"output_radius_mm":OUTPUT_RADIUS,
                 "idler_radius_mm":IDLER_RADIUS,"branches":routes,
                 "tensioners":["R_AB_tensioner_access","R_C_tensioner_access"],
                 "replaceable_anchor_access":"R_anchor_access",
                 "all_bends_pass":all(r["minimum_bend_mm"] is not None and
                   r["minimum_bend_mm"]>=MIN_BEND-.001 for r in routes.values())},
        "coupling":{"motor_order":["A","B","C"],"joint_order":["yaw","roll","pitch"],
           "q_from_motor_radians":J.tolist(),"determinant":float(np.linalg.det(J)),
           "condition_number":float(np.linalg.cond(J)),
           "domain":"ideal equal-radius, taut frictionless no-slip cables; linear local map only",
           "required_axis_angles_deg":{"yaw":[-30,30],"roll":[0,35],"pitch":[-55,0]},
           "singular_in_ideal_map":bool(abs(np.linalg.det(J))<1e-9)},
        "pose_angles_deg":POSES,"static_checks":static,"pose_checks":poses,
        "cable_brep_screen":"deferred after earlier package/support failure" if
            any(not row["pass_0p5_mm"] for row in static if "cable " not in row["pair"])
            else "performed for all six branch compounds in both operand orders",
        "first_failed_check":failures[0] if failures else None,
        "all_named_checks_pass":not failures,
        "limitations":["Cable branches are rounded 1 mm BRep centerline envelopes at neutral. Their moving pulley wrap, tie-off length, differential gearing and pretension are not solved; the Jacobian is the proposed equal-radius ideal constraint, not measured transmission behavior.",
                       "Cove skin is measured as hard shell; opaque images show appearance only. Floor reach is an upright torso arm configuration, not a hand-floor contact proof.",
                       "XL330 continuous torque, cable fatigue and temperature are unqualified. 0.52 Nm at 5 V is published stall torque only."],
    }
    (HERE/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    if "--measure-only" not in sys.argv:
        views(source, torso, skin_parts, hardware)
    print(json.dumps({"first_failed_check":report["first_failed_check"],
                      "all_named_checks_pass":report["all_named_checks_pass"],
                      "all_bends_pass":report["cable"]["all_bends_pass"]},indent=2))


if __name__ == "__main__":
    main()
