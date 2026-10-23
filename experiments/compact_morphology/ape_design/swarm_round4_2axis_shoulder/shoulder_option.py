"""Microban-inspired two-axis shoulder screen for the 350 mm sketch ape.

Architecture principle only: no Microban geometry is imported or copied.
The output is an exact source-shell envelope screen plus a sampled, geometric
workspace comparison; it is not a moving-clearance or torque qualification.
Run from repository root with .venv/bin/python <this file>.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import build123d as bd
import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[4]
TASK = Path(__file__).resolve().parent
OUT = TASK / "output"
sys.path.insert(0, str(ROOT / "experiments/sketch_ape_shell"))
from build import Study  # noqa: E402
import build as shell_source  # noqa: E402

PIVOT = np.array((66., 0., 246.))
ELBOW = np.array((96., 2., 194.5))
HAND = np.array((100., 0., 107.))
PACK_CENTER = (0., 5., 221.)
PACK_SIZE = (30., 60., 30.5)  # same 4S850 reserve as the round/mixed screen
COLORS = {
    "shell": [224, 216, 199, 255], "motor": [63, 116, 151, 255],
    "support": [188, 139, 76, 255], "pack": [85, 151, 105, 255],
    "service": [211, 174, 74, 150], "frame": [105, 119, 124, 255],
    "paw": [183, 110, 97, 255],
}
PART_CACHE: dict[float, dict] = {}
OUTER_CACHE: dict[str, bd.Shape] = {}


def source_outer_envelopes(wall: float) -> dict[str, bd.Shape]:
    """Rebuild the approved exterior lofts for a distinct containment test."""
    key = f"{wall:g}"
    if key in OUTER_CACHE: return OUTER_CACHE[key]
    study = Study(350, wall=wall, gap=.65)
    # These station profiles intentionally mirror only the sketch-06 source
    # loft dimensions; no Microban geometry enters this design.
    torso = shell_source.loft([
        (160,0,2,42,25),(173,0,2,49,29),(199,0,3,58,34),(229,0,4,63,35),
        (253,0,5,63,30),(264,0,5,60,27),(274,0,5,35,22),(278,0,5,23,18)]) & shell_source.above_waist()
    upper = study.hollow_loft([
        (199,93,2,14,17),(207,94,2,17,20),(228,89,3,20,22),
        (249,80,4,19,21),(260,73,4,14,17),(265,70,4,6,9)],crown=2.8)[1]
    fore = study.hollow_loft([
        (123,99,1,15,16),(133,102,1,18,19),(155,103,1,21,22),
        (178,101,1,18,20),(190,97,1,13,15)])[1]
    palm = shell_source.ellipsoid((99,-1,106),(16,13,20)) & shell_source.box(70,122,-25,23,93,120)
    elbow_cuff = shell_source.ellipsoid((96,2,194.5),(12.7,14.5,8)) & shell_source.box(75,115,-30,30,189,200)
    OUTER_CACHE[key] = {"torso": torso, "upper_arm": upper, "forearm": fore, "paw": palm,
                        "elbow_cuff": elbow_cuff}
    return OUTER_CACHE[key]


def volume(shape: bd.Shape) -> float:
    return round(sum(s.volume for s in shape.solids()), 3) if shape else 0.


def rot_matrix(axis: str, degrees: float) -> np.ndarray:
    t = math.radians(degrees); c, s = math.cos(t), math.sin(t)
    if axis == "x": return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "y": return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def shoulder_matrix(yaw: float, roll: float, pitch: float) -> np.ndarray:
    # Match the retained nested MJCF joints: local Z yaw, then local X roll,
    # then local Y pitch. Parent-to-child rotations multiply on the right.
    return rot_matrix("z", yaw) @ rot_matrix("x", roll) @ rot_matrix("y", pitch)


def endpoint(yaw: float, roll: float, pitch: float, elbow: float) -> np.ndarray:
    r = shoulder_matrix(yaw, roll, pitch)
    elbow_world = PIVOT + r @ (ELBOW - PIVOT)
    hand_at_zero = PIVOT + r @ (HAND - PIVOT)
    # Elbow flexion is about the simulator's local Y axis at the elbow.
    return elbow_world + r @ rot_matrix("y", elbow) @ r.T @ (hand_at_zero - elbow_world)


def nearest(target: np.ndarray, has_yaw: bool, shoulder_origin=PIVOT) -> dict:
    # Bounded deterministic grid over simulator limits; elbow and shoulder
    # ranges are copied from the retained v9 source-labelled MJCF.
    yaw_vals = np.deg2rad(np.linspace(-114.6, 114.6, 19)) if has_yaw else np.array([0.])
    roll_vals = np.deg2rad(np.linspace(-91.7, 91.7, 17))
    pitch_vals = np.deg2rad(np.linspace(-126.1, 126.1, 21))
    elbow_vals = np.deg2rad(np.linspace(-126.1, 8.6, 13))
    target_virtual = PIVOT + (target - np.asarray(shoulder_origin, float))
    best = (float("inf"), None)
    for yaw in yaw_vals:
        for roll in roll_vals:
            for pitch in pitch_vals:
                # Vectorize elbow sweep for this shoulder configuration.
                r = shoulder_matrix(math.degrees(yaw), math.degrees(roll), math.degrees(pitch))
                e = PIVOT + r @ (ELBOW - PIVOT)
                h0 = PIVOT + r @ (HAND - PIVOT)
                local = r.T @ (h0 - e)
                points = np.array([e + r @ rot_matrix("y", math.degrees(q)) @ local
                                   for q in elbow_vals])
                ds = np.linalg.norm(points - target_virtual, axis=1)
                idx = int(np.argmin(ds))
                if ds[idx] < best[0]: best = (float(ds[idx]), (yaw, roll, pitch, elbow_vals[idx], points[idx]))
    d, values = best
    yaw, roll, pitch, elbow, point = values
    world_point = point - PIVOT + np.asarray(shoulder_origin, float)
    return {"distance_mm": round(d, 2), "hand_mm": world_point.round(2).tolist(),
            "angles_deg": {"yaw": round(math.degrees(yaw), 1), "roll": round(math.degrees(roll), 1),
                           "pitch": round(math.degrees(pitch), 1), "elbow": round(math.degrees(elbow), 1)},
            "position_tolerance_20mm_reached": d <= 20.}


def cylinder_axis(point, axis, radius, depth):
    rot = {"x": bd.Rot(0, 90, 0), "y": bd.Rot(90, 0, 0), "z": bd.Rot(0, 0, 0)}[axis]
    return bd.Pos(*point) * rot * bd.Cylinder(radius, depth)


def case(center, size):
    return bd.Pos(*center) * bd.Box(*size)


def mesh_record(items, name, shape, group, owner="root", color=None):
    vertices, faces = shape.tessellate(.45, .45)
    m = trimesh.Trimesh(np.array([[v.X, v.Y, v.Z] for v in vertices]),
                        np.asarray(faces), process=False)
    m.merge_vertices()
    rgba = color or COLORS[group]
    m.visual.vertex_colors = rgba
    color_hex = "#%02x%02x%02x" % tuple(rgba[:3])
    items.append({"name": name, "group": group, "owner": owner, "color": color_hex,
                  "positions": m.vertices.round(4).ravel().tolist(),
                  "indices": m.faces.ravel().tolist()})


def transform(owner: str, pose: str) -> np.ndarray:
    angles = {
        "neutral": (0., 0., 0., 0.),
        "two_axis_reach": (0., 38., -66., -92.),
        "three_axis_reach": (-43., 38., -66., -92.),
        "two_axis_crawl": (0., 28., -42., -72.),
        "three_axis_crawl": (-38., 28., -42., -72.),
    }[pose]
    if owner == "root" or owner == "fixed": return np.eye(4)
    yaw, roll, pitch, elbow = angles
    if owner in ("upper_arm", "shoulder_drives"):
        r = shoulder_matrix(yaw, roll, pitch)
        t = np.eye(4); t[:3, :3] = r; t[:3, 3] = PIVOT - r @ PIVOT
        return t
    if owner == "forearm":
        r = shoulder_matrix(yaw, roll, pitch) @ rot_matrix("y", elbow)
        elbow_world = PIVOT + shoulder_matrix(yaw, roll, pitch) @ (ELBOW - PIVOT)
        t = np.eye(4); t[:3, :3] = r; t[:3, 3] = elbow_world - r @ ELBOW
        return t
    return np.eye(4)


def as_three_matrix(m):
    return m.T.flatten().round(9).tolist()


def build_wall(wall: float, scene_items: list | None = None) -> dict:
    if wall not in PART_CACHE:
        PART_CACHE[wall] = {p["name"]: p["shape"] for p in Study(350, wall=wall, gap=.65).build().parts}
    parts = PART_CACHE[wall]
    outer = source_outer_envelopes(wall)
    torso = bd.Compound(children=[parts[n] for n in ("torso_front", "torso_back")])
    skin_names = ("R_upper_arm_front", "R_upper_arm_back", "R_forearm_front", "R_forearm_back",
                  "R_palm_front", "R_palm_back", "R_elbow_cuff")
    arm_skin = {n: parts[n] for n in skin_names if n in parts}

    # Original source-size XL330 cases. Roll is torso-side; pitch, elbow and
    # paw cases follow their parent links. The dimensions are vendor-envelope
    # permutations, with a declared shaft/output hypothesis per axis.
    cases = {
        "shoulder_roll": case((38., 0., 243.), (34., 20., 26.)),
        "shoulder_pitch": case((38., 0., 216.), (34., 20., 26.)),
        "elbow_pitch": case((89., 1., 220.), (20., 26., 34.)),
        "paw_open": case((103., 1., 150.), (20., 26., 34.)),
    }
    pivots = {"shoulder_roll": ((66., 0., 246.), "x"),
              "shoulder_pitch": ((66., 0., 246.), "y"),
              "elbow_pitch": ((96., 2., 194.5), "y"),
              "paw_open": ((108., 0., 106.), "y")}
    support_shapes = {}
    for name, (p, axis) in pivots.items():
        # Separate bearing-seat pair and supported output shaft; rings are
        # clearance envelopes, not a selected bearing or printed seat.
        pair = []
        for offset in (-10., 10.):
            q = list(p); q["xyz".index(axis)] += offset
            ring = cylinder_axis(q, axis, 5., 4.) - cylinder_axis(q, axis, 2., 5.)
            pair.append(ring)
        shaft = cylinder_axis(p, axis, 2., 26.)
        support_shapes[name] = pair + [shaft]
    # Light carrier webs show where reactions close into the torso/upper-arm
    # structure. They remain hypotheses and are screened against the shell.
    carrier = case((66., 27., 246.), (28., 4., 16.))
    supports = {**support_shapes, "shoulder_carrier": [carrier]}
    pack = case(PACK_CENTER, PACK_SIZE)
    service = case((-27., 5., 221.), (20., 20., 16.))

    shell_hits = {}
    for name, shape in cases.items():
        targets = [("torso", torso)] if name.startswith("shoulder") else [
            ("right_arm", bd.Compound(children=list(arm_skin.values())))]
        shell_hits[name] = {region: volume(shape.intersect(target)) for region, target in targets}
    support_hits = {}
    for name, shapes in supports.items():
        targets = [("torso", torso)] if name.startswith("shoulder") else [
            ("right_arm", bd.Compound(children=list(arm_skin.values())))]
        support_hits[name] = {region: round(sum(volume(sh.intersect(target)) for sh in shapes), 3)
                              for region, target in targets}
    exterior_regions = {
        "shoulder_roll": bd.Compound(children=[outer["torso"], outer["upper_arm"]]),
        "shoulder_pitch": bd.Compound(children=[outer["torso"], outer["upper_arm"]]),
        "elbow_pitch": bd.Compound(children=[outer["upper_arm"], outer["forearm"], outer["elbow_cuff"]]),
        "paw_open": bd.Compound(children=[outer["forearm"], outer["paw"]]),
    }
    exterior = {"motor_case_outside_source_outer_mm3": {},
                "support_and_carrier_outside_source_outer_mm3": {},
                "shared_pack_and_service_outside_source_outer_mm3": {}}
    for name, shape in cases.items():
        exterior["motor_case_outside_source_outer_mm3"][name] = volume(shape - exterior_regions[name])
    support_owners = {"shoulder_carrier": "shoulder_roll", "shoulder_roll": "shoulder_roll",
                      "shoulder_pitch": "shoulder_pitch", "elbow_pitch": "elbow_pitch", "paw_open": "paw_open"}
    for name, shapes in supports.items():
        region = exterior_regions[support_owners[name]]
        exterior["support_and_carrier_outside_source_outer_mm3"][name] = [volume(sh - region) for sh in shapes]
    exterior["shared_pack_and_service_outside_source_outer_mm3"] = {
        "shared_4S850_pack": volume(pack - outer["torso"]),
        "shared_service_reserve": volume(service - outer["torso"])}
    pack_hits = {"battery_vs_torso_mm3": volume(pack.intersect(torso)),
                 "service_vs_torso_mm3": volume(service.intersect(torso))}
    component_hits = []
    all_components = [(k, v) for k, v in cases.items()] + [
        (f"{name}_support_{i}", shape) for name, shapes in supports.items()
        for i, shape in enumerate(shapes)] + [("shared_4S850_pack", pack), ("shared_service_reserve", service)]
    for i, (name, a) in enumerate(all_components):
        for other, b in all_components[i+1:]:
            if "support_" in name and "support_" in other and name.split("_support_")[0] == other.split("_support_")[0]:
                continue  # mating same-axis ring/shaft members
            overlap = volume(a.intersect(b))
            if overlap > 1.0: component_hits.append({"a": name, "b": other, "intersection_mm3": overlap})

    result = {"wall_mm": wall, "motor_case_shell_intersections_mm3": shell_hits,
              "support_and_carrier_shell_intersections_mm3": support_hits,
              "source_outer_envelope_containment": exterior,
              "shared_pack_and_service_intersections_mm3": pack_hits,
              "nonmating_component_intersections": component_hits,
              "case_dimensions_mm": {"XL330-M288-T": [20, 34, 26]},
              "case_mass_g_each": 18, "case_torque": "0.52 N m stall at 5 V; continuous torque unpublished"}

    if scene_items is not None:
        for name, sh in parts.items():
            if name.startswith(("torso_", "pelvis_", "head_", "neck_", "L_", "R_thigh", "R_shin", "R_ankle", "R_foot", "R_sole", "R_knee")):
                mesh_record(scene_items, name, sh, "shell", "root")
        for name, sh in arm_skin.items():
            owner = "upper_arm" if "upper_arm" in name or "elbow" in name else "forearm"
            if "palm" in name: owner = "forearm"
            mesh_record(scene_items, name, sh, "shell", owner)
        for name, sh in cases.items(): mesh_record(scene_items, name+"_XL330", sh, "motor", "shoulder_drives" if name.startswith("shoulder") else ("upper_arm" if name=="elbow_pitch" else "forearm"))
        for name, shapes in supports.items():
            owner = "shoulder_drives" if name.startswith("shoulder") else ("upper_arm" if name=="elbow_pitch" else "forearm")
            for i, sh in enumerate(shapes): mesh_record(scene_items, f"{name}_support_{i}", sh, "support", owner)
        mesh_record(scene_items, "shared_4S850_pack", pack, "pack", "fixed")
        mesh_record(scene_items, "shared_service_reserve", service, "service", "fixed")
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    screens = [build_wall(w) for w in (1., 2.)]
    poses = {
        "neutral": ((0., 0., 0., 0.), (0., 0., 0., 0.)),
        "two_axis_reach": ((0., 38., -66., -92.), (0., 38., -66., -92.)),
        "three_axis_reach": ((-43., 38., -66., -92.), (-43., 38., -66., -92.)),
        "two_axis_crawl": ((0., 28., -42., -72.), (0., 28., -42., -72.)),
        "three_axis_crawl": ((-38., 28., -42., -72.), (-38., 28., -42., -72.)),
    }
    # Targets are points in the source torso frame chosen to represent distinct
    # task regions. The score is positional reach only; grasp force, contact,
    # obstacle avoidance, torso crouch and self-collision are out of scope.
    targets = {
        "simple_grasp_forward": {"hand_mm": np.array((28., -70., 176.)), "shoulder_mm": PIVOT},
        "floor_reach_after_crouch": {"hand_mm": np.array((40., -40., 18.)), "shoulder_mm": np.array((66., -20., 155.))},
        "crawl_right_hand_support": {"hand_mm": np.array((96., -45., 45.)), "shoulder_mm": np.array((66., -20., 155.))},
        "cross_body_yaw_loss_probe": {"hand_mm": np.array((-31.2, -15., 243.2)), "shoulder_mm": PIVOT},
    }
    workspace = {}
    for name, spec in targets.items():
        target, origin = spec["hand_mm"], spec["shoulder_mm"]
        workspace[name] = {"target_hand_mm": target.tolist(), "assumed_shoulder_origin_mm": np.asarray(origin).tolist(),
                           "shoulder_3axis_baseline": nearest(target, True, origin),
                           "shoulder_2axis_option": nearest(target, False, origin)}
    # Separate option: rotate the whole trunk/robot about its vertical axis to
    # bring the neutral forward-grasp target into the yaw-free arm's workspace.
    # This carries an orientation/locomotion cost, so it is not counted as a
    # stationary equivalent to the shoulder-yaw axis.
    grasp = targets["simple_grasp_forward"]["hand_mm"]
    body_origin = np.array((0., 0., 164.))
    yaw_comp = []
    for body_yaw_deg in np.linspace(-30., 30., 13):
        body_yaw = math.radians(body_yaw_deg)
        R = rot_matrix("z", -body_yaw_deg)
        target_body = body_origin + R @ (grasp - body_origin)
        result = nearest(target_body, False, PIVOT)
        world_R = rot_matrix("z", body_yaw_deg)
        hand_world = body_origin + world_R @ (np.asarray(result["hand_mm"]) - body_origin)
        yaw_comp.append({"body_yaw_deg": round(float(body_yaw_deg),1),
                         "distance_mm": result["distance_mm"],
                         "hand_mm": hand_world.round(2).tolist(),
                         "angles_deg": result["angles_deg"]})
    best_yaw_comp = min(yaw_comp, key=lambda row: row["distance_mm"])
    # Sample uniform joint ranges to compare positional workspace bounding box
    # and convex-hull volume. A finite grid is a geometry summary, not IK proof.
    def sample_cloud(has_yaw):
        cloud=[]
        for yaw in (np.deg2rad(np.linspace(-114.6,114.6,9)) if has_yaw else [0.]):
            for roll in np.deg2rad(np.linspace(-91.7,91.7,9)):
                for pitch in np.deg2rad(np.linspace(-126.1,126.1,11)):
                    for elbow in np.deg2rad(np.linspace(-126.1,8.6,7)):
                        cloud.append(endpoint(math.degrees(yaw),math.degrees(roll),math.degrees(pitch),math.degrees(elbow)))
        return np.asarray(cloud)
    clouds={"three_axis_baseline":sample_cloud(True),"two_axis_option":sample_cloud(False)}
    cloud_stats={}
    for key, pts in clouds.items():
        cloud_stats[key]={"sample_count":len(pts),"min_mm":pts.min(0).round(2).tolist(),
                          "max_mm":pts.max(0).round(2).tolist(),
                          "mesh_wrist_reachable_radius_mm":round(float(np.linalg.norm(HAND-PIVOT)),2)}
    case_collision_details = {}
    # Main report transparently labels all unresolved geometry and sourcing.
    report={"stage":"concept; static source-shell package coupon and sampled kinematic comparison",
            "reference":"Microban official URDF shoulder pitch+roll+elbow architecture principle only; no reference CAD geometry used",
            "source_shell":"experiments/sketch_ape_shell/build.py Study(350, wall=1/2, gap=.65)",
            "axis_count":19,"removed_axes":["left_shoulder_yaw","right_shoulder_yaw"],
            "axis_map":{"each_leg":["hip_yaw","hip_roll","hip_pitch","knee_pitch","ankle_pitch"],
                        "each_arm":["shoulder_roll","shoulder_pitch","elbow_pitch","paw_open"],"head":["head_yaw"]},
            "retained_arm_axes":"shoulder roll + pitch + elbow pitch + one opening-paw axis per side",
            "actuator":"XL330-M288-T; 20x34x26 mm, 18 g, 0.52 N m stall at 5 V; continuous rating unpublished",
            "actuator_source":"https://emanual.robotis.com/docs/en/dxl/x/xl330-m288/",
            "motor_mass_g_all_19_axes":342,"observed_one_off_motor_listing_usd_all_19_axes":522.31,
            "three_axis_hardware_package_reference":{"path":"docs/design/ape-concepts/swarm-round3-shoulder-split.md",
                "verdict":"reject_at_static_package_gate","pitch_case_torso_intersection_mm3":{"1mm":872.907,"2mm":1744.879},
                "nonmating_component_overlaps":25,"note":"kinematic baseline is not a hardware-fit pass"},
            "same_pack":{"source":"Tattu 4S850 candidate", "case_mm":{"center":PACK_CENTER,"size":PACK_SIZE},
                         "mass_g":107,"service_reserve_center_mm":[-27,5,221],"service_reserve_size_mm":[20,20,16]},
            "screens":screens,"workspace_targets":workspace,"workspace_sample_summary":cloud_stats,
            "whole_body_yaw_compensation_for_forward_grasp":{"searched_body_yaw_deg":[-30,30],"step_deg":5,
                "best_yaw_free_arm_solution":best_yaw_comp,
                "interpretation":"the trunk must rotate by the reported body yaw; this changes robot heading and is not free or equivalent to local shoulder swivel"},
            "interpretation":{"floor_reach":"score uses an assumed shoulder origin lowered to z=155 mm after crouch; the rest of the body pose and its collision are not modeled",
                              "simple_grasp":"forward positional target only; paw opening angle and grasp success are not validated",
                              "cross_body_yaw_loss_probe":"deliberately probes a cross-body hand position that uses shoulder yaw in the baseline; not a required basic grasp pose",
                              "crawl":"one right-hand support target only; whole-body four-point stability is not modeled",
                              "lost_motion":"removing shoulder yaw fixes one shoulder swivel degree; compare target distances and cloud extents, do not call 2-axis generally equivalent"},
            "unresolved":["all parts are envelope/carrier hypotheses, not a released assembly","no continuous moving B-rep clearance or exterior containment sweep",
                          "bearing and shaft grades, fasteners, output datums, drive connection, hard stops and assembly access unresolved",
                          "no motor torque/duty qualification; 0.52 N m is stall","workspace scores omit grasp force, contact, self-collision and whole-body support"]}
    items=[]
    build_wall(2.,items)
    pose_mats={p:{owner:as_three_matrix(transform(owner,p)) for owner in ("root","fixed","upper_arm","forearm","shoulder_drives")}
               for p in poses}
    # Pose names describe the represented articulation; yaw is zero in the
    # two-axis option and deliberately nonzero in baseline comparison views.
    (OUT/"scene.json").write_text(json.dumps({"height_mm":350,"shell_wall_mm":2,"parts":items,
        "poses":pose_mats,"pose_names":list(poses)},separators=(",",":")))
    (OUT/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    viewer = (TASK/"viewer.html").read_text()
    viewer = viewer.replace("__SCENE__", "scene.json")
    viewer = viewer.replace("./vendor/three.module.js", "../../../../../experiments/sketch_ape_shell/vendor/three.module.js")
    viewer = viewer.replace("./vendor/OrbitControls.js", "../../../../../experiments/sketch_ape_shell/vendor/OrbitControls.js")
    (OUT/"index.html").write_text(viewer)
    print(json.dumps({"output":str(OUT),"wall_screens":len(screens),
                      "shell_hit_counts":{str(x["wall_mm"]):sum(v>0 for group in (x["motor_case_shell_intersections_mm3"],x["support_and_carrier_shell_intersections_mm3"]) for row in group.values() for v in row.values()) for x in screens},
                      "workspace":workspace},indent=2))


if __name__ == "__main__": main()
