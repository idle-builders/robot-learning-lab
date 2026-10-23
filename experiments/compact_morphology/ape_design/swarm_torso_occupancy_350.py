"""Combined 350 mm sketch-shell torso occupancy probe.

Run from the repository root with .venv/bin/python. Published case dimensions
are taken from the isolated joint coupons. Boards, frame, shaft supports,
connectors and service paths are explicitly design-space reservations.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import build123d as bd
import numpy as np
import trimesh

import hip_joint_coupon as hip
import shoulder_joint_coupon as shoulder

ROOT = Path(__file__).resolve().parents[3]
SOURCE_DIR = ROOT / "experiments/sketch_ape_shell"
sys.path.insert(0, str(SOURCE_DIR))
import build as source_shell

OUT = ROOT / "artifacts/ape-torso-occupancy-swarm-350"
BATTERY_SIZE = (60.0, 30.0, 30.5)  # Tattu 4S850 source candidate
BATTERY_CENTER = (0.0, 0.0, 195.0)  # relocated trial; not previous grid winner

COLORS = {
    "shell": [229, 221, 205, 60], "motor": [63, 105, 143, 255],
    "paired": [194, 119, 74, 255], "battery": [111, 171, 130, 255],
    "service": [222, 183, 82, 255], "spine": [115, 128, 139, 255],
    "corridor": [105, 165, 177, 115], "bearing": [177, 191, 198, 255],
    "round": [139, 102, 174, 255],
}


def vshape(center, size):
    return bd.Pos(*center) * bd.Box(*size)


def source_frame(center, size):
    """Transform coupon (forward,left,up) into source (right,rear,up) frame."""
    x, y, z = center
    sx, sy, sz = size
    return (y, -x, z), (sy, sx, sz)


def vol(a, b):
    common = a.intersect(b)
    return round(sum(s.volume for s in common), 2) if common else 0.0


def make_items():
    items = []
    # Exact case envelopes and placements are inherited from the independent
    # 350 mm coupons. Paired-axis output datums remain hypotheses.
    for item in hip.screen(350)["cases"]:
        center, size = source_frame(item["center_mm"], item["size_mm"])
        items.append({"name": item["name"], "center": center,
                      "size": size, "kind": "motor",
                      "layout": "common",
                      "source": "hip_joint_coupon.py; published XL330 / 2XC430 case envelope"})
    for row in shoulder.run(350)["scenarios"]:
        for item in row["paired"] if isinstance(row["paired"], list) else [row["paired"]]:
            center, size = source_frame(item["center_mm"], item["size_mm"])
            items.append({"name": item["name"], "center": center,
                          "size": size, "kind": "paired",
                          "layout": "paired_shoulder",
                          "source": "shoulder_joint_coupon.py; published 2XL430 outer case, output geometry unresolved"})
        for item in row["serial_direct"]:
            center, size = source_frame(item["center_mm"], item["size_mm"])
            items.append({"name": item["name"], "center": center,
                          "size": size, "kind": "motor",
                          "layout": "serial_shoulder",
                          "source": "shoulder_joint_coupon.py; published XL430 case envelope"})
    # One self-contained static candidate stack. The slim compute, regulator,
    # connector and spine volumes are assumptions, not selected hardware.
    reservations = [
        ("Tattu_4S850_candidate", BATTERY_CENTER, BATTERY_SIZE, "battery", "source: 60x30x30.5 mm, 107 g candidate"),
        ("compute_reserve", (0, 0, 229), (45, 35, 20), "service", "assumption: volume only; board/model unselected"),
        ("regulator_reserve", (22, 0, 209), (28, 26, 14), "service", "assumption: 12V/5V conversion, ratings unselected"),
        ("power_bus_connector_reserve", (0, 0, 249), (18, 24, 20), "service", "assumption: connector access volume"),
        ("inner_spine", (-20, 0, 197), (10, 24, 116), "spine", "hypothesis: continuous 10x24 mm structural web"),
        ("left_shoulder_service_corridor", (-4, 47, 252), (18, 14, 58), "corridor", "assumption: cable bend/service path"),
        ("right_shoulder_service_corridor", (-4, -47, 252), (18, 14, 58), "corridor", "assumption: cable bend/service path"),
        ("left_hip_service_corridor", (8, 34, 183), (22, 12, 46), "corridor", "assumption: cable bend/service path"),
        ("right_hip_service_corridor", (8, -34, 183), (22, 12, 46), "corridor", "assumption: cable bend/service path"),
        ("RS05_proximal_round_countercase", (0, 0, 193), (46, 46, 44), "round", "source package 46x46x44 mm, 191 g; provisional central-pelvis placement hypothesis; maker specifies heat sink; this screen omits thermal margin and connector/phase-wire exit"),
    ]
    for name, center, size, kind, source in reservations:
        center, size = source_frame(center, size)
        items.append({"name": name, "center": center, "size": size,
                      "kind": kind, "layout": "round_drive_countercase" if name.startswith("RS05") else "common", "source": source})
    return items


def compute(items):
    shapes = {i["name"]: vshape(i["center"], i["size"]) for i in items}
    torso_shapes = [p["shape"] for p in SOURCE_PARTS
                    if p["name"].startswith(("torso_", "pelvis_"))]
    # Cavity proxy is the source torso's exact inner loft before the outer-wall
    # subtraction, clipped by the same curved waist seam used by build.py.
    inner = source_shell.loft([(z, 0, 2, rx-2, ry-2) for z, rx, ry in
        [(160,42,25),(173,49,29),(199,58,34),(229,63,35),
         (253,63,30),(264,60,27),(274,35,22),(278,23,18)]])
    interior = inner & source_shell.above_waist()
    overlaps = []
    for n, a in enumerate(items):
        for b in items[n + 1:]:
            amount = vol(shapes[a["name"]], shapes[b["name"]])
            if amount > 0.05:
                overlaps.append({"a": a["name"], "b": b["name"], "intersection_mm3": amount})
    by_item = []
    for item in items:
        sh = shapes[item["name"]]
        cross = round(sum(vol(sh, torso) for torso in torso_shapes), 2)
        # Exact corner inclusion matches the previous screen's conservative
        # check, augmented by B-rep intersection with the shell skin.
        bb = sh.bounding_box()
        corners = [(x, y, z) for x in (bb.min.X, bb.max.X)
                   for y in (bb.min.Y, bb.max.Y) for z in (bb.min.Z, bb.max.Z)]
        outside = sum(not interior.is_inside(bd.Vector(x, y, z)) for x, y, z in corners)
        by_item.append({"name": item["name"], "torso_and_pelvis_skin_intersection_mm3": cross,
                        "corners_outside_nominal_inner_shell": outside,
                        "assumption_or_source": item["source"]})
    # Proxy free-volume figure: inner-shell volume minus clipped reservation
    # union. Overlapping objects are unioned first so space is never double-counted.
    inner_volume = round(interior.volume, 1)
    clipped = []
    for shape in shapes.values():
        clip = shape.intersect(interior)
        clipped.extend(list(clip))
    shapes_union = bd.Compound(children=clipped).fuse() if clipped else None
    occupied = round(shapes_union.volume, 1) if shapes_union else 0
    return {"nominal_inner_shell_volume_mm3": inner_volume,
            "reserved_union_inside_shell_mm3": occupied,
            "unreserved_proxy_volume_mm3": round(inner_volume - occupied, 1),
            "reservations": by_item, "positive_intersections": overlaps,
            "intersection_total_mm3_pairwise_nonexclusive": round(sum(x["intersection_mm3"] for x in overlaps), 2)}


def render(items):
    items = [i for i in items if i["layout"] in ("common", "paired_shoulder")]
    scene = trimesh.Scene()
    for p in SOURCE_PARTS:
        if p["name"].startswith(("torso_", "pelvis_", "L_upper_arm", "R_upper_arm", "L_thigh", "R_thigh")):
            add_shape(scene, p["name"], p["shape"], "shell")
    for item in items:
        add_shape(scene, item["name"], vshape(item["center"], item["size"]), item["kind"])
    # Service view: hide cosmetic shoulder/torso skin in a separate scene and
    # retain all reserved access volumes, so no overlap is hidden by opacity.
    service = trimesh.Scene()
    for item in items:
        add_shape(service, item["name"], vshape(item["center"], item["size"]), item["kind"])
    scene.export(OUT / "neutral.glb")
    service.export(OUT / "service-access.glb")
    serial = trimesh.Scene()
    for p in SOURCE_PARTS:
        if p["name"].startswith(("torso_", "pelvis_", "L_upper_arm", "R_upper_arm", "L_thigh", "R_thigh")):
            add_shape(serial, p["name"], p["shape"], "shell")
    for item in [i for i in make_items() if i["layout"] in ("common", "serial_shoulder")]:
        add_shape(serial, item["name"], vshape(item["center"], item["size"]), item["kind"])
    serial.export(OUT / "serial-shoulder-alternative.glb")
    round_scene = trimesh.Scene()
    for p in SOURCE_PARTS:
        if p["name"].startswith(("torso_", "pelvis_", "L_upper_arm", "R_upper_arm", "L_thigh", "R_thigh")):
            add_shape(round_scene, p["name"], p["shape"], "shell")
    round_items = [i for i in make_items() if i["layout"] in ("common", "paired_shoulder") and
                   not i["name"].startswith(("left_yaw", "right_yaw", "left_roll", "right_roll"))]
    round_items += [i for i in make_items() if i["layout"] == "round_drive_countercase"]
    for item in round_items:
        add_shape(round_scene, item["name"], vshape(item["center"], item["size"]), item["kind"])
    round_scene.export(OUT / "round-drive-countercase.glb")


def add_shape(scene, name, shape, kind):
    vertices, faces = shape.tessellate(0.15, 0.15)
    mesh = trimesh.Trimesh(vertices=np.array([[v.X, v.Y, v.Z] for v in vertices]),
                           faces=np.asarray(faces), process=False)
    mesh.visual.vertex_colors = COLORS[kind]
    scene.add_geometry(mesh, geom_name=name, node_name=name)


def shell_wall_sensitivity(items):
    thin_parts = source_shell.Study(350, wall=1, gap=.65).build().parts
    thin_torso = [p["shape"] for p in thin_parts
                  if p["name"].startswith(("torso_", "pelvis_"))]
    result = {}
    for layout in ("paired_shoulder", "serial_shoulder"):
        chosen = [i for i in items if i["layout"] in ("common", layout)]
        result[layout] = {
            item["name"]: round(sum(vol(vshape(item["center"], item["size"]), part)
                                    for part in thin_torso), 2)
            for item in chosen
        }
    return {"wall_mm": 1, "comparison_wall_mm": 2,
            "meaning": "same fixed case placements; exact torso and pelvis front/back B-rep intersections, no new fit optimization",
            "torso_and_pelvis_skin_intersection_mm3_by_layout_and_item": result}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    global SOURCE_PARTS
    SOURCE_PARTS = source_shell.Study(350, wall=2, gap=.65).build().parts
    # The old coupons compute their legacy corner-fit flags as a side effect;
    # this initializes their helper only for reusing their proposed centers.
    # All reported shell intersections below use the exact source B-reps above.
    hip.shell.build()
    items = make_items()
    report = {
        "height_mm": 350,
        "scope": "static occupancy against exact 350 mm Study source torso_front/back and pelvis_front/back B-reps; no full robot or moving pose gate",
        "shell_source": "experiments/sketch_ape_shell/build.py Study(height=350, wall=2, gap=0.65); exact torso and pelvis front/back B-reps used for shell intersections and GLB context",
        "coordinate_mapping": "coupon (forward,left,up) -> source sketch (right,rear,up): center (x,y,z)->(y,-x,z), box size (sx,sy,sz)->(sy,sx,sz); case placements remain hypotheses",
        "battery_candidate": "Tattu 4S850, source 60x30x30.5 mm, 107 g; shown rotated as 30x60x30.5 mm in source X/Y frame",
        "alternative_arrangements_screened": [
            "three serial shoulder case boxes: known case-to-case overlap in isolated coupon; retained as overlap competitor",
            "paired shoulder outer case, yaw relocation for hips, and shared battery/service/spine reservations: shown together here",
            "assumed compute and regulator sizes are transparent volume reservations, not component selections"],
        "hypothesis_limits": ["shoulder paired unit output axes and attachment are unresolved",
                              "hip yaw relocated output bearings/drive are conceptual; other 2XC output coordinates unavailable",
                              "spine and all cable/service corridors are geometric assumptions, not strength or bend-radius validation",
                              "no heat, battery retention, connector orientation, cover opening, or full motion sweep"],
        "layout_screens": {
            layout: compute([i for i in items if i["layout"] in ("common", layout)])
            for layout in ("paired_shoulder", "serial_shoulder")
        },
        "round_drive_countercase": compute([
            i for i in items if i["layout"] == "common" and
            not (i["name"].startswith(("left_yaw", "right_yaw", "left_roll", "right_roll")))
        ] + [i for i in items if i["layout"] == "paired_shoulder"] +
            [i for i in items if i["name"] == "RS05_proximal_round_countercase"]),
        "one_mm_wall_sensitivity": shell_wall_sensitivity(items),
        "items": items,
    }
    render(items)
    (OUT / "occupancy.json").write_text(json.dumps(report, indent=2) + "\n")
    shutil.copyfile(Path(__file__).with_name("torso_occupancy_viewer.html"), OUT / "index.html")
    print("items", len(items), "layout overlap pairs",
          {k: len(v["positive_intersections"]) for k, v in report["layout_screens"].items()},
          "free proxy volume mm3",
          {k: v["unreserved_proxy_volume_mm3"] for k, v in report["layout_screens"].items()})


if __name__ == "__main__":
    main()
