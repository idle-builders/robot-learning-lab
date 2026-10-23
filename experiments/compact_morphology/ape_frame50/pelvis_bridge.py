"""Connect the exposed bilateral fork hips to one provisional bare pelvis.

All geometry is in the source Cove coordinates (mm: lateral X, rear Y, up Z).
This is a load-path and moving-clearance experiment, not a printable assembly.
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
from hip_fork_exposed_probe import moved_hardware
from experiments.compact_morphology.ape_design.swarm_cove_fork_hip import build as hip

OUT = ROOT / "artifacts/ape-frame50-pelvis-bridge"


def fixed_pelvis():
    # Each cheek intentionally overlaps its source bulkhead at the inboard
    # front corner; the two front rails join the cheeks across the V pelvis.
    # The front routing leaves the shared battery and moved pitch case aft.
    parts = {}
    for level, z in (("low", 227.75), ("high", 245.75)):
        parts[f"{level}_crossbeam"] = hip.box((0, -48, z), (93, 6, 4))
        for side, x in (("left", -45.5), ("right", 45.5)):
            parts[f"{side}_{level}_cheek"] = hip.box((x, -32, z), (2, 34, 4))
    # A central upright connects the two rails and carries a forward torso
    # post. It is kept forward of the nominal 4S pack at rear Y=14 mm.
    parts["front_spine"] = hip.box((0, -48, 236.75), (12, 6, 23))
    parts["torso_post"] = hip.box((0, -48, 291), (12, 6, 90))
    # A removable battery/motor tray is only a candidate; no clamp interface
    # is claimed for the two yaw motor cases until the fasteners are designed.
    parts["yaw_motor_tray"] = hip.box((0, 1, 162), (94, 61, 4))
    parts["tray_front_tie"] = hip.box((0, -37, 196), (12, 23, 66))
    return parts


def mass_and_mesh(shape, color):
    vertices, faces = shape.tessellate(0.35, 0.4)
    mesh = trimesh.Trimesh(vertices=np.asarray([[v.X, v.Y, v.Z] for v in vertices]),
                           faces=np.asarray(faces), process=False)
    mesh.visual.vertex_colors = color
    return mesh


def overlap(a, b):
    aa, bb = a.bounding_box(), b.bounding_box()
    if any(getattr(aa.max, axis) < getattr(bb.min, axis) or
           getattr(bb.max, axis) < getattr(aa.min, axis) for axis in "XYZ"):
        return 0.0
    volumes, error = hip.both(a, b)
    if error:
        raise RuntimeError(error)
    return max(volumes)


def inspect():
    bridge = fixed_pelvis()
    hardware = moved_hardware()
    load_paths = []
    for side, sign in (("left", -1), ("right", 1)):
        fixed = {name: shape if sign == 1 else shape.mirror(bd.Plane.YZ)
                 for name, shape in hardware["fixed"].items()}
        for level in ("low", "high"):
            v = overlap(bridge[f"{side}_{level}_cheek"], fixed[f"yaw_bulkhead_{level}"])
            load_paths.append({"pair": [f"{side}_{level}_cheek", f"{side}_yaw_bulkhead_{level}"],
                               "intentional_interface_mm3": round(v, 3)})
        for level in ("low", "high"):
            v = overlap(bridge[f"{side}_{level}_cheek"], bridge[f"{level}_crossbeam"])
            load_paths.append({"pair": [f"{side}_{level}_cheek", f"{level}_crossbeam"],
                               "intentional_interface_mm3": round(v, 3)})
    for a, b in (("front_spine", "low_crossbeam"),
                 ("front_spine", "high_crossbeam"),
                 ("torso_post", "high_crossbeam"),
                 ("tray_front_tie", "yaw_motor_tray"),
                 ("tray_front_tie", "low_crossbeam")):
        load_paths.append({"pair": [a, b], "intentional_interface_mm3": round(overlap(bridge[a], bridge[b]), 3)})

    checks = {}
    source_internal = {}
    for pose, (yaw, roll, pitch) in hip.POSES.items():
        rows = []
        inherited_rows = []
        scene = trimesh.Scene()
        for name, shape in bridge.items():
            scene.add_geometry(mass_and_mesh(shape, (99, 110, 111, 255)),
                               geom_name=name, node_name=name)
        for sign, side in ((-1, "left"), (1, "right")):
            stages = hip.frame(hardware, sign, yaw, roll, pitch)
            for bulkhead in ("yaw_bulkhead_low", "yaw_bulkhead_high"):
                fixed_shape = stages["fixed"][bulkhead]
                for moving_stage in ("yaw", "roll", "thigh"):
                    for moving_name, moving_shape in stages[moving_stage].items():
                        # Shaft through the coaxial bearing is a deliberate
                        # mating interface, not a plate material clearance.
                        if moving_name == "yaw_spindle":
                            continue
                        v = overlap(fixed_shape, moving_shape)
                        if v > .05:
                            inherited_rows.append({"fixed": f"{side}_{bulkhead}",
                                                   "moving": f"{side}_{moving_stage}_{moving_name}",
                                                   "penetration_mm3": round(v, 3)})
            for stage, group in stages.items():
                for name, shape in group.items():
                    full = f"{side}_{stage}_{name}"
                    color = (73, 113, 144, 255) if "ST3250_case" in name else (
                        (201, 151, 78, 255) if stage != "fixed" else (112, 121, 125, 255))
                    scene.add_geometry(mass_and_mesh(shape, color), geom_name=full, node_name=full)
                    # Fixed source bulkheads and the new cheek/rail fusion are
                    # intentional. Every moving rigid part must clear the
                    # shared bridge, including the transported yaw carrier.
                    if stage == "fixed":
                        continue
                    for bridge_name, bridge_shape in bridge.items():
                        v = overlap(shape, bridge_shape)
                        if v > .05:
                            rows.append({"hardware": full, "bridge": bridge_name,
                                         "penetration_mm3": round(v, 3)})
        scene.export(OUT / f"pelvis-{pose}.glb")
        checks[pose] = rows
        source_internal[pose] = inherited_rows

    neutral = list(bridge.values())
    for sign in (-1, 1):
        neutral.extend(shape for stage in hip.frame(hardware, sign, 0, 0, 0).values()
                       for shape in stage.values())
    bd.export_step(bd.Compound(children=neutral), str(OUT / "pelvis-neutral.step"))
    report = {
        "stage": "bilateral fork-hip to shared-pelvis bridge; not a complete ape assembly",
        "body_datum": "approved 500 mm Cove/sketch ape",
        "coordinate_frame": "source Cove millimetres: X lateral right, Y rear, Z up",
        "source_axes_right_mm": {"yaw_roll": hip.PIVOT, "pitch": hip.PITCH},
        "source_hip_probe": "experiments/compact_morphology/ape_frame50/hip_fork_exposed_probe.py",
        "mass_limit": "Bridge solids are gross stock; real wall/infill, fasteners and battery mount remain open.",
        "bridge_gross_volume_mm3": round(sum(s.volume for s in bridge.values()), 2),
        "bridge_gross_petg_hf_mass_kg": round(sum(s.volume for s in bridge.values()) * 1.28 / 1e6, 4),
        "load_path_interfaces": load_paths,
        "inherited_source_bulkhead_hits": source_internal,
        "moving_to_shared_frame_hits": checks,
        "open_mechanics": ["source cheek to bulkhead fasteners", "yaw motor 59 mm coupler support",
                           "yaw motor tray/clamps", "torsion and bending strength of frame",
                           "knee/ankle/shoulder/elbow link integration", "full continuous swept clearance"],
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    shutil.copy2(Path(__file__).with_name("pelvis_bridge_viewer.html"), OUT / "index.html")
    return report


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    result = inspect()
    print(json.dumps({"interfaces": result["load_path_interfaces"],
                      "inherited_source_bulkhead_hit_counts": {name: len(rows) for name, rows in result["inherited_source_bulkhead_hits"].items()},
                      "pose_hit_counts": {name: len(rows) for name, rows in result["moving_to_shared_frame_hits"].items()},
                      "bridge_gross_petg_hf_mass_kg": result["bridge_gross_petg_hf_mass_kg"]}, indent=2))
