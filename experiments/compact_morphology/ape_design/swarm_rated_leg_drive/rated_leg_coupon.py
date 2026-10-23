"""Rated direct leg-drive package against the unmodified sketch-06 shell.

Run from the repository root with ``.venv/bin/python``.  All distances are mm.
This is a falsification coupon: boxes are conservative vendor-size motor cases;
the 25T adapter and printed bracket are geometrical reserves, not rated parts.
"""
from __future__ import annotations

import contextlib
import io
import json
import shutil
import sys
from pathlib import Path

import build123d as b
import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
OUT = ROOT / "artifacts/rated-leg-drive-20261005"
sys.path.insert(0, str(ROOT / "experiments/sketch_ape_shell"))
from build import Study
sys.path.insert(0, str(ROOT / "experiments/compact_morphology/ape_design/swarm_serial_scale"))
from serial_scale_case import shell_shapes

MOTORS = {
    "ST3250": dict(size=(45.22, 24.72, 35.0), mass_g=74.5,
                   rated_nm=16 * 0.0980665, no_load_s_60=0.133,
                   housing="aluminium", rail_v=12,
                   source="https://www.feetechrc.com/en/562636.html"),
    "ST3215": dict(size=(45.2, 24.7, 35.0), mass_g=55.0,
                   rated_nm=10 * 0.0980665, no_load_s_60=0.222,
                   housing="PA+GF", rail_v=12,
                   source="https://www.feetechrc.com/525603.html"),
}
BEARING = dict(id_mm=6, od_mm=13, width_mm=3.5, mass_g=1.9,
               static_n=224, dynamic_n=618,
               source="https://www.skf.com/binaries/pub12/Images/0901d19680406705-SKF-Stainless-steel-DGBB---11279_1-EN-Low-Res_tcm_12-259995.pdf")


def vol(shape):
    return sum(s.volume for s in shape.solids()) if shape else 0.0


def maybe_intersection(a, c):
    if not a or not c or vol(a) < 1e-9 or vol(c) < 1e-9:
        return 0.0
    aa, cc = a.bounding_box(), c.bounding_box()
    if any(getattr(aa.min, axis) >= getattr(cc.max, axis) or
           getattr(cc.min, axis) >= getattr(aa.max, axis) for axis in "XYZ"):
        return 0.0
    return vol(a.intersect(c))


def axis_cylinder(axis, radius, length, center):
    rotation = {"x": b.Rot(0, 90, 0), "y": b.Rot(90, 0, 0)}[axis]
    return b.Pos(*center) * rotation * b.Cylinder(radius, length)


def motor_box(axis, joint, model):
    """Place the 35 mm axial depth inward from output and 12.2 mm long-axis offset.

    The axial output face and 12.2 mm long-axis offset follow Feetech's drawing.
    The ST3215 case shares its published form factor; this is still a box reserve.
    """
    long, narrow, depth = MOTORS[model]["size"]
    x, y, z = joint
    if axis == "x":
        center = (x - depth / 2, y, z + 12.2)
        return b.Pos(*center) * b.Box(depth, narrow, long), center
    center = (x, y + depth / 2, z + 12.2)
    return b.Pos(*center) * b.Box(narrow, depth, long), center


def package(scale):
    # Same anatomical outputs as the corrected 21-axis ladder.  Use direct
    # drive to avoid granting an unqualified miniature belt a torque rating.
    spec = [
        ("hip_roll", "ST3250", "y", (32*scale, 0, 158*scale), "pelvis"),
        ("knee_pitch", "ST3250", "x", (42*scale, 3*scale, 86*scale), "thigh_R"),
        ("ankle_pitch", "ST3215", "x", (46*scale, 4*scale, 24*scale), "shin_R"),
    ]
    result = []
    for name, model, axis, joint, parent in spec:
        case, center = motor_box(axis, joint, model)
        # Two externally supported output stations, a through shaft and a
        # motor-output coupler reserve.  Both seats must attach to the parent
        # frame; the 2 mm web shown below is a proposed printed load path.
        stations = [7.0, 15.0]
        bearings = []
        seats = []
        for offset in stations:
            q = (joint[0]+offset, joint[1], joint[2]) if axis == "x" else (joint[0], joint[1]-offset, joint[2])
            bearings.append(axis_cylinder(axis, 6.5, 3.5, q) - axis_cylinder(axis, 3, 4.0, q))
            seats.append(axis_cylinder(axis, 8.5, 5.5, q) - axis_cylinder(axis, 6.5, 6.0, q))
        if axis == "x":
            shaft = axis_cylinder(axis, 3, 24, (joint[0]+6, joint[1], joint[2]))
            adapter = axis_cylinder(axis, 5, 5, (joint[0]+1.5, joint[1], joint[2]))
            bridge = b.Pos(joint[0]+7, joint[1]+8, joint[2]+16) * b.Box(20, 4, 32)
            moving_yoke = b.Pos(joint[0]+16, joint[1], joint[2]-8) * b.Box(3, 18, 21)
        else:
            shaft = axis_cylinder(axis, 3, 24, (joint[0], joint[1]-6, joint[2]))
            adapter = axis_cylinder(axis, 5, 5, (joint[0], joint[1]-1.5, joint[2]))
            bridge = b.Pos(joint[0]+8, joint[1]-7, joint[2]+16) * b.Box(4, 20, 32)
            moving_yoke = b.Pos(joint[0], joint[1]-16, joint[2]-8) * b.Box(18, 3, 21)
        parts = {"motor_case":case,"bearing_A":bearings[0],"bearing_B":bearings[1],
                 "seat_A":seats[0],"seat_B":seats[1],"shaft":shaft,
                 "25T_adapter_reserve":adapter,"parent_bridge":bridge,"moving_yoke":moving_yoke}
        result.append(dict(name=name,model=model,axis=axis,joint=joint,
                           motor_center=center,parent=parent,parts=parts))
    return result


def exact_source_skin(height):
    with contextlib.redirect_stdout(io.StringIO()):
        source = Study(height, 1, .65).build().parts
    names = ("pelvis_", "torso_", "R_thigh_", "R_shin_", "R_foot_", "R_sole", "R_toe_",
             "R_hip_cuff", "R_knee_cuff", "R_ankle_cuff")
    return {p["name"]:p["shape"] for p in source if p["name"].startswith(names)}


def measure(height):
    scale = height / 350
    skin = exact_source_skin(height)
    outer, _ = shell_shapes(height, 1)
    modules = package(scale)
    out = []
    for module in modules:
        name = module["name"]
        allowed = {"hip_roll":["pelvis","torso","hip_cuff_R","thigh_R"],
                   "knee_pitch":["thigh_R","knee_cuff_R","shin_R"],
                   "ankle_pitch":["shin_R","ankle_cuff_R","boot_R"]}[name]
        rows = []
        for part_name, shape in module["parts"].items():
            remain = shape
            for lobe in allowed:
                if lobe in outer and maybe_intersection(remain, outer[lobe][0]) > 0:
                    remain = remain.cut(outer[lobe][0])
            hits = []
            for skin_name, wall in skin.items():
                overlap = maybe_intersection(shape, wall)
                if overlap > .05:
                    hits.append(dict(source_part=skin_name, overlap_mm3=round(overlap, 3)))
            rows.append(dict(part=part_name, volume_mm3=round(vol(shape),3),
                             outside_adjacent_source_outer_mm3=round(vol(remain),3),
                             intact_source_skin_hits=hits))
        bridge_contacts={key:round(maybe_intersection(module["parts"]["parent_bridge"],module["parts"][key]),3)
                         for key in ("seat_A","seat_B")}
        out.append(dict(joint=name,model=module["model"],axis=module["axis"],
                        output_mm=[round(v,3) for v in module["joint"]],
                        motor_center_mm=[round(v,3) for v in module["motor_center"]],
                        parent=module["parent"],parts=rows,bridge_to_seat_contact_mm3=bridge_contacts,
                        motor_to_output="direct 1:1; rated gearbox in servo",
                        adapter="25T/OD5.9 mm to 6 mm shaft coupling is a size reserve, with no sourced torque/retention rating"))
    # Motor/structure clashes are separately checked; their intended contacts
    # are not cleared by subtracting housing volumes from one another.
    motor_conflicts=[]
    for i,a in enumerate(modules):
        for c in modules[i+1:]:
            v=maybe_intersection(a["parts"]["motor_case"],c["parts"]["motor_case"])
            if v>.05:motor_conflicts.append(dict(a=a["name"],b=c["name"],overlap_mm3=round(v,3)))
    hard_fail = any(p["outside_adjacent_source_outer_mm3"]>.05 or p["intact_source_skin_hits"]
                    for m in out for p in m["parts"])
    return dict(height_mm=height,wall_mm=1,source="unchanged Study(height,1,.65) exact 67-part BRep skin",
                modules=out,motor_motor_hits=motor_conflicts,
                static_gate_clear=not hard_fail and not motor_conflicts,
                coupled_leg_motion="not_run_static_gate_failed" if hard_fail or motor_conflicts else "not_run_requires_explicit_motion_builder"),modules,skin


def mesh(scene, name, shape, color):
    vertices, faces=shape.tessellate(.55,.55)
    tri=trimesh.Trimesh(np.array([[v.X,v.Y,v.Z] for v in vertices]),np.asarray(faces),process=False)
    tri.visual.vertex_colors=color
    scene.add_geometry(tri,geom_name=name,node_name=name)


def normalize_step(path):
    """Remove exporter-only trailing spaces; STEP syntax is whitespace agnostic."""
    path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines())+"\n")


def export_scene(modules,skin):
    scene=trimesh.Scene()
    for name,shape in skin.items():mesh(scene,"skin_"+name,shape,(226,219,204,85))
    colors={"motor_case":(48,101,153,255),"bearing_A":(64,65,69,255),"bearing_B":(64,65,69,255),
            "seat_A":(187,134,68,255),"seat_B":(187,134,68,255),"shaft":(80,80,82,255),
            "25T_adapter_reserve":(194,80,71,255),"parent_bridge":(168,137,82,255),"moving_yoke":(100,143,116,255)}
    for module in modules:
        for name,shape in module["parts"].items():mesh(scene,module["name"]+"_"+name,shape,colors[name])
    scene.export(OUT/"rated-leg-500.glb")
    # Separate solids retain the component identities in the STEP compound.
    step=OUT/"rated-leg-500.step"
    b.export_step(b.Compound(children=[part for m in modules for part in m["parts"].values()]),str(step))
    normalize_step(step)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    cases=[]
    for h in (350,450,500):
        print(f"checking {h} mm exact source shell",flush=True)
        report,modules,skin=measure(h)
        cases.append(report)
        print(f"  static pass: {report['static_gate_clear']}",flush=True)
        if h==500:export_scene(modules,skin)
    performance={name:{**data,"no_load_rpm":round(10/data["no_load_s_60"],2)} for name,data in MOTORS.items()}
    report=dict(cases=cases,manufacturer_data=performance,bearing=BEARING,
                known_motor_mass_per_leg_g=round(2*MOTORS["ST3250"]["mass_g"]+MOTORS["ST3215"]["mass_g"],1),
                six_external_bearings_per_leg_g=round(6*BEARING["mass_g"],1),
                mass_lower_bound={"basis":"source shell modeled material at illustrative solid-PLA 1.24 g/cm3 plus only six bilateral selected leg motors and twelve external bearings",
                                  "shell_350_cm3":304.582,"shell_500_cm3":645.919,
                                  "shell_350_g":round(304.582*1.24,1),"shell_500_g":round(645.919*1.24,1),
                                  "bilateral_selected_leg_motor_g":408.0,
                                  "bilateral_external_bearing_g":22.8,
                                  "shell_plus_selected_drive_350_g":round(304.582*1.24+430.8,1),
                                  "shell_plus_selected_drive_500_g":round(645.919*1.24+430.8,1),
                                  "excluded":"other axes, pack, chassis, brackets, adapters, shafts, electronics, cables and screws; no complete-body COM"},
                old_gait={"source":"artifacts/ape-physics-pair-20260930/forward-candidate.json",
                          "old_body_mass_kg":2.265,"ideal_servo_limit_nm":1.569,"hip_roll_saturation_fraction_L_R":[.73,.57],
                          "knee_saturation_fraction_L_R":[.036,.08],
                          "interpretation":"warning under the old 2.265 kg ideal-servo body; not a continuous-duty trace or a lower bound for a new, potentially lighter body"},
                transmission={"ratio":"1:1 direct","rated_torque_reserve_at_old_limit_nm":round(MOTORS["ST3250"]["rated_nm"]-1.569,4),
                              "25T_to_6mm_adapter":"unsourced torque and retention rating; unresolved",
                              "bearing_load":"W618/6 catalog ratings are radial; bracket, axle bending and bearing life not solved"},
                heat={"ST3250":"aluminium case gives a potential conductive path; mounting thermal resistance and loaded temperature curve unpublished",
                      "ST3215":"PA+GF case; no rated-duty thermal curve or mounted heat-path evidence"},
                scale_warning="50 cm source shell CAD material is 645919 vs 304582 mm3 at 35 cm; its 1.429x lever arm and 2.12x material-volume ratio suggest a 3.03x shell-only gravity-moment sensitivity, not a new whole-body gait torque prediction",
                verdict="No scale passes the unmodified source skin and component containment gate; coupled motion intentionally skipped.")
    (OUT/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    shutil.copyfile(HERE/"viewer.html",OUT/"index.html")
    print("wrote",OUT/"report.json")


if __name__=="__main__":main()
