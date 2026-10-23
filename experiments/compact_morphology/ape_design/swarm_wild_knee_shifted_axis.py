"""Shifted-axis, remote-drive knee geometry coupon for the 350 mm sketch ape.

Run from the repository root:
    .venv/bin/python experiments/compact_morphology/ape_design/swarm_wild_knee_shifted_axis.py

This is an envelope and pose screen, not a load-rated transmission design.
X is lateral, Y rearward, Z up; all lengths are millimetres. The drawn motor
envelope is the official XC330-T288-T 20 x 34 x 26 mm body, oriented with
its 34 mm dimension along Z. Its exact horn, cable exit, and mounts are not
represented. The tiny capstans, cable line and bearings are placeholders.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import build123d as bd
import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "artifacts/sketch-ape-shell/exports/350mm/ape_shell_assembly.step"
OUT = ROOT / "artifacts/swarm-wild-knee-shifted-axis-20261004"
AXIS = (42.0, 15.0, 94.0)
ANGLES = tuple(range(0, 121, 5))
MOTOR_CENTER = (39.0, 3.0, 126.0)
MOTOR_BODY = (20.0, 26.0, 34.0)  # X,Y,Z orientation of official 20 W x 34 H x 26 D
MOTOR_CLEARANCE = (22.0, 28.0, 36.0)  # illustrative 1 mm expansion per face
SERVO_SOURCE = "https://emanual.robotis.com/docs/en/dxl/x/xc330-t288/"


def box(center, size):
    return bd.Pos(*center) * bd.Box(*size)


def cyl_x(x, y, z, radius, length):
    return bd.Solid.make_cylinder(radius, length,
                                  bd.Plane(origin=(x, y, z), z_dir=(1, 0, 0)))


def rod_x(x, yz0, yz1, radius):
    dy, dz = yz1[0] - yz0[0], yz1[1] - yz0[1]
    return bd.Solid.make_cylinder(radius, math.hypot(dy, dz),
                                  bd.Plane(origin=(x, *yz0), z_dir=(0, dy, dz)))


def ellipsoid(center, radii):
    return bd.Solid.make_sphere(1).scale(radii).translate(center)


def turn(shape, degrees):
    return bd.Pos(*AXIS) * bd.Rot(degrees, 0, 0) * bd.Pos(*(-v for v in AXIS)) * shape


def volume(shape):
    return sum(s.volume for s in shape.solids()) if shape is not None else 0.0


def overlap(left, right):
    # Imported STEP compounds are checked solid by solid; a direct compound
    # intersection may silently omit children in OCC.
    return sum(volume(a & b) for a in left.solids() for b in right.solids())


def render_part(scene, name, shape, rgba):
    if volume(shape) < 1e-8:
        return
    vertices, faces = shape.tessellate(0.38, 0.32)
    mesh = trimesh.Trimesh(
        vertices=np.array([(v.X, v.Y, v.Z) for v in vertices]),
        faces=np.array(faces), process=False)
    mesh.visual.vertex_colors = rgba
    scene.add_geometry(mesh, geom_name=name, node_name=name)


def write_viewer():
    (OUT / "index.html").write_text("""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Shifted-axis ape knee coupon</title><style>
*{box-sizing:border-box}body{margin:0;background:#ebe8e1;color:#263431;font:14px/1.45 system-ui,sans-serif}
header{padding:19px 3vw;background:#273e3a;color:#f7f4ed}h1{margin:0;font-size:24px}
header p{margin:6px 0 0;max-width:950px;color:#d4e3dc}main{display:grid;grid-template-columns:minmax(480px,2fr) minmax(330px,1fr);gap:18px;padding:18px;max-width:1500px;margin:auto}
#stage{height:min(76vh,760px);min-height:430px;background:#d9d6ce;border-radius:13px;overflow:hidden}
.panel{background:#f8f6f0;border:1px solid #d8d3c8;border-radius:13px;padding:18px}
input{width:100%}h2{font-size:17px;margin:0 0 8px}p{margin:9px 0}dl{display:grid;grid-template-columns:1fr auto;gap:5px 12px}dd{margin:0;font-weight:700}
.warn{padding:10px;border-left:4px solid #b55b3a;background:#f5e4db}a{color:#1c6768}
@media(max-width:850px){main{grid-template-columns:1fr}#stage{height:52vh;min-height:350px}}
</style><script type="importmap">{"imports":{"three":"https://cdn.jsdelivr.net/npm/three@0.171.0/build/three.module.js","three/addons/":"https://cdn.jsdelivr.net/npm/three@0.171.0/examples/jsm/"}}</script></head><body>
<header><h1>Shifted-axis knee · 350 mm ape shell</h1><p>Posterior and superior pivot, compact actuator in the thigh, and the sketch's rounded frontal seam. Flex poses show a geometric lead with an unresolved force path.</p></header>
<main><div id="stage"></div><section class="panel"><h2>Flex the knee</h2>
<input id="angle" type="range" min="0" max="4" step="1" value="0"><p id="position"></p><dl id="metrics"></dl>
<p class="warn"><strong>Mechanism unqualified.</strong> The narrow link crosses the unmodified rear thigh skin in flex. The shown 3 mm capstans and cable line have no rated transmission, bearings, mounts, or cable anchors.</p>
<p>Ivory: source shell. Dark olive: frontal knee cowl. Blue: official-size motor envelope. Orange: proposed pivot/link. Teal: illustrative cable path. Orbit to inspect the rear opening.</p>
<p><a href="report.json">Quantitative report</a> · <a href="fixed-envelope.step">Fixed STEP</a> · <a href="moving-envelope.step">Moving STEP</a></p></section></main>
<script type="module">
import * as THREE from 'three';import {OrbitControls} from 'three/addons/controls/OrbitControls.js';import {GLTFLoader} from 'three/addons/loaders/GLTFLoader.js';
const report=await fetch('report.json').then(r=>r.json());const stage=document.querySelector('#stage');const scene=new THREE.Scene();scene.background=new THREE.Color(0xd9d6ce);scene.add(new THREE.HemisphereLight(0xffffff,0xa8aaa0,2));const key=new THREE.DirectionalLight(0xffffff,2.6);key.position.set(130,-120,200);scene.add(key);
const renderer=new THREE.WebGLRenderer({antialias:true});renderer.setPixelRatio(Math.min(devicePixelRatio,2));renderer.outputColorSpace=THREE.SRGBColorSpace;renderer.setSize(stage.clientWidth,stage.clientHeight);stage.append(renderer.domElement);
const camera=new THREE.PerspectiveCamera(37,stage.clientWidth/stage.clientHeight,.1,1000);camera.up.set(0,0,1);camera.position.set(190,-190,147);const orbit=new OrbitControls(camera,renderer.domElement);orbit.target.set(42,4,90);orbit.enableDamping=true;orbit.update();
const loader=new GLTFLoader(),cache=new Map();let active=null;const angles=[0,30,60,90,120],slider=document.querySelector('#angle');
async function show(){const deg=angles[+slider.value];let model=cache.get(deg);if(!model){model=(await loader.loadAsync(`knee-${String(deg).padStart(3,'0')}.glb`)).scene;model.traverse(o=>{if(o.isMesh){o.material.roughness=.84;o.material.side=THREE.DoubleSide}});cache.set(deg,model)}if(active)scene.remove(active);active=model;scene.add(active);const r=report.samples.find(x=>x.flex_deg===deg);document.querySelector('#position').textContent=`${deg}° knee flex`;document.querySelector('#metrics').innerHTML=`<dt>Trimmed skin vs fixed shell</dt><dd>${r.shin_foot_vs_fixed_mm3.toFixed(2)} mm³</dd><dt>Link vs rear thigh</dt><dd>${r.link_vs_thigh_mm3.toFixed(2)} mm³</dd><dt>Motor vs moving leg</dt><dd>${r.motor_vs_moving_mm3.toFixed(2)} mm³</dd>`}
slider.oninput=show;addEventListener('resize',()=>{renderer.setSize(stage.clientWidth,stage.clientHeight);camera.aspect=stage.clientWidth/stage.clientHeight;camera.updateProjectionMatrix()});function tick(){requestAnimationFrame(tick);orbit.update();renderer.render(scene,camera)}tick();await show();
</script></body></html>""")


def main():
    if not SOURCE.exists():
        raise FileNotFoundError(f"Build the 350 mm sketch shell first: {SOURCE}")
    OUT.mkdir(parents=True, exist_ok=True)
    parts = {child.label: child for child in bd.import_step(SOURCE).children}
    thigh = bd.Compound(children=[parts[f"R_thigh_{side}"] for side in ("front", "back")])
    # Keep the entire source shell below 75 mm. The source shin top is 83 mm.
    shin_cut = box((42, 3, 49), (130, 120, 52))
    shins = [parts[f"R_shin_{side}"] & shin_cut for side in ("front", "back")]
    feet = [parts[f"R_{name}"] for name in ("foot_dorsal", "toe_1", "toe_2", "toe_3", "sole")]
    moving_skin = bd.Compound(children=shins + feet)
    # Preserve the source cuff's frontal form, opening its back for flexion.
    front_cuff = parts["R_knee_cuff"] & box((42, -31, 86), (130, 80, 40))  # Y <= 9
    # A small fixed frontal lip covers the 75-79 mm slit in standing pose.
    lip = (ellipsoid((42, 3, 85), (15, 17, 10)) -
           ellipsoid((42, 3, 85), (13, 15, 8))) & box((42, -31, 77), (130, 68, 4))
    fixed_skin = bd.Compound(children=[*thigh.solids(), front_cuff, lip])
    motor = box(MOTOR_CENTER, MOTOR_BODY)
    clearance_box = box(MOTOR_CENTER, MOTOR_CLEARANCE)
    # Link and supports show occupancy only. No attachment or bearing load is proved.
    link = rod_x(42, (15, 94), (10, 72), 1.5)
    shaft = cyl_x(35, 15, 94, 1.5, 16)
    bearings = [cyl_x(x, 15, 94, 3.5, 2) - cyl_x(x, 15, 94, 1.6, 2)
                for x in (35, 46)]
    capstans = [cyl_x(49, y, z, 3, 2) for y, z in ((6, 126), (15, 94))]
    cable_line = rod_x(50, (6, 126), (15, 94), 0.45)
    # Reuse the sketch's parametric inner loft to check nominal cavity volume.
    sys.path.insert(0, str(ROOT / "experiments/sketch_ape_shell"))
    from build import loft
    stations = [(87, 42, 3, 16, 18), (98, 43, 3, 20, 22),
                (124, 39, 3, 24, 25), (145, 35, 4, 22, 23),
                (157, 33, 4, 16, 18), (163, 31, 4, 8, 10)]
    nominal_cavity = loft([(z, x, y, rx - 2, ry - 2) for z, x, y, rx, ry in stations])
    rows = []
    for deg in ANGLES:
        posed_skin = turn(moving_skin, deg)
        posed_link = turn(link, deg)
        row = {
            "flex_deg": deg,
            "shin_foot_vs_fixed_mm3": round(overlap(posed_skin, fixed_skin), 5),
            "link_vs_thigh_mm3": round(overlap(posed_link, thigh), 5),
            "link_vs_front_cowl_mm3": round(overlap(posed_link, bd.Compound(children=[front_cuff, lip])), 5),
            "motor_vs_moving_mm3": round(overlap(motor, bd.Compound(children=[*posed_skin.solids(), posed_link])), 5),
        }
        rows.append(row)
        if deg in (0, 30, 60, 90, 120):
            scene = trimesh.Scene()
            for name, shape, color in [
                ("thigh", thigh, (230, 224, 209, 255)),
                ("cuff", front_cuff, (78, 86, 70, 255)),
                ("frontal_lip", lip, (78, 86, 70, 255)),
                ("XC330_official_body_envelope", motor, (65, 106, 128, 190)),
                ("bearing_placeholders", bd.Compound(children=bearings), (83, 85, 80, 255)),
                ("capstan_placeholders", bd.Compound(children=capstans), (57, 141, 140, 255)),
                ("cable_centerline_illustrative", cable_line, (65, 163, 157, 255)),
                ("shin_and_foot", posed_skin, (239, 231, 214, 255)),
                ("moving_link", posed_link, (198, 97, 44, 255)),
                ("moving_shaft", shaft, (198, 97, 44, 255)),
            ]:
                render_part(scene, name, shape, color)
            scene.export(OUT / f"knee-{deg:03d}.glb")
    bd.export_step(bd.Compound(children=[*fixed_skin.solids(), motor, *bearings, *capstans]),
                   OUT / "fixed-envelope.step")
    bd.export_step(bd.Compound(children=[*moving_skin.solids(), link, shaft]),
                   OUT / "moving-envelope.step")
    report = {
        "verdict": "CONDITIONAL GEOMETRY ONLY — REJECT AS A LOAD-RATED KNEE",
        "source_step": str(SOURCE.relative_to(ROOT)),
        "axis_mm": AXIS, "standing_source_cuff_center_mm": (42, 3, 86),
        "axis_shift_mm": {"rearward": 12, "upward": 8},
        "source_shin_top_z_mm": 83, "trimmed_shin_top_z_mm": 75,
        "front_cuff_rear_cut_y_mm": 9,
        "retained_front_cuff_volume_pct": round(100 * front_cuff.volume / parts["R_knee_cuff"].volume, 2),
        "motor": {
            "model": "ROBOTIS XC330-T288-T", "source": SERVO_SOURCE,
            "official_body_dimensions_mm": "20 W x 34 H x 26 D",
            "modeled_xyz_mm": MOTOR_BODY, "center_mm": MOTOR_CENTER,
            "body_inside_nominal_thigh_cavity_pct": round(100 * overlap(motor, nominal_cavity) / motor.volume, 5),
            "one_mm_expanded_box_inside_nominal_cavity_pct": round(100 * overlap(clearance_box, nominal_cavity) / clearance_box.volume, 5),
            "body_vs_original_thigh_skin_mm3": round(overlap(motor, thigh), 5),
            "official_stall_torque_12V_Nm": 1.00,
            "continuous_torque_Nm": None,
            "torque_note": "ROBOTIS explicitly says stall torque differs from continuous output and real-world performance. No continuous knee margin is established.",
        },
        "illustrative_transmission": {
            "type": "parallel-axis 1:1 opposed-cable capstans; not built or rated",
            "capstan_radius_mm": 3,
            "tension_difference_at_official_stall_N": round(1000 / 3, 2),
            "note": "Even the momentary stall figure would demand 333 N cable tension difference at this radius, before losses. Cable, anchors, pulley support, bend life, and thermal duty are unspecified."
        },
        "samples": rows,
        "sample_step_deg": 5,
        "limitations": [
            "Skin is collision-free only at the 5-degree sampled poses; continuous sweep and positive manufacturing gap are unproved.",
            "The illustrative 3-mm moving link intersects the original rear thigh shell; a slot/relief and its structural effect remain unbuilt.",
            "Bearing rings are envelopes without a mounting load path or manufacturer rating; horn, fasteners, cables and connector egress are absent.",
            "The 1-mm motor expansion is a volume check against a nominal inner loft, not minimum surface clearance or assembled motor fit.",
            "Adjacent leg, hip, boot and ground contacts, mass distribution, knee strength and full-body motion are unscreened."
        ],
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    write_viewer()
    print(json.dumps({
        "artifact": str(OUT), "verdict": report["verdict"],
        "max_skin_overlap_mm3": max(r["shin_foot_vs_fixed_mm3"] for r in rows),
        "max_link_thigh_overlap_mm3": max(r["link_vs_thigh_mm3"] for r in rows),
        "motor_inside_nominal_cavity_pct": report["motor"]["body_inside_nominal_thigh_cavity_pct"],
        "stall_cable_delta_N": report["illustrative_transmission"]["tension_difference_at_official_stall_N"],
    }, indent=2))


if __name__ == "__main__":
    main()
