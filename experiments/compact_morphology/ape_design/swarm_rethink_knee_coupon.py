"""Shin-carried, belt-coupled knee coupon on the exact 350 mm ape shell.

The sourced STS3250 case is below the knee, inside the calf. The output drives
an equal 20-tooth knee pulley through a short 3M belt. This is a geometry and
load-path coupon, not a rated belt or bearing design. Units: mm, X lateral,
Y rear, Z up. Run with .venv/bin/python from the repository root.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import build123d as bd
import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "artifacts/sketch-ape-shell/exports/350mm/ape_shell_assembly.step"
OUT = ROOT / "artifacts/swarm-rethink-knee-20261004"
KNEE = (42.0, 25.0, 86.0)
DRIVE = (42.0, 3.0, 86.0-math.sqrt(37.5**2-22.0**2))
PITCH_RADIUS = 24.0 / math.pi  # 16 teeth x 3 mm pitch


def box(center, size):
    return bd.Pos(*center) * bd.Box(*size)


def clip(x0, x1, y0, y1, z0, z1):
    return bd.Solid.make_box(x1-x0, y1-y0, z1-z0,
                             bd.Plane(origin=(x0, y0, z0)))


def cyl_x(radius, x0, x1, center):
    return bd.Solid.make_cylinder(radius, x1-x0,
                                  bd.Plane(origin=(x0, center[1], center[2]),
                                           z_dir=(1, 0, 0)))


def rod(a, b, radius):
    v = bd.Vector(b) - bd.Vector(a)
    return bd.Solid.make_cylinder(radius, v.length,
                                  bd.Plane(origin=a, z_dir=v.normalized()))


def ellipsoid(center, radii):
    return bd.Solid.make_sphere(1).scale(radii).translate(center)


def comp(items):
    return bd.Compound(children=[s for item in items for s in item.solids()])


def volume(item):
    return sum(s.volume for s in item.solids()) if item else 0.0


def overlap(a, b):
    total = 0.0
    for u in a.solids():
        ub = u.bounding_box()
        for v in b.solids():
            vb = v.bounding_box()
            if any((ub.max.X < vb.min.X, vb.max.X < ub.min.X,
                    ub.max.Y < vb.min.Y, vb.max.Y < ub.min.Y,
                    ub.max.Z < vb.min.Z, vb.max.Z < ub.min.Z)):
                continue
            total += volume(u & v)
    return total


def clearance(a, b):
    return min(u.distance_to(v) for u in a.solids() for v in b.solids())


def turn(item, deg):
    return bd.Pos(*KNEE) * bd.Rot(float(deg), 0, 0) * \
           bd.Pos(*(-v for v in KNEE)) * item


def paint(scene, name, part, color):
    for index, solid in enumerate(part.solids()):
        if solid.volume < 1e-7:
            continue
        vertices, faces = solid.tessellate(0.55, 0.45)
        mesh = trimesh.Trimesh(
            vertices=np.array([(v.X, v.Y, v.Z) for v in vertices]),
            faces=np.asarray(faces), process=False)
        mesh.visual.vertex_colors = color
        scene.add_geometry(mesh, geom_name=f"{name}_{index}",
                           node_name=f"{name}_{index}")


def write_viewer():
    (OUT / "index.html").write_text("""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Shin-carried knee coupon</title>
<style>body{margin:0;background:#e9e7e0;color:#23332f;font:15px/1.5 system-ui}header{padding:18px 3vw;background:#253b35;color:white}h1{font-size:24px;margin:0}main{display:grid;grid-template-columns:2fr 1fr;gap:16px;padding:16px;max-width:1400px;margin:auto}#view{height:72vh;min-height:480px;background:#dcd9d0;border-radius:12px}.panel{background:#faf8f2;border-radius:12px;padding:19px}button{padding:10px 15px;margin:4px;border:0;border-radius:7px;background:#cdd8d2;cursor:pointer}button.on{background:#29584d;color:white}.warn{border-left:4px solid #b46137;padding-left:10px}a{color:#1d6666}@media(max-width:850px){main{display:block}#view{height:55vh;min-height:350px}}</style>
<script type="importmap">{"imports":{"three":"https://cdn.jsdelivr.net/npm/three@0.171.0/build/three.module.js","three/addons/":"https://cdn.jsdelivr.net/npm/three@0.171.0/examples/jsm/"}}</script></head><body>
<header><h1>Shin-carried ape knee · 350 mm shell</h1>Engineering clearance coupon for a distal servo and diagonal knee belt.</header>
<main><div id="view"></div><div class="panel"><h2>Flexion</h2><div id="buttons"></div><p id="pose"></p><p>Ivory: source thigh, shin, ankle and boot. Olive: retained front cuff and structural fork. Blue: Feetech-size case. Orange: belt pulleys and shin rails.</p><p class="warn"><b>Appearance rejected.</b> The external belt tube and rear cuff dominate the sketch silhouette; a flexible cover remains unresolved. The belt, spline adapter and bearings are not strength qualified.</p><p><a href="report.json">Geometry and ratings</a> · <a href="fixed.step">Fixed STEP</a> · <a href="moving.step">Moving STEP</a></p></div></main>
<script type="module">import * as T from 'three';import {OrbitControls} from 'three/addons/controls/OrbitControls.js';import {GLTFLoader} from 'three/addons/loaders/GLTFLoader.js';
const el=document.querySelector('#view'),scene=new T.Scene();scene.background=new T.Color(0xdcd9d0);scene.add(new T.HemisphereLight(0xffffff,0x9e9f93,2));const light=new T.DirectionalLight(0xffffff,2.4);light.position.set(150,-150,240);scene.add(light);const renderer=new T.WebGLRenderer({antialias:true});renderer.setPixelRatio(Math.min(devicePixelRatio,2));renderer.setSize(el.clientWidth,el.clientHeight);el.append(renderer.domElement);const camera=new T.PerspectiveCamera(34,el.clientWidth/el.clientHeight,.1,1000);camera.up.set(0,0,1);camera.position.set(180,-185,145);const control=new OrbitControls(camera,renderer.domElement);control.target.set(43,3,84);control.enableDamping=true;control.update();const loader=new GLTFLoader(),cache=new Map();let current;async function pose(d){let model=cache.get(d);if(!model){model=(await loader.loadAsync(`pose-${String(d).padStart(3,'0')}.glb`)).scene;model.traverse(o=>{if(o.isMesh){o.material.roughness=.82;o.material.side=T.DoubleSide}});cache.set(d,model)}if(current)scene.remove(current);current=model;scene.add(model);document.querySelector('#pose').textContent=`${d}° knee flexion`;document.querySelectorAll('button').forEach(b=>b.classList.toggle('on',+b.dataset.deg===d))}const buttons=document.querySelector('#buttons');for(const d of [0,60,120]){const b=document.createElement('button');b.textContent=d+'°';b.dataset.deg=d;b.onclick=()=>pose(d);buttons.append(b)}addEventListener('resize',()=>{renderer.setSize(el.clientWidth,el.clientHeight);camera.aspect=el.clientWidth/el.clientHeight;camera.updateProjectionMatrix()});function loop(){requestAnimationFrame(loop);control.update();renderer.render(scene,camera)}loop();await pose(0);</script></body></html>""")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    p = {q.label: q for q in bd.import_step(SOURCE).children}
    thigh = comp([p[f"R_thigh_{n}"] for n in ("front", "back")])
    source_shin = comp([p[f"R_shin_{n}"] for n in ("front", "back")])
    # The upper source shin interferes with the thigh in flexion. Keep the
    # source outside profile below this elevation; the belt side housing
    # reaches the knee structurally.
    shin = comp([s & clip(-30, 120, -70, 70, -10, 74)
                 for s in source_shin.solids()])
    ankle_boot = comp([p[f"R_{n}"] for n in
                       ("ankle_cuff", "foot_dorsal", "toe_1", "toe_2",
                        "toe_3", "sole")])
    front_cuff = p["R_knee_cuff"] & clip(-30, 120, -70, 3, 78, 100)

    # Official 45.22 x 24.72 x 35 mm envelope, arranged vertically in the
    # lower calf. Output axis is about 11.6 mm from one long-body end and
    # 15.8 mm from a depth edge in the linked vendor CAD. We use its entire
    # box, not any stall-torque figure, for clearance.
    case = box((42, DRIVE[1]+6.3, DRIVE[2]+1.4), (24.72, 37.4, 45.22))
    service = box((42, DRIVE[1]+6.3, DRIVE[2]+1.4), (26.72, 39.4, 47.22))
    relieved_shin = comp([s - service for s in shin.solids()])
    # A rounded two-millimetre cover replaces the opened calf shell. It is
    # deliberately larger than the rated body rather than treating the
    # servo as an exterior panel.
    cover_center = (42, DRIVE[1]+6.3, DRIVE[2]+1.4)
    outer = bd.fillet(box(cover_center, (30.72, 43.4, 51.22)).edges(), radius=5)
    inner = bd.fillet(box(cover_center, (26.72, 39.4, 47.22)).edges(), radius=3)
    calf_cover = outer - inner

    # 16-tooth, 3 mm pitch pulleys. Pitch length 123 mm gives a 37.5 mm
    # nominal center distance with equal pulleys. Cylinders represent pitch
    # and keep the belt visible; teeth, flanges and tensioner are omitted.
    drive_pulley = cyl_x(PITCH_RADIUS, 57.5, 63.5, DRIVE)
    fixed_knee_pulley = cyl_x(PITCH_RADIUS, 57.5, 63.5, KNEE)
    normal_y = -(KNEE[2]-DRIVE[2])/37.5
    normal_z = (KNEE[1]-DRIVE[1])/37.5
    belt = comp([rod((60.5, DRIVE[1]+sgn*PITCH_RADIUS*normal_y,
                       DRIVE[2]+sgn*PITCH_RADIUS*normal_z),
                     (60.5, KNEE[1]+sgn*PITCH_RADIUS*normal_y,
                       KNEE[2]+sgn*PITCH_RADIUS*normal_z), 1.2)
                 for sgn in (-1, 1)])
    belt_fairing = (rod((60.5, DRIVE[1], DRIVE[2]),
                        (60.5, KNEE[1], KNEE[2]), 10.5) -
                    rod((60.5, DRIVE[1], DRIVE[2]),
                        (60.5, KNEE[1], KNEE[2]), 8.5)) & \
                   clip(55.5, 80, -40, 60, 35, 100)

    # Moving shin frame straddles the case. It runs from the boot/ankle
    # structure up to both knee bearing positions. Fixed thigh cheeks run
    # from the knee axle to broad thigh-wall anchor points.
    moving_rails = comp([
        rod((27, 3, 32), (27, 25, 85), 1.8),
        rod((55, 3, 32), (55, 25, 85), 1.8),
        rod((27, 3, 32), (55, 3, 32), 2.2),
        rod((27, 12.5, 55), (31, 12.5, 55), 2.0),
        rod((55, 12.5, 55), (53, 12.5, 55), 2.0),
        rod((27, 3, 32), (30, 3, 24), 2.6),
        rod((55, 3, 32), (62, 3, 24), 2.6),
        rod((30, 3, 24), (30, -6, 5), 2.6),
        rod((62, 3, 24), (62, -6, 5), 2.6),
        rod((30, -6, 5), (62, -6, 5), 3.0),
        cyl_x(6.5, 24, 29, KNEE) - cyl_x(3.2, 24, 29, KNEE),
        cyl_x(6.5, 53, 58, KNEE) - cyl_x(3.2, 53, 58, KNEE),
    ])
    fixed_axle = cyl_x(3, 20, 70, KNEE)
    fork = comp([
        rod((23, 25, 86), (30, 3, 111), 3.0),
        rod((67, 25, 86), (58, 3, 111), 3.0),
        cyl_x(7.0, 20, 24, KNEE),
        cyl_x(7.0, 64, 70, KNEE),
    ])
    # Only the front of the source cuff stays fixed. The rear flexible
    # gaiter is not drawn because its fold path has not been established.
    # A rear lobe ties the shifted pivot visually into the source thigh.
    # It stays above the axis and is open below for the rotating shank.
    hood = (ellipsoid((42, 18, 92), (22, 18, 14)) -
            ellipsoid((42, 18, 92), (19.8, 15.8, 11.8))) & \
           clip(-30, 120, 4, 29, 86, 115)
    hood = hood - cyl_x(15, 18, 32, KNEE) - cyl_x(15, 52, 72, KNEE)
    fixed = comp([thigh, front_cuff, hood, fork, fixed_axle,
                  fixed_knee_pulley])
    moving = comp([relieved_shin, ankle_boot, calf_cover, belt_fairing,
                   case, drive_pulley, belt, moving_rails])

    report = {
        "verdict": "Engineering clearance clue only; reject for current cute covered sketch appearance.",
        "concept": "Servo in lower calf, diagonal equal-pulley belt to a rearward knee axle",
        "source_step": str(SOURCE.relative_to(ROOT)),
        "source_height_mm": 350,
        "knee_axis_mm": KNEE, "servo_output_axis_mm": DRIVE,
        "source_shin_trim_top_z_mm": 74,
        "source_shin_service_relief_volume_mm3": round(volume(shin)-volume(relieved_shin), 2),
        "case_vs_source_shin_shell_mm3": round(overlap(case, shin), 2),
        "case_vs_relieved_shin_shell_mm3": round(overlap(case, relieved_shin), 2),
        "calf_cover_vs_source_shin_mm3": round(overlap(calf_cover, shin), 2),
        "rail_anchor_to_source_shin_overlap_mm3": round(overlap(moving_rails, relieved_shin), 2),
        "rail_anchor_to_ankle_boot_overlap_mm3": round(overlap(moving_rails, ankle_boot), 2),
        "fork_anchor_to_source_thigh_overlap_mm3": round(overlap(fork, thigh), 2),
        "actuator": {
            "model": "Feetech STS3250 / ST-3250-C001",
            "manufacturer_datasheet": "https://www.gotronic.fr/pj2-38757-3355.pdf",
            "manufacturer_product_page": "https://www.feetechrc.com/en/562636.html",
            "rated_torque_kg_cm_at_12v": 16,
            "rated_torque_Nm_at_12v": round(16*0.0980665, 4),
            "rated_current_A": 1.4,
            "mass_g": 74.5,
            "actuator_only_knee_pitch_inertia_kg_m2_approx": round(
                0.0745 * (((DRIVE[1]+6.3-KNEE[1])/1000)**2 +
                          ((DRIVE[2]+1.4-KNEE[2])/1000)**2 +
                          ((37.4/1000)**2+(45.22/1000)**2)/12), 8),
            "inertia_note": "Uniform-box approximation for the 74.5 g servo only. Added cover, rails, belt, pulleys, ankle and boot inertia are excluded.",
            "peak_stall_kg_cm_at_12v_for_reference_only": 50,
            "official_envelope_mm": [45.22, 24.72, 35],
            "body_with_side_features_depth_mm": 37.4,
            "body_axis_location_note": "Output-axis offset and 37.4-mm feature span came from a reseller CAD download whose assembly label says ST-3235M, although the listing was for STS3250. Confirm against a purchased STS3250 before design use.",
            "nominal_drive_angle_range_deg": 360,
            "load_limit_note": "The 1.569 Nm rating is at the actuator. A 1:1 belt has no ideal torque margin for a 1.569 Nm knee target, and any belt loss lowers knee torque. Published duty/thermal assumptions need verification."
        },
        "belt": {
            "candidate": "Gates PowerGrip GT3 123-3MGT-06",
            "source": "https://www.gates.com/content/dam/documents-library/catalogs/power-transmission-catalog.pdf",
            "nominal_pitch_length_mm": 123,
            "pitch_mm": 3, "width_mm": 6,
            "each_pulley_teeth": 16,
            "pitch_radius_mm": round(PITCH_RADIUS, 4),
            "center_distance_mm": 37.5,
            "tension_difference_at_rated_servo_torque_N": round(16*0.0980665/(PITCH_RADIUS/1000), 1),
            "rating_note": "No belt tooth/cord or pulley-hub torque rating is established here. 123-mm belt availability is catalogued, capacity must be checked for shock and reversed knee duty."
        },
        "load_path": "Boot and shin shell attach to twin shin rails; rails reach two bearing rings around the fixed knee axle; axle and pulley attach through two thigh fork cheeks to the thigh frame. Servo case bolts to shin rails; its output spline drives the lower pulley, belt drives the fixed knee pulley. Axle/bearings carry foot bending and shear; the belt provides pitch torque.",
        "coupling_limit": "Spline adapter, tensioner, bearing size/retention, screw pattern, cable routing, shell mounting bosses and hip/other-leg clearance remain undrawn or unqualified.",
        "rear_cover_limit": "The 0/60/120 viewer shows an external long belt tube and dominant rear cuff that do not match the cute covered sketch. A flexible gaiter tied to both shells remains unresolved.",
        "motion_sample_step_deg": 10,
        "motion_limit": "10-degree B-rep samples do not prove continuous clearance or dynamic gaiter folding.",
        "samples": []
    }
    for deg in range(0, 121, 10):
        posed_skin = turn(comp([relieved_shin, ankle_boot]), deg)
        posed_case = turn(case, deg)
        posed_frame = turn(comp([moving_rails, drive_pulley, belt]), deg)
        posed_cover = turn(comp([calf_cover, belt_fairing]), deg)
        report["samples"].append({
            "flex_deg": deg,
            "skin_vs_original_thigh_mm3": round(overlap(posed_skin, thigh), 3),
            "case_vs_original_thigh_mm3": round(overlap(posed_case, thigh), 3),
            "frame_and_belt_vs_original_thigh_mm3": round(overlap(posed_frame, thigh), 3),
            "skin_vs_rear_hood_mm3": round(overlap(posed_skin, hood), 3),
            "case_vs_rear_hood_mm3": round(overlap(posed_case, hood), 3),
            "frame_and_belt_vs_rear_hood_mm3": round(overlap(posed_frame, hood), 3),
            "rounded_covers_vs_original_thigh_mm3": round(overlap(posed_cover, thigh), 3),
            "rounded_covers_vs_rear_hood_mm3": round(overlap(posed_cover, hood), 3),
            "boot_lowest_z_mm": round(turn(ankle_boot, deg).bounding_box().min.Z, 2),
        })
        if deg not in (0, 60, 120):
            continue
        report["samples"][-1]["hard_part_clearance_mm"] = {
            "case_to_source_thigh": round(clearance(posed_case, thigh), 3),
            "covers_to_source_thigh": round(clearance(posed_cover, thigh), 3),
            "case_to_rear_hood": round(clearance(posed_case, hood), 3),
        }
        scene = trimesh.Scene()
        for name, part, color in (
            ("source_thigh", thigh, (238, 230, 216, 255)),
            ("front_source_knee_cuff", front_cuff, (81, 89, 76, 255)),
            ("rounded_rear_hood", hood, (81, 89, 76, 255)),
            ("fixed_thigh_fork", fork, (85, 101, 91, 255)),
            ("fixed_axle", fixed_axle, (104, 109, 106, 255)),
            ("fixed_knee_pulley", fixed_knee_pulley, (201, 126, 75, 255)),
            ("source_shin_relieved", turn(relieved_shin, deg), (239, 231, 216, 255)),
            ("source_ankle_boot", turn(ankle_boot, deg), (235, 225, 205, 255)),
            ("rounded_moving_calf_cover", turn(calf_cover, deg), (235, 225, 205, 255)),
            ("rounded_belt_fairing", turn(belt_fairing, deg), (113, 119, 103, 255)),
            ("rated_servo_envelope", posed_case, (63, 112, 138, 255)),
            ("moving_rails", turn(moving_rails, deg), (127, 116, 96, 255)),
            ("drive_pulley", turn(drive_pulley, deg), (202, 108, 57, 255)),
            ("belt_runs", turn(belt, deg), (70, 142, 132, 255)),
        ):
            paint(scene, name, part, color)
        scene.export(OUT / f"pose-{deg:03d}.glb")
    bd.export_step(fixed, OUT / "fixed.step")
    bd.export_step(moving, OUT / "moving.step")
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    write_viewer()
    print(json.dumps({"samples": report["samples"],
                      "case_vs_source_shin_shell_mm3": report["case_vs_source_shin_shell_mm3"],
                      "source_shin_service_relief_volume_mm3": report["source_shin_service_relief_volume_mm3"]}, indent=2))


if __name__ == "__main__":
    main()
