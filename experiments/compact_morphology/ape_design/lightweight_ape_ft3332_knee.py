"""Source-shell 350 mm FT3332 short-cable knee coupon (millimetres).

This is a deliberately falsifiable packaging model, not released hardware. Run
with .venv/bin/python. X is the knee axis, Y points rearward, Z points up.
The vendor case, shaft offset and projecting spline follow the Feetech drawing;
the capstan adapter, bearing sizes, cable terminations, connector bend and
flexible rear closure remain design allocations.
"""
from __future__ import annotations

import base64
import json
import math
from pathlib import Path

import build123d as bd
import numpy as np
import trimesh


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "artifacts/lightweight-ape-20261004/source-shell-1mm/ape_shell_assembly.step"
OUT = ROOT / "artifacts/lightweight-ape-ft3332-knee-20261005"
SHELL_BUILDER = ROOT / "experiments/sketch_ape_shell/build.py"
VENDOR = "https://www.feetechrc.com/538886"
SHEET = "https://www.feetechrc.com/Data/feetechrc/upload/file/20260623/6391782192726766615980211.pdf"
KNEE = (42.0, 3.0, 99.0)
COLORS = {
    "skin": "#ded7c5", "frame": "#656a70", "motor": "#9d9e9b",
    "drive": "#bf8655", "cable": "#bb4a38", "bearing": "#425269",
    "rear": "#444849", "shin": "#d9d1bd",
}


def box(x0, x1, y0, y1, z0, z1):
    return bd.Solid.make_box(x1-x0, y1-y0, z1-z0,
                             bd.Plane(origin=(x0, y0, z0)))


def cyl_x(x0, x1, y, z, radius):
    return bd.Solid.make_cylinder(radius, x1-x0,
                                  bd.Plane(origin=(x0, y, z), z_dir=(1, 0, 0)))


def ring_x(x0, x1, y, z, outer, inner):
    return cyl_x(x0, x1, y, z, outer) - cyl_x(x0-.1, x1+.1, y, z, inner)


def loft(stations):
    return bd.Solid.make_loft([
        bd.Wire.make_ellipse(rx, ry, bd.Plane(origin=(x, y, z)))
        for z, x, y, rx, ry in stations])


def rail(stations):
    wires = []
    for z, x0, x1 in stations:
        wires.append(bd.Wire.make_polygon([
            (x0,-12.5,z),(x1,-12.5,z),(x1,-4.5,z),(x0,-4.5,z)
        ], close=True))
    return bd.Solid.make_loft(wires)


def volume(shape):
    return sum(s.volume for s in shape.solids())


def part(name, shape, kind, moving=False):
    if volume(shape) <= 0 or not shape.is_valid:
        raise ValueError(f"invalid {name}: {len(shape.solids())} solids, {volume(shape)} mm3")
    return dict(name=name, shape=shape, kind=kind, color=COLORS[kind], moving=moving)


def rotate(shape, deg):
    return shape.rotate(bd.Axis(KNEE, (1,0,0)), deg)


def bend_box_path(points, radius=.5):
    rods = []
    for a,b in zip(points[:-1],points[1:]):
        d = bd.Vector(b)-bd.Vector(a)
        rods.append(bd.Solid.make_cylinder(radius,d.length,
                    bd.Plane(origin=a,z_dir=d.normalized())))
    # Keep cable runs separately named and visibly connected at bends.
    return rods[0].fuse(*rods[1:])


def rear_bridge(angle):
    """An explicitly provisional, pose-deformed 1 mm rear sheet.

    The fold is routed behind the knee; this is a volume/appearance attempt,
    not a developed bellows pattern or proof it can follow intermediate poses.
    """
    a = np.array([20.0,111.0]) # fixed thigh rear lip, Y/Z
    q = rotate(bd.Vertex(42,18,88), angle)
    b = np.array([q.Y,q.Z])
    mid = (a+b)/2
    # A standing fold, then increasing posterior deployment during flexion.
    mid[0] += 6 + 11*math.sin(math.radians(angle/2))
    p = [a,mid,b]
    yz = [(float(y),float(z)) for y,z in p]
    yz += [(float(y-1),float(z)) for y,z in reversed(p)]
    w = bd.Wire.make_polygon([(34,y,z) for y,z in yz],close=True)
    return bd.Solid.extrude(bd.Face(w),(29,0,0))


def construct():
    if not SOURCE.exists():
        raise FileNotFoundError(f"Build exact source 1 mm shell first: {SOURCE}")
    source = {p.label:p for p in bd.import_step(SOURCE).children}
    if not all(name in source for name in ("R_thigh_front","R_thigh_back","R_shin_front","R_shin_back")):
        raise RuntimeError("Exact right-leg parts missing from source STEP")

    # The original stations remain unchanged at the top and ankle. Between
    # Z=104 and 145 the thigh receives a local cheek to make room for the
    # case/fork. This is the precise departure from the 1 mm source exterior.
    original = [(87,42,3,16,18),(98,43,3,20,22),(124,39,3,24,25),
                (145,35,4,22,23),(157,33,4,16,18),(163,31,4,8,10)]
    relieved = [(87,42,3,16,18),(98,43,3,21,22),(104,42,3,23,23),
                (112,41,3,27,24),(124,39,3,30,26),(138,36,3.6,29,25),
                (145,35,4,25,23),(157,33,4,16,18),(163,31,4,8,10)]
    thigh_original = loft(original) - loft([(z,x,y,rx-1,ry-1) for z,x,y,rx,ry in original])
    thigh_relief = loft(relieved) - loft([(z,x,y,rx-1,ry-1) for z,x,y,rx,ry in relieved])
    rear_cut = box(24,73,13.5,43,85,112)
    thigh_skin = thigh_relief - rear_cut

    # Vendor envelope: 34 x 15 x 36.5 mm, output X+, 8.55 mm from
    # one end of the 34 mm side. The 50 mm tab span and 42.6 mm total X span
    # also appear in the manufacturer's annotated outline.
    mx0,mx1 = 18.0,54.5
    my0,my1 = -4.5,10.5
    mz0,mz1 = 108.0,142.0
    output_z = mz1-8.55
    motor_case = box(mx0,mx1,my0,my1,mz0,mz1)
    motor_tabs = box(40,43,my0,my1,100,108.2).fuse(
                 box(40,43,my0,my1,141.8,150))
    motor_spline = cyl_x(mx1,60.6,3,output_z,2.975)
    # An unverified bend/plug envelope: translucent in the viewer and
    # deliberately included in the containment report.
    plug = box(12.5,18.1,-2,8,111,118)

    # Two opposed knee supports, linked continuously up the thigh's front
    # wall to a bridge above the servo. Bearings are generic 6x10x4 allocation.
    inner = ring_x(30.5,35.0,3,99,8.0,5.0)
    outer = ring_x(65.0,69.5,3,99,8.0,5.0)
    inner_rail = rail([(98.5,30.5,35),(111,25,29.5),
                       (124,19.5,24),(146,18.5,23)])
    outer_rail = rail([(98.5,65,69.5),(111,60.5,65),
                       (124,56,60.5),(146,51.5,56)])
    top_bridge = box(18.5,56,-12.5,-4.5,143.5,148)
    fork = inner.fuse(outer,inner_rail,outer_rail,top_bridge)
    # Generic bearing ring proxies are mechanical allocations, not a BOM.
    bearings = [ring_x(31,35,3,99,5,3.05),
                ring_x(65,69,3,99,5,3.05)]
    axle = cyl_x(30.7,69.3,3,99,3.0)
    drum = cyl_x(57.2,63.6,3,99,13.0)
    capstan = cyl_x(60.6,65.0,3,output_z,13.0)
    # Explicit dual cable strands: 25 mm pitch diameter with straight
    # tangents. End wraps, anchor bores and service tensioner are unresolved.
    cables = [bend_box_path([(62.0,3+sign*12.5,output_z),
                             (62.0,3+sign*12.5,99)],.42)
              for sign in (-1,1)]

    # Exact source calf exterior, locally shortened. A cap rib fuses its two
    # halves, and two moving spars connect it to the axle/root clamp.
    shin_a = source["R_shin_front"] & box(10,82,-60,60,20,75)
    shin_b = source["R_shin_back"] & box(10,82,-60,60,20,75)
    shin_rib = (loft([(70,44,3,21.5,22),(72,44,3,21.5,22)])
                - loft([(69.9,44,3,14.5,15.5),(72.1,44,3,14.5,15.5)]))
    root = cyl_x(37,51,3,99,7.5)
    root = root.fuse(box(37,40,-1,7,69,100),
                     box(48,51,-1,7,69,100),
                     box(24,65,-9,14,69,72.2))
    # Keep source shell and structural moving root independently inspectable:
    # the upper cap joins both skin halves but the structural attachment is
    # still only an assumed clamp/rib, not a strength demonstration.
    shin = shin_a.fuse(shin_b,shin_rib)
    # Thin moving front lip and rigid fixed side caps deliberately expose the
    # cost of covering the widened bearing/drum stack.
    front_lip = (ring_x(35.5,56,3,99,15.4,14.3)
                 & box(35,57,-20,3,88.5,120))
    side_caps = [cyl_x(28.5,30.4,3,99,15.3),
                 cyl_x(69.6,71.5,3,99,15.3)]
    fixed = [
        part("relieved_thigh_skin",thigh_skin,"skin"),
        part("continuous_double_bearing_fork",fork,"frame"),
        part("FT3332_case",motor_case,"motor"),
        part("FT3332_mounting_tabs",motor_tabs,"motor"),
        part("FT3332_spline",motor_spline,"drive"),
        part("connector_bend_envelope_assumed",plug,"motor"),
        part("motor_capstan_adapter_assumed",capstan,"drive"),
        part("cable_front_assumed",cables[0],"cable"),
        part("cable_rear_assumed",cables[1],"cable"),
        part("bearing_inboard_allocated",bearings[0],"bearing"),
        part("bearing_outboard_allocated",bearings[1],"bearing"),
        part("side_cap_inboard",side_caps[0],"skin"),
        part("side_cap_outboard",side_caps[1],"skin"),
    ]
    moving = [
        part("source_shin_trimmed",shin,"shin",True),
        part("shin_root_clamp_assumed",root,"frame",True),
        part("knee_axle_allocated",axle,"bearing",True),
        part("knee_drum_assumed",drum,"drive",True),
        part("moving_front_lip",front_lip,"skin",True),
    ]
    references = dict(source_thigh_wall_mm=1, source_thigh_volume_mm3=round(volume(thigh_original),2),
                      relieved_thigh_volume_mm3=round(volume(thigh_relief),2),
                      rear_cut_removed_mm3=round(volume(thigh_relief & rear_cut),2),
                      output_axis_point_mm=[mx1,3,round(output_z,2)],
                      output_axis_direction=[1,0,0],
                      motor_case_bounds_mm=[[mx0,my0,mz0],[mx1,my1,mz1]])
    return fixed,moving,rear_bridge,references,thigh_original,thigh_relief


def mesh_for(shape):
    verts,faces = shape.tessellate(.35,.22)
    return trimesh.Trimesh(vertices=np.asarray([tuple(v) for v in verts]),
                           faces=np.asarray(faces),process=False)


def export_pose(angle,fixed,moving,bridge):
    pose_dir=OUT/f"pose-{angle:03d}"
    pose_dir.mkdir(parents=True,exist_ok=True)
    pieces=fixed+[dict(p,shape=rotate(p["shape"],angle)) for p in moving]
    pieces.append(part("rear_fold_pose_proxy",bridge(angle),"rear"))
    for p in pieces:
        p["shape"].label=p["name"]
        p["shape"].color=bd.Color(p["color"])
    bd.export_step(bd.Compound(children=[p["shape"] for p in pieces]),
                   pose_dir/"ft3332_knee.step")
    scene=trimesh.Scene()
    viewer=[]
    for p in pieces:
        mesh=mesh_for(p["shape"])
        rgb=tuple(int(p["color"][i:i+2],16) for i in (1,3,5))
        mesh.visual.vertex_colors=np.tile(np.array([*rgb,255],dtype=np.uint8),(len(mesh.vertices),1))
        scene.add_geometry(mesh,node_name=p["name"],geom_name=p["name"])
        viewer.append(dict(name=p["name"],kind=p["kind"],color=p["color"],
                           v=np.round(mesh.vertices,3).flatten().tolist(),
                           f=mesh.faces.flatten().tolist()))
    (pose_dir/"ft3332_knee.glb").write_bytes(scene.export(file_type="glb"))
    return pieces,viewer


def overlap(a,b):
    try:
        intersection=a & b
        return round(volume(intersection),3) if intersection else 0.0
    except Exception:
        return None


def measures(angle,fixed,moving,bridge,source_original,source_relief):
    f={p["name"]:p["shape"] for p in fixed}
    m={p["name"]:rotate(p["shape"],angle) for p in moving}
    skin=f["relieved_thigh_skin"]
    shin=m["source_shin_trimmed"]
    root=m["shin_root_clamp_assumed"]
    fold=bridge(angle)
    cap=m["moving_front_lip"]
    fixed_cover=f["side_cap_outboard"]
    # These are sampled B-rep volume tests. Coaxial axle/bearing and the
    # capstan/cable mates are deliberately excluded as designed interfaces.
    checks={
        "thigh_skin__shin_skin_mm3":overlap(skin,shin),
        "thigh_skin__root_mm3":overlap(skin,root),
        "thigh_skin__moving_lip_mm3":overlap(skin,cap),
        "fixed_outboard_cap__shin_mm3":overlap(fixed_cover,shin),
        "rear_fold__shin_mm3":overlap(fold,shin),
        "rear_fold__root_mm3":overlap(fold,root),
        "rear_fold__motor_mm3":overlap(fold,f["FT3332_case"]),
        "rear_fold__fork_mm3":overlap(fold,f["continuous_double_bearing_fork"]),
    }
    original_source_shin_gap=round(source_original.distance_to(shin),3)
    original_rear_y=max(float(p[1]) for p in mesh_for(source_relief).vertices if 86 < p[2] < 114)
    fold_rear_y=float(fold.bounding_box().max.Y)
    return dict(angle_deg=angle,checks=checks,
                original_thigh_to_moving_shin_gap_mm=original_source_shin_gap,
                fold_rear_y_mm=round(fold_rear_y,2),
                fold_protrusion_beyond_relief_rear_mm=round(fold_rear_y-original_rear_y,2),
                any_measured_unintended_overlap=any(v is None or v>.01 for v in checks.values()))


def viewer_html(meshes,rows):
    vendor=ROOT/"experiments/sketch_ape_shell/vendor"
    def module(s):
        return "data:text/javascript;base64,"+base64.b64encode(s.encode()).decode()
    core=module((vendor/"three.core.js").read_text())
    three=module((vendor/"three.module.js").read_text().replace("./three.core.js","three-core"))
    orbit=module((vendor/"OrbitControls.js").read_text())
    payload=json.dumps(meshes,separators=(",",":"))
    options="".join(f'<option value="{a}">{a}°</option>' for a in (0,60,120))
    return f'''<!doctype html><html><head><meta charset="utf-8"><title>FT3332 ape knee coupon</title>
<style>body{{margin:0;background:#1d2429;color:#e9e5da;font:14px system-ui}}#bar{{position:absolute;z-index:2;top:12px;left:12px;background:#172026dd;padding:13px;border-radius:10px;max-width:320px}}select,button{{background:#33434b;color:white;border:1px solid #64767d;padding:5px}}#v{{width:100vw;height:100vh}}small{{display:block;line-height:1.4;max-width:300px;margin-top:8px}}</style>
<script type="importmap">{{"imports":{{"three-core":"{core}","three":"{three}","orbit":"{orbit}"}}}}</script></head>
<body><div id="bar"><strong>350 mm source-shell · FT3332 short-cable knee</strong><p>Pose <select id="pose">{options}</select> <button id="front">Front</button> <button id="side">Side</button> <button id="rear">Rear</button></p><small>Grey is the connected fork; copper is the assumed 1:1 cable drive. Dark rear fold is a per-pose closure attempt. Shaft adaptor, bearing duty and connector bend remain unverified.</small></div><div id="v"></div>
<script id="data" type="application/json">{payload}</script><script type="module">
import * as T from 'three'; import {{OrbitControls}} from 'orbit';
const data=JSON.parse(document.getElementById('data').textContent);const root=document.getElementById('v');
const renderer=new T.WebGLRenderer({{antialias:true}});renderer.setSize(innerWidth,innerHeight);renderer.setPixelRatio(devicePixelRatio);root.appendChild(renderer.domElement);
const scene=new T.Scene();scene.background=new T.Color('#1d2429');scene.add(new T.HemisphereLight(0xffffff,0x53606a,2.2));const light=new T.DirectionalLight(0xffffff,2);light.position.set(4,-5,8);scene.add(light);
const camera=new T.PerspectiveCamera(35,innerWidth/innerHeight,.1,2000);const orbit=new OrbitControls(camera,renderer.domElement);orbit.target.set(43,3,104);camera.position.set(180,-210,142);orbit.update();let group;
function show(){{if(group)scene.remove(group);group=new T.Group();for(const p of data[document.getElementById('pose').value]){{const g=new T.BufferGeometry();g.setAttribute('position',new T.Float32BufferAttribute(p.v,3));g.setIndex(p.f);g.computeVertexNormals();const m=new T.Mesh(g,new T.MeshStandardMaterial({{color:p.color,side:T.DoubleSide,metalness:p.kind==='motor'?.35:0,roughness:.75}}));m.name=p.name;group.add(m)}}scene.add(group)}}
document.getElementById('pose').onchange=show;document.getElementById('front').onclick=()=>{{camera.position.set(43,-220,104);orbit.update()}};document.getElementById('side').onclick=()=>{{camera.position.set(250,3,104);orbit.update()}};document.getElementById('rear').onclick=()=>{{camera.position.set(43,220,104);orbit.update()}};
addEventListener('resize',()=>{{camera.aspect=innerWidth/innerHeight;camera.updateProjectionMatrix();renderer.setSize(innerWidth,innerHeight)}});show();function animate(){{requestAnimationFrame(animate);orbit.update();renderer.render(scene,camera)}}animate();
</script></body></html>'''


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    fixed,moving,bridge,refs,source_original,source_relief=construct()
    f={p["name"]:p["shape"] for p in fixed}
    v={p["name"]:p["shape"] for p in moving}
    # A bounded diagnostic only: carrying the rear window 30 mm higher makes
    # room for the bent hard shin, but deletes a conspicuous stretch of skin.
    # It is not applied to the exported pose geometry.
    extra_rear_cut=box(22,72,18,45,112,142)
    current_skin=f["relieved_thigh_skin"]
    rescued_skin=current_skin-extra_rear_cut
    trial_angles=(0,60,90,120)
    rescue_trial={
        "additional_cut_bounds_mm":[[22,18,112],[72,45,142]],
        "additional_skin_removed_mm3":round(overlap(current_skin,extra_rear_cut),3),
        "fraction_of_current_skin_removed":round(overlap(current_skin,extra_rear_cut)/volume(current_skin),4),
        "sampled_shin_overlap_mm3":{str(a):overlap(rescued_skin,rotate(v["source_shin_trimmed"],a)) for a in trial_angles},
        "sampled_root_overlap_mm3":{str(a):overlap(rescued_skin,rotate(v["shin_root_clamp_assumed"],a)) for a in trial_angles},
        "status":"diagnostic rejected; large uncovered rear window and colliding fold remain",
    }
    pose_meshes={}
    rows=[]
    for angle in range(0,121,10):
        rows.append(measures(angle,fixed,moving,bridge,source_original,source_relief))
        if angle in (0,60,120):
            _,pose_meshes[str(angle)]=export_pose(angle,fixed,moving,bridge)
            print(f"exported {angle}°",flush=True)
    report={
        "status":"coupon only; no architecture selection",
        "source_shell":str(SOURCE.relative_to(ROOT)),
        "source_shell_builder":str(SHELL_BUILDER.relative_to(ROOT)),
        "manufacturer_product":VENDOR,"manufacturer_specification":SHEET,
        "rated_motor_torque_nm_at_8p4v":round(13.07*0.0980665,3),
        "servo_mass_g":51.5,"case_xyz_mm":[36.5,15,34],
        "output_offset_from_case_end_mm":8.55,
        "motor_output_axis_z_mm":refs["output_axis_point_mm"][2],
        "knee_axis_mm":KNEE,"nominal_pitch_radius_mm":12.5,
        "assumptions":["6 mm axle and 6x10x4 mm bearing allocation", "13 mm radius capstan/drum solids and 12.5 mm pitch radii", "shaft adapter and cable anchors/termination omitted", "connector bend is an unverified box", "pose-specific rear flexible sheet is not a developed manufacturable bellows", "cable tension, efficiency, fatigue, backlash, sensing, power-off hold and duty are unmeasured"],
        "geometry":refs,
        "source_knee_cuff_width_mm":30,
        "relieved_thigh_width_at_z124_mm":60,
        "side_cap_axial_stack_width_mm":43,
        "fixed_part_interferences_mm3":{
            "motor_case__thigh_skin":overlap(f["FT3332_case"],current_skin),
            "motor_case__fork":overlap(f["FT3332_case"],f["continuous_double_bearing_fork"]),
            "motor_capstan__thigh_skin":overlap(f["motor_capstan_adapter_assumed"],current_skin),
            "fork__thigh_skin":overlap(f["continuous_double_bearing_fork"],current_skin),
        },
        "rear_cut_trial":rescue_trial,
        "fixed_part_count":len(fixed),"moving_part_count":len(moving),
        "sampling_deg":10,"poses":rows,
        "rear_fold_max_protrusion_mm":max(r["fold_protrusion_beyond_relief_rear_mm"] for r in rows),
        "all_sampled_clear":not any(r["any_measured_unintended_overlap"] for r in rows),
    }
    (OUT/"metrics.json").write_text(json.dumps(report,indent=2)+"\n")
    (OUT/"index.html").write_text(viewer_html(pose_meshes,rows))
    print(json.dumps({"artifact":str(OUT),"all_sampled_clear":report["all_sampled_clear"],
                      "worst_overlap_mm3":max(v or 0 for r in rows for v in r["checks"].values()),
                      "rear_fold_max_protrusion_mm":report["rear_fold_max_protrusion_mm"]},indent=2))


if __name__=="__main__":
    main()
