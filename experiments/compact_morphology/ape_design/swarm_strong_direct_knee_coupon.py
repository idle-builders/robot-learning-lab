"""STS3250 supported-direct knee fit coupon on the exact 350 mm ape shell.

Run with .venv/bin/python. Coordinates are X lateral, Y rearward, Z upward,
millimetres. This is geometric feasibility work, not a torque-margin claim.
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
OUT = ROOT / "artifacts/swarm-strong-direct-knee-20261004"
AXIS = (42.0, 8.0, 99.0)
ANGLES = (0, 30, 60, 90, 120)
VIEW_ANGLES = (0, 60, 120)


def box(center, size):
    return bd.Pos(*center) * bd.Box(*size)


def clip(x0, x1, y0, y1, z0, z1):
    return bd.Solid.make_box(x1-x0, y1-y0, z1-z0, bd.Plane(origin=(x0,y0,z0)))


def sphere(center, radii):
    return bd.Solid.make_sphere(1).scale(tuple(radii)).translate(center)


def cyl_x(radius, x0, x1, y=AXIS[1], z=AXIS[2]):
    return bd.Solid.make_cylinder(radius, x1-x0,
                                  bd.Plane(origin=(x0,y,z), z_dir=(1,0,0)))


def rod(a,b,radius):
    v=bd.Vector(b)-bd.Vector(a)
    return bd.Solid.make_cylinder(radius,v.length,
                                  bd.Plane(origin=a,z_dir=v.normalized()))


def comp(shapes):
    return bd.Compound(children=[s for q in shapes for s in q.solids()])


def turn(shape, deg):
    return bd.Pos(*AXIS)*bd.Rot(float(deg),0,0)*bd.Pos(*(-v for v in AXIS))*shape


def gap(a,b):
    """Exact BREP distance and material overlap for compound solids."""
    minimum=float("inf")
    volume=0.
    for sa in a.solids():
        for sb in b.solids():
            # An axis-aligned bound is a strict lower bound on BREP distance.
            aa,bb=sa.bounding_box(),sb.bounding_box()
            low=math.sqrt(sum(max(
                getattr(bb.min,c)-getattr(aa.max,c),
                getattr(aa.min,c)-getattr(bb.max,c),0)**2 for c in "XYZ"))
            if low>minimum: continue
            minimum=min(minimum,sa.distance_to(sb))
            if low<1e-8:
                hit=sa&sb
                if hit: volume+=sum(s.volume for s in hit.solids())
    return round(minimum,4),round(volume,4)


def bounds(shape):
    b=shape.bounding_box()
    return [[round(getattr(b.min,c),3) for c in "XYZ"],
            [round(getattr(b.max,c),3) for c in "XYZ"]]


def add(scene,name,shape,rgba):
    for i,solid in enumerate(shape.solids()):
        if solid.volume<1e-5:continue
        vertices,faces=solid.tessellate(.35,.28)
        mesh=trimesh.Trimesh(vertices=np.array([(v.X,v.Y,v.Z) for v in vertices]),
                             faces=np.asarray(faces),process=False)
        mesh.visual.vertex_colors=rgba
        scene.add_geometry(mesh,geom_name=f"{name}_{i}",node_name=f"{name}_{i}")


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    parts={p.label:p for p in bd.import_step(SOURCE).children}
    source_thigh=comp([parts["R_thigh_front"],parts["R_thigh_back"]])
    source_cuff=parts["R_knee_cuff"]
    original_shin=comp([parts["R_shin_front"],parts["R_shin_back"]])
    shin=comp([parts[f"R_shin_{side}"] & clip(-50,140,-80,80,-10,64)
               for side in ("front","back")])
    boot=comp([parts[f"R_{name}"] for name in
               ("ankle_cuff","foot_dorsal","toe_1","toe_2","toe_3","sole")])

    # STS3250 manufacturer size 45.22 x 24.72 x 35 mm. The 35 mm dimension
    # excludes local tabs/cable and is extended to 37.4 in the vendor model;
    # model an intentionally conservative 37.4 mm vertical package.
    # Shaft is at near midlength and 12.4 mm above the bottom of the case.
    case=box((42,8,105.3),(24.72,45.22,37.4))
    service=box((42,8,105.3),(26.72,47.22,39.4))
    relieved_thigh=comp([s-service for s in source_thigh.solids()])

    # Sculpted fixed skin rounds the local enlarged lower-thigh profile. The
    # service space is cut from the solid so the motor is enclosed by a skin,
    # while source upper-thigh halves continue unmodified above z124.
    outer=(sphere((42,4,105),(23.7,30.5,28.5)) &
           clip(-50,140,-80,80,84,129))
    inner=sphere((42,4,105),(21.5,28.3,26.3))
    cover=outer-inner-service
    # A stationary front bow replaces the narrow original cuff under the
    # broadened lower thigh, leaving the rear for 120-degree calf travel.
    cover=cover & clip(-50,140,-80,25,84,129)
    lower_bow=(sphere((42,-5,78),(20,19,12))-
               sphere((42,-5,78),(17.8,16.8,9.8))-service) & \
              clip(-50,140,-80,4,68,90)
    cover=comp([cover,lower_bow])
    fixed_skin=comp([relieved_thigh,cover])

    # Independently supported 5 mm-bore, 13 mm-OD bearing envelopes on each
    # side of the actual case + opposite shaft projection. Their housings
    # connect to the exact thigh through stout inward cheek spars.
    bearings=comp([cyl_x(6.5,22.8,27.8)-cyl_x(2.55,22.7,27.9),
                   cyl_x(6.5,56.2,61.2)-cyl_x(2.55,56.1,61.3)])
    housings=comp([cyl_x(8.0,21.3,29.3)-cyl_x(6.6,21.2,29.4),
                   cyl_x(8.0,54.7,62.7)-cyl_x(6.6,54.6,62.8)])
    supports=comp([
        rod((25,8,100),(27,8,122),3.0),
        rod((59,8,100),(57,8,122),3.0)])

    # The yoke is separate from the case: two short bearing journals to
    # external side arms and a bridge embedded in the imported shin. The
    # torque spline-to-left hub and screws are purposefully left unmodeled.
    journals=comp([cyl_x(2.45,15,29.4),cyl_x(2.45,54.6,69)])
    hubs=comp([cyl_x(5.5,14.5,17.5),cyl_x(5.5,66.5,69.5)])
    arms=comp([rod((16,8,99),(26,-8,60),2.4),
               rod((68,8,99),(58,-8,60),2.4)])
    bridge=rod((26,-8,60),(58,-8,60),2.4)
    yoke=comp([journals,hubs,arms,bridge])

    # Rounded moving front lip continues the calf toward the stationary
    # cover. It is intentionally partial so the rear can open at high flex.
    moving_skin=comp([shin,boot])

    obstacles={
        "case":case,"relieved_thigh":relieved_thigh,
        "fixed_cover":cover,"bearing_housings":housings,
        "fixed_supports":supports}
    moving={"shin_ankle_boot_lip":moving_skin,"side_yoke":yoke}
    samples=[]
    for angle in ANGLES:
        pairs={}
        for mname,mshape in moving.items():
            posed=turn(mshape,angle)
            for fname,fshape in obstacles.items():
                pairs[mname+"__"+fname]=dict(zip(("gap_mm","overlap_mm3"),
                                                  gap(posed,fshape)))
        samples.append({"deg":angle,"min_nonmating_gap_mm":
                        min(p["gap_mm"] for p in pairs.values()),
                        "max_overlap_mm3":
                        max(p["overlap_mm3"] for p in pairs.values()),
                        "pairs":pairs})
        print(f"angle {angle}: gap {samples[-1]['min_nonmating_gap_mm']} overlap {samples[-1]['max_overlap_mm3']}",flush=True)
        if angle in VIEW_ANGLES and not os.environ.get("STRONG_DIRECT_QUICK"):
            scene=trimesh.Scene()
            for name,shape,color in (
                ("exact_thigh_relieved",relieved_thigh,(227,215,194,255)),
                ("sculpted_fixed_cover",cover,(101,93,75,255)),
                ("source_shin",turn(shin,angle),(235,222,200,255)),
                ("ankle_boot",turn(boot,angle),(219,204,178,255)),
                ("motor",case,(51,113,137,255)),
                ("bearings",bearings,(118,119,111,255)),
                ("housings",housings,(85,86,79,255)),
                ("supports",supports,(88,89,82,255)),
                ("moving_yoke",turn(yoke,angle),(178,112,64,255)),
            ): add(scene,name,shape,color)
            scene.export(OUT/f"swarm_strong_direct_knee_{angle:03d}.glb")

    static={
        "case_original_thigh":gap(case,source_thigh),
        "case_relieved_thigh":gap(case,relieved_thigh),
        "motor_cover":gap(case,cover),
        "housings_original_cuff":gap(housings,source_cuff),
        "support_original_thigh":gap(supports,source_thigh),
        "bridge_source_shin":gap(bridge,shin),
        "source_cuff_bounds":bounds(source_cuff),
        "motor_bounds":bounds(case),"housing_bounds":bounds(housings),
        "cover_bounds":bounds(cover),"original_shin_bounds":bounds(original_shin),
        "trimmed_shin_bounds":bounds(shin),
    }
    report={
        "motor":{
            "model":"Feetech STS3250",
            "manufacturer":"https://www.feetechrc.com/en/562636.html",
            "datasheet":"https://www.gotronic.fr/pj2-38757-3355.pdf",
            "rated_12v_Nm":1.569,"rated_12v_kgcm":16,
            "body_mm":[45.22,24.72,35],
            "case_model_xyz_mm":[24.72,45.22,37.4],
            "case_mass_g":74.5,
            "direct_drive":"1:1; rated torque is a source rating, no joint margin or duty inferred",
        },
        "source_step":str(SOURCE),"axis_xyz_mm":AXIS,
        "verdict":"REJECT: supported-direct package fits the motor after thigh relief, but the moving yoke intersects fixed thigh/cover/support; broad side yoke and rear opening fail enclosed sketch appearance.",
        "hard_clearance_finding":{
            "motor_to_trimmed_shin_120deg_gap_mm":2.9192,
            "trimmed_shin_top_z_mm":64,
            "source_cuff_bottom_z_mm":79,
            "neutral_rear_seam_gap_lower_bound_mm":15,
            "case_to_relieved_thigh_gap_mm":1,
            "case_to_fixed_cover_gap_mm":1,
            "full_assembly_positive_gap_verified":False,
            "limiting_nonmating_overlap_mm3":max(s["max_overlap_mm3"] for s in samples),
            "note":"The positive 2.9192 mm number applies only to the source shin versus motor case at 120 degrees; it is not a full-assembly clearance."
        },
        "width_finding":{
            "source_cuff_x_mm":30,
            "modeled_fixed_housing_stack_x_mm":41.4,
            "modeled_yoke_hub_stack_x_mm":55,
            "housing_extra_over_source_cuff_mm":11.4,
            "yoke_extra_over_source_cuff_mm":25,
            "idealized_support_stack_lower_bound_mm":37.72,
            "idealized_basis":"24.72 mm motor body axial size plus two 5 mm bearing stations plus two 1.5 mm outer housing walls; excludes shaft projections, adapters, tolerance and fasteners."
        },
        "static":static,
        "samples":samples,
        "notes":[
            "Bearing and housing sizes are envelopes, not selected/rated components.",
            "A vendor STEP available for a related ST-3235M is not a verified STS3250 model; the case box uses the STS3250 manufacturer dimensions with 2.4 mm height reserve.",
            "Source shin is cut at z64 and its ankle and full boot are preserved.",
            "This second attempt omits the rigid moving lip that collided heavily with the case in the first attempt. The resulting wide side yoke and open rear seam reject the requested hidden-hardware appearance.",
            "Motor tabs, cable, shaft adapter, fasteners, bearing retention, structural stiffness, moving-skin closure, full-body self collision and continuous-angle clearance remain unverified."
        ]}
    (OUT/"swarm_strong_direct_knee_report.json").write_text(json.dumps(report,indent=2)+"\n")
    if not os.environ.get("STRONG_DIRECT_QUICK"):
        bd.export_step(comp([fixed_skin,case,bearings,housings,supports]),
                       OUT/"swarm_strong_direct_knee_fixed.step")
        bd.export_step(comp([moving_skin,yoke]),
                       OUT/"swarm_strong_direct_knee_moving_standing.step")
        html=(ROOT/"artifacts/swarm-rethink-knee-20261004/index.html").read_text()
        html=html.replace("Shin-carried ape knee", "Supported-direct ape knee")
        html=html.replace("Engineering clearance coupon for a distal servo and diagonal knee belt.",
                          "Engineering clearance coupon for a thigh-housed STS3250 and supported direct axle.")
        html=html.replace("The external belt tube and rear cuff dominate the sketch silhouette; a flexible cover remains unresolved. The belt, spline adapter and bearings are not strength qualified.",
                          "The broad side yoke and open rear seam remain visible at flexion. This is a fit study, not an accepted enclosed knee.")
        html=html.replace("Blue: Feetech-size case. Orange: belt pulleys and shin rails.",
                          "Blue: Feetech-size case. Orange: supported shin yoke.")
        html=html.replace("pose-${String(d).padStart(3,'0')}.glb",
                          "swarm_strong_direct_knee_${String(d).padStart(3,'0')}.glb")
        html=html.replace("report.json","swarm_strong_direct_knee_report.json")
        html=html.replace("fixed.step","swarm_strong_direct_knee_fixed.step")
        html=html.replace("moving.step","swarm_strong_direct_knee_moving_standing.step")
        (OUT/"swarm_strong_direct_knee_index.html").write_text(html)
    print(json.dumps({"out":str(OUT),"static":static,
                      "minimum_gap":min(s["min_nonmating_gap_mm"] for s in samples),
                      "maximum_overlap":max(s["max_overlap_mm3"] for s in samples)},indent=2))


if __name__=="__main__":
    main()
