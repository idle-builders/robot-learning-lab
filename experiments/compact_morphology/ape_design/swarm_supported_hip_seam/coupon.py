"""One bounded nested-cup hip seam at 500 mm; exact solids, never a rated joint."""
from __future__ import annotations

import contextlib
import io
import json
import math
from pathlib import Path
import sys

import build123d as b
import trimesh

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
from experiments.sketch_ape_shell.build import Study, loft, above_waist, box
from experiments.compact_morphology.ape_design.swarm_rated_leg_skin_v2.skin_v2_case import integration_shapes
from experiments.compact_morphology.ape_design.swarm_rated_leg_drive.rated_leg_coupon import mesh

K = 500 / 350
PIVOT = (26.5*K, 0, 160*K+16)
POSES = (("stand", 0, 0), ("yaw_out", -20, 0), ("yaw_in", 20, 0),
         ("stride", 20, -35), ("squat", 0, -45))
GAP = .65
MARGIN = .5


def ellipsoid(center, radii):
    return b.Solid.make_sphere(1).scale(radii).translate(center)


def clip(x0=-200, x1=200, y0=-200, y1=200, z0=-200, z1=600):
    return b.Solid.make_box(x1-x0, y1-y0, z1-z0,
                            b.Plane(origin=(x0, y0, z0)))


def cyl_z(radius, z0, z1):
    return b.Solid.make_cylinder(radius, z1-z0,
                                 b.Plane(origin=(PIVOT[0], PIVOT[1], z0)))


def volume(shape):
    return sum(s.volume for s in shape.solids()) if shape else 0.


def bounds(shape):
    q = shape.bounding_box()
    return [[round(v, 3) for v in q.min], [round(v, 3) for v in q.max]]


def common(a, c):
    aa, cc = a.bounding_box(), c.bounding_box()
    if any(getattr(aa.min, p) >= getattr(cc.max, p) or
           getattr(cc.min, p) >= getattr(aa.max, p) for p in 'XYZ'):
        return [0., 0.]
    return [volume(a & c), volume(c & a)]


def placed(shape, yaw, pitch):
    center = b.Pos(*PIVOT)
    inverse = b.Pos(*(-v for v in PIVOT))
    return center * b.Rot(0, 0, yaw) * b.Rot(pitch, 0, 0) * inverse * shape


def shapes():
    with contextlib.redirect_stdout(io.StringIO()):
        source = Study(500, 2, GAP).build()
    original = {p['name']:p['shape'] for p in source.parts}
    knee = integration_shapes(500)

    # New fixed anatomy: broad central V bowl and a deep, lateral socket cheek.
    # The side of the bowl is *replaced* by the cheek rather than subtracting
    # a pose sweep. This keeps each front/back central panel in one piece.
    bowl_stations = [(130,0,5,9,11),(135,0,5,20,17),(145,0,5,34,23),
                     (155,0,4,43,26),(164,0,3,46,26),(178,0,3,49,29)]
    pelvis, _ = source.hollow_loft(bowl_stations)
    pelvis = (pelvis - above_waist(-source.gap)).scale(K)
    # Fair, wide side boundary under the cheek; bottom central V is retained.
    side_cut = clip(27, 200, -200, 200, -100, 250)
    pelvis = pelvis - side_cut
    fixed = {
        'pelvis_front': pelvis & clip(y1=4-GAP/2),
        'pelvis_back': pelvis & clip(y0=4+GAP/2),
    }
    socket = ellipsoid(PIVOT, (31, 34, 31)) - \
             ellipsoid(PIVOT, (29, 32, 29))
    # Fixed cheek wraps the upper root; lower lip is rounded by the ellipsoid.
    socket = socket & clip(x0=24, z0=224)
    # Concealed roots connect the cheek to both retained bowl panels.
    fixed['socket_cheek'] = socket.fuse(
        b.Pos(26,-29,241)*b.Box(22,10,10),
        b.Pos(26,36,241)*b.Box(22,12,10))

    # Revised femur narrows above the full source mid-thigh, then grows into a
    # shallow spherical root. All loft stations are in the 350 mm source frame.
    femur_stations = [(87,42,3,16,18),(98,43,3,20,22),(124,39,3,24,25),
                      (138,37,3,20,19),(150,35,2,14,13),(156,33,1,9,9)]
    femur, _ = source.hollow_loft(femur_stations)
    femur = femur.scale(K)
    root = ellipsoid(PIVOT, (27, 28, 27)) - \
           ellipsoid(PIVOT, (25, 26, 25))
    root = root & clip(z1=233)
    moving = {
        'thigh_front': femur & clip(y1=1-GAP/2),
        'thigh_back': femur & clip(y0=1+GAP/2),
        'root_cup': root,
        'offset_knee_motor': b.Pos(0,10,12) * knee['fixed_knee_hardware']['motor'],
    }

    # RSF-5B-50-IDT drawing: Ø25 × 35.8 mm; W618/6: Ø13 × 3.5 mm.
    # A short 1:1 transfer is a geometric reserve only. Two annular seats
    # connect to the central pelvis through a proposed fixed 4 mm bridge.
    motor = b.Solid.make_cylinder(12.5,35.8,
        b.Plane(origin=(13.5*K,-10*K,PIVOT[2])))
    shaft = cyl_z(3, PIVOT[2]-27, PIVOT[2]+2)
    bearings = {}
    seats = {}
    for label, dz in (('lower',-16),('upper',-4)):
        z = PIVOT[2]+dz
        bearings[label] = cyl_z(6.5,z,z+3.5) - cyl_z(3.1,z-.1,z+3.6)
        seats[label] = cyl_z(8.5,z-.6,z+4.1) - cyl_z(6.6,z-.7,z+4.2)
    fixed_hardware = {
        'RSF_yaw_body':motor,
        'bearing_lower':bearings['lower'], 'bearing_upper':bearings['upper'],
        'seat_lower':seats['lower'], 'seat_upper':seats['upper'],
        'support_bridge':b.Pos(23,-1,PIVOT[2]-10)*b.Box(33,4,24),
    }
    moving_hardware = {
        'yaw_shaft':shaft,
        'femur_flange':cyl_z(6,PIVOT[2]-29,PIVOT[2]-25),
    }
    obstacles = {
        'torso_front':original['torso_front'],
        'torso_back':original['torso_back'],
        'rear_shifted_4S_pack':b.Pos(0,14,200*K)*b.Box(60,30,30.5),
        'service_reserve':b.Pos(0,0,225*K)*b.Box(25,20,14),
    }
    return original, fixed, moving, fixed_hardware, moving_hardware, obstacles, femur_stations


def main():
    original, fixed, moving, fixed_hw, moving_hw, obstacles, stations = shapes()
    assembly = {**fixed, **fixed_hw, **obstacles}
    rows=[]
    for pose,yaw,pitch in POSES:
        hits=[]
        for mn,ms in {**moving, **moving_hw}.items():
            # The supported yaw shaft stays vertical in its fixed bearings.
            mm=placed(ms,yaw,0 if mn=='yaw_shaft' else pitch)
            for fn,fs in assembly.items():
                # Bearings/shaft and socket/root are intentional running or
                # structural interfaces; report them but do not infer air gap.
                ab,ba=common(mm,fs)
                dist=fs.distance_to(mm)
                if max(ab,ba)>.05 or dist<MARGIN:
                    hits.append(dict(moving=mn,fixed=fn,ab_mm3=round(ab,3),
                                     ba_mm3=round(ba,3),gap_mm=round(dist,4)))
        rows.append(dict(pose=pose,yaw_deg=yaw,pitch_deg=pitch,hits=hits))
    parts={name:dict(solids=len(s.solids()),volume_mm3=round(volume(s),3),
                     bounds_mm=bounds(s))
           for name,s in {**fixed,**moving,**fixed_hw,**moving_hw,**obstacles}.items()}
    report=dict(target_height_mm=500,source='sketch-06 Study(500,2,.65)',
                pivot_mm=PIVOT,pack_center_mm=(0,14,200*K),
                topology='central V bowl + separate fixed deep socket cheek + smaller moving root cup and redrawn upper femur',
                femur_stations_source_mm=stations,parts=parts,rows=rows,
                root_to_thigh_mating_mm3={n:round(max(common(moving['root_cup'],moving[n])),3)
                                          for n in ('thigh_front','thigh_back')},
                cheek_to_pelvis_mating_mm3={n:round(max(common(fixed['socket_cheek'],fixed[n])),3)
                                            for n in ('pelvis_front','pelvis_back')},
                notes='Sampled rigid BRep. Both Boolean orders logged. Bearings, seat, flange and shell attachment overlaps need a real fastening design; pitch/roll, strength, bus, heat and continuous path unqualified.')
    HERE.mkdir(parents=True,exist_ok=True)
    (HERE/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    for pose,yaw,pitch in POSES:
        for layer in ('candidate','source','service'):
            scene=trimesh.Scene()
            skins={**fixed,**obstacles} if layer!='source' else {
                n:original[n] for n in ('pelvis_front','pelvis_back','torso_front','torso_back')}
            for name,s in skins.items():
                if name in ('rear_shifted_4S_pack','service_reserve') and layer!='service':continue
                if name in ('rear_shifted_4S_pack','service_reserve'):
                    color=(115,96,80,255)
                else:color=(224,216,198,255)
                mesh(scene,name,s,color)
            skins_moving=moving if layer!='source' else {
                n:original[n] for n in ('R_thigh_front','R_thigh_back')}
            for name,s in skins_moving.items():
                if name=='offset_knee_motor' and layer!='service':continue
                mesh(scene,name,placed(s,yaw,pitch),(66,95,121,255) if name=='offset_knee_motor' else (230,221,203,255))
            if layer=='service':
                for name,s in fixed_hw.items():mesh(scene,name,s,(76,91,100,255))
                for name,s in moving_hw.items():mesh(scene,name,placed(s,yaw,0 if name=='yaw_shaft' else pitch),(124,102,79,255))
            scene.export(HERE/f'{layer}-{pose}.glb')
    print(json.dumps({'parts':{n:p['solids'] for n,p in parts.items()},
        'hits':{r['pose']:r['hits'][:12] for r in rows}},indent=2))


if __name__=='__main__':main()
