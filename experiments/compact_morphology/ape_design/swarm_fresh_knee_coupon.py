"""Fresh fixed-barrel knee screen on experiments/sketch_ape_shell (350 mm).

Run: .venv/bin/python experiments/compact_morphology/ape_design/swarm_fresh_knee_coupon.py
X lateral, Y rearward, Z up. Units are millimetres. This is a motion/fit coupon.
"""
from __future__ import annotations
import json, math, sys
from pathlib import Path
import build123d as bd
import numpy as np
import trimesh
ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT/'artifacts/sketch-ape-shell/exports/350mm/ape_shell_assembly.step'
OUT = ROOT/'artifacts/swarm-fresh-knee-new-shell-20261004'
AXIS = (42,3,86)
ANGLES = tuple(range(0,121,10))

def cyl_x(r,x0,length,y=3,z=86):
    return bd.Solid.make_cylinder(r,length,bd.Plane(origin=(x0,y,z),z_dir=(1,0,0)))

def turn(s,deg):
    return bd.Pos(*AXIS)*bd.Rot(deg,0,0)*bd.Pos(*(-v for v in AXIS))*s

def overlap(a,b):
    # OCC can silently intersect only the first child of a compound. Keep
    # each B-rep solid explicit so the shin halves and internal yoke count.
    total=0.0
    for left in a.solids():
        for right in b.solids():
            hit=left&right
            if hit is not None: total+=sum(s.volume for s in hit.solids())
    return total

def barrel():
    # Stationary frontal arc inside original knee cuff bounds; rear open.
    def p(ry,rz,a):
        return (26,3+ry*math.cos(math.radians(a)),86+rz*math.sin(math.radians(a)))
    aa=list(range(40,261,4))
    w=bd.Wire.make_polygon([p(17,7,a) for a in aa]+[p(15,5,a) for a in reversed(aa)],close=True)
    return bd.extrude(bd.Face(w),amount=32,dir=(1,0,0))

def add(scene,name,s,color):
    v,f=s.tessellate(.3,.25)
    m=trimesh.Trimesh(vertices=np.array([(x.X,x.Y,x.Z) for x in v]),faces=np.array(f),process=False)
    m.visual.vertex_colors=color
    scene.add_geometry(m,geom_name=name,node_name=name)

def main():
    if not SOURCE.exists(): raise FileNotFoundError(SOURCE)
    OUT.mkdir(parents=True,exist_ok=True)
    imported=bd.import_step(SOURCE)
    p={c.label:c for c in imported.children}
    thighs=[p[f'R_thigh_{s}'] for s in ('front','back')]
    shins=[p[f'R_shin_{s}'] for s in ('front','back')]
    foot=bd.Compound(children=[p[f'R_{name}'] for name in
        ('foot_dorsal','toe_1','toe_2','toe_3','sole')])
    cuff=p['R_knee_cuff']
    # Official-size ST-3250-C001 box. Bow-following tilt breaks shaft parallelism.
    motor=bd.Pos(39,3,124)*bd.Rot(0,-12,0)*bd.Box(35,24.72,45.22)
    sys.path.insert(0,str(ROOT/'experiments/sketch_ape_shell'))
    import build as study
    stations=[(87,42,3,16,18),(98,43,3,20,22),(124,39,3,24,25),
              (145,35,4,22,23),(157,33,4,16,18),(163,31,4,8,10)]
    inner=study.loft([(z,x,y,rx-2,ry-2) for z,x,y,rx,ry in stations])
    straight_motor=bd.Pos(39,3,124)*bd.Box(35,24.72,45.22)
    # Two link plates, one shaft, and bearing placeholders. Skin trimmed at z75.
    cutter=bd.Pos(42,3,32.5)*bd.Box(150,150,85)
    skin=[s&cutter for s in shins]
    yoke=(bd.Pos(29,3,80.5)*bd.Box(3,7,11)+bd.Pos(55,3,80.5)*bd.Box(3,7,11)+cyl_x(3.5,27,30))
    yoke+=bd.Pos(28,3,75)*bd.Box(9,8,5)
    yoke+=bd.Pos(56,3,75)*bd.Box(9,8,5)
    moving=bd.Compound(children=skin+[yoke])
    hood=barrel()
    bearings=[cyl_x(6.5,x,2)-cyl_x(3.7,x,2) for x in (26,56)]
    fixed_raw=bd.Compound(children=thighs+[hood]+bearings)
    raw=[]
    for deg in ANGLES:
        posed=turn(moving,deg)
        raw.append({'deg':deg,'shin_vs_original_thigh_mm3':round(sum(overlap(posed,t) for t in thighs),3),
                    'shin_vs_front_barrel_mm3':round(overlap(posed,hood),3),
                    'shin_vs_motor_mm3':round(overlap(posed,motor),3),
                    'foot_vs_original_thigh_mm3':round(sum(overlap(turn(foot,deg),t) for t in thighs),3),
                    'foot_vs_motor_mm3':round(overlap(turn(foot,deg),motor),3)})
    # An inverse sampled sweep allows one to see how much original rear skin
    # must be removed. It has zero tolerance and is not a manufacturing cut.
    relieved=[]
    for t in thighs:
        r=t-motor
        for deg in ANGLES:
            for s in moving.solids(): r-=turn(s,deg)
        relieved.append(r)
    fixed=bd.Compound(children=relieved+[hood]+bearings)
    checked=[]
    for deg in ANGLES:
        posed=turn(moving,deg)
        checked.append({'deg':deg,'moving_vs_relief_and_barrel_mm3':round(overlap(posed,fixed),3)})
        if deg in (0,30,60,90,120):
            scene=trimesh.Scene()
            for n,s,c in [('thigh_front',relieved[0],(234,225,210,255)),
                          ('thigh_back',relieved[1],(234,225,210,255)),
                          ('front_barrel',hood,(226,217,197,255)),
                          ('bearings',bd.Compound(children=bearings),(80,86,88,255)),
                          ('ST3250_official_size',motor,(65,90,105,160)),
                          ('moving_shin_yoke',posed,(240,228,210,255)),
                          ('moving_foot',turn(foot,deg),(221,211,193,255))]:
                add(scene,n,s,c)
            scene.export(OUT/f'knee-{deg:03d}.glb')
    bd.export_step(fixed,OUT/'fixed-relieved-thigh-and-barrel.step')
    bd.export_step(moving,OUT/'moving-shin-and-yoke.step')
    report={'source_new_shell':str(SOURCE.relative_to(ROOT)),
            'coordinate_system':'X lateral, Y rearward, Z up; mm',
            'concept':'fixed original-envelope front barrel; rear opening for internal shin yoke',
            'original_knee_cuff_bounds':str(cuff.bounding_box()),
            'axis_mm':AXIS,'motor':'Feetech ST-3250-C001 official 35x24.72x45.22 outer box',
            'motor_source':'https://www.feetechrc.com/en/562636.html',
            'motor_center_mm':(39,3,124),'motor_tilt_deg':-12,
            'motor_inside_nominal_new_thigh_pct':round(100*overlap(motor,inner)/motor.volume,3),
            'parallel_axis_zero_tilt_inside_nominal_new_thigh_pct':round(100*overlap(straight_motor,inner)/straight_motor.volume,3),
            'parallel_axis_zero_tilt_vs_original_thigh_skin_mm3':round(sum(overlap(straight_motor,t) for t in thighs),3),
            'motor_vs_original_thigh_skin_mm3':round(sum(overlap(motor,t) for t in thighs),3),
            'removed_original_thigh_skin_mm3':[round(a.volume-b.volume,3) for a,b in zip(thighs,relieved)],
            'barrel_outer_yz_radii_mm':(17,7),'rear_aperture_deg':140,
            'original_shin_trim_z_mm':75,'raw_sweep':raw,'after_sampled_relief':checked,
            'verdict':'REJECT at 120 degrees: retained shin hits motor and front barrel from deep flex; rear thigh relief is insufficient.',
            'cautions':['Motor tilt makes its output nonparallel to the knee axis; a rated cable drum/transmission is not designed.',
                        'Bearings, yoke attachment, output spool, cable routing, fasteners, positive gap and structure remain unverified.',
                        'No torque/thermal, impact, adjacent-leg, hip or foot clearance validation.',
                        'Boolean relief is exact at 10-degree samples; no between-sample clearance claim.']}
    (OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    (OUT/'index.html').write_text(Path(__file__).with_name('swarm_fresh_knee_viewer.html').read_text())
    print(json.dumps({'artifact':str(OUT),'motor_fit_pct':report['motor_inside_nominal_new_thigh_pct'],
        'raw_max_thigh_overlap':max(r['shin_vs_original_thigh_mm3'] for r in raw),
        'raw_max_barrel_overlap':max(r['shin_vs_front_barrel_mm3'] for r in raw),
        'raw_max_motor_overlap':max(r['shin_vs_motor_mm3'] for r in raw),
        'after_relief_max':max(r['moving_vs_relief_and_barrel_mm3'] for r in checked)},indent=2))
if __name__=='__main__': main()
