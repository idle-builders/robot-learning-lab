"""Bounded 350–600 mm source-ape fixed-hardware knee scale sweep.

350/500 use existing source STEP; 400/450 use the unchanged generator output;
550/600 are explicit extrapolations of the same generator with a fixed 2 mm
wall. The ST3250 and Gates 123-3MGT-15/20T hardware do not scale.
"""
from __future__ import annotations

import json,math
from pathlib import Path
import build123d as bd

from swarm_belt_knee_coupon import ROOT, box, cyl_x, compound, volume, overlap, gap, bbox

OUT=ROOT/'artifacts/swarm-scale-knee-20261004'
HEIGHTS=(350,400,450,500,550,600)
PITCH_RADIUS=60/(2*math.pi)


def source_files(h):
    folder=(ROOT/'artifacts/sketch-ape-shell/exports'/f'{h}mm'
            if h in (350,500) else OUT/f'source-{h}')
    return folder/'ape_shell_assembly.step',folder/'manifest.json'


def thigh_outer(h):
    s=h/350
    return bd.Solid.make_loft([bd.Wire.make_ellipse(rx*s,ry*s,
             bd.Plane(origin=(x*s,y*s,z*s)))
             for z,x,y,rx,ry in ((87,42,3,16,18),(98,43,3,20,22),
                                  (124,39,3,24,25),(145,35,4,22,23),
                                  (157,33,4,16,18),(163,31,4,8,10))])


def angled_capsule(radius,x0,x1,knee,drive):
    dy,dz=drive[1]-knee[1],drive[2]-knee[2]
    angle=math.degrees(math.atan2(-dy,dz))
    bar=bd.Pos((x0+x1)/2,(knee[1]+drive[1])/2,
               (knee[2]+drive[2])/2)*bd.Rot(angle,0,0)*bd.Box(x1-x0,2*radius,31.5)
    return cyl_x(radius,x0,x1,knee[1],knee[2]).fuse(
           cyl_x(radius,x0,x1,drive[1],drive[2]),bar)


def hardware(h):
    s=h/500
    knee=(60*s,18*s,155*s)
    drive=(knee[0],knee[1]-16,knee[2]+math.sqrt(31.5**2-16**2))
    motor=box((knee[0]-15,drive[1],drive[2]+12.2),(35,24.72,45.22))
    x0,x1=knee[0]+3,knee[0]+18
    belt=angled_capsule(PITCH_RADIUS+1.5,x0,x1,knee,drive)-\
         angled_capsule(PITCH_RADIUS-1,x0-.1,x1+.1,knee,drive)
    output_pulley=cyl_x(9,x0,x1,knee[1],knee[2])
    drive_pulley=cyl_x(9,x0,x1,drive[1],drive[2])
    shaft=cyl_x(4,knee[0]-30,knee[0]+26,knee[1],knee[2])
    bearings=compound([cyl_x(8,x,x+5,knee[1],knee[2])-
          cyl_x(4.1,x-.2,x+5.2,knee[1],knee[2])
          for x in (knee[0]-28,knee[0]+20)])
    seats=compound([cyl_x(10,x,x+6,knee[1],knee[2])-
          cyl_x(8.1,x-.2,x+6.2,knee[1],knee[2])
          for x in (knee[0]-28.5,knee[0]+19.5)])
    supports=compound([box((x,knee[1]+5,knee[2]+15),(5,10,20))
          for x in (knee[0]-25.5,knee[0]+22.5)])
    return locals()


def envelope_screen(h):
    step,manifest=source_files(h)
    if not step.exists() or not manifest.exists():
        return {'height_mm':h,'available':False}
    parts={p.label:p for p in bd.import_step(step).children}
    outer=thigh_outer(h)
    g=hardware(h)
    names=('motor','belt','output_pulley','drive_pulley','shaft','bearings','seats','supports')
    outside={name:round(volume(g[name]-outer),4) for name in names}
    thigh=compound([parts['R_thigh_front'],parts['R_thigh_back']])
    cuff=parts['R_knee_cuff']
    data=json.loads(manifest.read_text())
    return {'height_mm':h,'available':True,
            'source_kind':'validated generator' if h<=500 else 'explicit generator extrapolation',
            'source_step':str(step.relative_to(ROOT)),
            'source_shell_material_volume_mm3':round(sum(p['volume_mm3'] for p in data['parts']),3),
            'source_thigh_bbox_xyz_mm':bbox(thigh),
            'source_cuff_bbox_xyz_mm':bbox(cuff),
            'knee_axis_xyz_mm':g['knee'],'drive_axis_xyz_mm':g['drive'],
            'hardware_outside_unmodified_thigh_outer_mm3':outside,
            'motor_vs_unmodified_skin_mm3':round(overlap(g['motor'],thigh),3),
            'belt_vs_unmodified_skin_mm3':round(overlap(g['belt'],thigh),3),
            'all_hardware_internal_to_original_thigh_outer':max(outside.values())<.01}


if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    rows=[]
    for h in HEIGHTS:
        row=envelope_screen(h)
        rows.append(row)
        print(h,json.dumps({k:row.get(k) for k in
              ('available','hardware_outside_unmodified_thigh_outer_mm3',
               'all_hardware_internal_to_original_thigh_outer')}),flush=True)
    (OUT/'envelope-sweep.json').write_text(json.dumps(rows,indent=2)+'\n')
