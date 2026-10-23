"""Small source-skin pose screen for the fixed ST3250/123-3MGT-15 scale sweep.

These are unmodified source parts. This deliberately tests whether a covered
leg is even plausible before cutting a rear slit or fitting a fairing.
"""
from __future__ import annotations

import json
import build123d as bd

from swarm_belt_knee_coupon import ROOT, box, compound, overlap, gap, volume
from swarm_scale_knee_sweep import OUT, source_files, thigh_outer, hardware


def shifted(h, dz):
    g=hardware(h)
    for name in ('motor','belt','output_pulley','drive_pulley','shaft',
                 'bearings','seats','supports'):
        g[name]=bd.Pos(0,0,dz)*g[name]
    g['knee']=(g['knee'][0],g['knee'][1],g['knee'][2]+dz)
    g['drive']=(g['drive'][0],g['drive'][1],g['drive'][2]+dz)
    return g


def turn(shape, axis, angle):
    return bd.Pos(*axis)*bd.Rot(angle,0,0)*bd.Pos(*(-v for v in axis))*shape


def screen(h,dz):
    source,_=source_files(h)
    p={part.label:part for part in bd.import_step(source).children}
    thigh=compound([p['R_thigh_front'],p['R_thigh_back']])
    cuff=p['R_knee_cuff']
    shin=compound([p['R_shin_front'],p['R_shin_back']])
    foot=compound([p[f'R_{name}'] for name in
                   ('foot_dorsal','toe_1','toe_2','toe_3','sole')])
    moving=compound([shin,foot])
    fixed=compound([thigh,cuff])
    g=shifted(h,dz)
    outer=thigh_outer(h)
    outside={name:round(volume(g[name]-outer),3) for name in
             ('motor','belt','output_pulley','drive_pulley','shaft',
              'bearings','seats','supports')}
    rows=[]
    for angle in (0,30,60,90,120):
        skin=turn(moving,g['knee'],angle)
        rows.append({'angle_deg':angle,
                     'skin_vs_fixed_shell_overlap_mm3':round(overlap(skin,fixed),3),
                     'skin_vs_motor_overlap_mm3':round(overlap(skin,g['motor']),3),
                     'skin_vs_belt_overlap_mm3':round(overlap(skin,g['belt']),3),
                     'skin_to_motor_gap_mm':round(gap(skin,g['motor']),3),
                     'skin_to_belt_gap_mm':round(gap(skin,g['belt']),3),
                     'skin_to_fixed_shell_gap_mm':round(gap(skin,fixed),3)})
    return {'height_mm':h,'axis_z_shift_mm':dz,'knee_axis':g['knee'],
            'hardware_outside_thigh_outer_mm3':outside,'poses':rows}


if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    rows=[]
    for h,dz in ((500,0),(550,0),(600,0),(600,-20)):
        row=screen(h,dz)
        rows.append(row)
        print(h,dz,json.dumps(row),flush=True)
    (OUT/'unmodified-pose-screen.json').write_text(json.dumps(rows,indent=2)+'\n')
