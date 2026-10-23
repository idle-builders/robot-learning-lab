"""One bounded 600 mm internal-belt/rear-lid trial, not an approved design.

The 600 mm source is an explicit unchanged-generator extrapolation. Hardware
size is fixed. The only new seat change is a 1 mm inward placement per side.
The rigid lid deliberately tests whether the source rear silhouette can
cover the supported moving yoke without a deformable bridge.
"""
from __future__ import annotations

import json
import build123d as bd
import trimesh

from swarm_belt_knee_coupon import ROOT,box,cyl_x,compound,add,overlap,gap,volume,bbox
from swarm_belt_knee_500_v2_coupon import rod
from swarm_scale_knee_sweep import OUT,source_files,thigh_outer,hardware

H=600
S=H/500
SOURCE,_=source_files(H)


def turn(shape,axis,angle):
    return bd.Pos(*axis)*bd.Rot(angle,0,0)*bd.Pos(*(-v for v in axis))*shape


def rear_channel():
    outline=bd.Wire.make_polygon([
        (47.5*S,18*S,121*S),(47.5*S,55*S,121*S),
        (47.5*S,55*S,177*S),(47.5*S,30*S,177*S)],close=True)
    return bd.Solid.extrude(bd.Face(outline),(13*S,0,0))


def build():
    p={part.label:part for part in bd.import_step(SOURCE).children}
    g=hardware(H)
    k=g['knee']; d=g['drive']
    # At 600 the outermost original bearing-seat rings miss the source loft
    # by only 0.575 mm3. Shift each seat 1 mm medially, still surrounding
    # the unchanged two bearings, shaft and pulley.
    g['seats']=compound([
        cyl_x(10,k[0]-27.5,k[0]-21.5,k[1],k[2])-
        cyl_x(8.1,k[0]-27.7,k[0]-21.3,k[1],k[2]),
        cyl_x(10,k[0]+18.5,k[0]+24.5,k[1],k[2])-
        cyl_x(8.1,k[0]+18.3,k[0]+24.7,k[1],k[2])])
    thigh_orig=[p['R_thigh_front'],p['R_thigh_back']]
    cuff_orig=p['R_knee_cuff']
    shin_orig=[p['R_shin_front'],p['R_shin_back']]
    foot=compound([p[f'R_{name}'] for name in
                   ('foot_dorsal','toe_1','toe_2','toe_3','sole')])
    skin=compound([*shin_orig,foot])
    motor_keepout=box((k[0]-15,d[1],d[2]+12.2),(37,26.72,47.22))
    from swarm_scale_knee_sweep import angled_capsule
    belt_keepout=angled_capsule(12.5,k[0]+1,k[0]+20,k,d)
    channel=rear_channel()
    shaft_clearance=cyl_x(4.6,k[0]-31,k[0]+27,k[1],k[2])
    bearing_clearance=compound([cyl_x(8.3,x,x+7,k[1],k[2])
                                for x in (k[0]-29,k[0]+19)])
    thigh=[v-motor_keepout-belt_keepout-channel-shaft_clearance-
           bearing_clearance for v in thigh_orig]
    cuff_open=box((54*S,25*S,123*S),(13*S,48*S,24*S))
    cuff=cuff_orig-cuff_open
    fixed_skin=compound([*thigh,cuff])
    # Shaft collars and a rear pad attach two solid rods to the source shin.
    rods=[rod(x,4*S,108*S,18*S,128*S,2.5) for x in (52*S,56*S)]
    rods +=[rod(x,18*S,128*S,18*S,155*S,2.5) for x in (52*S,56*S)]
    pad=box((54*S,18*S,108*S),(13,25*S,4))
    collars=[cyl_x(5,x,x+4,k[1],k[2]) for x in (50*S,54*S)]
    yoke=compound([*rods,pad,*collars])
    # Flat 2 mm rigid closure at the source posterior outline. It shares
    # the edges of the opened thigh; any clash means it cannot stay rigid.
    lid=box((54*S,40*S,149*S),(13*S+2,2,56*S))
    return locals()


def run():
    OUT.mkdir(parents=True,exist_ok=True)
    g=build(); hw=g['g']; k=hw['knee']
    fixed_without_lid=compound([g['fixed_skin'],hw['motor'],hw['belt'],
        hw['drive_pulley'],hw['bearings'],hw['seats'],hw['supports']])
    fixed_with_lid=compound([fixed_without_lid,g['lid']])
    moving=compound([g['skin'],g['yoke']])
    rows=[]
    for a in (0,30,60,90,120):
        moved=turn(moving,k,a)
        rows.append({'angle_deg':a,
            'moving_vs_static_without_lid_overlap_mm3':round(overlap(moved,fixed_without_lid),3),
            'moving_to_static_without_lid_gap_mm':round(gap(moved,fixed_without_lid),3),
            'yoke_vs_rigid_rear_lid_overlap_mm3':round(overlap(turn(g['yoke'],k,a),g['lid']),3),
            'moving_vs_rigid_rear_lid_overlap_mm3':round(overlap(moved,g['lid']),3),
            'moving_to_rigid_rear_lid_gap_mm':round(gap(moved,g['lid']),3),
            'yoke_max_rear_y_mm':round(turn(g['yoke'],k,a).bounding_box().max.Y,3)})
        if a in (0,60,120):
            scene=trimesh.Scene()
            for name,shape,color in (
                ('modified_source_thigh',compound(g['thigh']),(228,220,204,255)),
                ('opened_source_cuff',g['cuff'],(103,98,87,255)),
                ('source_shin_foot',turn(g['skin'],k,a),(237,231,215,255)),
                ('attached_yoke',turn(g['yoke'],k,a),(199,116,68,255)),
                ('rear_rigid_lid_trial',g['lid'],(119,178,172,200)),
                ('ST3250',hw['motor'],(68,107,135,200)),
                ('Gates_belt',hw['belt'],(48,55,56,255)),
                ('drive_pulley',hw['drive_pulley'],(215,155,68,255)),
                ('output_pulley',turn(hw['output_pulley'],k,a),(215,155,68,255)),
                ('shaft',turn(hw['shaft'],k,a),(84,90,91,255)),
                ('bearings',hw['bearings'],(126,128,122,255)),
                ('seats',hw['seats'],(93,99,98,255)),
                ('supports',hw['supports'],(111,118,114,255))):
                add(scene,name,shape,color)
            scene.export(OUT/f'600-pose-{a:03d}.glb')
    source_scene=trimesh.Scene()
    for name,shape,color in (
        ('source_thigh',compound(g['thigh_orig']),(228,220,204,255)),
        ('source_cuff',g['cuff_orig'],(103,98,87,255)),
        ('source_shin_foot',g['skin'],(237,231,215,255))):
        add(source_scene,name,shape,color)
    source_scene.export(OUT/'600-source-standing.glb')
    outer=thigh_outer(H)
    data={'height_mm':H,'source_step':str(SOURCE.relative_to(ROOT)),
          'source_kind':'explicit 600 mm extrapolation of unchanged 350–500 mm generator; fixed 2 mm wall',
          'hardware_outside_source_thigh_outer_mm3':{
            n:round(volume(hw[n]-outer),4) for n in
            ('motor','belt','output_pulley','drive_pulley','shaft','bearings','seats','supports')},
          'shell_component_counts':[len(v.solids()) for v in g['thigh']]+[len(g['cuff'].solids())],
          'original_thigh_skin_volume_mm3':round(sum(map(volume,g['thigh_orig'])),3),
          'modified_thigh_skin_volume_mm3':round(sum(map(volume,g['thigh'])),3),
          'original_cuff_skin_volume_mm3':round(volume(g['cuff_orig']),3),
          'modified_cuff_skin_volume_mm3':round(volume(g['cuff']),3),
          'static_motor_to_modified_skin_gap_mm':round(gap(hw['motor'],g['fixed_skin']),3),
          'static_belt_to_modified_skin_gap_mm':round(gap(hw['belt'],g['fixed_skin']),3),
          'static_motor_vs_modified_skin_overlap_mm3':round(overlap(hw['motor'],g['fixed_skin']),3),
          'static_belt_vs_modified_skin_overlap_mm3':round(overlap(hw['belt'],g['fixed_skin']),3),
          'yoke_pad_vs_source_shin_overlap_mm3':round(overlap(g['pad'],g['skin']),3),
          'rear_channel_bbox_xyz_mm':bbox(g['channel']),
          'lid_bbox_xyz_mm':bbox(g['lid']),
          'lid_to_modified_thigh_skin_overlap_mm3':round(overlap(g['lid'],compound(g['thigh'])),3),
          'poses':rows}
    (OUT/'600-closure-trial.json').write_text(json.dumps(data,indent=2)+'\n')
    for name,shape in (
        ('600-modified-thigh',compound(g['thigh'])),('600-opened-cuff',g['cuff']),
        ('600-attached-yoke',g['yoke']),('600-rigid-rear-lid-trial',g['lid']),
        ('600-standing-fixed',fixed_with_lid),('600-standing-moving',moving)):
        bd.export_step(compound([shape]),OUT/f'{name}.step')
    print(json.dumps(data,indent=2),flush=True)


if __name__=='__main__':run()
