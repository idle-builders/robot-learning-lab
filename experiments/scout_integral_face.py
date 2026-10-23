"""Study a one-piece, finish-colored Scout face using retained head interfaces.

A process/assembly alternative, not a shared-tree promotion or lighting design.
Uses the shared variant for eye geometry and the retained integral-wing head.
"""
import argparse
import json
from math import degrees
from pathlib import Path
import traceback

from build123d import Box, Compound, Pos, Rot, export_step, export_stl, import_step
from lab.config import load_config, RobotConfig
from lab.robot import make_robot
from lab.mechanical import part_shape
from lab.camera_mount import removal_clearance
from tools.check_camera_support import copy_source, sha, write
from tools.compose_forearm_enclosure import verify_report


def volume(shape):
    return 0.0 if shape is None else sum(s.volume for s in shape.solids())


def union(shapes):
    result = shapes[0]
    for shape in shapes[1:]:
        result = result.fuse(shape)
    return result.clean()


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    report = dict(status='fail', physical_status='not_run', installed_status='not_run',
                  scope=__doc__, source_bindings={})
    try:
        inputs = [Path('experiments/scout_integral_face.py'), Path('variants/scout-shell-eyes.yaml'),
                  Path('robot/base.yaml'), Path('lab/config.py'), Path('lab/robot.py'),
                  Path('lab/styling.py'), Path('lab/mechanical.py')]
        report['source_hashes'] = copy_source(output, {str(p):p for p in inputs})
        source = Path('artifacts/scout-integral-head-wings-v4')
        parent = verify_report(source)
        report['retained_head_status'] = {k:parent.get(k) for k in ['status','nominal_status','motion_status','service_status','physical_status']}
        for p in [source/'report.json',source/'head_shell.step',source/'head_rear_cover.step',source/'resolved-config.json']:
            report['source_bindings'][str(p)] = sha(p)
        cfg = load_config(Path('variants/scout-shell-eyes.yaml'))
        before = RobotConfig.model_validate_json((source/'resolved-config.json').read_text())
        a,b = cfg.model_dump(),before.model_dump()
        for x in [a,b]:
            x.pop('name'); x['face'].pop('eye_shroud_wall_m')
        assert a == b, 'Retained head differs beyond name and eye wall'
        robot = make_robot(cfg)
        write(output/'resolved-config.json',cfg.model_dump())
        write(output/'interfaces.json',robot.interface())
        head = next(link for link in robot.links if link.name=='head')
        shell = import_step(source/'head_shell.step')
        rear = import_step(source/'head_rear_cover.step')
        components = [('head_shell',shell,[.85,.85,.81,1])]
        rows=[]
        for index,g in enumerate(head.geometry):
            if not g.role.startswith('eye_'):
                continue
            pose=Pos(*(v*1000 for v in g.pos))*Rot(*(degrees(v) for v in g.rpy))
            shape=pose*part_shape(robot,head,index)
            key=f'{g.role}-{index}'
            components.append((key,shape,list(g.color)))
            rows.append({'name':key,'volume_mm3':volume(shape),'head_overlap_mm3':volume(shape&shell)})
        fused=union([s for _,s,_ in components])
        original_sum=sum(volume(s) for _,s,_ in components)
        restored_path=output/'integral-face.step'
        export_step(fused,restored_path)
        restored=import_step(restored_path)
        export_stl(restored,output/'integral-face.stl',tolerance=.03,angular_tolerance=.08)
        # Independent stock accounting against the source components. Every source
        # solid must remain, and all output stock must belong to the original union.
        added=restored
        for _,s,_ in components:
            if added is not None:
                added=added-s
        missing=sum(volume(s-restored) for _,s,_ in components)
        cavity=removal_clearance(robot)
        # Optical marks are raised colored surface regions in this process option,
        # not separate lenses or diffusers. No transparent/light-emitting claim.
        meshes=[]
        def mesh(key,shape,color,view):
            vertices,faces=shape.tessellate(.08,.10)
            meshes.append({'name':key,'view':view,'vertices_m':[[v.X/1000,v.Y/1000,v.Z/1000] for v in vertices],'faces':faces,'color':color})
        mesh('integral_face',restored,[.70,.74,.73,1],'as_printed')
        # Partition surface stock with explicit precedence: white marks above dark
        # eye discs above accent housings above the head. This is finish guidance,
        # not overlapping printable meshes or evidence of a multi-material slice.
        remaining=restored
        for key,shape,color in reversed(components):
            region=remaining&shape
            if volume(region)>0.0001:
                mesh(key,region,color,'finish_concept')
                remaining=remaining-shape
        cut=restored & (Pos(0,-100,0)*Box(300,200,300))
        mesh('half_section',cut,[.70,.74,.73,1],'section')
        mesh('retained_rear',rear,[.45,.49,.5,.45],'as_printed')
        write(output/'meshes.json',meshes)
        checks={
            'one_valid_connected_solid':restored.is_valid and len(restored.solids())==1,
            'native_step_roundtrip':volume(fused-restored)<.001 and volume(restored-fused)<.001,
            'all_source_stock_retained':missing<.001,
            'no_stock_added_outside_source_union':volume(added)<.001,
            'camera_service_opening_clear':volume(restored&cavity)<.02,
            'rear_cover_no_overlap':volume(restored&rear)<.02,
            'finish_partition_covers_solid':volume(remaining)<.001,
        }
        report.update(status='partial' if all(checks.values()) else 'fail',
            geometry_status='pass' if all(checks.values()) else 'fail',checks=checks,
            original_group_count=len(components),candidate_printed_piece_count=len(restored.solids()),
            source_sum_mm3=original_sum,candidate_union_mm3=volume(restored),
            duplicated_source_stock_mm3=original_sum-volume(restored),
            missing_stock_mm3=missing,added_stock_mm3=volume(added),
            camera_service_overlap_mm3=volume(restored&cavity),rear_cover_gap_mm=restored.distance_to(rear),
            parts=rows,
            interface_choice='One head-owned printed front replaces the front shell and six fixed eye allocations; existing rear cover remains separate. Raised colors require finishing. No new screws, snap fits or adhesive joints.',
            limits=['Not installed in complete assembly or shared visual tree; seven old CAD groups remain in baseline.',
                'Fixed expression; this process option supplies no illumination or eye motion.',
                'No seam/fillet reinforcement added; connected solid is not strength proof.',
                'Finish labor, adhesion, color durability, print orientation/support and native whole-face slice remain open.',
                'No new occupied volume versus selected shell-eye union; inherited whole-robot failures persist.',
                'Rear gap and camera cutout checks do not prove complete service paths or printed tolerances.',
                'Source sum double-counts overlapping solids; its difference from union is accounting, not print saving.',
                'No price or whole-robot mass saving booked.'])
    except Exception as e:
        report['error']=str(e);(output/'error.log').write_text(traceback.format_exc())
    report['output_hashes']={str(p.relative_to(output)):sha(p) for p in output.rglob('*') if p.is_file() and 'source' not in p.relative_to(output).parts and p.name!='report.json'}
    write(output/'report.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ['source_hashes','source_bindings','output_hashes','parts']},indent=2))
    if report['status']=='fail':raise SystemExit(1)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    run(p.parse_args().output)
