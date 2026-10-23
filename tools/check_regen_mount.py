"""Check a proposed shunt tray, real nominal hardware and staged access probes."""
import argparse
from dataclasses import replace
from pathlib import Path
import itertools
import json
import hashlib
import traceback
import shutil
import numpy as np
from build123d import Box, Cylinder, Sphere, Pos, Rot, import_step, export_step, export_stl
from lab.artifact_store import snapshot_file
from lab.config import RobotConfig
from lab.robot import make_robot
from lab.power_pack import torso_containment
from lab.leg_mechanism import Part, transform_shape, overlap
from tools.regen_mount_geometry import geometry, along_x
from tools.check_neck_collar import ExactQueries
from tools.check_arm_episode import parts_from_step
from tools.compose_forearm_enclosure import verify_report

def run(out):
    out.mkdir(parents=True,exist_ok=False)
    write=lambda p,v:p.write_text(json.dumps(v,indent=2)+'\n')
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    c,shapes,cuts,interface=geometry();q=ExactQueries()
    report={'status':'partial','execution_status':'running','robots':{},'interface':interface,'physical_status':'not_run','thermal_status':'not_run','crossbeam_strength_status':'not_run'}
    source=['electronics/regen-mount.yaml','tools/regen_mount_geometry.py','tools/check_regen_mount.py','electronics/regen-clamp.yaml',c['module_source'],c['capacitor_clearance_probe']['source'],c['capacitor_clearance_probe']['vent_source'],'lab/pelvis.py','lab/robot.py','lab/mechanical.py','lab/power_pack.py','tools/check_neck_collar.py','tools/check_arm_episode.py']
    report['source_bindings']={p:sha(p) for p in source}
    for p in source:
        if p.startswith(('electronics/','tools/','lab/')):
            dest=out/'source'/p;dest.parent.mkdir(parents=True,exist_ok=True);snapshot_file(p,dest)
    try:
        assert shapes['tray'].is_valid and len(shapes['tray'].solids())==1
        export_step(shapes['tray'],out/'tray.step');export_stl(shapes['tray'],out/'tray.stl')
        readback=import_step(out/'tray.step')
        assert readback.is_valid and len(readback.solids())==1 and abs(readback.volume-shapes['tray'].volume)<1e-6
        report['tray']={'valid_solids':1,'volume_mm3':shapes['tray'].volume,'step_roundtrip':'pass'}
        for key,shape in shapes.items():
            if key != 'module':export_stl(shape,out/(key+'.stl'))
        # No blanket exclusions for nominal contacts: only volumetric interference is screened.
        parts=[Part(k,'torso',s,'candidate','power',(.5,.5,.5,1)) for k,s in shapes.items() if k not in ['capacitor_working_space','vent']]
        pairs=[]
        for a,b in itertools.combinations(parts,2):
            hits=q.collide(a.shape,[b])
            if hits:pairs.append({'a':a.name,'b':b.name,'overlap_mm3':hits[0]['overlap_mm3']})
        report['internal_interferences']=pairs
        insertion=[]
        for i,center in enumerate(interface['module_hole_centers_mm']):
            sign=1 if center[1]>0 else -1
            hits=[]
            for step in np.linspace(0,c['tray']['nut_insertion_travel_mm'],17):
                v=overlap(Pos(0,sign*float(step),0)*shapes[f'module_nut_{i}'],shapes['tray'])
                if v>.02:hits.append({'travel_mm':float(step),'overlap_mm3':v})
            insertion.append({'nut':i,'status':'pass' if not hits else 'fail','collisions':hits})
        report['nut_insertion']=insertion
        # Nominal axial thread coverage; physical threads and preload are not modeled.
        report['screw_engagement']=[dict(i,engagement_mm=max(0,min(i['screw_seat_mm'],i['nut_top_mm'])-max(i['screw_tip_mm'],i['nut_bottom_mm'])),blind_tip_space_mm=i['screw_tip_mm']-c['tray']['blind_bore_bottom_mm']) for i in interface['module_fasteners']]
        report['beam_screw_protrusion_mm']=c['arches']['screw_seat_x_mm']+c['hardware']['beam_screw_length_mm']-c['arches']['beam_front_face_x_mm']-c['hardware']['nut_height_mm']
        for name,version in [('pip','v3'),('scout','v2')]:
            print(name,'loading retained CAD',flush=True)
            folder=Path(f'artifacts/{name}-shin-cover-composed-{version}');retained=verify_report(folder)
            assert retained['assembly_step_roundtrip']=='pass'
            for p in ['assembly.step','report.json','resolved-config.json']:report['source_bindings'][str(folder/p)]=sha(folder/p)
            robot=make_robot(RobotConfig.model_validate_json((folder/'resolved-config.json').read_text()));frame=robot.forward_kinematics()['torso'];inverse=np.linalg.inv(frame)
            raw=parts_from_step(list(import_step(folder/'assembly.step').children),retained['parts'])
            front=next(p for p in raw if p.name=='torso_front');local_front=transform_shape(front.shape,inverse)
            modified=local_front
            for cut in cuts:modified=modified-cut
            assert modified.is_valid and len(modified.solids())==1
            removed=local_front.volume-modified.volume;assert removed>0
            targets=[replace(p,shape=transform_shape(modified,frame)) if p.name=='torso_front' else p for p in raw]
            battery=Pos(8,0,39)*Box(61.04,87.12,24.39)
            targets.append(Part('battery_reference_plus_1mm','torso',transform_shape(battery,frame),'allocation','power',(1,.6,0,1)))
            result={'source_assembly_status':retained['status'],'groups_checked':len(raw),'modified_front_valid_solids':1,'crossbeam_removed_mm3':removed,'checks':{},'tool_access':[]}
            result['uncut_crossbeam_negative_control']=[{'screw':i,'overlap_mm3':overlap(shapes[f'beam_screw_{i}'],local_front)} for i in range(len(c['arches']['y_centers_mm']))]
            assert all(i['overlap_mm3']>1 for i in result['uncut_crossbeam_negative_control'])
            for key,shape in shapes.items():
                hits=q.collide(transform_shape(shape,frame),targets)
                inside=torso_containment(robot,shape)
                result['checks'][key]={'status':'pass' if not hits and inside['status']=='pass' else 'fail','collisions':hits,'containment':inside}
            # Long straight shafts explicitly test both installed and disassembled access.
            service_targets=[p for p in targets if p.name not in ['torso_rear','carrier','battery_reference_plus_1mm']]
            for i,y in enumerate(c['arches']['y_centers_mm']):
                start=c['tool_probe']['rear_start_x_mm'];end=c['arches']['screw_seat_x_mm']-c['hardware']['head_height_mm']
                tool=along_x((start+end)/2,y,c['arches']['bolt_z_mm'],c['tool_probe']['shaft_diameter_mm']/2,end-start)
                result['tool_access'].append({'kind':'beam_screw','index':i,'installed_hits':q.collide(transform_shape(tool,frame),targets),'rear_cover_carrier_battery_removed_hits':q.collide(transform_shape(tool,frame),service_targets)})
            for i,(x,y) in enumerate(interface['module_hole_centers_mm']):
                base=interface['board_top_mm']+c['hardware']['head_height_mm']
                tool=Pos(x,y,(base+110)/2)*Cylinder(1.5,110-base)
                result['tool_access'].append({'kind':'module_screw','index':i,'installed_hits':q.collide(transform_shape(tool,frame),targets),'rear_cover_carrier_battery_removed_hits':q.collide(transform_shape(tool,frame),service_targets),'tray_hits':q.collide(tool,[Part('tray','torso',shapes['tray'],'candidate','power',(1,0,0,1))])})
            # Published overall L-key dimensions; conservative round shaft proxy.
            # The socket interior is not modeled: check only above the screw head.
            key=c['tool_probe']['l_key'];radius=key['envelope_diameter_mm']/2
            base=interface['board_top_mm']+c['hardware']['head_height_mm']
            elbow=base-key['socket_engagement_mm']+key['overall_short_mm']-radius
            added=[Part(k,'torso',transform_shape(s,frame),'candidate','power',(1,0,0,1)) for k,s in shapes.items() if k not in ['vent','capacitor_working_space']]
            result['l_key_access']={'elbow_center_z_mm':elbow,'radius_mm':radius,'sweeps':[],'insertion_status':'not_run','hand_clearance_status':'not_run'}
            for i,(x,y) in enumerate(interface['module_hole_centers_mm']):
                upright=Pos(x,y,(base+elbow)/2)*Cylinder(radius,elbow-base)
                handle=along_x(x-key['overall_long_mm']/2,y,elbow,radius,key['overall_long_mm'])+Pos(x,y,elbow)*Sphere(radius)
                for angle in range(-30,31,5):
                    probe=upright+Pos(x,y,0)*Rot(0,0,angle)*Pos(-x,-y,0)*handle
                    obstacles=service_targets+[p for p in added if p.name!=f'module_screw_{i}']
                    hits=q.collide(transform_shape(probe,frame),obstacles)
                    result['l_key_access']['sweeps'].append({'screw':i,'angle_deg':angle,'collisions':hits})
            result['l_key_access']['status']='fail' if any(r['collisions'] for r in result['l_key_access']['sweeps']) else 'pass'
            # Removal hypothesis: screws/capacitor/rear cover/carrier/battery removed;
            # lift off collars, slide forward below the beam, then lift the board.
            extraction=[]
            module_obstacles=service_targets+[p for p in added if p.name=='tray' or p.name.startswith('beam_')]
            waypoints=c['service_paths']['board_waypoints_mm']
            for start,end in zip(waypoints,waypoints[1:]):
                count=int(np.ceil(np.linalg.norm(np.array(end)-start)/c['service_paths']['sample_step_mm']))+1
                for fraction in np.linspace(0,1,count):
                    shift=[float(a+(b-a)*fraction) for a,b in zip(start,end)]
                    probe=Pos(*shift)*shapes['module']
                    hits=q.collide(transform_shape(probe,frame),module_obstacles)
                    if hits:extraction.append({'translation_mm':shift,'collisions':hits})
            result['board_removal']={'status':'fail' if extraction else 'pass','waypoints_mm':waypoints,'maximum_sample_step_mm':c['service_paths']['sample_step_mm'],'collisions':extraction,'continuous_sweep':'not_run','scope':'Candidate route after module screws, capacitor, rear cover, carrier and battery removed; wiring, screw extraction and key insertion not validated.'}
            tray_hits=[];tray_waypoints=c['service_paths']['tray_waypoints_mm']
            for start,end in zip(tray_waypoints,tray_waypoints[1:]):
                count=int(np.ceil(np.linalg.norm(np.array(end)-start)/c['service_paths']['sample_step_mm']))+1
                for fraction in np.linspace(0,1,count):
                    shift=[float(a+(b-a)*fraction) for a,b in zip(start,end)]
                    hits=q.collide(transform_shape(Pos(*shift)*shapes['tray'],frame),service_targets)
                    if hits:tray_hits.append({'translation_mm':shift,'collisions':hits})
            result['tray_removal']={'status':'fail' if tray_hits else 'pass','waypoints_mm':tray_waypoints,'maximum_sample_step_mm':c['service_paths']['sample_step_mm'],'collisions':tray_hits,'continuous_sweep':'not_run','scope':'After board/capacitor/beam hardware, rear cover, carrier and battery removed; sequential dependency, wiring and physical grip remain unqualified.'}
            result['neutral_status']='pass' if all(v['status']=='pass' for v in result['checks'].values()) and not pairs else 'fail'
            dest=out/name;dest.mkdir()
            export_step(modified,dest/'proposed-front.step');export_stl(modified,dest/'proposed-front.stl')
            write(dest/'report.json',result)
            report['robots'][name]=result;write(out/'progress.json',report)
            print(name,result['neutral_status'],{k:v['collisions'] for k,v in result['checks'].items() if v['status']=='fail'},flush=True)
        report['execution_status']='pass'
    except Exception as exc:
        report['execution_status']='fail';report['error']=str(exc);(out/'error.log').write_text(traceback.format_exc());raise
    finally:write(out/'report.json',report)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);run(parser.parse_args().output)
