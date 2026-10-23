"""Bounded terminal packaging study using the existing native CAD checker."""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import traceback

import numpy as np
from build123d import Box, Pos, export_step, export_stl, import_step
from lab.artifact_store import retain_dependency, snapshot_file
from lab.config import RobotConfig
from lab.robot import make_robot
from lab.leg_mechanism import Part, transform_shape
from tools.check_arm_episode import parts_from_step
from tools.check_neck_collar import ExactQueries
from tools.compose_forearm_enclosure import verify_report
from tools.regen_terminal_geometry import geometry


def part(name, shape):
    return Part(name, 'torso', shape, 'candidate', 'power', (.2,.6,.3,1))


def sample_route(route, step_mm):
    for start, end in zip(route, route[1:]):
        for t in np.linspace(0, 1, int(np.ceil(np.linalg.norm(np.array(end)-start)/step_mm))+1):
            yield (np.array(start)+(np.array(end)-start)*t).tolist()


def run(out):
    out.mkdir(parents=True, exist_ok=False)
    save=lambda n,v:(out/n).write_text(json.dumps(v,indent=2)+'\n')
    r={'status':'partial','execution_status':'running','physical_status':'not_run',
       'electrical_qualification':'not_run','full_wire_routing':'not_run',
       'tool_axis':'Assumed at terminal pin centers; actual screw offset not dimensioned.',
       'allowance_scope':'0.3 mm lateral/top study reserve; PCB seating plane excluded.'}
    try:
        c,m,shapes,cases=geometry()
        sources=['electronics/regen-terminal.yaml','electronics/regen-clamp-interface.json',
                 'electronics/capacitor-saddle.yaml','electronics/regen-nut-pockets.yaml',
                 'electronics/regen-mount.yaml','electronics/regen-clamp.yaml','electronics/motor-capacitor.yaml',
                 'tools/check_regen_terminal.py','tools/regen_terminal_geometry.py',
                 'tools/capacitor_saddle_geometry.py','tools/regen_nut_pocket_geometry.py',
                 'tools/regen_mount_geometry.py','tools/check_neck_collar.py','lab/leg_mechanism.py']
        r['source_bindings']={}
        for s in sources:
            r['source_bindings'][s]=hashlib.sha256(Path(s).read_bytes()).hexdigest()
            snapshot_file(s,out/'source'/s)
        q=ExactQueries()
        internal=[part(k,s) for k,s in shapes.items() if k not in ['original_tray','capacitor_working_space']]
        r['local']={}
        for name,case in cases.items():
            row={'placement':case['placement'],'cad':{}}
            d=out/name;d.mkdir()
            for k in ['body','allowance']:
                s=case[k];assert s.is_valid and len(s.solids())==1
                export_step(s,d/(k+'.step'));export_stl(s,d/(k+'.stl'))
                back=import_step(d/(k+'.step'))
                assert back.is_valid and len(back.solids())==1 and abs(back.volume-s.volume)<1e-5
                row['cad'][k]={'volume_mm3':float(s.volume),'roundtrip':'pass'}
                row[k]=q.collide(s,internal)
            for k in ['pins','solder']:
                row[k]={net:q.collide(s,internal) for net,s in case[k].items()}
            r['local'][name]=row
        save('progress.json',r)
        selected=cases['side-forward']
        additions={'terminal_body':selected['body'],**{'terminal_pin_'+n:s for n,s in selected['pins'].items()},
                   **{'solder_proxy_'+n:s for n,s in selected['solder'].items()}}
        r['robots']={}
        r['dependency_manifests']=[retain_dependency('artifacts/capacitor-saddle-v2'),retain_dependency('artifacts/regen-candidate-v1')]
        for name in ['pip','scout']:
            folder=Path(f'artifacts/{name}-shin-cover-composed-'+('v3' if name=='pip' else 'v2'))
            previous=Path('artifacts/regen-nut-pockets-v3')/name
            prior=verify_report(folder)
            assert prior['assembly_step_roundtrip']=='pass'
            robot=make_robot(RobotConfig.model_validate_json((folder/'resolved-config.json').read_text()))
            frame=robot.forward_kinematics()['torso']
            print(name,'loading retained assembly',flush=True)
            raw=parts_from_step(list(import_step(folder/'assembly.step').children),prior['parts'])
            front=import_step(previous/'front.step')
            targets=[replace(p,shape=transform_shape(front,frame)) if p.name=='torso_front' else p for p in raw]
            targets.append(part('battery_reference_plus_1mm',transform_shape(Pos(8,0,39)*Box(61.04,87.12,24.39),frame)))
            transform=lambda s:transform_shape(s,frame)
            state={'source_assembly_status':prior['status'],'groups_checked':len(raw),
                   'neutral':{k:q.collide(transform(s),targets) for k,s in additions.items()},
                   'allowance':q.collide(transform(selected['allowance']),targets)}
            service=[p for p in targets if p.name not in ['torso_rear','carrier','battery_reference_plus_1mm']]
            local_obstacles=[part(k,transform(s)) for k,s in shapes.items() if k not in ['original_tray','capacitor_working_space']]
            state['driver_probes']={label:{net:q.collide(transform(s),ts+local_obstacles) for net,s in selected['drivers'].items()}
                                    for label,ts in [('installed',targets),('rear_open_carrier_battery_removed',service)]}
            state['removal']={}
            for what in ['capacitor','module_with_terminal']:
                if what=='capacitor':
                    moving={'capacitor':shapes['capacitor']}
                    obstacles=service+[part(k,transform(s)) for k,s in shapes.items() if k not in ['original_tray','capacitor_working_space','capacitor','vent','strap','strap_head']]
                    obstacles += [part(k,transform(s)) for k,s in additions.items()]
                    route=c['parent']  # service path belongs to saddle source, not the terminal drawing
                    import yaml
                    route=yaml.safe_load(Path(c['parent']).read_text())['service']['capacitor_waypoints_mm']
                    prerequisites='Rear cover, carrier, battery and strap removed; capacitor disconnected. Terminal remains installed.'
                else:
                    moving={'module':shapes['module'],**additions}
                    obstacles=service+[part(k,transform(s)) for k,s in shapes.items() if k=='tray' or k.startswith('beam_')]
                    route=m['service_paths']['board_waypoints_mm']
                    prerequisites='Rear cover, carrier, battery, capacitor, strap and board screws removed; all wires disconnected. Terminal remains soldered to board.'
                hits=[];count=0
                for shift in sample_route(route,1):
                    count+=1
                    for k,s in moving.items():
                        found=q.collide(transform(Pos(*shift)*s),obstacles)
                        if found:hits.append({'moving':k,'shift_mm':shift,'collisions':found})
                state['removal'][what]={'status':'fail' if hits else 'pass','samples':count,'collisions':hits,
                                       'waypoints_mm':route,'prerequisites':prerequisites,'continuous_sweep':'not_run'}
            state['neutral_status']='fail' if any(state['neutral'].values()) or r['local']['side-forward']['body'] else 'pass'
            r['robots'][name]=state
            r['dependency_manifests'] += [retain_dependency(folder),retain_dependency(previous)]
            save('progress.json',r)
            print(name,'neutral',state['neutral_status'],'removal',[(k,v['status']) for k,v in state['removal'].items()],flush=True)
        r['execution_status']='pass'
    except Exception as exc:
        r['execution_status']='fail';r['error']=str(exc)
        (out/'error.log').write_text(traceback.format_exc())
        raise
    finally:
        save('report.json',r)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args().output)
