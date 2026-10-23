"""Screen uninstalled battery envelopes against retained covered robot CAD."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import traceback
import numpy as np
import yaml
from build123d import Box, Compound, Pos, export_step, import_step
from lab.config import RobotConfig
from lab.robot import make_robot
from lab.power_pack import torso_containment
from lab.leg_mechanism import Part, render_core, transform_shape
from tools.check_arm_episode import parts_from_step
from tools.check_camera_support import copy_source, sha, write
from tools.check_neck_collar import ExactQueries
from tools.compose_forearm_enclosure import verify_report

def run(output, study):
    output.mkdir(parents=True,exist_ok=False)
    report=dict(status='partial',scope=__doc__,physical_status='not_run',robots={},source_bindings={})
    paths=[study,Path(__file__).resolve().relative_to(Path.cwd()),Path('lab/power_pack.py'),Path('lab/robot.py'),Path('lab/mechanical.py'),Path('tools/check_neck_collar.py'),Path('tools/check_arm_episode.py'),Path('lab/leg_mechanism.py')]
    report['source_hashes']=copy_source(output,{str(p):p for p in paths})
    try:
        cfg=yaml.safe_load(study.read_text());center=cfg['position_torso_mm'];pad=cfg['clearance_allowance_mm']
        for name,version in [('pip','v3'),('scout','v2')]:
            folder=Path(f'artifacts/{name}-shin-cover-composed-{version}')
            print(name,'loading retained geometry',flush=True)
            retained=verify_report(folder)
            if retained['assembly_step_roundtrip']!='pass':raise ValueError('STEP roundtrip missing')
            for f in ['assembly.step','report.json','resolved-config.json']:
                report['source_bindings'][str(folder/f)]=sha(folder/f)
            robot=make_robot(RobotConfig.model_validate_json((folder/'resolved-config.json').read_text()))
            raw=parts_from_step(list(import_step(folder/'assembly.step').children),retained['parts'])
            frame=robot.forward_kinematics()['torso'];inverse=np.linalg.inv(frame)
            queries=ExactQueries();rows=[]
            for case in cfg['cases']:
                local=Pos(*center)*Box(*case['dimensions_torso_mm'])
                world=transform_shape(local,frame)
                roomy=transform_shape(Pos(*center)*Box(*(v+2*pad for v in case['dimensions_torso_mm'])),frame)
                containment=torso_containment(robot,local)
                collisions=queries.collide(world,raw);allowance_hits=queries.collide(roomy,raw)
                row=dict(case=case['id'],dimensions_torso_mm=case['dimensions_torso_mm'],position_torso_mm=center,containment=containment,collisions=collisions,clearance_allowance_mm=pad,clearance_allowance_collisions=allowance_hits,status='pass' if containment['status']=='pass' and not collisions else 'fail')
                rows.append(row);print(name,case['id'],row['status'],collisions,flush=True)
                dest=output/name/case['id'];dest.mkdir(parents=True)
                export_step(local,dest/'battery-envelope.step')
                visible=[replace(p,shape=transform_shape(p.shape,inverse),color=(.6,.65,.68,.16) if p.role=='shell' else (.35,.4,.44,1)) for p in raw if p.owner=='torso']
                visible.append(Part('battery_envelope','torso',local,'allocation','battery',(0.94,.6,.13,.85)))
                render_core(visible,{p.name:p.shape for p in visible},dest,robot,azimuth=135,elevation=-20)
                write(output/'progress.json',dict(robot=name,cases=rows))
            far=Pos(1000,0,39)*Box(64,74,24)
            far_check=torso_containment(robot,far)
            obstruction=queries.collide(world,[Part('injected_pack_obstruction','torso',world,'control','battery',(1,0,0,1))])
            if far_check['status']!='fail' or not obstruction:raise ValueError('Negative control missed')
            report['robots'][name]=dict(source_assembly=str(folder),source_assembly_status=retained['status'],assembly_groups_checked=len(raw),cases=rows,negative_controls={'outside_body':far_check,'injected_obstruction':obstruction})
        report['execution_status']='pass'
        report['limitations']=['Boxes represent body envelopes only; no connectors, leads, latch, insulation, retention or removal path.','No electrical qualification, battery purchase, mass model or robot variant promotion.','Retained covered assembly has loose spacer rings; later carrier overlays and complete trajectories are not qualified by this neutral screen.','Earlier assembly failures remain unresolved. A 1 mm inflated box is an explicit space allowance, not a manufacturing tolerance.']
    except Exception as exc:
        report['execution_status']='fail';report['error']=str(exc);(output/'error.log').write_text(traceback.format_exc());raise
    finally:
        report['output_hashes']={str(p.relative_to(output)):sha(p) for p in output.rglob('*') if p.is_file() and 'source' not in p.relative_to(output).parts and p.name!='report.json'}
        write(output/'report.json',report)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--study',type=Path,default=Path('experiments/battery-packaging-v1.yaml'));run(**vars(p.parse_args()))
