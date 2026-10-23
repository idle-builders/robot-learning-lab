"""Check configured carriers against the added/replaced shin parts at retained poses."""
import argparse
from dataclasses import replace
from pathlib import Path
from math import sqrt
from build123d import import_step
from lab.config import load_config
from lab.robot import make_robot
from lab.leg_mechanism import Part, transform_shape
from tools.check_camera_support import copy_source, sha, write
from tools.compose_forearm_enclosure import verify_report
from tools.check_neck_collar import ExactQueries
from tools.study_shin_cover import hits, POSES
from tools.qualify_shin_cover import POSES as EXTRA_POSES


def run(output):
    output.mkdir(parents=True,exist_ok=False)
    files=[Path(__file__).relative_to(Path.cwd()),Path('tools/study_shin_cover.py'),Path('tools/qualify_shin_cover.py'),Path('tools/check_neck_collar.py'),*Path('lab').glob('*.py')]
    report=dict(status='running',scope=__doc__,physical_status='not_run',source_hashes=copy_source(output,{str(p):p for p in files}),source_bindings={},robots={})
    study=Path('artifacts/shin-cover-study-v1');cover=verify_report(study)
    cost=verify_report(Path('artifacts/covered-carriers-cost-v2'))
    def bind(path):
        report['source_bindings'][str(path)]=sha(path)
        return path
    bind(study/'report.json');bind(Path('artifacts/covered-carriers-cost-v2/report.json'))
    for name in ['pip','scout']:
        folder=Path(f'artifacts/{name}-covered-carriers-v2');derived=verify_report(folder);bind(folder/'report.json')
        if cost['source_bindings'][str(folder/'report.json')]!=sha(folder/'report.json'):raise ValueError('Cost refers to different carriers')
        assembly=Path(cost['study']['assemblies'][name]);assembled=verify_report(assembly)
        if assembled['source_bindings'][str(study/'report.json')]!=sha(study/'report.json'):raise ValueError('Assembly uses different shin parts')
        bind(assembly/'report.json')
        variant=bind(Path(f'variants/{name}-carriers.yaml'));robot=make_robot(load_config(variant))
        carriers=[Part(p['name'],p['owner'],import_step(bind(folder/(p['name']+'.step'))),'spacer',p['owner'],(.2,.7,.5)) for p in derived['parts']]
        targets=[Part(p['name'],p['owner'],import_step(bind(study/p['step'])),p['role'],p['owner'],tuple(p['color'])) for p in cover['robots'][name]['parts']]
        assert len(carriers)==12 and len(targets)==20
        rows=[]
        for label,pose in [*POSES,*EXTRA_POSES]:
            frames=robot.forward_kinematics(pose);query=ExactQueries()
            posed=[replace(p,shape=transform_shape(p.shape,frames[p.owner])) for p in targets]
            checks=[]
            for part in carriers:
                shape=transform_shape(part.shape,frames[part.owner])
                collisions,exact=hits(shape,posed,query)
                a=query.bounds(shape)
                gaps=[]
                for target in posed:
                    b=query.bounds(target.shape)
                    gaps.append(sqrt(sum(max(0,lo_a-hi_b,lo_b-hi_a)**2 for lo_a,hi_a,lo_b,hi_b in zip(a.min,a.max,b.min,b.max,strict=True))))
                checks.append(dict(carrier=part.name,possible_pairs=len(posed),exact_queries=exact,hits=collisions,bound_clearance_lower_mm=min(gaps)))
            rows.append(dict(pose=label,joints_rad=pose,checks=checks,status='fail' if any(c['hits'] for c in checks) else 'pass'))
            report['robots'][name]=rows;write(output/'progress.json',report)
        print(name,[(r['pose'],r['status']) for r in rows],flush=True)
    report['status']='fail' if any(r['status']=='fail' for rows in report['robots'].values() for r in rows) else 'pass'
    report['limitations']=['Only twelve changed carrier groups versus twenty added/replaced shin-cover groups per robot. Other whole-robot pairs and earlier failures are not requalified.','Ten fixed leg poses; no continuous sweep, moving arm posture, cable route, removal path, tool access or tolerance/strength qualification.','Containing surface bounds and exact overlap gate 0.02 mm3 reuse the retained shin-cover checker. Nominal overlap absence does not establish clearance margin.']
    report['output_hashes']={'progress.json':sha(output/'progress.json')};write(output/'report.json',report)
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);r=run(p.parse_args().output);raise SystemExit(int(r['status']!='pass'))
