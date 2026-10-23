"""Apply the verified carrier delta to the retained covered robot cost inventory."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
from build123d import import_step
from lab.config import load_config
from lab.robot import make_robot
from lab.leg_mechanism import transform_shape
from tools.check_camera_support import copy_source, sha, write
from tools.check_neck_collar import ExactQueries
from tools.compose_forearm_enclosure import verify_report
from tools.print_cost import PrintAssumptions, calculate


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    baseline=Path('artifacts/covered-production-cost-v3')
    # The global BOM has since gained bench tools. Bind the archived analysis
    # and its outputs; validate its exact covered assembly against mass below.
    old=json.loads((baseline/'report.json').read_text())
    retained_review=json.loads(Path('artifacts/covered-production-review-v2/report.json').read_text())
    if retained_review['source_bindings'][str(baseline/'report.json')]!=sha(baseline/'report.json'):
        raise ValueError('Historical cost report differs from retained review')
    for name,digest in old['output_hashes'].items():
        if sha(baseline/name)!=digest: raise ValueError('Historical cost output changed')
    report=deepcopy(old)
    bom=json.loads((baseline/'prototype-bom.json').read_text())
    bindings={str(baseline/'report.json'):sha(baseline/'report.json')}
    scope='Covered robot plus configured linked carriers: incremental inventory and cost; no new complete STEP or physical qualification.'
    report['scope']=scope; bom['scope']=scope
    report['source_bindings']=bindings
    report['historical_input_drift']=[p for p,digest in old['source_bindings'].items() if not Path(p).is_file() or sha(p)!=digest]
    report['source_hashes']=copy_source(output,{str(p):p for p in [Path(__file__).relative_to(Path.cwd()),Path('tools/print_cost.py'),Path('tools/build_carrier_option.py')]})
    entry=next(e for e in bom['entries'] if e['id']=='print-spacers')
    entry['specification']='24 connected carriers in twelve generic modules plus 16 loose gripper rings; 40 printed solids. Experimental combination with shin covers; physical fit, strength and handling unqualified.'
    entry['dependencies']=['Printed seating/thickness and web-strength comparison','Complete combined-assembly clearance and service checks','Retained whole-robot failures remain open']
    entry['evidence_paths']=[]
    for robot in report['robots']:
        name=robot['name']; folder=Path(f'artifacts/{name}-covered-carriers-v2')
        derived=verify_report(folder)
        if derived['geometry_and_mass_status']!='pass': raise ValueError('Invalid geometry/mass derivation')
        source_mass=Path(f'artifacts/{name}-shin-covered-mass-v1')
        mass=verify_report(source_mass)
        source_path=str(source_mass/'report.json')
        if derived['source_hashes'][source_path]!=sha(source_path): raise ValueError('Wrong source mass')
        assembly=Path(old['study']['assemblies'][name])/'assembly.step'
        if mass['source_hashes'][str(assembly)]!=old['source_bindings'][str(assembly)]: raise ValueError('Cost/mass assembly differs')
        bindings[str(folder/'report.json')]=sha(folder/'report.json')
        bindings[source_path]=sha(source_path)
        cfg=load_config(Path(f'variants/{name}-carriers.yaml')); frames=make_robot(cfg).forward_kinematics()
        ledger=json.loads((source_mass/'part-ledger.json').read_text())
        old_by_name={p['name']:p for p in ledger['parts']}
        inventory=robot['inventory']; group=next(g for g in inventory['groups'] if g['group_id']=='print-spacers')
        replacements={p['name']:p for p in derived['parts']}
        new_solids=[s for s in group['solids'] if s['part_id'] not in replacements]
        if len(new_solids)!=16 or any('gripper' not in s['part_id'] for s in new_solids): raise ValueError('Gripper inventory differs')
        for key,row in replacements.items():
            original=[s for s in group['solids'] if s['part_id']==key]
            if len(original)!=8 or abs(sum(s['volume_mm3'] for s in original)-old_by_name[key]['volume_mm3'])>1e-4: raise ValueError('Source spacer inventory/ledger differs')
            local=import_step(folder/(key+'.step')); world=transform_shape(local,frames[row['owner']])
            if len(world.solids())!=2: raise ValueError('Carrier solid count differs')
            for i,solid in enumerate(world.solids()):
                box=ExactQueries().bounds(solid)
                new_solids.append(dict(part_id=key,solid_index=i,owner=row['owner'],role='spacer',volume_mm3=solid.volume,bounds_mm=[list(box.min),list(box.max)]))
        old_count=inventory['printed_solid_count']; old_volume=inventory['total_volume_mm3']
        group['solids']=new_solids; group['solid_count']=len(new_solids); group['total_volume_mm3']=sum(s['volume_mm3'] for s in new_solids)
        inventory['printed_solid_count']=sum(g['solid_count'] for g in inventory['groups'])
        inventory['total_volume_mm3']=sum(g['total_volume_mm3'] for g in inventory['groups'])
        if old_count-inventory['printed_solid_count']!=72: raise ValueError('Incorrect piece delta')
        delta=inventory['total_volume_mm3']-old_volume
        if abs(delta-sum(p['added_volume_mm3'] for p in derived['parts']))>1e-4: raise ValueError('Volume delta differs')
        robot['carrier_delta']=dict(printed_solids=-72,volume_mm3=delta,mass_profiles=derived['profiles'],source=str(folder/'report.json'))
        robot['print_delta_mm3']=inventory['total_volume_mm3']-robot['baseline_print_volume_mm3']
        for key in ['quantity_per_robot','geometric_solid_count']: entry[key][name]=40
        entry['nominal_cad_volume_mm3'][name]=group['total_volume_mm3'];entry['evidence_paths'].append(str(folder/'report.json'))
        for scenario,sample in zip(robot['cost']['scenarios'],robot['samples'],strict=True):
            updated=calculate(inventory['total_volume_mm3'],PrintAssumptions.model_validate(report['assumptions']),scenario,report['cost_assumptions']['contingency_fraction'],robot['cost']['target_usd'])
            sample.update(updated)
            actuator=next(x for x in scenario['lines'] if x['id']=='actuators')
            contingency=report['cost_assumptions']['contingency_fraction'];target=robot['cost']['target_usd']
            sample['actuator_unit_ceiling_usd']=[(target/(1+contingency)-(sample['example_total_usd'][i]/(1+contingency)-actuator['cost_usd'][i]))/robot['cost']['actuator_count'] for i in range(2)]
        bom['source_reports'][name]['carrier_delta_report']=str(folder/'report.json')
    report['limitations']=[x for x in old['limitations'] if 'Linked spacer' not in x]+[scope,'Other CAD, hardware quantities, handwork and assembly allowances unchanged. No measured labor saving booked.','Existing 3D/complete STEP links show the loose-spacer covered baseline; local replacement files are in the delta reports.']
    bom['limitations'].extend(report['limitations'])
    write(output/'prototype-bom.json',bom)
    report['output_hashes']={'prototype-bom.json':sha(output/'prototype-bom.json')}
    write(output/'report.json',report)
    print(json.dumps([{k:r[k] for k in ['name','carrier_delta','samples']} for r in report['robots']]),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True,type=Path);run(p.parse_args().output)
