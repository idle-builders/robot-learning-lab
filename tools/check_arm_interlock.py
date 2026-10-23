"""Retain native atopile pin checks and ngspice logic scenarios for the arm candidate.

XSPICE primitives express a functional hypothesis, not vendor electrical models.
The rail-valid input is injected: analog supply ramps and voltage margins are separate.
"""
import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import shutil
import subprocess
import traceback

import numpy as np

from lab.artifact_store import snapshot_file, retain_dependency
from tools.build_carrier import generate_parts
from tools.check_motor_disconnect import read_pins

ROOT = Path(__file__).absolute().parents[1]
SOURCE = ROOT/'electronics/arm-interlock'
EXPECTED = {
    'U1': {'1':'GND','2':'ARM_BUTTON_N','3':'ARM_RELEASED','4':'V3'},
    'U2': {'1':'RECOVERED','2':'GND','3':'STOP_CHAIN_OK','4':'CT','5':'V3','6':'V3'},
    'U3': {'1':'ARM_PRESS','2':'V3','3':None,'4':'GND','5':'ARM_AUTHORITY','6':'RECOVERED','7':'V3','8':'V3'},
    'U4': {'1':None,'2':'ARM_RELEASED','3':'GND','4':'ARM_PRESS','5':'V3'},
    'U5': {'1':'ARM_AUTHORITY','2':'GND','3':'RECOVERED','4':'MOTOR_ENABLE','5':'V3','6':'POWER_REQUEST'},
    'R1': {'1':'V3','2':'CT'}, 'R2': {'1':'V3','2':'RECOVERED'},
    'R3': {'1':'STOP_CHAIN_OK','2':'GND'},
    'R4': {'1':'POWER_REQUEST','2':'GND'},
    'R5': {'1':'MOTOR_ENABLE','2':'GND'},
    'R6': {'1':'V3','2':'ARM_BUTTON_N'},
    **{f'C{i}':{'1':'V3','2':'GND'} for i in range(1,6)},
}


def check_pins(pins):
    errors = [ref for ref in set(pins)|set(EXPECTED) if pins.get(ref) != EXPECTED.get(ref)]
    return {'status':'fail' if errors else 'pass','mismatched_components':sorted(errors)}


def scenarios():
    # Checks are intervals, not isolated samples. Each row: start, end, authority, enable.
    base = dict(supply=[(0,0),(.01,1)], stop=[(0,1)], button=[(0,1)], request=[(0,1)])
    definitions = [
        ('cold_no_press', {}, [(0.001,1.99,0,0)]),
        ('arm_before_request', dict(button=[(0,1),(.6,0),(.75,1)],request=[(0,0),(.95,1)]),
         [(.001,.59,0,0),(.69,.94,1,0),(.96,1.99,1,1)]),
        ('held_at_boot', dict(button=[(0,0),(.6,1),(.9,0)]),
         [(.001,.89,0,0),(1,1.99,1,1)]),
        ('stop_recovery', dict(button=[(0,1),(.6,0),(.75,1),(1.6,0)],stop=[(0,1),(.85,0),(.9,1)]),
         [(.69,.84,1,1),(.8501,1.59,0,0),(1.7,1.99,1,1)]),
        ('held_through_stop', dict(button=[(0,1),(.6,0),(1.5,1),(1.7,0)],stop=[(0,1),(.85,0),(.9,1)]),
         [(.69,.84,1,1),(.8501,1.69,0,0),(1.8,1.99,1,1)]),
        ('press_pending_during_stop', dict(button=[(0,1),(.86,0),(1.4,1),(1.6,0)],stop=[(0,1),(.85,0),(.9,1)]),
         [(.001,1.59,0,0),(1.7,1.99,1,1)]),
        ('button_bounce', dict(button=[(0,1),(.6,0),(.601,1),(.603,0),(.605,1),(.9,0)]),
         [(.001,.89,0,0),(1,1.99,1,1)]),
        ('rail_fault_recovery', dict(button=[(0,1),(.6,0),(.75,1),(1.6,0)],supply=[(0,0),(.01,1),(.85,0),(.9,1)]),
         [(.69,.84,1,1),(.8501,1.59,0,0),(1.7,1.99,1,1)]),
        ('request_removed', dict(button=[(0,1),(.6,0),(.75,1)],request=[(0,1),(.85,0),(.95,1)]),
         [(.69,.84,1,1),(.8501,.94,1,0),(.96,1.99,1,1)]),
        ('stop_open', dict(stop=[(0,0)],button=[(0,1),(.6,0),(.75,1)]),[(.001,1.99,0,0)]),
        ('stop_bounce', dict(button=[(0,1),(.6,0),(.75,1),(1.6,0)],stop=[(0,1),(.85,0),(.9,1),(.901,0),(.903,1)]),
         [(.69,.84,1,1),(.8501,1.59,0,0),(1.7,1.99,1,1)]),
    ]
    return [dict(name=name,**(base|overrides),checks=checks) for name,overrides,checks in definitions]


def pwl(points):
    result = [0, points[0][1]*3.3]
    previous = points[0][1]
    for time, level in points[1:]:
        result += [time, previous*3.3, time+1e-7, level*3.3]
        previous = level
    return ' '.join(f'{v:.10g}' for v in result)


def deck(case, debounce, recovery, release_clock=False):
    template = (SOURCE/'logic.cir.in').read_text()
    sources = '\n'.join(f'V{name} {name} 0 PWL({pwl(case[name])})' for name in ['supply','stop','button','request'])
    return template.replace('@SOURCES@',sources).replace('@DEBOUNCE@',str(debounce)).replace('@RECOVERY@',str(recovery)).replace('@CLOCK@','released' if release_clock else 'pressed')


def evaluate(data, case):
    if data.ndim != 2 or data.shape[1] != 9 or len(data)<100 or not np.isfinite(data).all() or data[-1,0]<2:
        raise ValueError('missing, malformed or incomplete native trace')
    if np.any(np.diff(data[:,0])<0):
        raise ValueError('native time reversed')
    checks=[]
    for start,end,armed,enable in case['checks']:
        section=data[(data[:,0]>=start)&(data[:,0]<=end)]
        good=len(section)>0 and np.all(np.abs(section[:,1]-armed*3.3)<.05) and np.all(np.abs(section[:,2]-enable*3.3)<.05)
        checks.append(dict(start_s=start,end_s=end,authority=armed,enable=enable,samples=len(section),status='pass' if good else 'fail'))
    return {'status':'pass' if all(c['status']=='pass' for c in checks) else 'fail','checks':checks}


def simulate(folder, case, debounce, recovery, release_clock=False):
    folder.mkdir(parents=True,exist_ok=False)
    (folder/'circuit.cir').write_text(deck(case,debounce,recovery,release_clock))
    result=subprocess.run(['/opt/homebrew/bin/ngspice','-b','circuit.cir'],cwd=folder,capture_output=True,text=True,timeout=20)
    (folder/'native.log').write_text(result.stdout+result.stderr)
    result.check_returncode()
    data=np.loadtxt(folder/'trace.dat',skiprows=1)
    return dict(name=case['name'],debounce_s=debounce,recovery_s=recovery,**evaluate(data,case),trace=str(folder/'trace.dat'))


def run(out):
    out=out.absolute();out.mkdir(parents=True,exist_ok=False)
    report=dict(status='fail',execution_status='fail',physical_status='not_run',analog_power_ramps='not_run',production_readiness='hold',model='ngspice XSPICE functional primitives; no vendor electrical models',steps=[])
    try:
        report['source_hashes']={}
        for p in [*SOURCE.rglob('*'),Path(__file__),ROOT/'tools/check_motor_disconnect.py',ROOT/'tools/build_carrier.py',ROOT/'lab/artifact_store.py']:
            if not p.is_file():continue
            rel=p.absolute().relative_to(ROOT)
            report['source_hashes'][str(rel)]=hashlib.sha256(p.read_bytes()).hexdigest()
            snapshot_file(p,out/'source'/rel)
        compiled=out/'compiled';shutil.copytree(SOURCE,compiled)
        generate_parts(compiled)
        cmd=[str(ROOT/'.venv-ato/bin/ato'),'build',str(compiled)]
        result=subprocess.run(cmd,env=dict(os.environ,FBRK_TELEMETRY='0',ATO_NON_INTERACTIVE='1'),capture_output=True,text=True,timeout=180)
        (out/'atopile.log').write_text(result.stdout+result.stderr)
        report['steps'].append(dict(command=cmd,exit_code=result.returncode))
        result.check_returncode()
        parts=json.loads((compiled/'parts.json').read_text());components=json.loads((compiled/'components.json').read_text())
        pins=read_pins(compiled/'build/builds/default/pinout',components)
        (out/'compiled-pad-nets.json').write_text(json.dumps(pins,indent=2)+'\n')
        report['pin_check']=check_pins(pins)
        if report['pin_check']['status']!='pass':raise ValueError('compiled circuit differs; refused simulation')
        report['cases']=[]
        for db,rec in itertools.product([.02,.05,.08],[.18,.3,.42]):
            for case in scenarios():
                folder=out/'logic'/f"{case['name']}-d{int(db*1000)}-r{int(rec*1000)}"
                item=simulate(folder,case,db,rec);item['trace']=str(Path(item['trace']).relative_to(out))
                report['cases'].append(item)
        report['negative_control']=simulate(out/'negative-release-clock',next(c for c in scenarios() if c['name']=='held_at_boot'),.08,.18,True)
        report['negative_control']['trace']=str(Path(report['negative_control']['trace']).relative_to(out))
        report['timing_separation_s']=.18-.08
        # Conservative static compatibility calculation, not a power-ramp proof.
        report['static_margin']={'min_supervisor_falling_threshold_v':3.07*(1-.015),
            'debouncer_to_inverter_high_margin_v':3.07*(1-.015)-1.0-2.0,
            'note':'Only 24 mV before supervisor assertion, excluding transient droop; insufficient to qualify actual wiring/noise. Needs bench or revised parts.',
            'status':'not_run'}
        report['component_bom']=[dict(mpn=p['mpn'],manufacturer=p['manufacturer'],references=[r for r,k in components.items() if k==kind],quantity_per_candidate_board=sum(k==kind for k in components.values()),installed_quantity=0,procurement_status='hold',unit_price_usd_1000_robots=None,unit_price_usd_5000_robots=None) for kind,p in parts.items()]
        report['dependency_manifests']={'firmware_sequence':retain_dependency('artifacts/target-power-sequence-v1'),'motor_disconnect':retain_dependency('artifacts/motor-disconnect-v2')}
        report['execution_status']='pass'
        report['status']='pass' if all(c['status']=='pass' for c in report['cases']) and report['negative_control']['status']=='fail' else 'fail'
    except Exception:
        report['traceback']=traceback.format_exc()
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    (out/'report.sha256').write_text(hashlib.sha256((out/'report.json').read_bytes()).hexdigest()+'\n')
    print(json.dumps({k:report[k] for k in ['status','execution_status','physical_status']}))
    if report['status']!='pass':raise RuntimeError(report.get('traceback','Circuit scenarios failed; retained report'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    run(p.parse_args().output)
