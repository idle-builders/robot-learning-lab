"""Bounded native Isaac Lab installation smoke and robot conversion; no provisioning."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import signal
import shutil
import subprocess
import sys
import time
import traceback

COMMIT='ae37b028ea415c91ea2bc32609efcd759ed2b974'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def command_run(command,cwd,log,timeout):
    """Retain stdout/stderr and stop the whole child group on deadline."""
    start=time.monotonic()
    with log.open('xb') as stream:
        process=subprocess.Popen(command,cwd=cwd,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        timed_out=False
        try:code=process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out=True;os.killpg(process.pid,signal.SIGTERM)
            try:code=process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL);code=process.wait()
    return dict(command=list(map(str,command)),exit_code=code,timed_out=timed_out,elapsed_s=time.monotonic()-start,status='pass' if code==0 and not timed_out else 'fail',log=log.name)


def smoke_commands(cli):
    common=['--task','Isaac-Cartpole','--num_envs','16','--viz','none','physics=newton_mjwarp']
    return [('zero-agent',[str(cli),'zero_agent',*common,'--max_steps','360']),
            ('random-agent',[str(cli),'random_agent',*common,'--max_steps','360']),
            ('train',[str(cli),'train',*common,'--rl_library','rsl_rl','--max_iterations','5','--seed','17','--run_name','robot_lab_worker_smoke'])]


def run(checkout,bundle,expected_sha256,output,timeout=600):
    output=output.resolve();output.mkdir(parents=True,exist_ok=False)
    report=dict(status='partial',scope=__doc__,platform=platform.platform(),stages=[],robot_training='not_run',usd_articulation_parity='not_run',physical_status='not_run')
    source=output/'source';source.mkdir()
    for name in ['isaac_pilot.py','isaac_convert.py','isaac-pilot.json','asset_probe.py']:
        shutil.copy2(Path(__file__).with_name(name),source/name)
    try:
        if platform.system()!='Linux':
            report['reason']='Native pilot requires the selected Linux/NVIDIA worker; no backend substitute launched'
            return report
        checkout=checkout.resolve();bundle=bundle.resolve()
        head=subprocess.check_output(['git','-C',str(checkout),'rev-parse','HEAD'],text=True).strip()
        if head!=COMMIT:raise ValueError('Isaac Lab checkout revision mismatch')
        subprocess.run(['git','-C',str(checkout),'diff','--quiet','HEAD','--'],check=True)
        policy=json.loads(Path(__file__).with_name('isaac-pilot.json').read_text())
        if sha(checkout/'uv.lock')!=policy['upstream_lock_sha256']:raise ValueError('Upstream dependency lock mismatch')
        python=checkout/'.venv/bin/python';cli=checkout/'.venv/bin/isaaclab'
        if not python.is_file() or not cli.is_file():raise ValueError('Provision the pinned upstream environment first')
        report['checkout_commit']=head;report['lock_sha256']=sha(checkout/'uv.lock')
        report['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'],text=True,timeout=15).strip()
        packages=list(policy['package_versions'])
        version_code='import importlib.metadata,json;print(json.dumps({p:importlib.metadata.version(p) for p in '+repr(packages)+'}))'
        versions=json.loads(subprocess.check_output([str(python),'-c',version_code],text=True,timeout=30));report['package_versions']=versions
        for name,expected in policy['package_versions'].items():
            actual=versions[name].split('+')[0]
            if not (actual.startswith(expected[:-1]) if expected.endswith('*') else actual==expected):raise ValueError('Worker package version mismatch: '+name)
        report['bundle_manifest_sha256']=expected_sha256
        # Execute the trusted worker probe, which verifies the bundle before use.
        # Never run code from an input bundle before verifying its identity.
        stages=[('asset-preflight',[str(python),'-I',str(Path(__file__).with_name('asset_probe.py').resolve()),str(bundle),'--expected-sha256',expected_sha256,'--output',str(output/'asset-preflight.json')])]
        stages+=smoke_commands(cli)
        stages+=[('convert',[str(python),str(Path(__file__).with_name('isaac_convert.py').resolve()),'--bundle',str(bundle),'--output',str(output/'converted'),'--viz','none'])]
        for name,command in stages:
            row=command_run(command,output,output/(name+'.log'),timeout);row['stage']=name;report['stages'].append(row)
            (output/'progress.json').write_text(json.dumps(report,indent=2)+'\n')
            if row['status']!='pass':raise RuntimeError('Worker stage did not complete: '+name)
            if name=='asset-preflight' and json.loads((output/'asset-preflight.json').read_text())['status']!='pass':raise RuntimeError('Asset preflight did not pass')
            if name=='train':
                checkpoints=[p for p in (output/'logs').rglob('*.pt') if p.is_file() and p.stat().st_size>0]
                if not checkpoints or 'Training time:' not in (output/'train.log').read_text(errors='replace'):raise RuntimeError('Native training completion/checkpoint missing')
                report['stock_checkpoints']=[str(p.relative_to(output)) for p in checkpoints]
            if name=='convert':
                converted=json.loads((output/'converted/report.json').read_text())
                if converted['status']!='pass' or len(converted['models'])!=2:raise RuntimeError('Both converted assets required')
        report['native_smoke_and_conversion']='pass'
        # Overall remains partial until robot/articulation/controller parity exists.
    except Exception:
        report['status']='fail';report['error']=traceback.format_exc()
    finally:
        report['output_hashes']={str(p.relative_to(output)):sha(p) for p in output.rglob('*') if p.is_file() and p!=output/'report.json'}
        (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['checkout','bundle','output']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--expected-sha256',required=True);p.add_argument('--timeout',type=int,default=600,choices=range(30,1801))
    a=p.parse_args();r=run(a.checkout,a.bundle,a.expected_sha256,a.output,a.timeout);print(json.dumps(r,indent=2));raise SystemExit(1 if r['status']=='fail' else 0 if r.get('native_smoke_and_conversion')=='pass' else 2)
