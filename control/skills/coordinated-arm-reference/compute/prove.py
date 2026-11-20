"""Run the actual extracted worker/entry startup, zero native activity."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
from common import DATA, HERE, ROOT, ref, write


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--archive',type=Path,required=True);parser.add_argument('--work',type=Path,required=True)
    args=parser.parse_args();work=args.work.resolve();work.mkdir(parents=True,exist_ok=False)
    extracted=work/'extracted';extracted.mkdir()
    with tarfile.open(args.archive)as archive:
        for member in archive:
            if not member.isfile()or not(extracted/member.name).resolve().is_relative_to(extracted):raise ValueError('Unsafe source capsule')
        archive.extractall(extracted,filter='data')
    env={k:v for k,v in os.environ.items()if k not in ('PYTHONPATH','PYTHONHOME')}
    worker=extracted/'control/skills/coordinated-arm-reference/compute/worker.py'
    result=subprocess.run([sys.executable,'-I','-B',str(worker),'--source-only-root',str(extracted)],
        cwd=extracted,env=env,capture_output=True,text=True)
    if result.returncode:
        write(DATA/'worker-source-proof-failure.json',dict(archive=ref(args.archive.resolve()),
            command=result.args,exit_code=result.returncode,stdout=result.stdout,stderr=result.stderr,
            scope='Source startup failed before any native/provider execution; original extraction retained'))
        result.check_returncode()
    startup=json.loads(result.stdout)
    if startup['imports_containing_science']or startup['stages']!=1:raise ValueError('Unexpected startup activity')
    if any(not Path(startup[k]).is_relative_to(extracted)for k in ('common_source','entry_source','reference_source')):
        raise ValueError('Startup used preloaded checkout')
    receipt=dict(schema='cove50.coordinated-arm-reference-extracted-startup-proof.v1',
        archive=ref(args.archive.resolve()),worker_source=ref(HERE/'worker.py'),source=ref(Path(__file__)),startup=startup,
        command=[sys.executable,'-I','-B','<extracted>/control/skills/coordinated-arm-reference/compute/worker.py','--source-only-root','<extracted>'],
        retained_environment_proof=ref(ROOT/'data/forward-stop-learning-20261030/source-proof.json'),
        scope='Actual isolated extracted worker imports its own common, guarded native entry and reference, constructs exact one-child command and loads unchanged helper source. Existing forward wrapper interruption path reused unchanged. No PT deserialization/NN/models/FK/physics/provider.',
        disposable_extraction_removed=True)
    write(DATA/'worker-source-proof.json',receipt)
    shutil.rmtree(work)
    print(json.dumps(dict(proof=ref(DATA/'worker-source-proof.json'),stages=startup['stages'])))


if __name__=='__main__':main()
