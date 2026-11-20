"""Actual isolated extracted worker→entry source startup, no scientific imports."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
from common import DATA,HERE,ROOT,ref,write


def main():
    p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True);p.add_argument('--work',type=Path,required=True)
    args=p.parse_args();work=args.work.resolve();work.mkdir(parents=True,exist_ok=False)
    extracted=work/'extracted';extracted.mkdir()
    with tarfile.open(args.archive)as packed:
        for member in packed:
            if not member.isfile()or not(extracted/member.name).resolve().is_relative_to(extracted):raise ValueError('Unsafe source capsule')
        packed.extractall(extracted,filter='data')
    worker=extracted/'control/skills/coordinated-arm-feedback/compute/worker.py'
    result=subprocess.run([sys.executable,'-I','-B',str(worker),'--source-only-root',str(extracted)],cwd=extracted,
        env={k:v for k,v in os.environ.items()if k not in ('PYTHONPATH','PYTHONHOME')},capture_output=True,text=True)
    if result.returncode:
        write(DATA/'worker-source-proof-failure.json',dict(archive=ref(args.archive.resolve()),command=result.args,
            returncode=result.returncode,stdout=result.stdout,stderr=result.stderr,scope='Source-only startup; no native/model/provider.'))
        result.check_returncode()
    startup=json.loads(result.stdout)
    if startup['imports_containing_science']or startup['stages']!=9:raise ValueError('Unexpected actual startup')
    if any(not Path(startup[k]).is_relative_to(extracted)for k in ('common_source','entry_source','interface_source')):
        raise ValueError('Startup used preloaded checkout')
    receipt=dict(schema='cove50.coordinated-arm-feedback-extracted-startup-proof.v1',archive=ref(args.archive.resolve()),
        worker=ref(HERE/'worker.py'),source=ref(Path(__file__)),startup=startup,
        scope='Actual isolated extracted composed worker/entry/importclosure and exact9childcommands only;0PT/NN/native/models/physics/provider.',
        disposable_extraction_removed=True)
    write(DATA/'worker-source-proof.json',receipt);shutil.rmtree(work)
    print(json.dumps(dict(proof=ref(DATA/'worker-source-proof.json'),stages=startup['stages'])))


if __name__=='__main__':main()
