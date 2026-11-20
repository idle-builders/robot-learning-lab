"""Exact extracted startup and real interrupted-wrapper path, zero native work."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
from common import DATA, HERE, ROOT, ref, write


def main(archive, work):
    work.mkdir(parents=True,exist_ok=False)
    extracted=work/'extracted';extracted.mkdir()
    with tarfile.open(archive)as tf:
        for member in tf:
            if not member.isfile()or not(extracted/member.name).resolve().is_relative_to(extracted):raise ValueError('Unsafe proof capsule')
        tf.extractall(extracted,filter='data')
    env=dict(os.environ);env.pop('PYTHONPATH',None);env.pop('PYTHONHOME',None)
    # -I removes checkout, script-dir and user-site imports. The actual extracted
    # worker loads its own common/native/schedule source through explicit routes.
    worker=extracted/'control/skills/forward-stop-learning/compute/worker.py'
    result=subprocess.run([sys.executable,'-I','-B',str(worker),'--source-only-root',str(extracted)],cwd=extracted,env=env,capture_output=True,text=True,check=True)
    startup=json.loads(result.stdout)
    if startup['imports_containing_science']or startup['stages']!=21:raise ValueError('Unexpected extracted startup scope')
    plan=json.loads((extracted/'control/skills/forward-stop-learning/setup/worker-plan.json').read_text())
    fixture=extracted/plan['dictionary_cache_fixture']['path']
    fixture_output=work/'dictionary-cache.json'
    subprocess.run([str(ROOT/'.venv-control/bin/python'),'-B',str(fixture),'--schedule',
        str(extracted/'control/skills/forward-stop-learning/schedule.py'),'--out',str(fixture_output)],
        cwd=extracted,env=env,capture_output=True,text=True,check=True)
    dictionary_fixture=json.loads(fixture_output.read_text())
    fixture=work/'lifecycle';fixture.mkdir();probe=fixture/'probe.py'
    code=r'''
import json,sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from lab import storage_lifecycle as storage
root=Path(sys.argv[2]);storage.ROOT=root;storage.WORK=root/'.cache/work';storage.RECORDS=root/'data/storage-runs'
(root/'required-input.bin').write_bytes(b'active-input')
def forbidden(*args,**kwargs):raise AssertionError('Seal/sync/clean reached before consumers complete')
storage.seal=storage.sync_one=storage.clean=forbidden
try:
 storage.run('interruption-proof',[sys.executable,'-B',sys.argv[3],'--release',str(root/'unused-release.json'),'--consumer-timeout-s','.05','--lifecycle-probe'])
except KeyboardInterrupt:
 records=[json.loads(p.read_text())for p in storage.RECORDS.glob('*.json')]
 print(json.dumps(dict(record_state=records[0]['state'],active_input_present=(root/'required-input.bin').read_bytes()==b'active-input',work_preserved=Path(root/records[0]['work']).exists(),normal_seal_reached=False)))
else:raise AssertionError('Expected actual wrapper interruption path')
'''
    probe.write_text(code)
    result=subprocess.run([str(ROOT/'.venv/bin/python'),'-B',str(probe),str(ROOT),str(fixture),str(HERE/'session.py')],cwd=fixture,env=env,capture_output=True,text=True,timeout=15,check=True)
    lifecycle=json.loads(result.stdout)
    if lifecycle!=dict(record_state='interrupted',active_input_present=True,work_preserved=True,normal_seal_reached=False):raise ValueError('Actual lifecycle probe differs')
    receipt=dict(schema='cove50.forward-stop-source-proof.v1',archive=ref(archive),source=ref(Path(__file__)),
        worker_source=ref(HERE/'worker.py'),session_source=ref(HERE/'session.py'),startup=startup,lifecycle=lifecycle,
        actual_extracted_dictionary_cache_fixture=dictionary_fixture,
        extraction_command=[sys.executable,'-I','-B','<fresh-extracted>/control/skills/forward-stop-learning/compute/worker.py','--source-only-root','<fresh-extracted>'],
        actual_wrapper=ref(ROOT/'infra/storage/storage_lifecycle.py'),scope='Actual extracted worker/native-entry/schedule startup, separate actual extracted Schedule fixture on real CPU tensors, and wrapper subprocess interruption under temporary paths. No NN/checkpoint/model/physics/provider/private-storage calls in proof.',
        disposable_extraction_and_probe_removed=True)
    write(DATA/'source-proof.json',receipt)
    shutil.rmtree(work)
    print(json.dumps(dict(proof=ref(DATA/'source-proof.json'),startup_stages=startup['stages'],lifecycle=lifecycle)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True);p.add_argument('--work',type=Path,required=True)
    a=p.parse_args();main(a.archive.resolve(),a.work.resolve())
