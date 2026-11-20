"""One attached source owner, exact root start marker, then retained consumer wait."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from common import DATA, HERE, ROOT, activation, load, read, ref, verify, write


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--release',type=Path)
    parser.add_argument('--consumer-timeout-s',type=float,default=21600.)
    parser.add_argument('--lifecycle-probe',action='store_true')
    args=parser.parse_args()
    retained=load('control/skills/forward-stop-learning/compute/session.py','heading_retained_session')
    if args.lifecycle_probe:
        retained.main()
        return
    if not args.prepare:
        parser.error('Source preparation first; later provider start requires an exact root marker')
    evidence=Path(os.environ['LAB_EVIDENCE_DIR']).resolve();work=Path(os.environ['LAB_WORK_DIR']).resolve()
    began=time.monotonic()
    try:
        if not 0<args.consumer_timeout_s<=43200:
            raise ValueError('Finite owner timeout at most12hours')
        owned=evidence/'source/reference-baselines'
        subprocess.run([sys.executable,'-B',str(HERE/'own_inputs.py'),'--out',str(owned)],cwd=ROOT,check=True)
        record=dict(state='source_inputs_owned_no_provider',pid=os.getpid(),parent_pid=os.getppid(),
            managed_work=work.relative_to(ROOT).as_posix(),managed_evidence=evidence.relative_to(ROOT).as_posix(),
            baseline_ownership=ref(owned/'ownership.json'),provider_requests=0,consumer_timeout_s=args.consumer_timeout_s)
        write(evidence/'source-inputs-ready.json',record)
        write(DATA/'execution/preparation-active.json',dict(run_id=os.environ['LAB_RUN_ID'],**record))
        print(json.dumps(dict(source_inputs_ready=ref(evidence/'source-inputs-ready.json'),provider_requests=0)),flush=True)
        marker=evidence/'start-authorized.json'
        while not marker.exists():
            if time.monotonic()-began>=args.consumer_timeout_s:
                retained.interrupted('Root start/local consumers deadline reached',evidence)
            time.sleep(1.)
        start=read(marker)
        if start.get('root_authorized_start')is not True or start.get('exclusive_native_lane_released')is not True:
            raise PermissionError('Root must release the exact native lane')
        release=ROOT/start['release']['path'];archive=ROOT/start['archive']['path'];proof=ROOT/start['proof']['path']
        if ref(release)!=start['release'] or ref(archive)!=start['archive'] or ref(proof)!=start['proof']:
            raise ValueError('Exact released capsule/proof differs')
        if read(proof)['archive']!=start['archive']:
            raise ValueError('Extracted proof belongs to another capsule')
        activation(verify(manifest=False),read(release))
        remaining=args.consumer_timeout_s-(time.monotonic()-began)
        if remaining<=0:
            retained.interrupted('Consumer budget exhausted before submission',evidence)
        sys.argv=[str(HERE/'session.py'),'--release',str(release),'--capsule',str(archive),
            '--consumer-timeout-s',str(remaining)]
        retained.main()
    except BaseException as error:
        if not(evidence/'local-session-interrupted.json').exists():
            retained.interrupted(type(error).__name__+': '+str(error),evidence)
        raise


if __name__=='__main__':
    main()
