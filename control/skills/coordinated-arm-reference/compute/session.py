"""Managed source prep, explicit root start, then unchanged active-consumer wait."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from common import DATA, FORWARD, HERE, ROOT, activation, load, read, ref, verify, write


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--consumer-timeout-s',type=float,default=21600.)
    args=parser.parse_args()
    if not args.prepare:parser.error('Start with --prepare; provider work needs the later exact root marker')
    evidence=Path(os.environ['LAB_EVIDENCE_DIR']).resolve();work=Path(os.environ['LAB_WORK_DIR']).resolve()
    retained=load(FORWARD+'/session.py','coordinated_arm_retained_session')
    began=time.monotonic()
    def command(argv):
        subprocess.run([sys.executable,'-B',*map(str,argv)],cwd=ROOT,check=True)
    try:
        if not 0<args.consumer_timeout_s<=43200:raise ValueError('Finite local owner at most12hours')
        package=work/'preparation'
        command([HERE/'prepare.py','--package','--work',package])
        command([HERE/'prove.py','--archive',package/'input.tar.gz','--work',work/'source-proof'])
        write(evidence/'source-ready.json',dict(state='source_ready_no_provider',pid=os.getpid(),parent_pid=os.getppid(),
            managed_work=work.relative_to(ROOT).as_posix(),managed_evidence=evidence.relative_to(ROOT).as_posix(),
            archive=ref(package/'input.tar.gz'),proof=ref(DATA/'worker-source-proof.json'),worker_plan=ref(DATA/'worker-plan.json'),
            native_counts=dict(controls=0,physics=0,optimizer=0),provider_requests=0))
        print(json.dumps(dict(source_ready=ref(evidence/'source-ready.json'),provider_requests=0)),flush=True)
        marker=evidence/'start-authorized.json'
        while not marker.exists():
            if time.monotonic()-began>=args.consumer_timeout_s:
                retained.interrupted('Root start/local consumers deadline reached',evidence)
            time.sleep(1.)
        start=read(marker)
        if start.get('root_authorized_start')is not True or start.get('exclusive_native_lane_released')is not True:
            raise PermissionError('Root must release this exact native lane')
        release=ROOT/start['release']['path']
        if ref(release)!=start['release']:raise ValueError('Exact root release differs')
        activation(verify(manifest=False),read(release))
        write(DATA/'initial-worker-source-proof.json',read(DATA/'worker-source-proof.json'))
        command([HERE/'prepare.py','--package','--work',package,'--release',release])
        command([HERE/'prove.py','--archive',package/'input.tar.gz','--work',work/'released-source-proof'])
        remaining=args.consumer_timeout_s-(time.monotonic()-began)
        if remaining<=0:retained.interrupted('Local consumer budget exhausted before submission',evidence)
        # The proven retained session performs exactly one submit, then only a
        # local consumer-release wait; all timeout paths interrupt the wrapper.
        sys.argv=[str(HERE/'session.py'),'--release',str(release),'--capsule',str(package/'input.tar.gz'),
            '--consumer-timeout-s',str(remaining)]
        retained.main()
    except BaseException as error:
        if not(evidence/'local-session-interrupted.json').exists():
            retained.interrupted(type(error).__name__+': '+str(error),evidence)
        raise

if __name__ == '__main__':
    main()
