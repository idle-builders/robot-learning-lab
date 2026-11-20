"""Attached managed owner: one submission, then finite LOCAL consumer wait."""
import argparse
import datetime
import json
import os
from pathlib import Path
import signal
import time
from common import ROOT, ref, write


def interrupted(reason, evidence):
    write(evidence/'local-session-interrupted.json',dict(reason=reason,at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        inputs_remain_active=True,provider_requests_by_wait=0,storage_wrapper_parent=os.getppid()))
    # A normal nonzero child exit would make the existing wrapper seal/offload.
    # Interrupt its actual blocking subprocess.run instead. No lifecycle edits.
    os.kill(os.getppid(),signal.SIGINT)
    raise KeyboardInterrupt(reason)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--release',type=Path,required=True)
    parser.add_argument('--capsule',type=Path)
    parser.add_argument('--consumer-timeout-s',type=float,default=21600.)
    parser.add_argument('--lifecycle-probe',action='store_true');args=parser.parse_args()
    evidence=Path(os.environ['LAB_EVIDENCE_DIR']).resolve();work=Path(os.environ['LAB_WORK_DIR']).resolve()
    started=time.monotonic()
    try:
        if not 0<args.consumer_timeout_s<=43200:raise ValueError('Finite consumer wait at most12hours')
        if not args.lifecycle_probe:
            from launch import submit
            if args.capsule is None:raise ValueError('Exact reviewed capsule path required')
            os.link(args.capsule.resolve(),work/'input.tar.gz')
            write(evidence/'local-session.json',dict(state='attached',pid=os.getpid(),parent_pid=os.getppid(),
                release=ref(args.release),work=str(work),evidence=str(evidence),consumer_timeout_s=args.consumer_timeout_s,
                wait_scope='Local consumers-complete file only; no provider/status/storage polling in this wait'))
            print(json.dumps(submit(evidence,work,args.release.resolve())),flush=True)
        marker=evidence/'consumers-complete.json'
        while not marker.exists():
            if time.monotonic()-started>=args.consumer_timeout_s:interrupted('Local consumers-complete deadline reached',evidence)
            time.sleep(min(1.,max(.001,args.consumer_timeout_s-(time.monotonic()-started))))
        closed=json.loads(marker.read_text())
        if closed.get('consumers_complete')is not True or closed.get('root_authorized_closure')is not True:
            raise ValueError('Exact root consumers-complete release required')
        if not args.lifecycle_probe and closed.get('native_index')!=ref(evidence/'closed-native-index.json'):
            raise ValueError('Consumer release must bind the actual closed native index')
        write(evidence/'local-session-completed.json',dict(consumers_release=ref(marker),elapsed_s=time.monotonic()-started,
            wrapper_may_now_seal=True,provider_requests_by_wait=0))
    except BaseException as exc:
        if not (evidence/'local-session-interrupted.json').exists():interrupted(type(exc).__name__+': '+str(exc),evidence)
        raise


if __name__=='__main__':main()
