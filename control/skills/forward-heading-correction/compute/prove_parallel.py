"""Finite stdlib exercise of the exact paired parent used in the remote worker."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import types
sys.path.insert(0, str(Path(__file__).resolve().parent))
from parallel_stage import PAIR_SOURCE


def main(out):
    root=out.parent/'parallel-fixture-work'
    root.mkdir(parents=True,exist_ok=False)
    results=[]
    try:
        for kind in ('both_complete','one_nonzero','one_timeout','exit_at_signal','first_log_close_error'):
            folder=root/kind;folder.mkdir();evidence=folder/'evidence';evidence.mkdir()
            cases=[dict(name=n,mode='replay')for n in ('left','right')]
            children=[dict(name=n,process_s=.1 if kind=='one_timeout'and n=='left'else 3,join_s=1,pack_s=1)for n in ('left','right')]
            published=[];completed=[];pids=[]
            class Common:
                def command(self,case,checkpoint,evidence,work):
                    delay=2 if kind=='one_timeout'and case['name']=='left'else .22
                    code=7 if kind=='one_nonzero'and case['name']=='left'else 0
                    return [sys.executable,'-I','-B','-c',f'import time;time.sleep({delay});raise SystemExit({code})']
                def completed(self,case,path):completed.append(case['name'])
            report=dict(stages=[],case_disposition={})
            processes=[];race=[]
            def popen(*args,**kwargs):
                process=subprocess.Popen(*args,**kwargs);processes.append(process)
                if kind in ('exit_at_signal','first_log_close_error')and len(processes)==1:
                    original=process.poll;calls=[0]
                    def poll():
                        calls[0]+=1
                        if calls[0]==1:raise RuntimeError('Injected parent unwind with second child live')
                        if calls[0]==2:return None
                        return original()
                    process.poll=poll
                return process
            def killpg(pid,signum):
                if kind=='exit_at_signal'and pid==processes[0].pid:
                    processes[0].wait();race.append('first exited before signal');raise ProcessLookupError()
                return os.killpg(pid,signum)
            class Log:
                def __init__(self,stream):self.stream=stream
                def fileno(self):return self.stream.fileno()
                def close(self):
                    self.stream.close()
                    assert all(p.returncode is not None for p in processes)
                    raise OSError('Injected first log close error after all children joined')
            class Evidence:
                def __truediv__(self,name):
                    path=evidence/name
                    if kind=='first_log_close_error'and name=='left.log':
                        return types.SimpleNamespace(open=lambda mode:Log(path.open(mode)))
                    return path
            def publish(name,paths):
                assert all(p.returncode is not None for p in processes)
                published.append(name)
            scope=dict(plan=dict(transport_reserve_s=0),children=children,closed=set(),budget=100,
                began=time.monotonic(),time=time,common=Common(),evidence=Evidence(),work=folder,
                root=folder,report=report,subprocess=types.SimpleNamespace(Popen=popen,TimeoutExpired=subprocess.TimeoutExpired,STDOUT=subprocess.STDOUT),
                os=types.SimpleNamespace(environ=os.environ,killpg=killpg),signal=signal,json=json,
                progress=lambda:None,publish=publish,
                safe_error=lambda exc:dict(type=type(exc).__name__,message=str(exc)))
            exec(compile('def fixture():\n'+PAIR_SOURCE+'    return pair_stages\n',__file__+':actual-paired-parent','exec'),scope)
            pair=scope['fixture']();error=None
            try:pair(cases,Path('unused-candidate'))
            except RuntimeError as exc:error=str(exc)
            rows=report['stages'];pids=[r['pid']for r in rows]
            for pid in pids:
                try:os.kill(pid,0)
                except ProcessLookupError:continue
                raise AssertionError('Unjoined fixture child')
            assert published==['left','right'] and scope['closed']=={'left','right'}
            assert report['evaluation_peak_children']==2 and len(rows)==2
            if kind in ('both_complete','one_nonzero','one_timeout'):
                assert max(r['spawned_worker_s']for r in rows)<min(r['exit_detected_worker_s']for r in rows)
            if kind=='both_complete':assert error is None and completed==['left','right']
            elif kind in ('one_nonzero','one_timeout'):assert error and completed==['right']
            else:assert error
            if kind=='one_timeout':assert rows[0]['operator_budget_censored']
            if kind=='exit_at_signal':assert race==['first exited before signal']
            if kind=='first_log_close_error':assert 'log_close_error'in rows[0]
            results.append(dict(case=kind,rows=rows,parent_serial_publication=published,
                completed=completed,error=error,all_spawned_children_joined=True))
        raw=Path(__file__).read_bytes();paired=Path(__file__).with_name('parallel_stage.py').read_bytes()
        out.write_text(json.dumps(dict(schema='cove50.paired-evaluation-parent-fixture.v1',
            source=dict(path=str(Path(__file__).resolve()),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw)),
            paired_source_sha256=hashlib.sha256(paired).hexdigest(),cases=results,
                scope='Actual PAIR_SOURCE parent, ten tiny stdlib child processes, including unwind with sibling live, ESRCH race and isolated log-close error. No native/NN/models/checkpoints/physics/provider. Timing illustrates overlap only, not native speedup.',
            disposable_fixture_work_removed=True),indent=2)+'\n')
    finally:shutil.rmtree(root)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    main(parser.parse_args().out.resolve())
