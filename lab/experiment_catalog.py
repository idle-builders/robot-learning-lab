"""Read-only discovery of retained experiments and local bounded launch helpers."""
from __future__ import annotations
import atexit
import json
import os
import signal
from pathlib import Path
import subprocess
import time
ROOT=Path(__file__).resolve().parents[1]
_viewer=None


def experiments():
    result={}
    for path in sorted((ROOT/'artifacts').glob('control-stack-*/report.json'),key=lambda p:p.stat().st_mtime,reverse=True):
        report=json.loads(path.read_text())
        if 'backend' in report:result[path.parent.name]=report
    return result


def retained_run(name):
    from lab.household_catalog import runs as household_runs
    if name not in experiments() and name not in household_runs():raise ValueError('Choose a retained control or household run')
    return ROOT/'artifacts'/name


def launch(kind,steps,parent=None,forward=None,yaw=0):
    choices={'Reference controller':'reference','Frozen Isaac policy in MuJoCo':'isaac',
        'Adapt motor policy with PPO':'adapt','Continue PPO checkpoint':'resume','LeRobot ACT imitation':'act',
        'Continue ACT checkpoint':'act-resume'}
    if kind not in choices:raise ValueError('Choose a supported experiment')
    mode=choices[kind]
    if mode.startswith('act'):
        command=[str(ROOT/'.venv-lerobot/bin/python'),str(ROOT/'tools/lerobot_experiment.py'),'--steps',str(int(steps))]
        if mode=='act-resume':command+=['--checkpoint',str(retained_run(parent)/'pretrained_model')]
    else:
        command=[str(ROOT/'.venv-learning/bin/python'),str(ROOT/'tools/control_experiment.py'),'--mode',mode,'--steps',str(int(steps))]
        if mode=='resume':command+=['--checkpoint',str(retained_run(parent)/'policy.zip')]
        if mode in {'reference','isaac'} and forward is not None:command+=['--forward',str(float(forward)),'--yaw',str(float(yaw))]
    # The child writes failure reports and CAS evidence even when an experiment fails.
    result=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,timeout=300)
    outputs=[line.strip() for line in result.stdout.splitlines() if line.startswith(str(ROOT/'artifacts/control-stack-'))]
    if not outputs:raise RuntimeError('Runner stopped before publishing a report: '+result.stderr[-500:])
    name=Path(outputs[-1]).name
    path=retained_run(name)
    # Keep interpreter diagnostics as separate evidence without mutating the sealed run.
    from lab.artifact_store import plan_migration,apply_migration,verify_manifest
    import uuid
    logs=ROOT/'artifacts'/('control-launch-'+uuid.uuid4().hex[:12]);logs.mkdir()
    (logs/'stdout.log').write_text(result.stdout);(logs/'stderr.log').write_text(result.stderr)
    (logs/'launch.json').write_text(json.dumps({'run':name,'command':command,'exit_code':result.returncode},indent=2))
    manifest=plan_migration(ROOT,[logs]);apply_migration(ROOT,manifest);verify_manifest(ROOT,manifest)
    return path


def stop_viewer():
    global _viewer
    if _viewer is not None and _viewer.poll() is None:
        os.killpg(_viewer.pid,signal.SIGTERM)
        try:_viewer.wait(timeout=5)
        except subprocess.TimeoutExpired:os.killpg(_viewer.pid,signal.SIGKILL);_viewer.wait()
    _viewer=None


def open_replay(name):
    return open_recording(retained_run(name)/'episode.rrd')


def open_recording(path):
    global _viewer
    path=Path(path).absolute()
    if not path.is_relative_to(ROOT/'artifacts'):raise ValueError('Recording must be retained in artifacts')
    if not path.is_file():raise ValueError('This run has no 3D replay; choose a rendered run')
    stop_viewer()
    with open('/tmp/pip-alternative-rerun.log','w') as log:
        _viewer=subprocess.Popen([str(ROOT/'.venv-observe/bin/rerun'),'--bind','127.0.0.1','--serve-web','--port','9882',
            '--web-viewer-port','9096',str(path)],stdout=log,stderr=subprocess.STDOUT,cwd=ROOT,start_new_session=True)
    time.sleep(1)
    if _viewer.poll() is not None:raise RuntimeError('Replay server could not start; inspect /tmp/pip-alternative-rerun.log')
    return 'http://127.0.0.1:9096/?url=rerun%2Bhttp%3A%2F%2F127.0.0.1%3A9882%2Fproxy'


atexit.register(stop_viewer)
