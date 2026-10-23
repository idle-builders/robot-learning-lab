"""Existing Gradio/Rerun adapters for inspectable household task attempts."""
import json,subprocess,uuid
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def runs():
 result={}
 for path in sorted((ROOT/'artifacts').glob('household-interaction-*/report.json'),key=lambda p:p.stat().st_mtime,reverse=True):
  result[path.parent.name]=json.loads(path.read_text())
 return result


def retained(name):
 if name not in runs():raise ValueError('Choose a retained household experiment')
 return ROOT/'artifacts'/name


def launch(parent,case,distance,lateral,train=False,steps=2048):
 path=retained(parent)
 if not (path/'policy.zip').is_file():raise ValueError('This attempt has no learned checkpoint')
 if case not in {'far','near','settle','vision_loss','target_moves_closer'}:raise ValueError('Unknown trial')
 if not .34<=float(distance)<=.85 or abs(float(lateral))>.12:raise ValueError('Outside proxy test envelope')
 command=[str(ROOT/'.venv-learning/bin/python'),str(ROOT/'tools/household_experiment.py'),'--parent',str(path/'policy.zip'),
          '--mode','train' if train else 'evaluate','--steps',str(int(steps))]
 if not train:command+=['--case',case,'--distance',str(float(distance)),'--lateral',str(float(lateral))]
 logs=ROOT/'artifacts'/('household-launch-'+uuid.uuid4().hex[:12]);logs.mkdir()
 result=None
 try:
  result=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,timeout=600)
  (logs/'stdout.log').write_text(result.stdout);(logs/'stderr.log').write_text(result.stderr)
  outputs=[line for line in result.stdout.splitlines() if line.startswith(str(ROOT/'artifacts/household-interaction-'))]
  if not outputs:raise RuntimeError('Task runner stopped before publishing output; see '+str(logs))
  output=retained(Path(outputs[-1]).name)
  return output
 except subprocess.TimeoutExpired as exc:
  (logs/'stdout.log').write_bytes(exc.stdout or b'');(logs/'stderr.log').write_bytes(exc.stderr or b'')
  raise RuntimeError('Task exceeded ten-minute local budget; partial evidence retained in '+str(logs)) from None
 finally:
  from lab.artifact_store import plan_migration,apply_migration,verify_manifest
  (logs/'launch.json').write_text(json.dumps({'command':command,'exit_code':result.returncode if result else None},indent=2))
  manifest=plan_migration(ROOT,[logs]);apply_migration(ROOT,manifest);verify_manifest(ROOT,manifest)
