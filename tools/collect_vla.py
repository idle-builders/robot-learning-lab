"""Collect declared teacher demonstrations into native LeRobot, with causal RGB/state."""
import json
from pathlib import Path
import sys
import uuid
import time
import numpy as np
import mujoco
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lab.vla_control import VisualRobot, evaluate, FPS, INSTRUCTION, HOLD
from lab.floor_pickup import CleanupReference
from lab.artifact_store import snapshot_file, retain_dependency
from tools.probe_vla import finalize


def run():
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    out = ROOT/'artifacts'/('vla-demonstrations-'+uuid.uuid4().hex[:12])
    out.mkdir()
    print(out, flush=True)
    parent = ROOT/'artifacts/floor-learning-2f86cd18caaa'
    snapshot_file(parent/'scene.xml', out/'scene.xml')
    for name in ['tools/collect_vla.py', 'lab/vla_control.py', 'lab/floor_pickup.py', 'lab/mujoco_learning.py', 'lab/electromechanical.py', 'learning/pip-control-v1.json', 'learning/pip-electromechanical-v1.json']:
        snapshot_file(ROOT/name, out/'source'/name)
    snapshot_file(ROOT/'artifacts/floor-cleanup-preflight-18686e4733a4/report.json', out/'teacher-posture-source.json')
    (out/'dependencies.json').write_text(json.dumps(retain_dependency(parent), indent=2))
    robot = VisualRobot(out/'scene.xml')
    contract = robot.contract()
    (out/'contract.json').write_text(json.dumps(contract, indent=2))
    cases = [dict(task='cleanup', x=x, y=.14) for x in [.013,.015,.017,.019]] + [dict(task='hold', x=x, y=.14) for x in [.013,.017]]
    # Fixed evaluation cases are declared before collection/training. No held-out
    # episode is collected into the native training dataset.
    evaluation = [{'name':'nominal_cleanup','task':'cleanup','x':.016,'y':.140,'instruction':INSTRUCTION},
                  {'name':'lateral_cleanup','task':'cleanup','x':.015,'y':.143,'instruction':INSTRUCTION},
                  {'name':'hold','task':'hold','x':.016,'y':.140,'instruction':HOLD}]
    (out/'evaluation-cases.json').write_text(json.dumps(evaluation, indent=2))
    ds = LeRobotDataset.create('local/pip-visual-control', fps=FPS, root=out/'dataset', robot_type='pip-floor-reach-v1', use_videos=False,
        features={'observation.state': {'dtype':'float32','shape':(25,),'names':contract['state_names']},
                  'observation.images.head': {'dtype':'image','shape':(180,320,3),'names':['height','width','channels']},
                  'action': {'dtype':'float32','shape':(17,),'names':contract['action_names']}})
    report = {'status':'running','teacher':'Authored CleanupReference; demonstration source only. No learned task success claimed.', 'episodes':[]}
    started = time.monotonic()
    try:
        for episode, case in enumerate(cases):
            robot.reset(case['x'],case['y'])
            teacher = CleanupReference(robot.motor)
            rows = []
            alignment = []
            try:
                for step in range(2000 if case['task']=='cleanup' else 400):
                    m = robot.motor
                    sample = step % 5 == 0
                    if sample:
                        state, rgb, acquired = robot.state(), robot.image(), float(m.data.time)
                    if case['task']=='cleanup':
                        if step % 10 == 0:
                            teacher.capture()
                        target, speed, _ = teacher.command()
                        m.heading_error = teacher.yaw
                    else:
                        target, speed = m.rest.copy(), 0.
                        m.heading_error = 0.
                    if sample:
                        action = robot.encode_action(target,speed,float(np.clip(-3*m.heading_error,-.15,.15)))
                        ds.add_frame({'observation.state':state,'observation.images.head':rgb,'action':action,
                                      'task':INSTRUCTION if case['task']=='cleanup' else HOLD})
                        alignment.append({'frame':len(alignment),'acquired_s':acquired,'command_s':float(m.data.time)})
                    row = m.advance(target,speed)
                    addr = m.model.jnt_dofadr[m.model.joint('task_block_free').id]
                    row['block_velocity_m_s'] = m.data.qvel[addr:addr+3].tolist()
                    row['teacher_phase'] = teacher.phase
                    rows.append(row)
                    if case['task']=='cleanup' and teacher.phase=='done' or abs(row['pitch_rad'])>.28:
                        break
                ds.save_episode()
                result = evaluate(rows,initial=(case['x'],case['y'],.015),task=case['task'])
                result.update(case=case,frames=len(alignment),episode=episode)
                report['episodes'].append(result)
                (out/f'episode-{episode}-alignment.json').write_text(json.dumps(alignment))
                (out/f'episode-{episode}-truth.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
                print(result,flush=True)
            finally:
                teacher.close()
        report['status']='collected'
        report['training_episodes']=[r['episode'] for r in report['episodes'] if r['success']]
        # Failed teachers are retained, but excluded explicitly from imitation.
        report['excluded_failed_episodes']=[r['episode'] for r in report['episodes'] if not r['success']]
    except Exception as exc:
        import traceback
        report.update(status='fail', error_type=type(exc).__name__)
        (out/'error.log').write_text(traceback.format_exc())
        raise
    finally:
        ds.finalize()
        robot.close()
        import subprocess
        (out/'environment.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
        report['wall_s']=time.monotonic()-started
        (out/'report.json').write_text(json.dumps(report,indent=2))
        finalize(out)
    return out

if __name__=='__main__':
    print(run())
