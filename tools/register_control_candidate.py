"""Index a retained native training attempt; never promote a failed skill."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lab.artifact_store import snapshot_file

p = argparse.ArgumentParser()
p.add_argument('run', type=Path)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
evidence = a.run / 'evidence'
seal = json.loads((evidence / 'bootstrap.json').read_text())
for name, expected in seal['files'].items():
    if hashlib.sha256((evidence / name).read_bytes()).hexdigest() != expected:
        raise ValueError('Changed evidence: ' + name)
skill = evidence / 'control/camera/skills'
report = json.loads((skill / 'report.json').read_text())
checkpoint = skill / 'selector.zip'
if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != report['selector_checkpoint_sha256']:
    raise ValueError('Selector checkpoint does not match evaluation')
inputs = json.loads((a.run / 'inputs.json').read_text())
motor = a.run / 'submitted-source/control/motor.pt'
control_report=json.loads((evidence/'control/report.json').read_text())
if control_report.get('checkpoint'):
    relative=Path(control_report['checkpoint'])
    if relative.is_absolute() or '..' in relative.parts:raise ValueError('Unsafe produced checkpoint path')
    motor=evidence/'control'/relative
if hashlib.sha256(motor.read_bytes()).hexdigest() != report['motor_checkpoint_sha256']:
    raise ValueError('Motor checkpoint does not match evaluation')
request=json.loads((a.run/'submitted-source/control/request.json').read_text())
if report.get('selector_reused'):
    supplied=a.run/'submitted-source/control/selector.zip'
    if (hashlib.sha256(supplied.read_bytes()).hexdigest()!=report['selector_checkpoint_sha256']
            or request.get('selector_checkpoint_sha256')!=report['selector_checkpoint_sha256']):
        raise ValueError('Re-evaluated selector does not match retained input')
a.output.mkdir(parents=True, exist_ok=False)
for source, name in [(checkpoint, 'selector.zip'), (motor, 'motor.pt'),
                     (skill / 'report.json', 'evaluation.json'),
                     (a.run / 'inputs.json', 'inputs.json'),
                     (Path(__file__), 'register_control_candidate.py')]:
    snapshot_file(source, a.output / name)
eligible = report['evaluation_success'] and report['evaluation_falls'] == 0 and report['lost_vision_hold']
actuator_scope='Delayed DC motor/shared-pack proxy; electrical values are estimates.' if request.get('dynamics') else 'Ideal actuators.'
entry = dict(
    skill_id='pip-visible-marker-approach', origin='re-evaluation of retained native PPO selector' if report.get('selector_reused') else 'authored bounded proposal; native PPO training',
    training_steps_this_run=report['training_steps'], selector_checkpoint_origin=request.get('selector_checkpoint_origin'),
    candidate_id=report['selector_checkpoint_sha256'], input_payload_sha256=inputs['payload_sha256'],
    source_run=str(a.run.absolute()), evaluation_status='pass' if eligible else 'fail',
    eligible=eligible, checkpoint_reloaded=report['checkpoint_reloaded'],
    selector='selector.zip', motor='motor.pt', report='evaluation.json',
    observations=['visible', 'camera_forward_m', 'camera_left_m', 'episode_time_div_20'],
    discrete_skills=['hold', 'approach', 'align'],
    rates_hz=dict(physics=500, motor=50, vision=10, selector=5),
    scope='Simulation candidate only; fixed neutral head, known marker. '+actuator_scope+' No language model or hardware deployment.',
)
(a.output / 'registry.json').write_text(json.dumps(dict(schema_version=1, status='pass',
    scope='Evidence integrity and candidate indexing; entry eligibility is separate.', entries=[entry]), indent=2))
print(json.dumps(dict(status='pass', candidate_eligible=eligible, output=str(a.output))))
